"""Causal replay of a bag into an estimator + full evaluation of one bag / many bags.

Estimator API (duck-typed; every method is optional, a fresh instance is created per bag)
-----------------------------------------------------------------------------------------
    on_front(stamp, v_kmh, t_bag)            front bogie speed [km/h]   (~10 Hz)
    on_rear(stamp, v_kmh, t_bag)             rear bogie speed  [km/h]   (~10 Hz)
    on_cmd(stamp, notch, t_bag)              driver controller notch, int -15..15 (~20 Hz)
    on_gnss_fix(stamp, lat, lon, alt, status, t_bag, antenna)   only first N s ('master'/'rover')
    on_gnss_vel(stamp, vx, vy, vz, t_bag, antenna)              only first N s, ENU [m/s]
    on_tick(t_bag)                           timer at EvalConfig.tick_hz (if > 0)

``stamp`` is the message header.stamp [s], ``t_bag`` the bag receive time (= sim clock under
``ros2 bag play --clock``). Messages arrive strictly in bag-time order. Each callback may return
None, one output or a list of outputs. An output is an :class:`Output` (or a tuple
``(stamp, v_mps, x, y, z)``, or a dict with those keys). ``stamp`` is what the node would write
into header.stamp of /result/velocity and /result/position.

Constructor: ``Estimator(**params)``. If the constructor accepts ``frame`` it receives the
reference frame kind ('enu'/'utm'); if the class sets ``needs_oracle_ref = True`` it receives
``oracle_ref=<Reference>`` (cheating - for upper-bound baselines only).
"""
from __future__ import annotations

import importlib
import inspect
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Union

import numpy as np

from .loader import BagData, load_bag, T_BAG, T_HDR, V_COL, LAT, LON, ALT, STATUS, VX, VY, VZ
from .reference import RefConfig, Reference, build_reference
from . import metrics as M


class Output(NamedTuple):
    stamp: float
    v: float
    x: float
    y: float
    z: float
    frame_id: Optional[str] = 'map'
    cov: Optional[tuple] = None        # (var_x, var_y, var_z) [m^2]
    v_var: Optional[float] = None      # speed variance [m^2/s^2]
    slip: Optional[float] = None       # slip flag / probability (diagnostic)
    yaw: Optional[float] = None        # heading [rad] in the frame (0 = +x, CCW) -> Odometry orientation


@dataclass
class EvalConfig:
    gnss_seconds: float = 5.0          # GNSS delivered only while t_bag - t_start <= gnss_seconds
    gnss_mode: str = 'first_n'         # 'first_n' | 'none' | 'all' (debug only)
    gnss_topics: tuple = ('fix_master', 'fix_rover', 'vel_master', 'vel_rover')
    tick_hz: float = 0.0               # > 0: call on_tick(t_bag) at this rate (bag clock)
    tol: float = 0.05                  # nearest-stamp matching tolerance [s]
    direction: str = 'ref2out'         # 'ref2out' | 'out2ref'
    ref: RefConfig = field(default_factory=RefConfig)
    faults: Optional[list] = None      # fault injection specs, see faults.py
    fault_seed: int = 0

    def to_dict(self):
        d = asdict(self)
        return d


# ----------------------------------------------------------------------------------------------
# Estimator construction
# ----------------------------------------------------------------------------------------------
def resolve_factory(spec: Union[str, Callable]) -> Callable:
    """'package.module:ClassName' (or 'path/to/file.py:ClassName') -> class/callable."""
    if not isinstance(spec, str):
        return spec
    mod, _, name = spec.rpartition(':') if spec.count(':') > 1 else spec.partition(':')
    if mod.endswith('.py'):
        from importlib import util as _util
        from pathlib import Path
        p = Path(mod).resolve()
        s = _util.spec_from_file_location(p.stem, p)
        m = _util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, name)


def make_estimator(factory: Callable, params: dict = None, frame: str = 'enu', ref: Reference = None):
    params = dict(params or {})
    try:
        sig = inspect.signature(factory)
        accepts = sig.parameters
        var_kw = any(p.kind == p.VAR_KEYWORD for p in accepts.values())
    except (TypeError, ValueError):
        accepts, var_kw = {}, False
    if ('frame' in accepts or var_kw) and 'frame' not in params:
        params['frame'] = frame
    if getattr(factory, 'needs_oracle_ref', False):
        params['oracle_ref'] = ref
    return factory(**params)


# ----------------------------------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------------------------------
K_FRONT, K_REAR, K_CMD, K_FIXM, K_FIXR, K_VELM, K_VELR, K_TICK = range(8)
_KIND_TOPIC = {K_FRONT: 'front', K_REAR: 'rear', K_CMD: 'cmd', K_FIXM: 'fix_master', K_FIXR: 'fix_rover',
               K_VELM: 'vel_master', K_VELR: 'vel_rover'}


def build_events(bag: BagData, cfg: EvalConfig):
    """Merged event list sorted by bag time: arrays (t_bag, kind, row)."""
    ts, ks, rs = [], [], []
    for k, topic in _KIND_TOPIC.items():
        a = bag[topic]
        if len(a) == 0:
            continue
        rows = np.arange(len(a))
        if topic.startswith(('fix', 'vel')):
            if cfg.gnss_mode == 'none' or topic not in cfg.gnss_topics:
                continue
            if cfg.gnss_mode == 'first_n':
                rows = rows[a[:, T_BAG] - bag.t_start <= cfg.gnss_seconds]
        ts.append(a[rows, T_BAG])
        ks.append(np.full(len(rows), k, np.int8))
        rs.append(rows)
    if cfg.tick_hz and cfg.tick_hz > 0:
        tt = np.arange(bag.t_start, bag.t_end + 1e-9, 1.0 / cfg.tick_hz)
        ts.append(tt)
        ks.append(np.full(len(tt), K_TICK, np.int8))
        rs.append(np.arange(len(tt)))
    t = np.concatenate(ts)
    k = np.concatenate(ks)
    r = np.concatenate(rs)
    o = np.lexsort((k, t))          # by time, ties: inputs before GNSS before ticks
    return t[o], k[o], r[o]


def _as_list(ret):
    if ret is None:
        return ()
    if isinstance(ret, list):
        return ret
    return (ret,)


def _unpack(o):
    if isinstance(o, dict):
        return (o['stamp'], o['v'], o['x'], o['y'], o['z'], o.get('frame_id', 'map'), o.get('cov'),
                o.get('v_var'), o.get('slip'), o.get('yaw'))
    t = tuple(o)
    if len(t) >= 10:
        return t[:10]
    return t + (None,) * (10 - len(t))     # plain (stamp, v, x, y, z): frame_id unknown


def replay(bag: BagData, estimator, cfg: EvalConfig = None) -> M.OutputLog:
    """Stream the bag into ``estimator`` in bag-time order and record everything it publishes."""
    cfg = cfg or EvalConfig()
    t_ev, k_ev, r_ev = build_events(bag, cfg)
    lists = {kk: bag[tp].tolist() for kk, tp in _KIND_TOPIC.items()}
    h = {
        K_FRONT: getattr(estimator, 'on_front', None),
        K_REAR: getattr(estimator, 'on_rear', None),
        K_CMD: getattr(estimator, 'on_cmd', None),
        K_FIXM: getattr(estimator, 'on_gnss_fix', None),
        K_FIXR: getattr(estimator, 'on_gnss_fix', None),
        K_VELM: getattr(estimator, 'on_gnss_vel', None),
        K_VELR: getattr(estimator, 'on_gnss_vel', None),
        K_TICK: getattr(estimator, 'on_tick', None),
    }
    perf = time.perf_counter
    recs = []
    proc = np.zeros(len(t_ev))
    called = np.zeros(len(t_ev), bool)
    kl = k_ev.tolist()
    rl = r_ev.tolist()
    tl = t_ev.tolist()
    for e in range(len(kl)):
        k = kl[e]
        fn = h[k]
        if fn is None:
            continue
        r = rl[e]
        if k == K_TICK:
            args = (tl[e],)
        else:
            row = lists[k][r]
            if k <= K_REAR:
                args = (row[T_HDR], row[V_COL], row[T_BAG])
            elif k == K_CMD:
                args = (row[T_HDR], int(row[V_COL]), row[T_BAG])
            elif k in (K_FIXM, K_FIXR):
                args = (row[T_HDR], row[LAT], row[LON], row[ALT], int(row[STATUS]), row[T_BAG],
                        'master' if k == K_FIXM else 'rover')
            else:
                args = (row[T_HDR], row[VX], row[VY], row[VZ], row[T_BAG], 'master' if k == K_VELM else 'rover')
        t0 = perf()
        ret = fn(*args)
        dt = perf() - t0
        proc[e] = dt
        called[e] = True
        if ret is not None:
            for o in _as_list(ret):
                recs.append((_unpack(o), tl[e], dt))
    return _to_log(recs, proc[called])


def _to_log(recs, proc_all) -> M.OutputLog:
    n = len(recs)
    stamp = np.empty(n)
    v = np.empty(n)
    xyz = np.empty((n, 3))
    emit = np.empty(n)
    ep = np.empty(n)
    fid = []
    has_cov = any(r[0][6] is not None for r in recs)
    has_vv = any(r[0][7] is not None for r in recs)
    has_slip = any(r[0][8] is not None for r in recs)
    has_yaw = any(r[0][9] is not None for r in recs)
    yaw = np.full(n, np.nan) if has_yaw else None
    cov = np.full((n, 3), np.nan) if has_cov else None
    vv = np.full(n, np.nan) if has_vv else None
    sl = np.full(n, np.nan) if has_slip else None
    for i, (o, te, dt) in enumerate(recs):
        stamp[i] = o[0] if o[0] is not None else np.nan
        v[i] = o[1] if o[1] is not None else np.nan
        xyz[i, 0] = o[2] if o[2] is not None else np.nan
        xyz[i, 1] = o[3] if o[3] is not None else np.nan
        xyz[i, 2] = o[4] if o[4] is not None else np.nan
        emit[i] = te
        ep[i] = dt
        fid.append(o[5])
        if has_cov and o[6] is not None:
            c = np.asarray(o[6], float).ravel()
            cov[i] = c[:3] if len(c) >= 3 else np.r_[c, [np.nan] * (3 - len(c))]
        if has_vv and o[7] is not None:
            vv[i] = o[7]
        if has_slip and o[8] is not None:
            sl[i] = float(o[8])
        if has_yaw and o[9] is not None:
            yaw[i] = float(o[9])
    return M.OutputLog(stamp=stamp, v=v, xyz=xyz, emit_tbag=emit, emit_proc=ep, frame_id=fid, cov=cov,
                       v_var=vv, slip=sl, yaw=yaw, proc_all=np.asarray(proc_all), n_callbacks=int(len(proc_all)))


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def score_log(bag: BagData, ref: Reference, log: M.OutputLog, cfg: EvalConfig, eval_bag: BagData = None) -> dict:
    """All metrics for one replay. ``eval_bag`` is the (possibly fault-injected) bag the estimator
    saw - used for the naive-odometry comparison in robustness metrics; ``bag`` is the clean one."""
    eval_bag = eval_bag if eval_bag is not None else bag
    regimes = M.speed_regimes(ref, bag)
    anomalies = M.anomaly_regimes(ref, eval_bag)
    res = {
        'bag': bag.name, 'vehicle': bag.vehicle, 'duration_s': bag.duration,
        'ref_diag': ref.diag,
        'speed': M.speed_metrics(ref, log, cfg.tol, cfg.direction, regimes),
        'pos': M.position_metrics(ref, log, cfg.tol, cfg.direction),
        'robust': M.robustness_metrics(ref, eval_bag, log, cfg.tol, cfg.direction, anomalies),
        'rt': M.realtime_metrics(log, bag),
        'valid': M.validity_metrics(log, bag),
    }
    cov = M.covariance_metrics(ref, log, cfg.tol, cfg.direction)
    if cov:
        res['cov'] = cov
    hd = M.heading_metrics(ref, log, cfg.tol, cfg.direction)
    if hd:
        res['heading'] = hd
    res['summary'] = M.headline(res)
    return res


def evaluate_bag(bag_name: str, factory: Union[str, Callable], params: dict = None, cfg: EvalConfig = None,
                 keep_log: bool = False) -> dict:
    """Replay + score one bag. Returns a JSON-serialisable dict (plus '_log'/'_ref' if keep_log)."""
    cfg = cfg or EvalConfig()
    try:
        bag = load_bag(bag_name)
        try:
            ref = build_reference(bag, cfg.ref)
        except ValueError:            # no GNSS in this bag: replay anyway (crash / rate / validity checks)
            ref = None
        run_bag = bag
        if cfg.faults:
            from .faults import apply_faults
            run_bag = apply_faults(bag, cfg.faults, seed=cfg.fault_seed)
        est = make_estimator(resolve_factory(factory), params, cfg.ref.frame, ref)
        t0 = time.perf_counter()
        log = replay(run_bag, est, cfg)
        wall = time.perf_counter() - t0
        if ref is None:
            res = {'bag': bag.name, 'vehicle': bag.vehicle, 'duration_s': bag.duration, 'no_reference': True,
                   'rt': M.realtime_metrics(log, bag), 'valid': M.validity_metrics(log, bag)}
            res['summary'] = M.headline(res)
        else:
            res = score_log(bag, ref, log, cfg, run_bag)
        res['replay_wall_s'] = wall
        if keep_log:
            res['_log'] = log
            res['_ref'] = ref
            res['_bag'] = run_bag
        return res
    except Exception as ex:  # keep going on other bags, but report loudly
        return {'bag': bag_name, 'error': f'{type(ex).__name__}: {ex}', 'traceback': traceback.format_exc()}


def _eval_worker(args):
    bag_name, factory, params, cfg = args
    return evaluate_bag(bag_name, factory, params, cfg)


def evaluate_many(bags: Sequence[str], factory: Union[str, Callable], params: dict = None,
                  cfg: EvalConfig = None, jobs: int = 1) -> dict:
    """Evaluate several bags (in parallel if jobs > 1; ``factory`` must then be a 'module:Class'
    string or an importable top-level class). Returns {'config', 'bags': [...], 'aggregate'}."""
    cfg = cfg or EvalConfig()
    tasks = [(b, factory, params, cfg) for b in bags]
    if jobs > 1 and len(bags) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_eval_worker, tasks))
    else:
        results = [_eval_worker(t) for t in tasks]
    return {'config': cfg.to_dict(), 'estimator': factory if isinstance(factory, str) else repr(factory),
            'params': params or {}, 'bags': results, 'aggregate': M.aggregate(results)}
