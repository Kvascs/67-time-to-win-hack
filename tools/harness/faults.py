"""Fault injection on the estimator INPUTS (criterion 3: slip / slide / dropouts / outliers).

The reference (GNSS truth) is never modified, so errors inside the fault windows measure how
well the estimator rejects bad odometry and recovers afterwards.

A fault is a dict ``{'kind', 'topic', 't0', 'dur', ...params}`` or a compact string
``"kind:topic@t0+dur[:key=val,...]"``; ``t0`` is seconds from the bag start or a percentage of
the bag duration (``"40%"``). ``topic`` in {front, rear, cmd, wheels (= front+rear), inputs}.

kinds
  dropout     messages removed
  freeze      value held at its first in-window value (stuck sensor)
  zero        value forced to 0 (sensor reads 0 while moving)
  scale       value * factor (default 1.2)
  slip        wheel spin: value * (1 + peak * tri(t)), peak default 0.4 (triangular profile)
  slide       wheel slide/lock: value * (1 - peak * tri(t)), peak default 0.6
              optional ``phase=1`` (traction only: notch > 0) / ``phase=-1`` (braking only: notch < 0)
              makes slip/slide/scale physically consistent with the driver command
  spike       with probability p (0.05) value += U(-amp, amp) (amp 20 km/h), clipped at 0
  noise       value += N(0, sigma) (sigma 1.0 km/h), clipped at 0 (notch: rounded)
  nan         value = NaN with probability p (default 1)
  delay       bag time += lag (0.3 s): messages arrive late (and possibly out of order)
  stamp_jump  header.stamp += offset (1.0 s) - clock glitch
  stamp_zero  header.stamp = 0
  dup         every message delivered twice
Suites: ``suite:basic`` (see SUITES) expands to a representative mix.
"""
from __future__ import annotations

from typing import List, Union

import numpy as np

from .loader import BagData, T_BAG, T_HDR, V_COL

TOPIC_GROUPS = {'front': ['front'], 'rear': ['rear'], 'cmd': ['cmd'], 'wheels': ['front', 'rear'],
                'inputs': ['front', 'rear', 'cmd']}

SUITES = {
    'basic': [
        'dropout:wheels@25%+3',          # both bogies silent 3 s
        'slip:front@40%+6:peak=0.4',     # front bogie spins +40 %
        'slide:rear@55%+4:peak=0.6',     # rear bogie slides -60 %
        'spike:wheels@65%+60:p=0.05:amp=20',
        'nan:front@75%+2',
        'freeze:rear@80%+10',
        'dropout:cmd@85%+10',
        'slip:wheels@90%+5:peak=0.25',   # both bogies wrong together: only the model can help
    ],
    'dropouts': ['dropout:wheels@20%+1', 'dropout:wheels@40%+5', 'dropout:wheels@60%+15',
                 'dropout:front@70%+30', 'dropout:inputs@85%+3'],
    'slip': ['slip:front@20%+5:peak=0.3', 'slip:rear@35%+5:peak=0.5', 'slide:front@50%+4:peak=0.8',
             'slide:wheels@65%+3:peak=0.5', 'slip:wheels@80%+8:peak=0.2'],
    # physically consistent: spin only under traction, slide only under braking, long stale bogie
    'realistic': [
        'slip:front@15%+8:peak=0.3:phase=1', 'slide:rear@30%+5:peak=0.5:phase=-1',
        'dropout:rear@40%+90', 'slip:wheels@60%+6:peak=0.15:phase=1', 'slide:wheels@75%+4:peak=0.4:phase=-1',
        'spike:front@85%+30:p=0.03:amp=15',
    ],
    'garbage': ['nan:wheels@20%+3', 'zero:front@35%+5', 'spike:wheels@50%+60:p=0.1:amp=40',
                'stamp_jump:wheels@65%+20:offset=1.0', 'stamp_zero:cmd@75%+5', 'dup:inputs@80%+30',
                'delay:wheels@90%+10:lag=0.3'],
}


def parse_fault(spec: Union[str, dict]) -> dict:
    if isinstance(spec, dict):
        return dict(spec)
    kind, _, rest = spec.partition(':')
    parts = rest.split(':')
    topic, _, win = parts[0].partition('@')
    t0, _, dur = win.partition('+')
    f = {'kind': kind, 'topic': topic, 't0': t0 if t0.endswith('%') else float(t0 or 0), 'dur': float(dur or 1)}
    for kv in parts[1:]:
        k, _, v = kv.partition('=')
        f[k] = float(v)
    return f


def expand(specs) -> List[dict]:
    out = []
    for s in specs or []:
        if isinstance(s, str) and s.startswith('suite:'):
            out += [parse_fault(x) for x in SUITES[s.split(':', 1)[1]]]
        else:
            out.append(parse_fault(s))
    return out


def _notch_at(bag: BagData, t: np.ndarray) -> np.ndarray:
    cmd = bag['cmd']
    if len(cmd) == 0:
        return np.zeros(len(t))
    j = np.searchsorted(cmd[:, T_BAG], t, side='right') - 1
    return np.where(j >= 0, cmd[np.clip(j, 0, None), V_COL], 0.0)


def _window(bag: BagData, f: dict):
    """Fault window [a, b] on the bag clock. With ``phase`` the start is moved to the first
    moment at/after t0 where the notch has the requested sign and the tram moves (> 1 m/s)."""
    t0 = f['t0']
    if isinstance(t0, str):
        t0 = float(t0.rstrip('%')) / 100.0 * bag.duration
    a = bag.t_start + float(t0)
    if 'phase' in f and len(bag['front']):
        w = bag['front']
        cand = (w[:, T_BAG] >= a) & (w[:, V_COL] > 3.6)
        notch = _notch_at(bag, w[:, T_BAG])
        cand &= (notch > 0) if f['phase'] > 0 else (notch < 0)
        idx = np.flatnonzero(cand)
        if len(idx):
            a = float(w[idx[0], T_BAG])
    return a, a + float(f['dur'])


def apply_faults(bag: BagData, specs, seed: int = 0) -> BagData:
    """Return a modified copy of ``bag`` (GNSS topics untouched)."""
    rng = np.random.default_rng(seed)
    out = bag.copy()
    out.meta['faults'] = []
    for f in expand(specs):
        a, b = _window(out, f)
        out.meta['faults'].append({**f, 't_start': a, 't_end': b})
        for topic in TOPIC_GROUPS[f['topic']]:
            arr = out.topics[topic]
            if len(arr) == 0:
                continue
            m = (arr[:, T_BAG] >= a) & (arr[:, T_BAG] <= b)
            if not np.any(m):
                continue
            k = f['kind']
            is_cmd = topic == 'cmd'
            if k == 'dropout':
                arr = arr[~m]
            elif k == 'freeze':
                arr[m, V_COL] = arr[np.flatnonzero(m)[0], V_COL]
            elif k == 'zero':
                arr[m, V_COL] = 0.0
            elif k == 'scale':
                arr[m, V_COL] *= f.get('factor', 1.2)
            elif k in ('slip', 'slide'):
                if 'phase' in f:
                    notch = _notch_at(bag, arr[:, T_BAG])
                    m &= (notch > 0) if f['phase'] > 0 else (notch < 0)
                    if not np.any(m):
                        continue
                u = (arr[m, T_BAG] - a) / max(b - a, 1e-9)
                tri = 1.0 - np.abs(2.0 * u - 1.0)
                peak = f.get('peak', 0.4 if k == 'slip' else 0.6)
                arr[m, V_COL] *= (1.0 + peak * tri) if k == 'slip' else (1.0 - peak * tri)
            elif k == 'spike':
                hit = m & (rng.random(len(arr)) < f.get('p', 0.05))
                amp = f.get('amp', 20.0 if not is_cmd else 10.0)
                arr[hit, V_COL] += rng.uniform(-amp, amp, hit.sum())
                if is_cmd:
                    arr[hit, V_COL] = np.clip(np.round(arr[hit, V_COL]), -15, 15)
                else:
                    arr[hit, V_COL] = np.maximum(arr[hit, V_COL], 0.0)
            elif k == 'noise':
                arr[m, V_COL] += rng.normal(0.0, f.get('sigma', 1.0), m.sum())
                arr[m, V_COL] = np.clip(np.round(arr[m, V_COL]), -15, 15) if is_cmd else np.maximum(arr[m, V_COL], 0.0)
            elif k == 'nan':
                hit = m & (rng.random(len(arr)) < f.get('p', 1.0))
                arr[hit, V_COL] = 127.0 if is_cmd else np.nan   # int8 notch cannot be NaN: garbage
            elif k == 'delay':
                arr[m, T_BAG] += f.get('lag', 0.3)
            elif k == 'stamp_jump':
                arr[m, T_HDR] += f.get('offset', 1.0)
            elif k == 'stamp_zero':
                arr[m, T_HDR] = 0.0
            elif k == 'dup':
                arr = np.vstack([arr, arr[m]])
            else:
                raise ValueError(f'unknown fault kind {k!r}')
            if k in ('delay', 'dup'):
                arr = arr[np.argsort(arr[:, T_BAG], kind='stable')]
            out.topics[topic] = arr
    return out
