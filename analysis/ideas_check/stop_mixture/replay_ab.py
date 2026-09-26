"""Closed-loop A/B of the mixture landmark file on the VAL bags with the native C++ replay.

Same configuration as tools/replay/quick_eval.py (train-only validation maps, cut-offs, branches,
GNSS only in the first 5 s), only the landmark file / one parameter changes:
  cur  : analysis/validation_maps/landmarks.csv                       (current honest-validation file)
  mix  : analysis/ideas_check/stop_mixture/landmarks_mixture_train.csv (+ train mixture rows)
  q0   : cur with landmark_assoc_q=0                                    (before the association widening)
  extra variants can be given as name=path_to_landmark_file on the command line.
The binary is a private copy of build_core/tbo_replay.exe and all temporary files live in the session
scratch directory (quick_eval writes to the shared build_core/replay_tmp). TBO_DEBUG_LM=1 makes the
estimator print every place-fix attempt (t, predicted s, association sd, candidates, p_known); the
attempts are matched with the GNSS truth s (RTK master antenna projected on the train map).
Run: python analysis/ideas_check/stop_mixture/replay_ab.py [name=landmarks.csv ...]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
sys.path.insert(0, str(ROOT / 'analysis' / 'map_build'))
sys.path.insert(0, str(HERE))
import cpp_bridge as CB  # noqa: E402
import quick_eval as QE  # noqa: E402

SCRATCH = Path(os.environ.get('SMX_SCRATCH', r'<session-scratch>'
                                             r'\53977e3d-3af2-4796-9c03-8895f50bc4dd\scratchpad'))
TMP = SCRATCH / 'replay'
EXE = SCRATCH / 'tbo_replay.exe'
LM_RE = re.compile(r'LM t=([\d.]+) s_map=([-\d.]+) sd=([\d.]+) n=(\d+) d0=([-\d.]+) p_known=([\d.]+) k=([-\d.]+)')


def run_variant(bag, name, sets):
    ev = TMP / f'{bag}_events.csv'
    out = TMP / f'{bag}_{name}_out.csv'
    base = {'output_frame': 'enu', 'base_link_along_m': 0, 'base_link_height_m': 0,
            'landmark_file': str(QE.LANDMARKS), 'cutoff_file': str(QE.CUTOFFS)}
    base.update(sets)
    cmd = [str(EXE), '--in', str(ev), '--out', str(out), '--map', str(QE.MAP),
           '--traction', str(CB.PKG / 'config' / 'traction_lut.csv'), '--branches', QE.BRANCHES]
    for k, v in base.items():
        cmd += ['--set', f'{k}={v}']
    res = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, TBO_DEBUG_LM='1'))
    if res.returncode != 0:
        raise RuntimeError(f'{bag} {name}: {res.stderr[-2000:]}')
    o = pd.read_csv(out)
    lm = sorted({m.groups() for m in LM_RE.finditer(res.stderr)}, key=lambda g: float(g[0]))
    fixes = pd.DataFrame([[float(v) for v in g] for g in lm], columns=['t', 's_map', 'sd', 'n', 'd0', 'p_known', 'k'])
    out.unlink()
    return bag, name, metrics(bag, o), o, fixes


def metrics(bag, o):
    """position part of tools/replay/quick_eval.eval_bag (same matching and along/cross definition)"""
    d = np.load(CB.NPZ / f'{bag}.npz')
    mf = d['sensing__gnss__master__fix']
    mf = mf[np.isfinite(mf[:, 2])]
    p_ref = QE.enu(mf[:, 2], mf[:, 3], mf[:, 4], mf[0, 2], mf[0, 3], mf[0, 4])
    op = o[o.pos_valid == 1] if 'pos_valid' in o.columns else o
    out_tp = op.stamp_ns.to_numpy() * 1e-9
    pick, ok = QE.match_nearest(mf[:, 1], out_tp)
    po = op[['x', 'y', 'z']].to_numpy()[pick][ok]
    yaw = op.yaw.to_numpy()[pick][ok]
    e = po - p_ref[ok]
    along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
    cross = -e[:, 0] * np.sin(yaw) + e[:, 1] * np.cos(yaw)
    e3 = np.linalg.norm(e, axis=1)
    return dict(along_rmse=float(np.sqrt(np.mean(along ** 2))), along_max=float(np.abs(along).max()),
                cross_rmse=float(np.sqrt(np.mean(cross ** 2))), p3_rmse=float(np.sqrt(np.mean(e3 ** 2))),
                end_err=float(e3[-1]), ref_rtk=float((mf[:, 5] == 2).mean()))


def truth_s(bags):
    """GNSS truth s(t) on the train main cycle: good (RTK) master fixes projected on map_train/main."""
    import runs as R
    from validate import MapProjector
    mp = MapProjector(ROOT / 'analysis' / 'map_build' / 'map_train')
    E = mp.polys['main']
    out = {}
    for b, r in R.load_runs(bags).items():
        s, dd, dist, seg, dpsi = E.project(r.x, r.y, r.psi, max_d=3.0, max_dpsi=np.radians(60))
        m = np.isfinite(s) & r.good
        out[b] = (r.th[m], s[m])
    return out, float(E.length)


def main():
    from fit_mixture import Tee
    sys.stdout = Tee(HERE / 'replay_ab_log.txt')
    splits = json.loads((ROOT / 'data' / 'splits.json').read_text())
    bags = splits['val']
    TMP.mkdir(parents=True, exist_ok=True)
    if not EXE.exists():
        shutil.copy2(ROOT / 'build_core' / 'tbo_replay.exe', EXE)
    variants = {'cur': {}, 'mix': {'landmark_file': str(HERE / 'landmarks_mixture_train.csv')},
                'q0': {'landmark_assoc_q': 0}}
    for a in sys.argv[1:]:
        k, v = a.split('=', 1)
        variants[k] = {'landmark_file': str(Path(v).resolve())}
    for b in bags:
        if not (TMP / f'{b}_events.csv').exists():
            CB.export_events(b, TMP / f'{b}_events.csv')
    jobs = [(b, n, s) for b in bags for n, s in variants.items()]
    rows, fixes, outs = [], [], {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for bag, name, met, o, fx in ex.map(lambda j: run_variant(*j), jobs):
            rows.append(dict(bag=bag, variant=name, **met))
            fx['bag'], fx['variant'] = bag, name
            fixes.append(fx)
            outs[(bag, name)] = o[['stamp_ns', 's', 's_var', 's_map']].copy()
            print(f'{bag} {name}: along_rmse {met["along_rmse"]:.3f}  fixes {len(fx)}', flush=True)
    df = pd.DataFrame(rows)
    fx = pd.concat(fixes, ignore_index=True)
    # truth at every fix attempt + the estimate right before / after it
    tru, L = truth_s(bags)
    rec = []
    for (bag, name), g in fx.groupby(['bag', 'variant']):
        tt, ss = tru[bag]
        o = outs[(bag, name)]
        ot = o.stamp_ns.to_numpy() * 1e-9
        for r in g.itertuples(index=False):
            i = np.searchsorted(tt, r.t)
            if i == 0 or i >= len(tt) or tt[i] - tt[i - 1] > 1.0:
                s_true = np.nan
            else:
                s_true = float(np.interp(r.t, tt[i - 1:i + 1], np.unwrap(ss[i - 1:i + 1], period=L)) % L)
            j0 = max(np.searchsorted(ot, r.t - 0.05) - 1, 0)          # published state just before the attempt
            j1 = min(np.searchsorted(ot, r.t + 0.5), len(ot) - 1)      # and 0.5 s later (after the update)
            rec.append(dict(bag=bag, variant=name, t=r.t, s_map=r.s_map, sd=r.sd, n=int(r.n), d0=r.d0, p_known=r.p_known,
                            accepted=bool(r.p_known >= 0.6 and r.n > 0), s_true=s_true,
                            err_pred=((r.s_map - s_true + L / 2) % L) - L / 2,
                            s_var_before=float(o.s_var.iloc[j0]), s_map_after=float(o.s_map.iloc[j1])))
    fr = pd.DataFrame(rec)
    fr['err_after'] = ((fr.s_map_after - fr.s_true + L / 2) % L) - L / 2
    fr.to_csv(HERE / 'replay_ab_fixes.csv', index=False)
    piv = df.pivot(index='bag', columns='variant', values='along_rmse')
    for v in variants:
        if v != 'cur':
            piv[f'{v}-cur'] = piv[v] - piv['cur']
    ref = df[df.variant == 'cur'].set_index('bag').ref_rtk
    piv['ref_rtk'] = ref
    ck = pd.read_csv(ROOT / 'analysis' / 'timing_reference' / 'clocks_per_bag.csv').set_index('bag')
    piv['clock_anom'] = [float(ck.gnss_vs_veh_anom_frac.get(b, 0.0)) for b in piv.index]
    piv['ref_good'] = (piv.ref_rtk > 0.8) & ~(piv.clock_anom > 0.01)   # quick_eval GOOD-REF rule
    pd.set_option('display.width', 250)
    for metric in ('along_rmse', 'p3_rmse', 'end_err'):
        pm = df.pivot(index='bag', columns='variant', values=metric)
        for v in variants:
            if v != 'cur':
                pm[f'{v}-cur'] = pm[v] - pm['cur']
        pm['ref_good'] = piv.ref_good
        print(f'\n{metric} [m] per val bag')
        print(pm.round(3).to_string())
        print(f'mean {metric}:', pm[list(variants)].mean().round(4).to_dict(),
              f'| GOOD-REF ({int(pm.ref_good.sum())} bags):', pm[pm.ref_good][list(variants)].mean().round(4).to_dict())
    df.to_csv(HERE / 'replay_ab_metrics.csv', index=False)
    # fix attempts that differ between cur and mix (same bag, same attempt time)
    a = fr[fr.variant == 'cur'].set_index(['bag', 't'])
    for v in variants:
        if v == 'cur':
            continue
        b = fr[fr.variant == v].set_index(['bag', 't'])
        j = a.join(b, how='outer', lsuffix='_cur', rsuffix=f'_{v}')
        diff = j[(j[f'n_{v}'] != j.n_cur) | ((j[f'p_known_{v}'] - j.p_known_cur).abs() > 0.01) |
                 ((j[f'err_after_{v}'] - j.err_after_cur).abs() > 0.05)]
        print(f'\nfix attempts that differ, cur vs {v}: {len(diff)} of {len(j)}')
        print(diff[['s_true_cur', 'err_pred_cur', 'sd_cur', 'n_cur', 'p_known_cur', 'err_after_cur',
                    f'err_pred_{v}', f'sd_{v}', f'n_{v}', f'p_known_{v}', f'err_after_{v}']].round(2).to_string())


if __name__ == '__main__':
    main()
