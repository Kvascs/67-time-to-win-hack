"""Where does the D33 (bogie-ratio map corrections) speed-error increase come from?

Runs the rm3 replay on the 17 val bags twice (D33 on / ratio_enable=0) with the exact quick_eval settings
(validation maps, speed/position output delay 0), collects the accepted corrections from stderr
(TBO_DEBUG_RATIO=1) and compares the speed error vs GNSS Doppler (horizontal |v| of master/vel, nearest
output within 0.05 s) inside/outside post-correction windows and as a time profile after each fix.

usage: python d33_speed.py [--reuse]      (--reuse: skip replays whose outputs already exist)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools' / 'replay'))
import cpp_bridge  # noqa: E402

cpp_bridge.REPLAY_EXE = Path(__file__).resolve().parents[3] / 'build_core_rm3' / 'tbo_replay.exe'
import quick_eval as qe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TMP = ROOT / 'build_core' / 'd33_speed_tmp'
qe.TMP = TMP
OUT_DIR = Path(__file__).resolve().parent
V = str(ROOT / 'analysis' / 'validation_maps')
SETS = {'wheel_epochs_file': V + r'\wheel_epochs.csv', 'ratio_map_file': V + r'\ratio_map.csv',
        'speed_output_delay_s': '0', 'position_output_delay_s': '0'}
VARIANTS = {'on': {}, 'off': {'ratio_enable': '0'}}
EXTRA = {'uk': {'ratio_update_k': '1'}}   # --uk: D33 with the correction allowed to move k through P(k,s)
RE_FIX = re.compile(r'RATIO t=([-\d.]+) s_map=([-\d.]+) d=([-+\d.]+) sig=([-\d.e+]+) margin=([-\d.e+]+) n=(\d+) sd_s=([-\d.e+]+)')
W_PRE, W_POST = 5.0, 15.0
BINS = [0, 1, 2, 3, 5, 10, 15, 20]


def run(bag: str, var: str, reuse: bool):
    ev = TMP / f'{bag}_events.csv'
    out = TMP / f'{bag}_{var}_out.csv'
    err = TMP / f'{bag}_{var}_stderr.txt'
    if reuse and out.exists() and err.exists():
        return
    sets = {'output_frame': 'enu', 'base_link_along_m': 0, 'base_link_height_m': 0,
            'landmark_file': str(qe.LANDMARKS), 'cutoff_file': str(qe.CUTOFFS), **SETS, **{**VARIANTS, **EXTRA}[var]}
    o = cpp_bridge.run_replay(ev, out, map_csv=qe.MAP, traction_csv=cpp_bridge.PKG / 'config' / 'traction_lut.csv',
                              sets=sets, branches=qe.BRANCHES)
    err.write_text(o.attrs.get('stderr', ''), encoding='utf-8')


def load(bag: str, var: str):
    o = pd.read_csv(TMP / f'{bag}_{var}_out.csv',
                    usecols=['stamp_ns', 'v', 's', 'd', 'k', 'g', 'a_model', 'yaw', 's_map', 'mu3'])
    fixes = [tuple(float(x) for x in m.groups()) for m in RE_FIX.finditer((TMP / f'{bag}_{var}_stderr.txt').read_text())]
    fx = pd.DataFrame(fixes, columns=['t', 's_map', 'd', 'sig', 'margin', 'n', 'sd_s'])
    return o, fx


def speed_samples(bag, o_on, o_off):
    d = np.load(cpp_bridge.NPZ / f'{bag}.npz')
    mv = d['sensing__gnss__master__vel']
    ref_t, ref_v = mv[:, 1], np.hypot(mv[:, 2], mv[:, 3])
    res = {}
    for name, o in (('on', o_on), ('off', o_off)):
        out_t = o.stamp_ns.to_numpy() * 1e-9
        pick, ok = qe.match_nearest(ref_t, out_t)
        res[name] = (pick, ok)
    ok = res['on'][1] & res['off'][1]
    t = ref_t[ok]
    s = pd.DataFrame({'t': t, 'ref': ref_v[ok]})
    for name, o in (('on', o_on), ('off', o_off)):
        p = res[name][0][ok]
        s[f'v_{name}'] = o.v.to_numpy()[p]
        for c in ('s', 'd', 'k', 'g', 'a_model'):
            s[f'{c}_{name}'] = o[c].to_numpy()[p]
    s['e_on'] = s.v_on - s.ref
    s['e_off'] = s.v_off - s.ref
    # per-sample rmse on the full match of each variant (exactly like quick_eval)
    full = {}
    for name, o in (('on', o_on), ('off', o_off)):
        pick, okk = res[name]
        e = o.v.to_numpy()[pick][okk] - ref_v[okk]
        full[name] = float(np.sqrt(np.mean(e ** 2)))
    return s, full


def curvature_at(o, t_fix):
    t = o.stamp_ns.to_numpy() * 1e-9
    m = (t > t_fix - 3) & (t < t_fix + 3)
    if m.sum() < 3:
        return np.nan
    yaw = np.unwrap(o.yaw.to_numpy()[m])
    ds = o.s.to_numpy()[m][-1] - o.s.to_numpy()[m][0]
    return float(abs(yaw[-1] - yaw[0]) / ds) if ds > 2.0 else np.nan


def main():
    reuse = '--reuse' in sys.argv
    t0 = time.time()
    TMP.mkdir(parents=True, exist_ok=True)
    os.environ['TBO_DEBUG_RATIO'] = '1'
    bags = json.load(open(ROOT / 'data' / 'splits.json'))['val']
    for b in bags:
        if not (reuse and (TMP / f'{b}_events.csv').exists()):
            cpp_bridge.export_events(b, TMP / f'{b}_events.csv')
    print(f'exported {len(bags)} bags in {time.time() - t0:.0f} s', flush=True)
    jobs = [(b, v) for b in bags for v in (list(VARIANTS) + (list(EXTRA) if '--uk' in sys.argv else []))]
    with ThreadPoolExecutor(max_workers=2) as ex:   # at most 2 replay processes
        list(ex.map(lambda j: run(j[0], j[1], reuse), jobs))
    print(f'replays done in {time.time() - t0:.0f} s', flush=True)

    lines = []
    P = lambda *a: lines.append(' '.join(str(x) for x in a))  # noqa: E731
    per_bag, allS, allF = [], [], []
    for b in bags:
        o_on, fx = load(b, 'on')
        o_off, fx_off = load(b, 'off')
        s, full = speed_samples(b, o_on, o_off)
        fx['kappa'] = [curvature_at(o_on, t) for t in fx.t]
        fx['bag'] = b
        # windows around the fixes of the ON run (same windows applied to the OFF run)
        tf = fx.t.to_numpy()
        in_w = np.zeros(len(s), bool)
        for t in tf:
            in_w |= (s.t.to_numpy() >= t - W_PRE) & (s.t.to_numpy() <= t + W_POST)
        idx = np.searchsorted(tf, s.t.to_numpy(), side='right') - 1   # last fix at or before the sample
        s['since'] = np.where(idx >= 0, s.t.to_numpy() - tf[np.clip(idx, 0, None)] if len(tf) else np.inf, np.inf)
        s['last_absd'] = np.where(idx >= 0, np.abs(fx.d.to_numpy()[np.clip(idx, 0, None)]) if len(tf) else np.nan, np.nan)
        s['last_kappa'] = np.where(idx >= 0, fx.kappa.to_numpy()[np.clip(idx, 0, None)] if len(tf) else np.nan, np.nan)
        s['in_w'] = in_w
        s['bag'] = b
        s['moving'] = s.ref > 0.5
        per_bag.append({'bag': b, 'rmse_on': full['on'], 'rmse_off': full['off'], 'n_fix': len(fx),
                        'n_fix_off_run': len(fx_off), 'cover_w': float(in_w.mean()),
                        'dsse': float(((s.e_on ** 2) - (s.e_off ** 2)).sum()),
                        'dsse_in': float(((s.e_on ** 2) - (s.e_off ** 2))[in_w].sum()), 'n': len(s)})
        allS.append(s)
        allF.append(fx)
    S = pd.concat(allS, ignore_index=True)
    F = pd.concat(allF, ignore_index=True)
    pb = pd.DataFrame(per_bag)
    pd.set_option('display.width', 220)

    P('# D33 speed check (rm3 binary, val 17 bags, speed/position output delay 0, validation maps)')
    P(f'windows: [t_fix - {W_PRE:.0f} s, t_fix + {W_POST:.0f} s] around every accepted correction of the ON run\n')
    P('## A) per-bag speed RMSE vs Doppler (quick_eval matching)')
    pb['diff'] = pb.rmse_on - pb.rmse_off
    P(pb[['bag', 'rmse_off', 'rmse_on', 'diff', 'n_fix', 'cover_w', 'dsse', 'dsse_in']].round(4).to_string(index=False))
    P(f'median off {pb.rmse_off.median():.4f}  on {pb.rmse_on.median():.4f} | mean off {pb.rmse_off.mean():.4f}  '
      f'on {pb.rmse_on.mean():.4f} | bags worse {int((pb["diff"] > 0).sum())}/{len(pb)}')
    P(f'fixes total {len(F)}; |d| < 0.5 m: {int((F.d.abs() < 0.5).sum())}, >= 0.5 m: {int((F.d.abs() >= 0.5).sum())}; '
      f'median |d| {F.d.abs().median():.2f} m\n')

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float('nan')

    def row(name, m):
        e_on, e_off = S.e_on[m], S.e_off[m]
        dsse = float((e_on ** 2 - e_off ** 2).sum())
        return (f'{name:<34} n={int(m.sum()):6d}  rmse_off {rms(e_off):.4f}  rmse_on {rms(e_on):.4f}  '
                f'd_rmse {rms(e_on) - rms(e_off):+.4f}  dSSE {dsse:+8.2f} ({100 * dsse / tot:+6.1f}% of total)  '
                f'rms(v_on-v_off) {rms(S.v_on[m] - S.v_off[m]):.4f}')

    tot = float((S.e_on ** 2 - S.e_off ** 2).sum())
    P('## B) pooled samples; dSSE = sum(e_on^2 - e_off^2); share = dSSE / total dSSE')
    P(f'total dSSE {tot:+.2f} over {len(S)} samples; window coverage {S.in_w.mean():.3f}; moving share {S.moving.mean():.3f}')
    allm = np.ones(len(S), bool)
    P(row('all', allm))
    P(row('inside windows', S.in_w.to_numpy()))
    P(row('outside windows', ~S.in_w.to_numpy()))
    P(row('outside windows, moving', (~S.in_w & S.moving).to_numpy()))
    P(row('before first fix', np.isinf(S.since).to_numpy()))
    P('-- by time since the last fix (ON-run fixes)')
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        P(row(f'since in [{lo},{hi}) s', ((S.since >= lo) & (S.since < hi)).to_numpy()))
    P(row('since >= 20 s', ((S.since >= 20) & np.isfinite(S.since)).to_numpy()))
    P('-- post-fix [0,15) s split by the last fix |d|')
    post = ((S.since >= 0) & (S.since < 15)).to_numpy()
    P(row('|d| < 0.5 m', post & (S.last_absd < 0.5).to_numpy()))
    P(row('|d| >= 0.5 m', post & (S.last_absd >= 0.5).to_numpy()))
    kmed = float(np.nanmedian(F.kappa))
    P(f'-- post-fix [0,15) s split by curvature at the fix (|dyaw/ds| over +-3 s; median {kmed:.5f} 1/m)')
    P(row('kappa < 0.002 1/m (straight)', post & (S.last_kappa < 0.002).to_numpy()))
    P(row('kappa >= 0.002 1/m (curve)', post & (S.last_kappa >= 0.002).to_numpy()))
    P('-- moving only')
    P(row('inside windows, moving', (S.in_w & S.moving).to_numpy()))
    P('')

    # C) time profile of v_on - v_off around each fix (output rows of both runs, 0.1 s grid)
    P('## C) time profile around each fix: mean over fixes of quantities at t_fix + tau (Doppler-matched samples)')
    P('   dv = v_on - v_off; ddv = dv(tau) - dv(-0.5..0 s mean) (the fix\'s own increment); |ds| = |s_on - s_off|;')
    P('   dd, dg, dam = on-off difference of the disturbance state d, grade state g, model accel a_model')
    taus = np.arange(-5, 20.0001, 1.0)
    recs = []
    for b, fxb in F.groupby('bag'):
        sb = S[S.bag == b]
        t = sb.t.to_numpy()
        dv = (sb.v_on - sb.v_off).to_numpy()
        for r in fxb.itertuples():
            pre = (t >= r.t - 0.5) & (t < r.t)
            if not pre.any():
                continue
            dv0 = dv[pre].mean()
            for tau in taus:
                m = (t >= r.t + tau) & (t < r.t + tau + 1.0)
                if not m.any():
                    continue
                recs.append({'tau': tau, 'absd': abs(r.d), 'dv': dv[m].mean(), 'ddv': dv[m].mean() - dv0,
                             'eon2': (sb.e_on.to_numpy()[m] ** 2).mean(), 'eoff2': (sb.e_off.to_numpy()[m] ** 2).mean(),
                             'ds': np.abs(sb.s_on.to_numpy()[m] - sb.s_off.to_numpy()[m]).mean(),
                             'dd': (sb.d_on - sb.d_off).to_numpy()[m].mean(),
                             'dg': (sb.g_on - sb.g_off).to_numpy()[m].mean(),
                             'dam': (sb.a_model_on - sb.a_model_off).to_numpy()[m].mean(),
                             'sgn_ddv': np.sign(r.d) * (dv[m].mean() - dv0)})
    R = pd.DataFrame(recs)
    prof = R.groupby('tau').agg(n=('dv', 'size'), mean_ddv=('ddv', 'mean'), rms_ddv=('ddv', lambda x: rms(x)),
                                sgn_ddv=('sgn_ddv', 'mean'), rms_dv=('dv', lambda x: rms(x)),
                                eon2=('eon2', 'mean'), eoff2=('eoff2', 'mean'), ds=('ds', 'mean'),
                                rms_dd=('dd', lambda x: rms(x)), rms_dg=('dg', lambda x: rms(x)),
                                rms_dam=('dam', lambda x: rms(x)))
    prof['eon2-eoff2'] = prof.eon2 - prof.eoff2
    P(prof.round(5).to_string())
    P('\n-- same, fixes with |d| >= 0.5 m only')
    Rb = R[R.absd >= 0.5]
    if len(Rb):
        p2 = Rb.groupby('tau').agg(n=('dv', 'size'), mean_ddv=('ddv', 'mean'), rms_ddv=('ddv', lambda x: rms(x)),
                                   sgn_ddv=('sgn_ddv', 'mean'), eon2=('eon2', 'mean'), eoff2=('eoff2', 'mean'))
        p2['eon2-eoff2'] = p2.eon2 - p2.eoff2
        P(p2.round(5).to_string())
    P('')
    # D) mechanism: the wheel scale k. The wheels measure z = (1+k) v, so with the same wheels a different k
    # gives v_on - v_off = -v dk / (1+k_on). k moves only at place (landmark) fixes through P(k,s) (and the
    # quantum fuse); the ratio fix moves s (and shrinks P(k,s) by the Joseph form), so the next place fix sees a
    # different innovation and gain -> different k -> different v until the next place fix.
    P('## D) mechanism check: speed difference explained by the wheel-scale difference dk = k_on - k_off')
    dv_all = (S.v_on - S.v_off).to_numpy()
    dk_all = (S.k_on - S.k_off).to_numpy()
    dv_k = -S.v_off.to_numpy() * dk_all / (1.0 + S.k_on.to_numpy())
    res_k = dv_all - dv_k
    e_off = S.e_off.to_numpy()
    for name, m in (('all', np.ones(len(S), bool)), ('inside windows', S.in_w.to_numpy()),
                    ('outside windows', ~S.in_w.to_numpy())):
        cross = float(np.sum(2 * e_off[m] * dv_all[m]))
        sq = float(np.sum(dv_all[m] ** 2))
        dsse_k = float(np.sum((e_off[m] + dv_k[m]) ** 2 - e_off[m] ** 2))
        dsse_r = float(np.sum((e_off[m] + res_k[m]) ** 2 - e_off[m] ** 2))
        c = np.corrcoef(dv_all[m], dv_k[m])[0, 1]
        P(f'{name:<16} dSSE {cross + sq:+7.2f} = cross 2*e_off*dv {cross:+7.2f} + dv^2 {sq:+7.2f} | '
          f'rms dv {rms(dv_all[m]):.4f}  rms dv_k {rms(dv_k[m]):.4f}  rms(dv - dv_k) {rms(res_k[m]):.4f}  '
          f'corr(dv, dv_k) {c:.3f} | dSSE from dv_k only {dsse_k:+7.2f}, from the rest {dsse_r:+7.2f}  '
          f'rms dk {rms(dk_all[m]) * 1e4:.2f}e-4')
    # time profile of dk around the fixes
    recs = []
    for b, fxb in F.groupby('bag'):
        sb = S[S.bag == b]
        t = sb.t.to_numpy()
        dk = (sb.k_on - sb.k_off).to_numpy()
        dvb = (sb.v_on - sb.v_off).to_numpy()
        dvk = -sb.v_off.to_numpy() * dk / (1.0 + sb.k_on.to_numpy())
        for r in fxb.itertuples():
            pre = (t >= r.t - 0.5) & (t < r.t)
            if not pre.any():
                continue
            for tau in taus:
                m = (t >= r.t + tau) & (t < r.t + tau + 1.0)
                if m.any():
                    recs.append({'tau': tau, 'ddk': dk[m].mean() - dk[pre].mean(),
                                 'sgn_ddk': np.sign(r.d) * (dk[m].mean() - dk[pre].mean()),
                                 'ddv': dvb[m].mean() - dvb[pre].mean(),
                                 'ddv_k': dvk[m].mean() - dvk[pre].mean(),
                                 'ddv_rest': (dvb[m] - dvk[m]).mean() - (dvb[pre] - dvk[pre]).mean()})
    Rk = pd.DataFrame(recs)
    pk = Rk.groupby('tau').agg(n=('ddk', 'size'), rms_ddk_e4=('ddk', lambda x: 1e4 * rms(x)),
                               sgn_ddk_e4=('sgn_ddk', lambda x: 1e4 * x.mean()), rms_ddv=('ddv', lambda x: rms(x)),
                               rms_ddv_k=('ddv_k', lambda x: rms(x)), rms_ddv_rest=('ddv_rest', lambda x: rms(x)))
    P('-- profile after each fix (increments vs the 0.5 s before the fix): dk in 1e-4, dv split into the k part and the rest')
    P(pk.loc[[0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 19.0]].round(5).to_string())
    P('')
    if '--uk' in sys.argv:   # E) the same D33 but the correction also moves k (ratio_update_k=1)
        P('## E) D33 with ratio_update_k=1 (correction moves k through P(k,s)); speed RMSE per bag')
        d_ = {}
        for b in bags:
            o = pd.read_csv(TMP / f'{b}_uk_out.csv', usecols=['stamp_ns', 'v'])
            mv = np.load(cpp_bridge.NPZ / f'{b}.npz')['sensing__gnss__master__vel']
            pick, okk = qe.match_nearest(mv[:, 1], o.stamp_ns.to_numpy() * 1e-9)
            d_[b] = float(np.sqrt(np.mean((o.v.to_numpy()[pick][okk] - np.hypot(mv[:, 2], mv[:, 3])[okk]) ** 2)))
        pb['rmse_uk'] = pb.bag.map(d_)
        P(pb[['bag', 'rmse_off', 'rmse_on', 'rmse_uk']].round(4).to_string(index=False))
        P(f'median off {pb.rmse_off.median():.4f}  on {pb.rmse_on.median():.4f}  uk {pb.rmse_uk.median():.4f} | '
          f'mean off {pb.rmse_off.mean():.4f}  on {pb.rmse_on.mean():.4f}  uk {pb.rmse_uk.mean():.4f} | '
          f'uk worse than off on {int((pb.rmse_uk > pb.rmse_off + 1e-5).sum())}/{len(pb)} bags')
        P('-- position (quick_eval position block: published outputs vs master fix, ENU), median / mean over bags')
        pos = []
        for b in bags:
            dz = np.load(cpp_bridge.NPZ / f'{b}.npz')
            mf = dz['sensing__gnss__master__fix']
            mf = mf[np.isfinite(mf[:, 2])]
            p_ref = qe.enu(mf[:, 2], mf[:, 3], mf[:, 4], mf[0, 2], mf[0, 3], mf[0, 4])
            for var in ('off', 'on', 'uk'):
                o = pd.read_csv(TMP / f'{b}_{var}_out.csv', usecols=['stamp_ns', 'x', 'y', 'z', 'yaw', 'pos_valid'])
                op = o[o.pos_valid == 1]
                pick, okk = qe.match_nearest(mf[:, 1], op.stamp_ns.to_numpy() * 1e-9)
                e = op[['x', 'y', 'z']].to_numpy()[pick][okk] - p_ref[okk]
                yaw = op.yaw.to_numpy()[pick][okk]
                along = e[:, 0] * np.cos(yaw) + e[:, 1] * np.sin(yaw)
                e3 = np.linalg.norm(e, axis=1)
                pos.append({'bag': b, 'var': var, 'along_rmse': rms(along), 'p3_rmse': rms(e3), 'end_err': float(e3[-1])})
        pos = pd.DataFrame(pos)
        for var in ('off', 'on', 'uk'):
            q = pos[pos['var'] == var]
            P(f'{var:<4} along_rmse {q.along_rmse.median():.3f} / {q.along_rmse.mean():.3f}   p3_rmse {q.p3_rmse.median():.3f} / '
              f'{q.p3_rmse.mean():.3f}   end_err {q.end_err.median():.3f} / {q.end_err.mean():.3f}')
        P('')
    P('## per-bag rms(v_on - v_off) inside vs outside windows (divergence of the two runs)')
    for b in bags:
        sb = S[S.bag == b]
        P(f'{b}: in {rms((sb.v_on - sb.v_off)[sb.in_w]):.4f}  out {rms((sb.v_on - sb.v_off)[~sb.in_w]):.4f}  '
          f'before-first-fix {rms((sb.v_on - sb.v_off)[np.isinf(sb.since)]):.4f}')
    P(f'\nelapsed {time.time() - t0:.0f} s')
    txt = '\n'.join(lines)
    (OUT_DIR / 'd33_speed.txt').write_text(txt + '\n', encoding='utf-8')
    F.to_csv(TMP / 'fixes.csv', index=False)
    print(txt)


if __name__ == '__main__':
    main()
