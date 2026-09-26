#!/usr/bin/env python
"""Evidence files for the C++ anomaly-suite report (results_cpp/diagnostics/*.csv|json).

* natural_wheel_stats.json - natural bogie-speed noise (front-rear at shared stamps) and runs of
  bit-identical readings while moving over train+val bags (false-alarm budget of a "stuck" detector);
* trace_<case>.csv          - output excerpts (scenario vs clean-input replay) around each root-cause case,
  for the variants in which the case was analysed.

Usage (from C:\\MosTransHack\\tools):  python anomaly_sim/cpp_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cpp_suite as R  # noqa: E402

OUT = R.RESULTS / 'diagnostics'

# (name, variant, scenario, bag, t_from_rel, t_to_rel, what it shows)
CASES = [
    ('freeze_after_vehicle_gap', 'base', 'S18_combined_severe', '30618_2f104a1d', 955.0, 1000.0,
     'all vehicle topics silent 960.1-969.8 s: every later stamp is > latest+max_future_s -> rejected forever'),
    ('freeze_fixed_mf20', 'mf20', 'S18_combined_severe', '30618_2f104a1d', 955.0, 975.0,
     'same gap with max_future_s=20: bridged (outputs of the gap are back-filled at the first new input)'),
    ('frozen_front_under_braking', 'base', 'X20_frozen_front_hold10', '30618_2f104a1d', 388.0, 405.0,
     'front frozen at 8.57 m/s while braking: IMM picks rear-bad, stuck detector circular, lock-up guard blocks recovery'),
    ('noise_lockout', 'base', 'S12_noise_increase', '30618_0f120b35', 490.0, 540.0,
     'sigma 2.4 km/h noise: model-only lock-out, recovery needs |f-r|<0.35 m/s for 3 s, d up to +0.8 -> runaway'),
    ('ramp_absorbed_by_d', 'base', 'X05_joint_slip_ramp20', '30618_6cb3280a', 1036.5, 1046.0,
     '+20 % ramp: d rises to the 0.6 clamp within 0.4 s, CUSUM reference follows the slip, no alarm'),
    ('ramp_slow_reference', 'fixA_jointref', 'X05_joint_slip_ramp20', '30618_6cb3280a', 1036.5, 1046.0,
     'same with fix_joint_d_tau_s=20: joint latch after ~1.1 s, model bridges, slip ratio 0.20 published'),
    ('single_bogie_query_glitch', 'base', 'S09_outliers_spikes', '30618_0f120b35', 502.0, 506.0,
     'rear spike charges the rear CUSUM; W1-only queries see n_av=1 -> transient joint latch at wheel stamps'),
    ('standstill_no_wheels', 'mf20', 'S07_dropout_long', '30639_d927f360', 270.0, 312.0,
     'front dropout + natural rear silence at departure: standstill never cleared, single returning bogie rejected'),
    ('low_speed_lock', 'base', 'S05_slide_wheel_lock', '30618_22c1c589', 946.5, 952.5,
     'both bogies lock at 3 m/s: maneuver mode follows, filter < standstill_max_v, false standstill'),
    ('slide_spinup_cmd_fault', 'base', 'S04_slide_braking_wsp', '30618_2f104a1d', 713.0, 725.0,
     'joint slide: IMM follows the less-sliding bogie; spin-up raises cmd_inconsistent + unmodeled_accel (8 s hold)'),
]


def flag_names(f: int) -> str:
    return ';'.join(k for k, b in R.BIT.items() if (int(f) >> b) & 1)


def trace(variant, scn, bag, t0, t1) -> pd.DataFrame | None:
    f = R.RESULTS / variant / 'out' / scn / f'{bag}.npz'
    fc = R.RESULTS / variant / 'clean_out' / f'{bag}.npz'
    if not f.exists() or not fc.exists():
        return None
    o, oc, ref = R.load_compact(f), R.load_compact(fc), R.load_ref(bag)
    T0 = ref['t_first_veh']
    t = o['stamp_ns'] * 1e-9
    pk, ok = R.match_nearest(t, oc['stamp_ns'] * 1e-9, 0.1)
    sel = np.flatnonzero((t - T0 >= t0) & (t - T0 <= t1))
    keep = np.isfinite(ref['v'])
    vref = np.interp(t[sel], ref['t'][keep], ref['v'][keep])
    return pd.DataFrame({
        't_rel': np.round(t[sel] - T0, 3), 'recv_rel': np.round(o['recv_ns'][sel] * 1e-9 - T0, 3),
        'v': o['v'][sel], 'v_clean_run': np.where(ok[sel], oc['v'][pk[sel]], np.nan), 'v_gnss': vref,
        'ds_vs_clean': np.where(ok[sel], o['s'][sel] - oc['s'][pk[sel]], np.nan),
        **{f'mu{j}': o['mu'][sel, j] for j in range(5)},
        'd': o['d'][sel], 'a_model': o['a_model'][sel], 'slip_f': o['slip_f'][sel], 'slip_r': o['slip_r'][sel],
        'flags': o['flags'][sel], 'flag_names': [flag_names(x) for x in o['flags'][sel]]})


def natural_wheel_stats() -> dict:
    splits = json.loads(Path(R.SPLITS_JSON).read_text())
    bags = splits['train'] + splits['val']
    runs, noise = [], []
    hours = 0.0
    for bag in bags:
        d = np.load(R.NPZ_DIR / f'{bag}.npz')
        fr, rr = d['vehicle__front_bogie_velocity'], d['vehicle__rear_bogie_velocity']
        if len(fr) > 50 and len(rr) > 50:
            _, fi, ri = np.intersect1d(fr[:, 1], rr[:, 1], return_indices=True)
            mv = (fr[fi, 2] > 5) & (rr[ri, 2] > 5)
            x = fr[fi, 2][mv] - rr[ri, 2][mv]
            if len(x) > 100:
                hp = np.diff(x) / np.sqrt(2)
                noise.append({'bag': bag, 'front_minus_rear_std_kmh': float(np.std(x)),
                              'white_sigma_per_wheel_kmh': float(1.4826 * np.median(np.abs(hp - np.median(hp))) / np.sqrt(2))})
        for key in ('vehicle__front_bogie_velocity', 'vehicle__rear_bogie_velocity'):
            a = d[key]
            if len(a) < 10:
                continue
            t, v = a[:, 1], a[:, 2]
            hours += (t[-1] - t[0]) / 3600.0
            same = np.r_[False, (np.diff(v) == 0) & (v[1:] > 0.3 * 3.6)]
            i, n = 0, len(v)
            while i < n:
                if same[i]:
                    j = i
                    while j + 1 < n and same[j + 1]:
                        j += 1
                    runs.append({'bag': bag, 'bogie': key.split('__')[1][:5], 'dur_s': float(t[j] - t[i - 1]),
                                 'v_ms': float(v[i]) / 3.6, 't_rel': float(t[i - 1] - t[0])})
                    i = j + 1
                else:
                    i += 1
    r = np.array([x['dur_s'] for x in runs])
    nz = pd.DataFrame(noise)
    return {'bags': len(bags), 'sensor_hours': round(hours, 2),
            'identical_runs_moving': {f'>= {thr}s': int((r >= thr).sum()) for thr in (0.3, 0.5, 0.8, 1.0, 1.2, 2.0)},
            'longest_identical_runs': sorted(runs, key=lambda x: -x['dur_s'])[:8],
            'noise_front_minus_rear_std_kmh_median': float(nz.front_minus_rear_std_kmh.median()),
            'noise_white_sigma_per_wheel_kmh_median': float(nz.white_sigma_per_wheel_kmh.median()),
            'noise_per_bag': noise}


LOO_REC = ['max_future_s=20', 'rate_to_maneuver=0.01', 'stuck_min_change=0', 'stuck_time_s=0.8', 'cusum_h=0.2']
LOO_FIX = ['fix_future_confirm=1', 'fix_joint_d_tau_s=20', 'fix_joint_need_both=1', 'fix_cmd_check=1',
           'fix_standstill_exit=1', 'fix_single_recover=1', 'fix_agree_tau_s=2', 'fix_d_clamp_bothbad=1',
           'fix_lockup_guard_s=4', 'fix_stuck_reset=1']
LOO_CASES = [('S05_slide_wheel_lock', '30618_2f104a1d', 262, 290), ('S09_outliers_spikes', '30618_6cb3280a', 576, 600),
             ('X21_frozen_wheels_zero3', '30618_0f120b35', 932, 970), ('X23_spikes_only', '30618_0f120b35', 0, 1e9),
             ('S09_outliers_spikes', '30618_0f120b35', 0, 1e9)]


def loo_skip_bad() -> pd.DataFrame:
    """fix_cusum_skip_bad modes on the cases where the first all-fixes run regressed (patched2 binary),
    measured against the clean-input replay of the SAME configuration."""
    import subprocess
    exe = R.BIN / 'tbo_replay_patched2.exe'
    rows = []
    tmp = R.RESULTS / 'tmp_diag'
    tmp.mkdir(parents=True, exist_ok=True)
    configs = [('snapshot_defaults', []), ('fixes+skip_bad=1', LOO_REC + LOO_FIX + ['fix_cusum_skip_bad=1']),
               ('fixes+skip_bad=2', LOO_REC + LOO_FIX + ['fix_cusum_skip_bad=2']), ('fixes_no_skip', LOO_REC + LOO_FIX)]

    def replay(ev, sets, out):
        cmd = [str(exe), '--in', str(ev), '--out', str(out), '--map', str(R.MAP), '--traction', str(R.TRACTION),
               '--branches', R.BRANCHES, '--set', f'landmark_file={R.LANDMARKS}']
        for s in sets:
            cmd += ['--set', s]
        subprocess.run(cmd, capture_output=True)
        return R.read_output(out)

    for name, sets in configs:
        for scn, bag, t0, t1 in LOO_CASES:
            oc = replay(R.CACHE / 'events' / 'S00_baseline_gnss_cut' / f'{bag}.csv', sets, tmp / 'c.csv')
            o = replay(R.CACHE / 'events' / scn / f'{bag}.csv', sets, tmp / 'o.csv')
            T0 = R.load_ref(bag)['t_first_veh']
            t = o['stamp_ns'] * 1e-9 - T0
            pk, ok = R.match_nearest(o['stamp_ns'] * 1e-9, oc['stamp_ns'] * 1e-9, 0.1)
            sel = (t >= t0) & (t <= t1) & ok
            dv = o['v'][sel] - oc['v'][pk[sel]]
            ds = o['s'][sel] - oc['s'][pk[sel]]
            rows.append({'config': name, 'scenario': scn, 'bag': bag, 't_from_rel': t0, 't_to_rel': min(t1, float(t[-1])),
                         'dv_max': float(np.max(np.abs(dv))), 'ds_max': float(np.max(np.abs(ds))),
                         'n_glitch_0p3': R.sanity_metrics(o)['n_glitch_0p3'],
                         'cmd_inconsistent_frac': float(((o['flags'] & R.FAM['cmd_inconsistent']) != 0).mean())})
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'skip_bad_modes.csv', index=False)
    return df


def landmark_after_reanchor() -> pd.DataFrame:
    """S05 lock on 30618_2f104a1d with the recommended params + fixes: re-anchoring by pinVelocity() drops the
    s-v covariance, sigma_s stays ~16 m at the following stop and a landmark 17 m away is accepted."""
    import subprocess
    exe = R.BIN / 'tbo_replay_patched2.exe'
    bag = '30618_2f104a1d'
    ref = R.load_ref(bag)
    T0 = ref['t_first_veh']
    tmp = R.RESULTS / 'tmp_diag'
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg, sets in (('snapshot_defaults', []), ('rec_params+fixes', LOO_REC + LOO_FIX)):
        for scn in ('S00_baseline_gnss_cut', 'S05_slide_wheel_lock'):
            cmd = [str(exe), '--in', str(R.CACHE / 'events' / scn / f'{bag}.csv'), '--out', str(tmp / 'o.csv'),
                   '--map', str(R.MAP), '--traction', str(R.TRACTION), '--branches', R.BRANCHES,
                   '--set', f'landmark_file={R.LANDMARKS}']
            for s in sets:
                cmd += ['--set', s]
            subprocess.run(cmd, capture_output=True)
            o = R.read_output(tmp / 'o.csv')
            t = o['stamp_ns'] * 1e-9
            fs, ok = R.match_nearest(ref['t_fix'], t)
            e = np.column_stack([o['x'], o['y']])[fs] - ref['p_fix'][:, :2]
            along = np.where(ok, e[:, 0] * np.cos(o['yaw'][fs]) + e[:, 1] * np.sin(o['yaw'][fs]), np.nan)
            for a in (262, 266, 270, 272, 273.5, 274.5, 276, 300, 800, 1150, 1160):
                j = int(np.argmin(np.abs(t - T0 - a)))
                k = int(np.argmin(np.abs(ref['t_fix'] - T0 - a)))
                rows.append({'config': cfg, 'scenario': scn, 't_rel': a, 'v': o['v'][j], 's_sd': float(np.sqrt(o['s_var'][j])),
                             'mu_model_only': o['mu3'][j], 'flags': flag_names(o['flags'][j]),
                             'along_err_vs_gnss': float(along[k])})
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'landmark_after_reanchor.csv', index=False)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if '--loo' in sys.argv:
        print(loo_skip_bad().round(3).to_string())
        return
    if '--landmark' in sys.argv:
        print(landmark_after_reanchor().round(3).to_string())
        return
    idx = []
    for name, variant, scn, bag, t0, t1, what in CASES:
        df = trace(variant, scn, bag, t0, t1)
        if df is None:
            idx.append({'case': name, 'variant': variant, 'scenario': scn, 'bag': bag, 'file': '', 'note': 'missing'})
            continue
        fn = OUT / f'trace_{name}.csv'
        df.to_csv(fn, index=False, float_format='%.4f')
        idx.append({'case': name, 'variant': variant, 'scenario': scn, 'bag': bag, 't_from_rel': t0, 't_to_rel': t1,
                    'file': fn.name, 'what': what, 'max_abs_v_minus_clean': float(np.nanmax(np.abs(df.v - df.v_clean_run))),
                    'max_abs_ds_vs_clean': float(np.nanmax(np.abs(df.ds_vs_clean)))})
    pd.DataFrame(idx).to_csv(OUT / 'trace_index.csv', index=False)
    (OUT / 'natural_wheel_stats.json').write_text(json.dumps(natural_wheel_stats(), indent=1), encoding='utf-8')
    print(pd.DataFrame(idx)[['case', 'variant', 'file', 'max_abs_v_minus_clean', 'max_abs_ds_vs_clean']].to_string())


if __name__ == '__main__':
    main()
