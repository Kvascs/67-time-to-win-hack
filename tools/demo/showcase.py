#!/usr/bin/env python3
"""Default fault-injection showcase for the pitch and the jury report (see docs/DEMO.md).

Three validation bags (two vehicles), the same physically consistent mix on each, windows placed on real
traction / braking episodes (found with ``fault_demo.py --suggest``):
  1. joint slip of both bogies under traction at >= 30 km/h   slip:wheels@T+3:peak=0.25:phase=1
  2. joint slide of both bogies under braking from >= 35 km/h  slide:wheels@T+4:peak=0.5:phase=-1
  3. frozen bogie reading at the start of a braking            freeze:<front|rear>@T+12
  4. both bogies silent while accelerating from a stop         dropout:wheels@T+5
  5. random spikes on both bogies over traction + braking      spike:wheels@T+30:p=0.05:amp=20
plus, on the first bag, a known limit: a slow joint slip at low speed (looks like real acceleration).
A metrics-only sensitivity run (joint slip / slide of different severity on one bag) shows where the
estimator stops telling a joint wheel anomaly from real motion.

    python tools/demo/showcase.py                 # everything -> tools/demo/out/
    python tools/demo/showcase.py --only 30618_2f104a1d --no-sensitivity
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_demo as FD  # noqa: E402

SHOWCASE = [
    {'bag': '30618_2f104a1d', 'gif_window': 2, 'faults': [
        'slip:wheels@851+3:peak=0.25:phase=1',      # traction 30 -> 52 km/h
        'slide:wheels@337+4:peak=0.5:phase=-1',     # braking from 47 km/h
        'freeze:front@699+12',                      # braking from 52 km/h
        'dropout:wheels@784+5',                     # start from a stop, notch +9
        'spike:wheels@1084+30:p=0.05:amp=20',       # traction 13 -> 50 km/h, then braking
        'slip:wheels@169+5:peak=0.3:phase=1',       # LIMIT: slow joint slip at 10-27 km/h
    ]},
    {'bag': '30618_22c1c589', 'gif_window': 1, 'faults': [
        'slip:wheels@782+3:peak=0.25:phase=1',      # traction 35 -> 51 km/h
        'slide:wheels@227+4:peak=0.5:phase=-1',     # braking from 41 km/h
        'freeze:rear@278+12',                       # braking from 47 km/h
        'dropout:wheels@639+5',                     # start from a stop
        'spike:wheels@1015+30:p=0.05:amp=20',       # traction 30 -> 50 km/h, then braking
    ]},
    {'bag': '30639_d927f360', 'gif_window': 4, 'faults': [
        'slip:wheels@601+3:peak=0.25:phase=1',      # traction 0 -> 49 km/h (second vehicle)
        'slide:wheels@910+4:peak=0.5:phase=-1',     # braking from 47 km/h
        'freeze:front@223+12',                      # braking from 45 km/h
        'dropout:wheels@146+5',                     # start from a stop
        'spike:wheels@838+30:p=0.05:amp=20',        # traction 12 -> 35 km/h, then braking
    ]},
]
# joint slip / slide of different severity (speed, peak, ramp) on one bag, metrics only
SENSITIVITY = {'bag': '30618_2f104a1d', 'faults': [
    'slide:wheels@186+4:peak=0.3:phase=-1', 'slip:wheels@226+3:peak=0.3:phase=1',
    'slide:wheels@262+3:peak=0.5:phase=-1', 'slip:wheels@320+2:peak=0.3:phase=1',
    'slide:wheels@390+4:peak=0.3:phase=-1', 'slip:wheels@440+4:peak=0.3:phase=1',
    'slip:wheels@686+3:peak=0.25:phase=1', 'slip:wheels@853+4:peak=0.2:phase=1',
    'slide:wheels@977+4:peak=0.4:phase=-1', 'slip:wheels@1051+3:peak=0.3:phase=1',
    'slide:wheels@1115+4:peak=0.25:phase=-1',
]}


def _f(x, fmt='{:.2f}'):
    return '—' if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def extra_accel(r: dict) -> float:
    """Apparent extra wheel acceleration of a triangular slip / slide ramp: peak * v0 / (dur / 2) [m/s^2]."""
    spec = r['spec']
    peak = next((float(p.split('=')[1]) for p in spec.split(':') if p.startswith('peak=')),
                0.4 if r['kind'] == 'slip' else 0.6)
    dur = r['t_end_s'] - r['t_start_s']
    return peak * r['v_start_kmh'] / 3.6 / max(dur / 2.0, 1e-6)


def summary_md(results, sens, after_s: float) -> str:
    out = ['| Bag | # | Сбой | Режим, км/ч | RMSE скорости в окне, м/с<br>оценка / наивная '
           f'| Ошибка вдоль пути через {after_s:g} с после окна, м<br>оценка / наивная | Реакция, с |',
           '|---|---|---|---|---|---|---|']
    for res in results:
        for r in res['windows']:
            out.append(f"| {res['bag']} | {r['window']} | {r['description']} "
                       f"| {r['regime']}, {r['v_start_kmh']:.0f}→{r['v_end_kmh']:.0f} "
                       f"| **{_f(r['v_rmse_in_est'])}** / {_f(r['v_rmse_in_naive'])} "
                       f"| **{_f(r['along_after_est'], '{:+.2f}')}** / {_f(r['along_after_naive'], '{:+.2f}')} "
                       f"| {_f(r['reaction_s'])} |")
    if sens:
        out += ['', f"Чувствительность к совместному буксованию / юзу ({sens['bag']}):", '',
                '| # | Сбой | v в начале, км/ч | Лишнее ускорение колёс, м/с² | RMSE в окне, м/с<br>оценка / наивная '
                f'| Вдоль пути +{after_s:g} с, м<br>оценка / наивная | «Только модель», с |',
                '|---|---|---|---|---|---|---|']
        rows = sorted(sens['windows'], key=lambda r: (r['kind'], extra_accel(r)))
        for r in rows:
            out.append(f"| {r['window']} | {r['spec']} | {r['v_start_kmh']:.0f} | {extra_accel(r):.2f} "
                       f"| **{_f(r['v_rmse_in_est'])}** / {_f(r['v_rmse_in_naive'])} "
                       f"| **{_f(r['along_after_est'], '{:+.2f}')}** / {_f(r['along_after_naive'], '{:+.2f}')} "
                       f"| {r['model_only_s']:.1f} |")
    return '\n'.join(out) + '\n'


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(FD.DEFAULT_OUT))
    ap.add_argument('--only', nargs='*', help='run only these bags of the showcase')
    ap.add_argument('--no-sensitivity', action='store_true')
    ap.add_argument('--exe', default=str(FD.DEFAULT_EXE))
    ap.add_argument('--maps', default=str(FD.DEFAULT_MAPS))
    ap.add_argument('--after', type=float, default=10.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    setup = FD.ReplaySetup(exe=Path(a.exe), maps=Path(a.maps))
    rdir = Path(tempfile.mkdtemp(prefix='fault_showcase_'))      # shared: clean replays are reused
    t0 = time.perf_counter()
    results, sens = [], None
    try:
        for item in SHOWCASE:
            if a.only and item['bag'] not in a.only:
                continue
            results.append(FD.run_demo(item['bag'], item['faults'], out / item['bag'], setup=setup,
                                       after_s=a.after, gif_window=item.get('gif_window'), replay_dir=rdir,
                                       reuse=True))
        if not a.no_sensitivity and (not a.only or SENSITIVITY['bag'] in a.only):
            sens = FD.run_demo(SENSITIVITY['bag'], SENSITIVITY['faults'], out / f"sensitivity_{SENSITIVITY['bag']}",
                               setup=setup, after_s=a.after, gif=False, png=False, replay_dir=rdir, reuse=True)
    finally:
        shutil.rmtree(rdir, ignore_errors=True)
    rows = [dict(r, group='showcase') for res in results for r in res['windows']]
    if sens:
        rows += [dict(r, group='sensitivity', extra_accel_mps2=extra_accel(r)) for r in sens['windows']]
    pd.DataFrame(rows).reindex(columns=['group'] + FD.CSV_COLS + ['extra_accel_mps2']).to_csv(
        out / 'showcase_metrics.csv', index=False, float_format='%.6g')
    (out / 'showcase_metrics.md').write_text(
        '# Инъекция сбоев: сводка витрины\n\nСгенерировано `python tools/demo/showcase.py`. Подробности и '
        'определения — `docs/DEMO.md`.\n\n' + summary_md(results, sens, a.after), encoding='utf-8')
    meta = {'setup': setup.provenance(), 'sets': setup.binary_sets(), 'after_s': a.after,
            'bags': {r['bag']: {'faults': r['faults'], 'gif': r['gif'], 'reference': r['reference']} for r in results},
            'sensitivity': SENSITIVITY if sens else None, 'wall_s': time.perf_counter() - t0}
    (out / 'showcase.json').write_text(json.dumps(FD.clean_json(meta), indent=1, ensure_ascii=False, default=FD._json_default),
                                       encoding='utf-8')
    print(summary_md(results, sens, a.after))
    print(f'showcase done in {meta["wall_s"]:.0f} s -> {FD._rel(out)}')


if __name__ == '__main__':
    main()
