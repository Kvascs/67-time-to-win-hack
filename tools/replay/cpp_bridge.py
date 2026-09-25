"""Bridge between extracted bags (data/npz/*.npz) and the C++ estimator replay (tbo_replay).

export_events(): writes the bag as an arrival-ordered CSV event stream. GNSS messages are
kept only during the first `gnss_seconds` of the run (like the jury's test bags).
run_replay():    runs tbo_replay and returns its published outputs as a DataFrame.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / 'data' / 'npz'
REPLAY_EXE = ROOT / 'build_core' / ('tbo_replay.exe' if os.name == 'nt' else 'tbo_replay')
PKG = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry'


def export_events(bag: str, out_csv: Path, gnss_seconds: float | None = 5.0) -> dict:
    d = np.load(NPZ / f'{bag}.npz')
    parts = []

    def add(key, typ, fmt_cols):
        a = d[key]
        if len(a) == 0:
            return
        df = pd.DataFrame({'type': typ, 'recv': (a[:, 0] * 1e9).round().astype('int64'),
                           'stamp': (a[:, 1] * 1e9).round().astype('int64')})
        for name, col in fmt_cols:
            df[name] = a[:, col]
        parts.append(df)

    add('vehicle__front_bogie_velocity', 'W0', [('a', 2)])
    add('vehicle__rear_bogie_velocity', 'W1', [('a', 2)])
    add('vehicle__driver_position_cmd', 'C', [('a', 2)])
    for src, key in ((0, 'sensing__gnss__master__fix'), (1, 'sensing__gnss__rover__fix')):
        a = d[key]
        if len(a):
            parts.append(pd.DataFrame({'type': 'GF', 'recv': (a[:, 0] * 1e9).round().astype('int64'),
                                       'stamp': (a[:, 1] * 1e9).round().astype('int64'), 'a': src,
                                       'b': a[:, 2], 'c': a[:, 3], 'e': a[:, 4], 'f': a[:, 5]}))
    ev = pd.concat(parts, ignore_index=True)
    t_first = ev.loc[ev.type.isin(['W0', 'W1', 'C']), 'recv'].min()
    if gnss_seconds is not None:
        ev = ev[(ev.type != 'GF') | (ev.recv <= t_first + int(gnss_seconds * 1e9))]
    ev = ev.sort_values('recv', kind='stable')
    with open(out_csv, 'w', newline='\n') as fh:
        for r in ev.itertuples(index=False):
            if r.type == 'GF':
                fh.write(f'GF,{r.recv},{r.stamp},{int(r.a)},{r.b:.10f},{r.c:.10f},{r.e:.4f},{int(r.f)}\n')
            elif r.type == 'C':
                fh.write(f'C,{r.recv},{r.stamp},{int(r.a)}\n')
            else:
                fh.write(f'{r.type},{r.recv},{r.stamp},{r.a:.6f}\n')
    return {'events': len(ev), 't_first_ns': int(t_first)}


def run_replay(events_csv: Path, out_csv: Path, params: Path | None = None, map_csv: Path | None = None,
               traction_csv: Path | None = None, sets: dict | None = None,
               branches: str | None = None) -> pd.DataFrame:
    cmd = [str(REPLAY_EXE), '--in', str(events_csv), '--out', str(out_csv)]
    if params:
        cmd += ['--params', str(params)]
    if map_csv:
        cmd += ['--map', str(map_csv)]
    if traction_csv:
        cmd += ['--traction', str(traction_csv)]
    if branches:
        cmd += ['--branches', branches]
    for k, v in (sets or {}).items():
        cmd += ['--set', f'{k}={v}']
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f'tbo_replay failed: {res.stderr}')
    df = pd.read_csv(out_csv)
    df.attrs['stderr'] = res.stderr.strip()
    return df
