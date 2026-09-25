"""Export the identified traction/brake table (analysis/traction_id) into the estimator format.

Input : analysis/traction_id/traction_lut_table.csv  (rows = notch -15..15, columns v0, v0.5, ... m/s;
        grade-compensated steady-state acceleration, output-error fit on train bags)
Output: ros2_ws/src/tram_backup_odometry/config/traction_lut.csv  (header "notch,<v>...")
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'analysis' / 'traction_id'
OUT = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'config' / 'traction_lut.csv'


def main():
    t = pd.read_csv(SRC / 'traction_lut_table.csv', index_col=0)
    speeds = [float(c.lstrip('v')) for c in t.columns]
    params = json.loads((SRC / 'traction_model_params.json').read_text(encoding='utf-8'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='\n') as fh:
        fh.write('# steady-state drive acceleration a*(notch, v) on level track, m/s^2 (running resistance\n')
        fh.write('# included, grade excluded: the estimator adds -kg*grade(s) from the map).\n')
        fh.write(f"# identified by output-error fit on train bags; tau={params['tau']:.4f} s, "
                 f"kg(brake,coast,traction)={params['kg']}\n")
        fh.write('notch,' + ','.join(f'{v:g}' for v in speeds) + '\n')
        for notch, row in t.sort_index().iterrows():
            fh.write(f'{int(notch)},' + ','.join(f'{a:.5f}' for a in row.to_numpy()) + '\n')
    print(f'wrote {OUT} ({len(t)} notches x {len(speeds)} speeds); tau={params["tau"]}, kg={params["kg"]}')


if __name__ == '__main__':
    main()
