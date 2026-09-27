"""Speed of saved replay outputs against the GNSS doppler (horizontal |v| of master vel at its header stamps),
nearest output within 0.05 s, per bag; two tags side by side.

  python tools/replay/speed_vs_doppler.py qk0_val qk_val      # outputs of eval_base_link.py --tag ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpp_bridge import NPZ, ROOT  # noqa: E402
from quick_eval import match_nearest  # noqa: E402


def speed_rmse(out_csv, bag):
    o = pd.read_csv(out_csv, usecols=['stamp_ns', 'v']).drop_duplicates('stamp_ns').sort_values('stamp_ns')
    mv = np.load(NPZ / f'{bag}.npz')['sensing__gnss__master__vel']
    pick, ok = match_nearest(mv[:, 1], o.stamp_ns.to_numpy() * 1e-9)
    err = o.v.to_numpy()[pick][ok] - np.hypot(mv[:, 2], mv[:, 3])[ok]
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    tags = sys.argv[1:]
    rows = {}
    for tag in tags:
        for f in sorted((ROOT / 'build_core' / 'replay_tmp' / f'bl_{tag}').glob('*_out.csv')):
            bag = f.name[:-8]
            rows.setdefault(bag, {})[tag] = speed_rmse(f, bag)
    df = pd.DataFrame(rows).T[tags]
    print(df.round(4).to_string())
    print('median', df.median().round(4).to_dict(), ' mean', df.mean().round(4).to_dict())
    if len(tags) == 2:
        d = df[tags[1]] - df[tags[0]]
        print(f'{tags[1]} vs {tags[0]}: better {int((d < -1e-4).sum())}, worse {int((d > 1e-4).sum())}')


if __name__ == '__main__':
    main()
