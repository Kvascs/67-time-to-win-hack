"""Shape of the written fields relative to the median field (same train samples, same exe):
correlation, amplitude slope (least squares of field on median field), |d| percentiles.

    python analysis/ideas_check/rpca_dfield/field_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    ref = pd.read_csv(HERE / 'dfield_median_final2.csv', comment='#')
    files = {'median_repo': ROOT / 'analysis' / 'validation_maps' / 'dfield.csv',
             **{p.stem.replace('dfield_', ''): p for p in sorted(HERE.glob('dfield_*.csv'))}}
    print(f'{"field":16s} {"corr":>6s} {"slope":>6s} {"|d|p50":>7s} {"|d|p95":>7s} {"rms":>7s}  header')
    for k, p in files.items():
        f = pd.read_csv(p, comment='#')
        d = np.interp(ref.s, f.s, f.iloc[:, 1])
        c = np.corrcoef(d, ref.d)[0, 1]
        slope = float(np.dot(d, ref.d) / np.dot(ref.d, ref.d))
        head = open(p, encoding='utf-8').readline().strip()[:150]
        print(f'{k:16s} {c:6.3f} {slope:6.3f} {np.percentile(np.abs(d), 50):7.4f} {np.percentile(np.abs(d), 95):7.4f} '
              f'{np.sqrt(np.mean(d ** 2)):7.4f}  {head}')


if __name__ == '__main__':
    main()
