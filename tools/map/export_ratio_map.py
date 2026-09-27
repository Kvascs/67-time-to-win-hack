"""Learned bogie-ratio map with reliability -> ratio_map.csv for the estimator.

Source: analysis/ideas_check/ratio_map_correction (build_map.py -> ratio_map_parts.npz: per run and 1 m bin of the
main-cycle antenna arc the sums of z = log(v_front/v_rear) / sigma_v(v); reliab.py -> reliab_cells.csv: 25 m cells
where a leave-one-out correction over the train runs made the estimate worse, widened by one cell).
Bins with < 10 samples get mu 0, sd 1; sd is floored at 0.5 (as in the study).

  python tools/map/export_ratio_map.py
    ros2_ws/src/tram_backup_odometry/maps/ratio_map.csv   map from all runs (shipped)
    analysis/validation_maps/ratio_map.csv                map from train runs only (honest validation)
  Both use the train leave-one-out reliability.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RM = ROOT / 'analysis' / 'ideas_check' / 'ratio_map_correction'
sys.path.insert(0, str(RM))
sys.argv = sys.argv[:1]  # regress.py (imported by rm_common) reads sys.argv[1]
import rm_common as M  # noqa: E402


def write(path, rmap, rel, what):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(f'# bogie-ratio map ({what}): z = log(v_front/v_rear) / sigma_v(v) per 1 m of the main-cycle antenna arc;\n')
        fh.write('# rel = 0: no corrections there (leave-one-out over train runs made the estimate worse nearby)\n')
        fh.write(f'# L={rmap.L:.4f}\n')
        for v, s in zip(*rmap.sv_tab):
            fh.write(f'#sv,{v:.3f},{s:.7f}\n')
        fh.write('bin,mu,sd,rel\n')
        for i in range(rmap.nb):
            fh.write(f'{i},{rmap.mu[i]:.5f},{rmap.sd[i]:.5f},{int(rel[i])}\n')
    print(path.relative_to(ROOT), rmap.nb, 'bins, reliable', int(np.sum(rel)))


def main():
    z = np.load(RM / 'ratio_map_parts.npz')
    split = np.array(z['split'])
    L = float(z['L'])
    sv = (z['sv_v'], z['sv_s'])
    cells = pd.read_csv(RM / 'reliab_cells.csv')
    cell_len = float(cells.cell_start_m.iloc[1] - cells.cell_start_m.iloc[0])
    unrel = cells.unreliable.to_numpy().astype(bool)
    for sel, path, what in ((np.ones(len(split), bool), ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps' / 'ratio_map.csv', 'all runs'),
                            (split == 'train', ROOT / 'analysis' / 'validation_maps' / 'ratio_map.csv', 'train runs only')):
        rmap = M.RatioMap(z['n'][sel].sum(0), z['s1'][sel].sum(0), z['s2'][sel].sum(0), L, sv)
        arc = (np.arange(rmap.nb) + 0.5) * L / rmap.nb
        rel = ~unrel[(arc // cell_len).astype(int) % len(unrel)]
        write(path, rmap, rel, what)


if __name__ == '__main__':
    main()
