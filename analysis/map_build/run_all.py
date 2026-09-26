"""Full reproducible pipeline:  python run_all.py
  1. per-run GNSS preprocessing cache (runs.py)
  2. map_train/  (train only)  -> validation on val (validate.py, init_test.py, stops on map_train)
  3. map/        (train + val) -> deployment map + stops + finalize
  4. plots (make_plots.py)
"""
import subprocess
import sys
import time

STEPS = [
    ['build_map.py', 'train'],
    ['validate.py'],
    ['init_test.py'],
    ['stops_analysis.py', 'map_train'],
    ['finalize_map.py', 'map_train'],
    ['build_map.py', 'all'],
    ['stops_analysis.py', 'map'],
    ['wheel_curvature.py', 'map'],
    ['drift_sim.py'],
    ['finalize_map.py', 'map'],
    ['make_plots.py'],
    ['selftest_track_map.py'],
]

if __name__ == '__main__':
    for st in STEPS:
        t0 = time.time()
        print('>>>', ' '.join(st), flush=True)
        r = subprocess.run([sys.executable, '-u'] + st)
        print(f'<<< {st[0]} exit={r.returncode} ({time.time() - t0:.0f} s)', flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)
