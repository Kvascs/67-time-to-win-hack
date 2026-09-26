#!/bin/bash
# Run 2 (frozen map snapshot, pinned for every stage): scan -> L-BFGS-B -> val error budget -> val checks ->
# determinism recheck. Never more than 2 replay processes (each stage uses a pool of 2, stages are sequential).
cd /c/MosTransHack/analysis/ideas_check/quasi_newton
export QN_MAPS='<session-scratch>'
python a_tune.py scan > a_scan_stdout.txt 2>&1
python a_tune.py lbfgsb 0.1 7 > a_lbfgsb_stdout.txt 2>&1
python c_error_budget.py > c_error_budget_stdout.txt 2>&1
python c_episodes.py > c_episodes_stdout.txt 2>&1
python c_upper_bound.py > c_upper_bound_stdout.txt 2>&1
python a_tune.py val best > a_val_stdout.txt 2>&1
python a_tune.py val scanbest > a_val_scanbest_stdout.txt 2>&1
python a_recheck.py > a_recheck_stdout.txt 2>&1
python a_noise_fit.py > a_noise_fit_stdout.txt 2>&1
echo chain2-done
