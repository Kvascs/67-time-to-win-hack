#!/bin/bash
# Sequential replay jobs (never more than 2 replay processes): wait for the scan, then
# L-BFGS-B -> val error budget (keeps outputs) -> val check of the best point -> in-filter LUT A/B.
cd /c/MosTransHack/analysis/ideas_check/quasi_newton
until grep -q "replays run in this call" a_scan_stdout.txt || grep -q Traceback a_scan_stdout.txt; do sleep 20; done
python a_tune.py lbfgsb 0.1 7 > a_lbfgsb_stdout.txt 2>&1
python c_error_budget.py > c_error_budget_stdout.txt 2>&1
python a_tune.py val best > a_val_stdout.txt 2>&1
python b_filter_ab.py b_oe_lbfgs_warm_best.pt > b_filter_ab_stdout.txt 2>&1
echo chain-done
