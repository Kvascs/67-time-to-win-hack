"""Runs tools/replay/eval_par.py on VAL for each field variant, sequentially (2 replay workers),
with --dropwin 60:10 (model-only windows) and/or clean. Results: build_core/eval/rpca_<mode>_<variant>.csv.

Tokens: 'variant' (dw, then clean) or 'mode:variant' with mode = dw | clean. Variants: none (dfield_file=""),
median_repo (analysis/validation_maps/dfield.csv) or X for dfield_X.csv next to this script.

    python analysis/ideas_check/rpca_dfield/run_evals.py none median_repo dw:pcp_lam1 clean:pcp_lam1
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXE = ROOT / 'build_core' / 'tbo_replay_final2.exe'
EXTRA = {'dw': ['--dropwin', '60:10'], 'clean': []}


def field_path(v: str) -> str:
    if v == 'none':
        return ''
    if v == 'median_repo':
        return str(ROOT / 'analysis' / 'validation_maps' / 'dfield.csv')
    p = HERE / f'dfield_{v}.csv'
    assert p.exists(), p
    return str(p)


def main():
    jobs = []
    for tok in sys.argv[1:]:
        if ':' in tok:
            mode, v = tok.split(':', 1)
            jobs.append((mode, v))
        else:
            jobs += [('dw', tok), ('clean', tok)]
    for mode, v in jobs:
        tag = f'rpca_{mode}_{v}'
        cmd = [sys.executable, str(ROOT / 'tools' / 'replay' / 'eval_par.py'), '--tag', tag, '--exe', str(EXE),
               '--split', 'val', '--jobs', '2', *EXTRA[mode], '--set', f'dfield_file={field_path(v)}']
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        log = HERE / 'logs' / f'{tag}.log'
        log.parent.mkdir(exist_ok=True)
        log.write_text(' '.join(cmd) + '\n' + res.stdout + res.stderr, encoding='utf-8')
        print(f'{tag}: rc={res.returncode} {time.time() - t0:.0f} s', flush=True)
        print(res.stdout.strip(), flush=True)


if __name__ == '__main__':
    main()
