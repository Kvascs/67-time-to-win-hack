"""Command line interface: ``python -m anomaly_sim <command> ...`` (run from ``C:\\MosTransHack\\tools``).

Commands
  list          scenarios of a suite (and injector types with --injectors)
  run           apply scenarios to bags -> corrupted npz (+ optional rosbag2 output)
  tobag         convert corrupted npz file(s) to rosbag2 directories for ``ros2 bag play``
  plot          overview/zoom plot of a corrupted npz
  validate      schema-check npz files and read back / compare rosbag2 outputs
  characterize  statistics of the natural anomalies in the clean dataset
  evaluate      run the reference estimators on generated scenarios and tabulate robustness metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .constants import BAG_DIR, DEFAULT_OUT, DEFAULT_SUITE, NPZ_DIR


def _cmd_list(a):
    from .scenario import list_injectors, load_suite
    if a.injectors:
        for name, info in list_injectors().items():
            print(f'{name:14s} stage {info["stage"]:2d}  {info["doc"]}')
            print(f'{"":14s} defaults: {json.dumps(info["defaults"])}')
        return 0
    for s in load_suite(a.suite):
        types = [x['type'] for x in s.specs()]
        print(f'{s.name:32s} seed={s.seed:5d} gnss_keep={s.gnss_keep_s}  {types}')
        if a.verbose:
            print(f'    {s.description}')
    return 0


def _cmd_run(a):
    from .scenario import bags_from_spec, generate, load_suite
    suite = load_suite(a.suite)
    if a.scenarios and a.scenarios != ['all']:
        sel = []
        for want in a.scenarios:
            sel += [s for s in suite if s.name == want or s.name.split('_')[0] == want]
        suite = sel
    bags = bags_from_spec(a.bags)
    missing = [b for b in bags if not (Path(a.npz_dir) / f'{b}.npz').exists()]
    if missing:
        print(f'missing npz for {missing}', file=sys.stderr)
        return 2
    t0 = time.time()
    out = Path(a.out) / 'npz'
    res = generate(suite, bags, out, workers=a.workers, npz_dir=a.npz_dir)
    print(f'{len(res)} runs written to {out} in {time.time() - t0:.1f}s')
    if a.write_bags:
        from .bagio import write_bag
        from .run import load_run
        for r in res:
            run = load_run(r['path'])
            dst = Path(a.out) / 'bags' / run.meta['scenario'] / run.name
            rep = write_bag(run, dst, fmt=a.fmt, overwrite=True, verify=not a.no_verify)
            print(f"bag {dst}  msgs={rep['messages']}  verify={rep.get('verify')}")
    return 0


def _cmd_tobag(a):
    from .bagio import write_bag
    from .run import load_run
    for p in a.npz:
        run = load_run(p)
        scen = run.meta.get('scenario', 'custom')
        dst = Path(a.out) / scen / run.name if a.out else Path(p).with_suffix('')
        rep = write_bag(run, dst, src_bag=a.src_bag, fmt=a.fmt, overwrite=a.overwrite, verify=not a.no_verify)
        print(json.dumps({k: v for k, v in rep.items() if k != 'topics'}))
    return 0


def _cmd_plot(a):
    from .plotting import plot_run
    from .run import load_pair
    for p in a.npz:
        bad, clean = load_pair(p)
        out = Path(a.out) if a.out else Path(p).with_suffix('.png')
        if a.out and len(a.npz) > 1:
            out = Path(a.out) / (Path(p).parent.name + '_' + Path(p).stem + '.png')
        print(plot_run(bad, clean, out, max_events=a.max_events))
    return 0


def _cmd_validate(a):
    from .run import check_npz, load_run
    bad = 0
    files = []
    for p in a.paths:
        p = Path(p)
        files += sorted(p.rglob('*.npz')) if p.is_dir() else [p]
    for f in files:
        probs = check_npz(f)
        if probs:
            bad += 1
            print(f'FAIL {f}: {probs}')
        elif a.verbose:
            print(f'ok   {f}')
    print(f'{len(files) - bad}/{len(files)} npz files OK')
    if a.bags:
        from .bagio import verify_bag
        for bdir in sorted(Path(a.bags).rglob('metadata.yaml')):
            bag = bdir.parent
            scen = bag.parent.name
            npz = Path(a.npz_root) / scen / f'{bag.name}.npz' if a.npz_root else None
            if npz and npz.exists():
                probs = verify_bag(bag, load_run(npz))
                print(('ok   ' if not probs else 'FAIL ') + f'{bag} {probs or ""}')
                bad += bool(probs)
    return 1 if bad else 0


def _cmd_characterize(a):
    from .characterize import characterize_dataset
    characterize_dataset(out_dir=Path(a.out), bags=a.bags, workers=a.workers)
    return 0


def _cmd_evaluate(a):
    from .evaluate import evaluate_suite
    evaluate_suite(Path(a.npz_root), out_dir=Path(a.out), scenarios=a.scenarios, workers=a.workers)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='anomaly_sim', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('list', help='list scenarios / injectors')
    p.add_argument('--suite', default=str(DEFAULT_SUITE))
    p.add_argument('--injectors', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(fn=_cmd_list)

    p = sub.add_parser('run', help='generate corrupted npz (and optionally bags)')
    p.add_argument('--suite', default=str(DEFAULT_SUITE))
    p.add_argument('--scenarios', nargs='*', default=['all'], help='names or S-prefixes, or "all"')
    p.add_argument('--bags', nargs='+', default=['val'], help='bag names and/or splits: train val no_gnss_long all')
    p.add_argument('--out', default=str(DEFAULT_OUT))
    p.add_argument('--npz-dir', default=str(NPZ_DIR))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--write-bags', action='store_true', help='also write rosbag2 dirs to <out>/bags/<scenario>/<bag>')
    p.add_argument('--fmt', default='humble', choices=['humble', 'rosbags'])
    p.add_argument('--no-verify', action='store_true')
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser('tobag', help='corrupted npz -> rosbag2')
    p.add_argument('npz', nargs='+')
    p.add_argument('--out', default=None, help='root dir; bag goes to <out>/<scenario>/<bag>')
    p.add_argument('--src-bag', default=None, help=f'original bag dir (default {BAG_DIR}/<bag>)')
    p.add_argument('--fmt', default='humble', choices=['humble', 'rosbags'])
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-verify', action='store_true')
    p.set_defaults(fn=_cmd_tobag)

    p = sub.add_parser('plot', help='plot corrupted npz vs clean')
    p.add_argument('npz', nargs='+')
    p.add_argument('--out', default=None)
    p.add_argument('--max-events', type=int, default=6)
    p.set_defaults(fn=_cmd_plot)

    p = sub.add_parser('validate', help='check npz files (and bags)')
    p.add_argument('paths', nargs='+')
    p.add_argument('--bags', default=None, help='root of written bags to verify against npz')
    p.add_argument('--npz-root', default=None, help='root of npz (<root>/<scenario>/<bag>.npz)')
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(fn=_cmd_validate)

    p = sub.add_parser('characterize', help='natural anomaly statistics of the clean dataset')
    p.add_argument('--out', default=str(DEFAULT_OUT / 'natural'))
    p.add_argument('--bags', nargs='+', default=['all'])
    p.add_argument('--workers', type=int, default=6)
    p.set_defaults(fn=_cmd_characterize)

    p = sub.add_parser('evaluate', help='reference estimators on generated scenarios')
    p.add_argument('--npz-root', default=str(DEFAULT_OUT / 'npz'))
    p.add_argument('--out', default=str(DEFAULT_OUT / 'eval'))
    p.add_argument('--scenarios', nargs='*', default=None)
    p.add_argument('--workers', type=int, default=4)
    p.set_defaults(fn=_cmd_evaluate)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
