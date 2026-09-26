"""Scenario suites: named, seeded lists of injector specs, applied to clean runs."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .constants import DEFAULT_SUITE, NPZ_DIR, SPLITS_JSON, Label
from .context import Context
from .injectors import REGISTRY, build_injector
from .run import Run, load_run, save_run
from .sampling import make_rng


@dataclass
class Scenario:
    name: str
    seed: int
    description: str = ''
    injectors: list[dict] = field(default_factory=list)
    gnss_keep_s: float | None = 5.0
    tags: list[str] = field(default_factory=list)

    def specs(self) -> list[dict]:
        """Injector specs including the implicit GNSS cut."""
        specs = [dict(s) for s in self.injectors if s.get('enabled', True)]
        if self.gnss_keep_s is not None and not any(s['type'] == 'gnss_cut' for s in specs):
            specs.append({'type': 'gnss_cut', 'keep_s': self.gnss_keep_s})
        return specs

    def validate(self) -> None:
        for s in self.specs():
            build_injector(s)


def load_suite(path: str | Path = DEFAULT_SUITE) -> list[Scenario]:
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    doc = json.loads(text) if path.suffix == '.json' else yaml.safe_load(text)
    defaults = doc.get('defaults', {}) or {}
    out: list[Scenario] = []
    names = set()
    for sc in doc['scenarios']:
        if sc['name'] in names:
            raise ValueError(f'duplicate scenario name {sc["name"]}')
        names.add(sc['name'])
        s = Scenario(name=sc['name'], seed=int(sc['seed']), description=sc.get('description', ''),
                     injectors=sc.get('injectors', []) or [],
                     gnss_keep_s=sc.get('gnss_keep_s', defaults.get('gnss_keep_s', 5.0)),
                     tags=sc.get('tags', []) or [])
        s.validate()
        out.append(s)
    return out


def get_scenario(suite: list[Scenario], name: str) -> Scenario:
    for s in suite:
        if s.name == name or s.name.split('_')[0] == name:
            return s
    raise KeyError(f'scenario {name!r} not in suite: {[s.name for s in suite]}')


def apply_scenario(clean: Run, scenario: Scenario, seed: int | None = None) -> Run:
    """Return a corrupted copy of ``clean`` (``clean`` itself is not modified)."""
    seed = scenario.seed if seed is None else int(seed)
    run = clean.copy()
    ctx = Context.build(clean)
    specs = scenario.specs()
    injectors = [build_injector(s) for s in specs]
    order = sorted(range(len(injectors)), key=lambda i: (injectors[i].stage, i))
    events: list[dict] = []
    placement = {}
    for i in order:
        rng = make_rng(seed, clean.name, i)
        for e in injectors[i].apply(run, ctx, rng):
            e['injector'] = i
            events.append(e)
        if getattr(injectors[i], 'placement', None):
            placement[str(i)] = {'type': injectors[i].type_name, **injectors[i].placement}
    for k in list(run.streams):
        run.streams[k] = run.streams[k].sort()
    events.sort(key=lambda e: (e['t0'], e['type']))
    for j, e in enumerate(events):
        e['id'] = j
    run.events = events
    run.meta = {
        'bag': clean.name,
        'scenario': scenario.name,
        'description': scenario.description,
        'seed': seed,
        'tags': scenario.tags,
        'injectors': specs,
        'source_npz': clean.meta.get('source_npz', str(NPZ_DIR / f'{clean.name}.npz')),
        'gnss_keep_s': scenario.gnss_keep_s,
        'placement': placement,
        't_start': clean.t_start,
        'generator': f'anomaly_sim {__version__}',
        **({'crop': clean.meta['crop']} if 'crop' in clean.meta else {}),
    }
    return run


def summarize(run: Run) -> dict:
    """Compact per-run summary: events per type and labelled rows per topic/flag."""
    ev: dict[str, int] = {}
    for e in run.events:
        ev[e['type']] = ev.get(e['type'], 0) + 1
    rows = {}
    for key, s in run.streams.items():
        if not len(s) or not s.label.any():
            continue
        rows[key] = {f.name: int(((s.label & f.value) > 0).sum()) for f in Label if f.value and (s.label & f.value).any()}
    return {'bag': run.name, 'scenario': run.meta.get('scenario'), 'events': ev, 'labelled_rows': rows,
            'n_msgs': {k: len(s) for k, s in run.streams.items()}}


# ---------------------------------------------------------------------------------- batch

def bags_from_spec(spec: list[str] | str) -> list[str]:
    """Bag names from explicit names and/or split names (train, val, no_gnss_long, short, all)."""
    if isinstance(spec, str):
        spec = [spec]
    splits = json.loads(Path(SPLITS_JSON).read_text()) if Path(SPLITS_JSON).exists() else {}
    out: list[str] = []
    for item in spec:
        if item == 'all':
            names = [i['bag'] for i in splits.get('info', [])]
        elif item in splits and isinstance(splits[item], list):
            names = list(splits[item])
        else:
            names = [item]
        for n in names:
            if n not in out:
                out.append(n)
    return out


def _generate_one(args) -> dict:
    scenario, bag, out_dir, npz_dir = args
    t0 = time.time()
    clean = load_run(Path(npz_dir) / f'{bag}.npz', name=bag)
    run = apply_scenario(clean, scenario)
    path = Path(out_dir) / scenario.name / f'{bag}.npz'
    save_run(run, path)
    (path.with_suffix('.events.json')).write_text(json.dumps({'meta': run.meta, 'events': run.events}, indent=1,
                                                             default=_np_default), encoding='utf-8')
    summ = summarize(run)
    summ['seconds'] = round(time.time() - t0, 2)
    summ['path'] = str(path)
    return summ


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def generate(suite: list[Scenario], bags: list[str], out_dir: str | Path, workers: int = 4,
             npz_dir: str | Path = NPZ_DIR, log=print) -> list[dict]:
    """Apply every scenario of ``suite`` to every bag; writes ``<out>/<scenario>/<bag>.npz``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(sc, b, str(out_dir), str(npz_dir)) for sc in suite for b in bags]
    results: list[dict] = []
    if workers <= 1:
        for j in jobs:
            r = _generate_one(j)
            log(f"{r['scenario']:32s} {r['bag']}  events={r['events']}  {r['seconds']}s")
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_generate_one, jobs):
                log(f"{r['scenario']:32s} {r['bag']}  events={r['events']}  {r['seconds']}s")
                results.append(r)
    # merge with an existing manifest (partial re-generation must not drop other runs)
    man_path = out_dir / 'manifest.json'
    merged: dict[tuple[str, str], dict] = {}
    if man_path.exists():
        try:
            for r in json.loads(man_path.read_text(encoding='utf-8')):
                merged[(r['scenario'], r['bag'])] = r
        except (json.JSONDecodeError, KeyError):
            merged = {}
    for r in results:
        merged[(r['scenario'], r['bag'])] = r
    man_path.write_text(json.dumps(sorted(merged.values(), key=lambda r: (r['scenario'], r['bag'])), indent=1,
                                   default=_np_default), encoding='utf-8')
    return results


def list_injectors() -> dict[str, dict]:
    """Injector types with their default parameters (for docs / CLI)."""
    out = {}
    for name, cls in sorted(REGISTRY.items()):
        out[name] = {'stage': cls.stage, 'doc': (cls.__doc__ or '').strip().splitlines()[0],
                     'defaults': {**cls.placement_defaults, **cls.defaults}}
    return out
