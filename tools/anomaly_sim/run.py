"""In-memory representation of one recording (``Run``) made of per-topic ``Stream`` tables.

A ``Stream`` keeps, for every message that will be delivered to the estimator:

* ``t_bag``   - bag receive time [s] (defines the ``ros2 bag play`` order),
* ``t_hdr``   - header.stamp [s] as it will be published (possibly corrupted),
* ``val``     - payload columns (possibly corrupted),
* ``clean``   - payload the message would have had without value corruption,
* ``t_hdr0``  - header stamp before timing corruption,
* ``label``   - uint32 bit mask of :class:`~anomaly_sim.constants.Label` flags,
* ``src``     - row index of the originating message in the clean source (-1 = synthetic).

On disk a run is an ``.npz`` with exactly the same topic keys / column layout as the clean
files written by ``tools/extract_bags.py`` (so any existing loader works unchanged), plus
underscore-prefixed side arrays (``_label__<key>``, ``_clean__<key>``, ``_hdr0__<key>``,
``_src__<key>``) and ``_meta_json`` (scenario, seed, injected events, source path).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import ALL_TOPICS, COLUMNS, NPZ_DIR, VEHICLE

SIDE_PREFIXES = ('_label__', '_clean__', '_hdr0__', '_src__')


@dataclass
class Stream:
    key: str
    t_bag: np.ndarray
    t_hdr: np.ndarray
    val: np.ndarray
    clean: np.ndarray
    t_hdr0: np.ndarray
    label: np.ndarray
    src: np.ndarray

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_array(cls, key: str, arr: np.ndarray) -> 'Stream':
        arr = np.asarray(arr, dtype=np.float64)
        if key in COLUMNS:
            ncol = len(COLUMNS[key])
        else:
            ncol = max(arr.shape[1] - 2, 1) if arr.ndim == 2 else 1
        if arr.size == 0:  # the extractor stores empty topics as (0, 3)
            arr = np.zeros((0, 2 + ncol))
        val = arr[:, 2:].copy()
        n = len(arr)
        return cls(key=key, t_bag=arr[:, 0].copy(), t_hdr=arr[:, 1].copy(), val=val, clean=val.copy(),
                   t_hdr0=arr[:, 1].copy(), label=np.zeros(n, np.uint32), src=np.arange(n, dtype=np.int64))

    def to_array(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 2 + self.val.shape[1]))
        return np.column_stack([self.t_bag, self.t_hdr, self.val])

    def copy(self) -> 'Stream':
        return Stream(self.key, self.t_bag.copy(), self.t_hdr.copy(), self.val.copy(), self.clean.copy(),
                      self.t_hdr0.copy(), self.label.copy(), self.src.copy())

    def __len__(self) -> int:
        return len(self.t_bag)

    # ------------------------------------------------------------------ row operations
    def take(self, idx) -> 'Stream':
        idx = np.asarray(idx)
        return Stream(self.key, self.t_bag[idx], self.t_hdr[idx], self.val[idx], self.clean[idx],
                      self.t_hdr0[idx], self.label[idx], self.src[idx])

    def keep(self, mask: np.ndarray) -> 'Stream':
        return self.take(np.flatnonzero(mask))

    @staticmethod
    def concat(parts: list['Stream']) -> 'Stream':
        p0 = parts[0]
        return Stream(p0.key, *(np.concatenate([getattr(p, f) for p in parts])
                                for f in ('t_bag', 't_hdr', 'val', 'clean', 't_hdr0', 'label', 'src')))

    def sort(self) -> 'Stream':
        """Stable sort by bag receive time (= publication order during ``ros2 bag play``)."""
        order = np.argsort(self.t_bag, kind='stable')
        if np.all(order == np.arange(len(order))):
            return self
        return self.take(order)

    @property
    def v(self) -> np.ndarray:
        """First payload column (wheel velocity / notch)."""
        return self.val[:, 0]


@dataclass
class Run:
    name: str
    streams: dict[str, Stream]
    meta: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def copy(self) -> 'Run':
        return Run(self.name, {k: s.copy() for k, s in self.streams.items()}, json.loads(json.dumps(self.meta)),
                   [dict(e) for e in self.events])

    def __getitem__(self, key: str) -> Stream:
        return self.streams[key]

    def __contains__(self, key: str) -> bool:
        return key in self.streams

    @property
    def t_start(self) -> float:
        """Bag time of the first message of the recording (all topics)."""
        firsts = [s.t_bag.min() for s in self.streams.values() if len(s)]
        return float(min(firsts)) if firsts else 0.0

    @property
    def t_end(self) -> float:
        lasts = [s.t_bag.max() for s in self.streams.values() if len(s)]
        return float(max(lasts)) if lasts else 0.0

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def has_gnss(self) -> bool:
        return any(len(self.streams.get(k, ())) > 0 for k in ('sensing__gnss__master__fix', 'sensing__gnss__rover__fix'))

    def crop(self, t0_rel: float, t1_rel: float) -> 'Run':
        """Keep messages whose bag time lies in [t_start + t0_rel, t_start + t1_rel] (for tests/demos)."""
        t0 = self.t_start + t0_rel
        t1 = self.t_start + t1_rel
        out = self.copy()
        for k, s in out.streams.items():
            out.streams[k] = s.keep((s.t_bag >= t0) & (s.t_bag <= t1))
        out.meta['crop'] = [t0_rel, t1_rel]
        return out


# ---------------------------------------------------------------------------------------- I/O

def load_run(path: str | Path, name: str | None = None) -> Run:
    """Load a clean (extracted) or corrupted npz into a :class:`Run`.

    ``path`` may also be a bare bag name, in which case ``data/npz/<name>.npz`` is used.
    """
    path = Path(path)
    if not path.suffix and not path.exists():
        path = NPZ_DIR / f'{path.name}.npz'
    with np.load(path, allow_pickle=False) as d:
        files = set(d.files)
        meta = json.loads(str(d['_meta_json'])) if '_meta_json' in files else {}
        streams: dict[str, Stream] = {}
        for key in d.files:
            if key.startswith('_'):
                continue
            s = Stream.from_array(key, d[key])
            if f'_label__{key}' in files:
                s.label = d[f'_label__{key}'].astype(np.uint32)
            if f'_clean__{key}' in files:
                s.clean = d[f'_clean__{key}'].astype(np.float64).reshape(len(s), -1)
            if f'_hdr0__{key}' in files:
                s.t_hdr0 = d[f'_hdr0__{key}'].astype(np.float64)
            if f'_src__{key}' in files:
                s.src = d[f'_src__{key}'].astype(np.int64)
            streams[key] = s
    events = meta.pop('events', []) if isinstance(meta, dict) else []
    if 'source_npz' not in meta and not meta.get('scenario'):
        meta['source_npz'] = str(path)
    return Run(name=name or meta.get('bag') or path.stem, streams=streams, meta=meta, events=events)


def save_run(run: Run, path: str | Path, side_arrays: bool = True) -> Path:
    """Save a run as npz with the extractor's layout plus side arrays and ``_meta_json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, s in run.streams.items():
        arrays[key] = s.to_array()
        if side_arrays:
            arrays[f'_label__{key}'] = s.label.astype(np.uint32)
            arrays[f'_src__{key}'] = s.src.astype(np.int64)
            if key in VEHICLE:
                arrays[f'_clean__{key}'] = s.clean[:, 0].copy() if s.clean.shape[1] == 1 else s.clean
                arrays[f'_hdr0__{key}'] = s.t_hdr0
    meta = dict(run.meta)
    meta['bag'] = run.name
    meta['events'] = run.events
    arrays['_meta_json'] = np.array(json.dumps(meta, default=_json_default))
    np.savez_compressed(path, **arrays)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serialisable: {type(o)}')


def load_pair(corrupted_path: str | Path) -> tuple[Run, Run]:
    """Load a corrupted run and the clean run it was generated from (for reference/metrics)."""
    cr = load_run(corrupted_path)
    src = cr.meta.get('source_npz')
    if not src or not Path(src).exists():
        src = NPZ_DIR / f'{cr.name}.npz'
    clean = load_run(src, name=cr.name)
    if 'crop' in cr.meta:
        clean = clean.crop(*cr.meta['crop'])
    return cr, clean


def check_npz(path: str | Path) -> list[str]:
    """Schema check of a (clean or corrupted) npz. Returns a list of problems (empty = OK)."""
    problems: list[str] = []
    with np.load(path, allow_pickle=False) as d:
        keys = [k for k in d.files if not k.startswith('_')]
        for key in ALL_TOPICS:
            if key not in keys:
                problems.append(f'missing topic {key}')
        for key in keys:
            a = d[key]
            if a.ndim != 2 or (len(a) and a.shape[1] != 2 + len(COLUMNS.get(key, (0,)))):
                problems.append(f'{key}: bad shape {a.shape}')
                continue
            if a.dtype != np.float64:
                problems.append(f'{key}: dtype {a.dtype}')
            if len(a) > 1 and np.any(np.diff(a[:, 0]) < 0):
                problems.append(f'{key}: bag times not sorted')
            if not np.all(np.isfinite(a[:, :2])):
                problems.append(f'{key}: non-finite times')
            for side in SIDE_PREFIXES:
                sk = side + key
                if sk in d.files and len(d[sk]) != len(a):
                    problems.append(f'{sk}: length {len(d[sk])} != {len(a)}')
        if '_meta_json' in d.files:
            try:
                json.loads(str(d['_meta_json']))
            except json.JSONDecodeError as e:  # pragma: no cover
                problems.append(f'_meta_json invalid: {e}')
    return problems
