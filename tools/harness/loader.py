"""Loading of extracted bags (.npz) and dataset splits.

Each ``data/npz/<bag>.npz`` (written by ``tools/extract_bags.py``) holds one float64 array per
topic, rows sorted by bag receive time::

    vehicle__front_bogie_velocity / vehicle__rear_bogie_velocity : [t_bag, t_header, v_kmh]
    vehicle__driver_position_cmd                                 : [t_bag, t_header, notch]
    sensing__gnss__{master,rover}__fix : [t_bag, t_header, lat, lon, alt, status, cov_xx, cov_yy, cov_zz]
    sensing__gnss__{master,rover}__vel : [t_bag, t_header, vx_east, vy_north, vz_up, wz]

NOTE: wheel speeds are in km/h (the README says m/s, the data says otherwise: wheel/GNSS ~ 3.6).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # C:\MosTransHack
DATA_DIR = ROOT / 'data'
NPZ_DIR = DATA_DIR / 'npz'
SPLITS_PATH = DATA_DIR / 'splits.json'

# short name -> npz key
TOPIC_KEYS: Dict[str, str] = {
    'front': 'vehicle__front_bogie_velocity',
    'rear': 'vehicle__rear_bogie_velocity',
    'cmd': 'vehicle__driver_position_cmd',
    'fix_master': 'sensing__gnss__master__fix',
    'fix_rover': 'sensing__gnss__rover__fix',
    'vel_master': 'sensing__gnss__master__vel',
    'vel_rover': 'sensing__gnss__rover__vel',
}
NCOLS = {'front': 3, 'rear': 3, 'cmd': 3, 'fix_master': 9, 'fix_rover': 9, 'vel_master': 6, 'vel_rover': 6}
INPUT_TOPICS = ('front', 'rear', 'cmd')
GNSS_TOPICS = ('fix_master', 'fix_rover', 'vel_master', 'vel_rover')

# column indices
T_BAG, T_HDR = 0, 1
V_COL = 2                                     # wheel speed / notch
LAT, LON, ALT, STATUS = 2, 3, 4, 5            # fix
VX, VY, VZ = 2, 3, 4                          # vel


@dataclass
class BagData:
    """All topics of one bag as numpy arrays (see module docstring for column layout)."""
    name: str
    topics: Dict[str, np.ndarray]
    vehicle: str = ''
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.vehicle:
            self.vehicle = self.name.split('_')[0]
        starts = [a[0, T_BAG] for a in self.topics.values() if len(a)]
        ends = [a[-1, T_BAG] for a in self.topics.values() if len(a)]
        self.t_start = float(min(starts)) if starts else 0.0
        self.t_end = float(max(ends)) if ends else 0.0

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def __getitem__(self, key: str) -> np.ndarray:
        return self.topics[key]

    def has_gnss(self) -> bool:
        return len(self.topics['fix_master']) > 0

    def copy(self) -> 'BagData':
        return BagData(self.name, {k: v.copy() for k, v in self.topics.items()}, self.vehicle, dict(self.meta))


def _empty(key: str) -> np.ndarray:
    return np.zeros((0, NCOLS[key]))


def load_bag(bag: Union[str, Path], npz_dir: Union[str, Path] = NPZ_DIR) -> BagData:
    """Load ``<npz_dir>/<bag>.npz`` (or an explicit .npz path) into :class:`BagData`.

    Rows of every topic are (stable-)sorted by bag receive time.
    """
    p = Path(bag)
    if p.suffix != '.npz':
        p = Path(npz_dir) / f'{bag}.npz'
    with np.load(p) as z:
        topics = {}
        for short, key in TOPIC_KEYS.items():
            a = np.asarray(z[key], dtype=np.float64) if key in z.files else _empty(short)
            if a.ndim != 2 or a.shape[0] == 0:
                a = _empty(short)
            elif np.any(np.diff(a[:, T_BAG]) < 0):
                a = a[np.argsort(a[:, T_BAG], kind='stable')]
            topics[short] = a
    return BagData(p.stem, topics)


def load_splits(path: Union[str, Path] = SPLITS_PATH) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_bags(spec: Union[str, Sequence[str]], splits: dict = None) -> List[str]:
    """Turn a split spec into a list of bag names.

    ``spec`` may be a split name (``'val'``, ``'train'``, ``'no_gnss_long'``, ``'short'``),
    a '+'-joined combination (``'train+val'``), ``'gnss'`` (= train+val), or explicit bag names
    (list or comma-separated string). Duplicate bags (see splits['duplicates']) are dropped.
    """
    splits = splits or load_splits()
    if isinstance(spec, str):
        parts = [s for s in spec.replace(',', '+').split('+') if s]
    else:
        parts = list(spec)
    out: List[str] = []
    for part in parts:
        if part == 'gnss':
            out += splits['train'] + splits['val']
        elif part in splits and isinstance(splits[part], list) and part != 'info':
            out += splits[part]
        else:
            out.append(part)
    dups = set(splits.get('duplicates', {}).keys())
    seen, res = set(), []
    for b in out:
        if b in seen or b in dups:
            continue
        seen.add(b)
        res.append(b)
    return res


def bag_info(splits: dict = None) -> Dict[str, dict]:
    splits = splits or load_splits()
    return {r['bag']: r for r in splits.get('info', [])}
