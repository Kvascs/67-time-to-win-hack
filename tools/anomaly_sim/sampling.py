"""Parameter specs used in scenario files and their random sampling.

Conventions (YAML / JSON):

* ``3.5``                      -> fixed value
* ``[1.0, 10.0]``              -> uniform float in [lo, hi]  (``sample_int``: integer in [lo, hi])
* ``{loguniform: [0.2, 10]}``  -> log-uniform
* ``{normal: [mu, sigma]}``    -> Gaussian
* ``{choice: [a, b, c]}``      -> uniform choice among values
* ``{front: 0.4, rear: 0.4, both: 0.2}`` -> weighted categorical choice (``sample_choice``)
* ``[front, rear]``            -> uniform categorical choice (``sample_choice``)
"""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np

_DIST_KEYS = {'loguniform', 'normal', 'choice', 'uniform'}


def _is_num(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def sample_float(spec: Any, rng: np.random.Generator) -> float:
    if spec is None:
        return None  # type: ignore[return-value]
    if _is_num(spec):
        return float(spec)
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(_is_num(x) for x in spec):
        lo, hi = float(spec[0]), float(spec[1])
        return float(rng.uniform(min(lo, hi), max(lo, hi)))
    if isinstance(spec, dict) and len(spec) == 1:
        (k, v), = spec.items()
        if k == 'uniform':
            return float(rng.uniform(*v))
        if k == 'loguniform':
            lo, hi = np.log(v[0]), np.log(v[1])
            return float(np.exp(rng.uniform(lo, hi)))
        if k == 'normal':
            return float(rng.normal(v[0], v[1]))
        if k == 'choice':
            return float(v[rng.integers(len(v))])
    raise ValueError(f'cannot sample a float from {spec!r}')


def sample_int(spec: Any, rng: np.random.Generator) -> int:
    if _is_num(spec):
        return int(round(float(spec)))
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(_is_num(x) for x in spec):
        lo, hi = int(spec[0]), int(spec[1])
        return int(rng.integers(min(lo, hi), max(lo, hi) + 1))
    if isinstance(spec, dict) and len(spec) == 1 and 'choice' in spec:
        v = spec['choice']
        return int(v[rng.integers(len(v))])
    return int(round(sample_float(spec, rng)))


def sample_choice(spec: Any, rng: np.random.Generator):
    if isinstance(spec, dict) and not (_DIST_KEYS & set(spec)):
        keys = list(spec)
        w = np.array([float(spec[k]) for k in keys])
        if w.sum() <= 0:
            raise ValueError(f'weights must be positive: {spec!r}')
        return keys[int(rng.choice(len(keys), p=w / w.sum()))]
    if isinstance(spec, dict) and 'choice' in spec:
        v = spec['choice']
        return v[rng.integers(len(v))]
    if isinstance(spec, (list, tuple)):
        return spec[rng.integers(len(spec))]
    return spec


def sample_bool(spec: Any, rng: np.random.Generator) -> bool:
    """``true``/``false`` or a probability in [0, 1]."""
    if isinstance(spec, bool):
        return spec
    return bool(rng.random() < float(spec))


def make_rng(seed: int, bag: str, *extra: int) -> np.random.Generator:
    """Deterministic generator for (scenario seed, bag name, injector index...)."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(zlib.crc32(bag.encode()), *map(int, extra)))
    return np.random.default_rng(ss)
