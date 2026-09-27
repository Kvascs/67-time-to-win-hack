"""Multi-objective tuning of the bogie-ratio corrections (D33) with IBEA-eps+ (Zitzler & Kuenzli, 2004).

Protocol as in tools/harness/tbo_sweep.py: tune on TRAIN bags with an RTK reference (maps built from
train, i.e. in-sample), confirm the chosen point once on VAL (train-only maps), on the organisers' bag
with their metric (tools/replay/eval_checker.py) and on the fault suite (tools/anomaly_sim).

Objectives, all minimised over the tuning bags:
  f1  median along-track RMSE, m        (typical bag)
  f2  mean along-track RMSE, m          (tail: a few bags with large errors dominate it)
  f3  median speed RMSE vs Doppler, m/s (the corrections must not cost speed)

IBEA-eps+: additive eps-indicator on objectives normalised to [0, 1] by the bounds of the merged
population; fitness F(x) = sum_{y != x} -exp(-I(y, x) / (c * kappa)), c = max |I|, kappa = 0.05;
environmental selection removes the worst individual one at a time and updates the others;
binary tournament mating; SBX crossover (eta 15, p 0.9) and polynomial mutation (eta 20, p 1/n)
on the parameters scaled to [0, 1]. The hand-set values are one individual of the first population.

    python tools/tune/ibea_d33.py search --pop 10 --gens 6 --workers 5      # raw criteria, new journal
    python tools/tune/ibea_d33.py resume --pop 10 --until 21:45 --workers 6  # continue the journal, normalised criteria
    python tools/tune/ibea_d33.py select                                    # protocol: candidates from the journal
    python tools/tune/ibea_d33.py cliff --point "ratio_margin=2.5"           # protocol: each parameter +-10 %
    python tools/tune/ibea_d33.py confirm --split val --point "ratio_step_m=40,ratio_margin=2.5"
    python tools/tune/ibea_d33.py plot                                      # docs/img/ibea_front.png
Run on 27.09: 20 evaluations with search, 18 with resume, 36 with tools/tune/mootation_d33.py (analysis/tuning/ibea_d33/, PROTOCOL.md).
Continuation with MOOtation's implementation of the same algorithm: tools/tune/mootation_d33.py.
The journal is the D33 tuning: it was evaluated with the D33 build, where a correction does not move the wheel
scale (ratio_update_k = 0). D34 changed that default to 1 (ratio_k_gain 0.3), so BASE pins ratio_update_k = 0:
with a D34 binary the evaluations stay D33 (the code path is then identical) and the journal is not mixed.
--exe must be a build with the bogie-ratio corrections (D33 or later); BASE is not part of the journal key.

Writes analysis/tuning/ibea_d33/: evals.csv (every evaluated point), bags.csv (per-bag rows),
front.csv (final population with the non-dominated flag), summary.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))
import cpp_bridge  # noqa: E402
import quick_eval  # noqa: E402

OUT = ROOT / 'analysis' / 'tuning' / 'ibea_d33'
V = ROOT / 'analysis' / 'validation_maps'
BASE = {'wheel_epochs_file': str(V / 'wheel_epochs.csv'), 'ratio_map_file': str(V / 'ratio_map.csv'),
        'speed_output_delay_s': 0, 'position_output_delay_s': 0,
        'ratio_update_k': 0}  # D33 behaviour on any build: the journal is the D33 tuning (see the docstring)
PARAMS = [  # name, lower, upper, hand-set value
    ('ratio_step_m', 20.0, 100.0, 50.0),
    ('ratio_window_m', 100.0, 300.0, 200.0),
    ('ratio_margin', 1.0, 6.0, 3.0),
    ('ratio_sigma_max', 0.4, 1.5, 0.8),
    ('ratio_sigma_min', 0.1, 0.8, 0.35),
    ('ratio_dmax', 1.5, 6.0, 3.0),
    ('ratio_gate_sd', 2.0, 5.0, 3.0),
]
NAMES = [p[0] for p in PARAMS]
LO = np.array([p[1] for p in PARAMS])
HI = np.array([p[2] for p in PARAMS])
X0 = np.array([p[3] for p in PARAMS])
OBJ = ['f1_along_med', 'f2_along_mean', 'f3_v_med']
KAPPA = 0.05


def decode(u):
    return LO + np.clip(u, 0.0, 1.0) * (HI - LO)


def as_sets(x):
    return {n: round(float(v), 3) for n, v in zip(NAMES, x)}


def _init(exe):
    cpp_bridge.REPLAY_EXE = Path(exe)
    quick_eval.TMP = ROOT / 'build_core' / 'ga_tmp' / f'w{os.getpid()}'


def _job(arg):
    cid, bag, sets = arg
    r, _ = quick_eval.eval_bag(bag, sets=sets)
    return cid, r


def rtk_bags(split):
    bags = json.load(open(ROOT / 'data' / 'splits.json'))[split]
    keep = []
    for b in bags:
        d = np.load(cpp_bridge.NPZ / f'{b}.npz')
        mf = d['sensing__gnss__master__fix']
        mf = mf[np.isfinite(mf[:, 2])]
        if len(mf) and (mf[:, 5] == 2).mean() > 0.8:
            keep.append(b)
    return keep


def objectives(rows):
    df = pd.DataFrame(rows)
    return np.array([df.along_rmse.median(), df.along_rmse.mean(), df.v_rmse.median()]), df


KEY0 = json.dumps(as_sets(X0), sort_keys=True)


class Evaluator:
    """Evaluates a batch of points: all (point, bag) jobs go to one process pool; every evaluation is appended to
    the journal (evals.csv, bags.csv), and the journal is read back as a cache, so a stopped search resumes.
    norm=True: objectives are per-bag ratios to the hand-set point (PROTOCOL.md): median and p90 of the along-track
    RMSE ratio, median of the speed RMSE ratio; a single heavy bag cannot drive them."""

    def __init__(self, bags, pool, tag, norm=False):
        self.bags, self.pool, self.tag, self.norm = bags, pool, tag, norm
        self.rows, self.n = {}, 0
        OUT.mkdir(parents=True, exist_ok=True)
        if (OUT / 'evals.csv').exists() and (OUT / 'bags.csv').exists():
            e, b = pd.read_csv(OUT / 'evals.csv'), pd.read_csv(OUT / 'bags.csv')
            self.n = int(e['eval'].max())
            for _, r in e.iterrows():
                g = b[b['eval'] == r['eval']].drop_duplicates('bag').set_index('bag')
                if set(bags) <= set(g.index):
                    self.rows[json.dumps({nm: round(float(r[nm]), 3) for nm in NAMES}, sort_keys=True)] = g.loc[bags]

    def objectives(self, df):
        if not self.norm:
            return np.array([df.along_rmse.median(), df.along_rmse.mean(), df.v_rmse.median()])
        base = self.rows[KEY0]
        ra = df.along_rmse.to_numpy() / base.along_rmse.to_numpy()
        rv = df.v_rmse.to_numpy() / base.v_rmse.to_numpy()
        return np.array([np.median(ra), np.quantile(ra, 0.9), np.median(rv)])

    def __call__(self, xs):
        keys = [json.dumps(as_sets(x), sort_keys=True) for x in xs]
        todo = [k for k in dict.fromkeys(keys) if k not in self.rows]
        jobs = [(k, b, {**BASE, **json.loads(k)}) for k in todo for b in self.bags]
        per = {k: [] for k in todo}
        for k, r in self.pool.map(_job, jobs, chunksize=1):
            per[k].append(r)
        for k in todo:
            df = pd.DataFrame(per[k]).set_index('bag').loc[self.bags]
            self.rows[k] = df
            self.n += 1
            raw = np.array([df.along_rmse.median(), df.along_rmse.mean(), df.v_rmse.median()])
            row = {'eval': self.n, 'tag': self.tag, **json.loads(k), **dict(zip(OBJ, raw)),
                   'p3_mean': df.p3_rmse.mean(), 'v_mean': df.v_rmse.mean()}
            out = df.reset_index()[['bag', 'along_rmse', 'p3_rmse', 'v_rmse', 'end_err']]
            out.insert(0, 'eval', self.n)
            for name, frame in (('evals.csv', pd.DataFrame([row])), ('bags.csv', out)):
                p = OUT / name
                frame.to_csv(p, mode='a', header=not p.exists(), index=False)
        return np.array([self.objectives(self.rows[k]) for k in keys])


def normalise(F):
    lo, hi = F.min(axis=0), F.max(axis=0)
    return (F - lo) / np.where(hi > lo, hi - lo, 1.0)


def ibea_fitness(F):
    Fn = normalise(F)
    indicator = (Fn[:, None, :] - Fn[None, :, :]).max(axis=2)   # I(a, b): shift of a to weakly dominate b
    c = max(np.abs(indicator).max(), 1e-12)
    E = np.exp(-indicator / (c * KAPPA))
    np.fill_diagonal(E, 0.0)
    return -E.sum(axis=0), E


def environmental_selection(F, alpha):
    fit, E = ibea_fitness(F)
    alive = np.ones(len(F), bool)
    while alive.sum() > alpha:
        cand = np.flatnonzero(alive)
        w = cand[np.argmin(fit[cand])]
        alive[w] = False
        fit = fit + E[w]
    return np.flatnonzero(alive)


def sbx(p1, p2, rng, eta=15.0, pc=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-12:
            continue
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def poly_mutation(x, rng, eta=20.0):
    y = x.copy()
    for i in range(len(x)):
        if rng.random() > 1.0 / len(x):
            continue
        u = rng.random()
        d = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        y[i] = np.clip(y[i] + d, 0, 1)
    return y


def nondominated(F):
    nd = np.ones(len(F), bool)
    for i in range(len(F)):
        nd[i] = not np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
    return nd


def search(a):
    rng = np.random.default_rng(a.seed)
    bags = rtk_bags(a.split)[:a.max_bags or None]
    print(f'{len(bags)} {a.split} bags with RTK; pop {a.pop}, {a.gens} generations', flush=True)
    for name in ('evals.csv', 'bags.csv'):
        (OUT / name).unlink(missing_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.exe,)) as pool:
        ev = Evaluator(bags, pool, a.split)
        # first population: the hand-set point plus a Latin hypercube
        lhs = (rng.permuted(np.tile(np.arange(a.pop - 1), (len(NAMES), 1)), axis=1).T
               + rng.random((a.pop - 1, len(NAMES)))) / (a.pop - 1)
        U = np.vstack([(X0 - LO) / (HI - LO), lhs])
        F = ev([decode(u) for u in U])
        print(f'gen 0: {time.time() - t0:.0f} s, hand-set {np.round(F[0], 4)}, best f2 {F[:, 1].min():.4f}', flush=True)
        for g in range(1, a.gens + 1):
            fit, _ = ibea_fitness(F)
            kids = []
            while len(kids) < a.pop:
                i, j = rng.integers(len(U), size=2), rng.integers(len(U), size=2)
                p1 = U[i[0]] if fit[i[0]] > fit[i[1]] else U[i[1]]
                p2 = U[j[0]] if fit[j[0]] > fit[j[1]] else U[j[1]]
                for c in sbx(p1, p2, rng):
                    kids.append(poly_mutation(c, rng))
            K = np.array(kids[:a.pop])
            FK = ev([decode(u) for u in K])
            U, F = np.vstack([U, K]), np.vstack([F, FK])
            keep = environmental_selection(F, a.pop)
            U, F = U[keep], F[keep]
            print(f'gen {g}: {time.time() - t0:.0f} s, evals {ev.n}, best f1 {F[:, 0].min():.4f} '
                  f'f2 {F[:, 1].min():.4f} f3 {F[:, 2].min():.4f}', flush=True)
    X = np.array([decode(u) for u in U])
    front = pd.DataFrame(X, columns=NAMES).round(3)
    for k, name in enumerate(OBJ):
        front[name] = F[:, k]
    front['nondominated'] = nondominated(F)
    front.sort_values('f2_along_mean').to_csv(OUT / 'front.csv', index=False)
    print(front.sort_values('f2_along_mean').round(4).to_string(index=False))


def resume(a):
    """Continue the search from the journal with the normalised objectives (PROTOCOL.md) until --until HH:MM."""
    import datetime as dt
    rng = np.random.default_rng(a.seed + 1)
    bags = rtk_bags(a.split)[:a.max_bags or None]
    hh, mm = (int(t) for t in a.until.split(':'))
    deadline = dt.datetime.now().replace(hour=hh, minute=mm, second=0).timestamp()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.exe,)) as pool:
        ev = Evaluator(bags, pool, a.split + '-norm', norm=True)
        ev([X0])
        keys = list(ev.rows)
        U = np.array([(np.array([json.loads(k)[nm] for nm in NAMES]) - LO) / (HI - LO) for k in keys])
        F = np.array([ev.objectives(ev.rows[k]) for k in keys])
        keep = environmental_selection(F, a.pop) if len(U) > a.pop else np.arange(len(U))
        U, F = U[keep], F[keep]
        print(f'resumed from {len(keys)} evaluated points, population {len(U)}', flush=True)
        g, dur = 0, 0.0
        while time.time() + dur < deadline:
            tg = time.time()
            fit, _ = ibea_fitness(F)
            kids = []
            while len(kids) < a.pop:
                i, j = rng.integers(len(U), size=2), rng.integers(len(U), size=2)
                p1 = U[i[0]] if fit[i[0]] > fit[i[1]] else U[i[1]]
                p2 = U[j[0]] if fit[j[0]] > fit[j[1]] else U[j[1]]
                for c in sbx(p1, p2, rng):
                    kids.append(poly_mutation(c, rng))
            K = np.array(kids[:a.pop])
            FK = ev([decode(u) for u in K])
            U, F = np.vstack([U, K]), np.vstack([F, FK])
            keep = environmental_selection(F, a.pop)
            U, F = U[keep], F[keep]
            g += 1
            dur = time.time() - tg
            print(f'gen +{g}: {time.time() - t0:.0f} s, evals {ev.n}, best median ratio {F[:, 0].min():.4f} '
                  f'p90 {F[:, 1].min():.4f} speed {F[:, 2].min():.4f}', flush=True)
    print('stopped at the deadline' if time.time() + dur >= deadline else 'done', flush=True)


def confirm(a):
    bags = json.load(open(ROOT / 'data' / 'splits.json'))[a.split]
    rtk = set(rtk_bags(a.split))
    pts = {'hand-set': as_sets(X0)}
    for i, s in enumerate(a.point):
        pts[f'p{i + 1}'] = {**as_sets(X0), **{k: float(v) for k, v in (kv.split('=') for kv in s.split(','))}}
    jobs = [(name, b, {**BASE, **sets}) for name, sets in pts.items() for b in bags]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.exe,)) as pool:
        res = list(pool.map(_job, jobs, chunksize=1))
    df = pd.DataFrame([{'point': n, **r} for n, r in res])
    df.to_csv(OUT / f'confirm_{a.split}.csv', index=False)
    for n in pts:
        d = df[df.point == n]
        dr = d[d.bag.isin(rtk)]
        print(f'{n:9s} along median {dr.along_rmse.median():.4f} mean {dr.along_rmse.mean():.4f} | 3-D median '
              f'{dr.p3_rmse.median():.4f} mean {dr.p3_rmse.mean():.4f} | speed median {d.v_rmse.median():.4f} '
              f'mean {d.v_rmse.mean():.4f} ({len(dr)} RTK / {len(d)} bags)  {pts[n]}', flush=True)


def normalised(bags_csv=None):
    """Per-bag ratios to the hand-set point for every evaluated point (analysis/tuning/ibea_d33/PROTOCOL.md)."""
    e = pd.read_csv(OUT / 'evals.csv')
    b = pd.read_csv(bags_csv or OUT / 'bags.csv')
    hand = int(e.loc[np.all(np.isclose(e[NAMES].to_numpy(), X0), axis=1), 'eval'].iloc[0])
    base = b[b['eval'] == hand].set_index('bag')
    rows = []
    for ev, g in b.groupby('eval'):
        g = g.set_index('bag')
        ra = g.along_rmse / base.along_rmse.reindex(g.index)
        rv = g.v_rmse / base.v_rmse.reindex(g.index)
        rows.append({'eval': ev, 'along_ratio_med': ra.median(), 'along_ratio_p90': ra.quantile(0.9),
                     'v_ratio_med': rv.median(), 'better': int((ra < 0.999).sum()), 'worse': int((ra > 1.001).sum())})
    n = pd.DataFrame(rows).merge(e, on='eval')
    n['nondominated'] = nondominated(n[['along_ratio_med', 'along_ratio_p90', 'v_ratio_med']].to_numpy())
    return n, hand


def select(a):
    n, hand = normalised()
    pd.set_option('display.width', 250)
    cols = ['eval', 'along_ratio_med', 'along_ratio_p90', 'v_ratio_med', 'better', 'worse', 'f1_along_med',
            'f2_along_mean', 'nondominated'] + NAMES
    ok = n[n.nondominated & (n.v_ratio_med <= 1.0) & (n.along_ratio_p90 <= 1.0)].sort_values('along_ratio_med')
    print(f'{len(n)} evaluated, {int(n.nondominated.sum())} non-dominated; hand-set = eval {hand}')
    print(n[n.nondominated].sort_values('along_ratio_med')[cols].round(4).to_string(index=False))
    print()
    print('candidates by the protocol (speed ratio <= 1, tail <= 1), best first:')
    print(ok[cols].round(4).to_string(index=False) if len(ok) else '  none')
    n.to_csv(OUT / 'normalised.csv', index=False)


def cliff(a):
    """Each parameter of the given point +-10 % (protocol step 2), on the tuning bags."""
    point = {**as_sets(X0), **{k: float(v) for k, v in (kv.split('=') for kv in a.point[0].split(','))}}
    x0 = np.array([point[nm] for nm in NAMES])
    xs = [x0]
    for i in range(len(NAMES)):
        for f in (0.9, 1.1):
            x = x0.copy()
            x[i] *= f
            xs.append(x)
    bags = rtk_bags(a.split)[:a.max_bags or None]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.exe,)) as pool:
        Evaluator(bags, pool, f'cliff-{a.split}')(xs)
    n, hand = normalised()
    last = n.tail(len(xs))
    print(last[['eval', 'along_ratio_med', 'along_ratio_p90', 'v_ratio_med', 'better', 'worse'] + NAMES].round(4).to_string(index=False))


def plot(a):
    """docs/img/ibea_front.png: every evaluated point, the non-dominated ones and the hand-set point."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    surface, ink, ink2, muted, grid, axis, accent = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#2a78d6'
    plt.rcParams.update({
        'figure.facecolor': surface, 'axes.facecolor': surface, 'savefig.facecolor': surface,
        'axes.edgecolor': axis, 'axes.labelcolor': ink2, 'xtick.color': ink2, 'ytick.color': ink2,
        'text.color': ink, 'axes.grid': True, 'grid.color': grid, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'legend.frameon': False})
    n, hand = normalised()
    n = n[n.tag.astype(str).str.startswith('train')]
    nd = n.nondominated.to_numpy()
    h = n[n['eval'] == hand].iloc[0]
    cap = 1.25
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for ax, (col, label) in zip(axes, (('along_ratio_p90', 'хвост: p90 отношения по бэгам'),
                                       ('v_ratio_med', 'скорость: медиана отношения RMSE'))):
        x = n.along_ratio_med.clip(upper=cap)
        y = n[col].clip(upper=cap)
        ax.axvline(1.0, color=axis, lw=1.0, zorder=1)
        ax.axhline(1.0, color=axis, lw=1.0, zorder=1)
        ax.scatter(x[~nd], y[~nd], s=26, color=muted, alpha=0.7, label='оценённые точки', zorder=2)
        ax.scatter(x[nd], y[nd], s=40, color=accent, edgecolor=surface, linewidth=1.2,
                   label='недоминируемые (3 критерия)', zorder=3)
        ax.scatter([h.along_ratio_med], [h[col]], s=120, facecolor='none', edgecolor=ink, linewidth=1.6,
                   label='ручные значения (1; 1)', zorder=4)
        n_cap = int(((n.along_ratio_med > cap) | (n[col] > cap)).sum())
        if n_cap:
            ax.text(0.02, 0.97, f'{n_cap} точ. за краем (обрыв) прижаты к {cap}', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8, color=ink2)
        ax.set_xlabel('медиана отношения вдоль пути')
        ax.set_ylabel(label)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8.5)
    fig.suptitle(f'IBEA-ε+ по 7 параметрам D33 (оценок — {len(n)}, 41 бэг train): лучше ручных — левее и ниже (1; 1)',
                 fontsize=10.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    img = ROOT / 'docs' / 'img' / 'ibea_front.png'
    fig.savefig(img, dpi=110)
    print('wrote', img)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['search', 'resume', 'confirm', 'plot', 'select', 'cliff'])
    ap.add_argument('--until', default='21:50', help='resume: stop before a generation that would end later')
    ap.add_argument('--split', default='train')
    ap.add_argument('--pop', type=int, default=12)
    ap.add_argument('--gens', type=int, default=6)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=33)
    ap.add_argument('--exe', default=str(ROOT / 'build_core' / ('tbo_replay.exe' if os.name == 'nt' else 'tbo_replay')))
    ap.add_argument('--point', action='append', default=[])
    ap.add_argument('--max-bags', type=int, default=0, help='smoke test: first N bags only')
    a = ap.parse_args()
    {'search': search, 'resume': resume, 'confirm': confirm, 'plot': plot, 'select': select, 'cliff': cliff}[a.mode](a)
