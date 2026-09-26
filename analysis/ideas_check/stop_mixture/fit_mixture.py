"""Stop places as Gaussian mixtures (hypothesis test, nothing is installed anywhere).

Idea under test: model each stop landmark as a 1-D mixture along s (EM on TRAIN stop events), so that
secondary stop modes (e.g. a "queue" position a few metres before a platform) become explicit
PDA candidates instead of being pulled onto the platform.

Steps
  1. For every landmark: TRAIN stop events within +-12 m, EM fits K=1..3 (a) pure Gaussian mixture and
     (b) Gaussians + uniform background over the window; BIC selection; classify components
     (primary / an already existing neighbouring landmark / real secondary / weak).
  2. VAL recurrence of every train secondary mode (+-max(1 m, 2 sigma)) against a background null.
  3. Offline benefit: val stop events near landmarks, PDA emulation (same formulas and parameters as
     Estimator::placeUpdate) with the current landmark file vs landmarks + mixture rows, on a grid of
     prediction errors / association variances; "snap" errors and wrong-component rates;
     stops in the 1.5-4 m "unknown place" band explained by a mixture component.
  4. Files in the landmark format (s,sigma,p_stop,cls): the current rows unchanged + one extra row per
     real secondary mode; sidecar *_components.csv with the fit details.

Inputs : analysis/validation_maps/landmarks.csv (train map), analysis/map_build/map_train|map/{stops,stops_events}.csv,
         analysis/map_build/cache/run_*.pkl (pass counts for p_stop), data/splits.json
Outputs: this directory (see README-less file list printed at the end; log in fit_mixture_log.txt)
Run    : python analysis/ideas_check/stop_mixture/fit_mixture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MB = ROOT / 'analysis' / 'map_build'
sys.path.insert(0, str(MB))

WIN = 12.0          # half window around a landmark [m]
SIG_FLOOR = 0.10    # EM sigma floor [m] (~ sklearn reg_covar=0.01)
KMAX = 3
# selection rules for a "real" secondary mode (same for train and all)
MIN_SEP = 1.0       # |offset from the landmark| [m]
MAX_SIG = 1.0       # component sigma [m] (landmark rule: robust std <= 1 m)
MIN_EV = 3          # hard-assigned events
MIN_RUNS = 3        # distinct runs (landmark rule: n_runs >= 3)
KNOWN_TOL = 1.0     # a component within max(1 m, 2 sigma) of another landmark is that landmark
# PDA parameters (config/params.yaml)
PDA = dict(g=3.0, extra=0.3, p_random=0.15, min_prob=0.6, q=0.008)


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8', newline='\n')
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.o.flush()
        self.f.flush()


def cyc(a, L):
    """signed cyclic difference wrapped to (-L/2, L/2]"""
    x = np.mod(a, L)
    return np.where(x > L / 2, x - L, x)


# ------------------------------------------------------------------------------------------------
# 1-D EM: K Gaussians (+ optional uniform background over a window of width W)
# ------------------------------------------------------------------------------------------------
def _npdf(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)


def em_fit(x, K, bg, W=2 * WIN, n_init=40, seed=0, max_iter=2000, tol=1e-9):
    """EM with n_init restarts run side by side (vectorised); returns the best local optimum."""
    x = np.asarray(x, float)
    n = len(x)
    if K == 0:
        return dict(K=0, bg=True, ll=n * np.log(1.0 / W), pi=np.zeros(0), mu=np.zeros(0), sig=np.zeros(0),
                    pi0=1.0, p=0)
    if n < K:
        return None
    rng = np.random.default_rng(seed + 17 * K + (1 if bg else 0))
    I = n_init
    mu = np.empty((I, K))
    mu[0] = np.quantile(x, (np.arange(K) + 0.5) / K)
    for i in range(1, I):
        mu[i] = rng.choice(x, K, replace=False) + rng.normal(0, 0.05, K)
    sig = np.full((I, K), 0.5)
    sig[I // 2:] = rng.uniform(0.1, 2.0, (I - I // 2, K))
    pi0 = np.full(I, 0.1 if bg else 0.0)
    pi = np.repeat(((1 - pi0) / K)[:, None], K, 1)
    alive = np.ones(I, bool)
    ll_prev = np.full(I, -np.inf)
    xx = x[None, :, None]
    for _ in range(max_iter):
        dens = pi[:, None, :] * _npdf(xx, mu[:, None, :], sig[:, None, :])      # I x n x K
        tot = dens.sum(2) + (pi0 / W)[:, None]
        alive &= np.all(tot > 0, 1)
        tot = np.where(tot > 0, tot, 1e-300)
        ll = np.log(tot).sum(1)
        r = dens / tot[:, :, None]
        Nk = r.sum(1)
        alive &= np.all(Nk > 1e-8, 1)
        Nk = np.maximum(Nk, 1e-12)
        pi = Nk / n
        if bg:
            pi0 = ((pi0 / W)[:, None] / tot).sum(1) / n
        mu = (r * x[None, :, None]).sum(1) / Nk
        sig = np.sqrt(np.maximum((r * (x[None, :, None] - mu[:, None, :]) ** 2).sum(1) / Nk, SIG_FLOOR ** 2))
        if np.all(np.abs(ll - ll_prev)[alive] < tol):
            break
        ll_prev = ll
    dens = pi[:, None, :] * _npdf(xx, mu[:, None, :], sig[:, None, :])
    ll = np.where(alive, np.log(dens.sum(2) + (pi0 / W)[:, None]).sum(1), -np.inf)
    if not np.isfinite(ll).any():
        return None
    b = int(np.argmax(ll))
    o = np.argsort(mu[b])
    return dict(K=K, bg=bg, ll=float(ll[b]), pi=pi[b][o], mu=mu[b][o], sig=sig[b][o], pi0=float(pi0[b]),
                p=3 * K if bg else 3 * K - 1)


def bic(fit, n):
    return -2 * fit['ll'] + fit['p'] * np.log(n)


def responsibilities(fit, x, W=2 * WIN):
    dens = fit['pi'] * _npdf(np.asarray(x, float)[:, None], fit['mu'], fit['sig'])
    tot = dens.sum(1) + fit['pi0'] / W
    return dens / tot[:, None], (fit['pi0'] / W) / tot


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def select_landmarks(stops_csv):
    """export_core_map.py rules (tools/map/export_core_map.py)"""
    st = pd.read_csv(stops_csv)
    lm = st[(st.edge == 'main') & st.cls.isin(['platform', 'signal', 'terminal']) &
            (st.n_runs >= 3) & (st.s_robust_std <= 1.0)].sort_values('s_median')
    return pd.DataFrame(dict(s=lm.s_median.round(3).values, sigma=np.maximum(lm.s_robust_std, 0.2).round(3).values,
                             p_stop=lm.p_stop.fillna(0.3).round(3).values, cls=lm.cls.values,
                             cluster=lm.cluster.values))


def load_passes(map_dir, bags):
    import data_io as D  # noqa: F401
    import runs as R
    import stops_analysis as SA
    from validate import MapProjector
    mp = MapProjector(MB / map_dir)
    L = float(mp.polys['main'].length)
    runs = R.load_runs(bags)
    passes = {b: SA.passes_on_main(mp, r) for b, r in runs.items()}
    return L, passes


def n_covering(passes, bags, s, L):
    import stops_analysis as SA
    return int(sum(SA.covered(passes[b], s, L) for b in bags if b in passes))


# ------------------------------------------------------------------------------------------------
# step 1: per-landmark mixture fits
# ------------------------------------------------------------------------------------------------
def fit_all(lm, ev, L, passes, bags, tag, rtk_only=False):
    """lm: landmark rows (s,sigma,p_stop,cls); ev: events used for fitting (main edge)."""
    if rtk_only:
        ev = ev[ev.n_good >= 3]
    fits, comps = [], []
    for li, l in lm.iterrows():
        o = cyc(ev.s.values - l.s, L)
        m = np.abs(o) <= WIN
        w = ev[m].copy()
        w['o'] = o[m]
        x = w.o.values
        n = len(x)
        row = dict(tag=tag, lm_s=l.s, lm_cls=l.cls, lm_sigma=l.sigma, lm_p=l.p_stop, n=n, runs=w.bag.nunique(),
                   n_rtk=int((w.n_good >= 3).sum()))
        res = {}
        for bgf in (False, True):
            best_k, best_b = None, np.inf
            for K in range(0 if bgf else 1, KMAX + 1):
                if n < max(K, 2):
                    continue
                f = em_fit(x, K, bgf)
                if f is None:
                    continue
                b = bic(f, n)
                row[f"bic_{'bg' if bgf else 'gm'}{K}"] = round(b, 2)
                if b < best_b:
                    best_k, best_b, res[bgf] = K, b, f
            row[f"K_{'bg' if bgf else 'gm'}"] = best_k
            if bgf in res:
                ff = res[bgf]
                row[f"comp_{'bg' if bgf else 'gm'}"] = ' '.join(
                    f'{m:+.2f}/{s_:.2f}/{p_:.2f}' for m, s_, p_ in zip(ff['mu'], ff['sig'], ff['pi'])) + \
                    (f" bg={ff['pi0']:.2f}" if bgf else '')
        fits.append(row)
        f = res.get(True)
        if f is None or f['K'] == 0:
            continue
        R, R0 = responsibilities(f, x)
        hard = np.where(R.max(1) >= R0, R.argmax(1), -1)
        tight = f['sig'] <= MAX_SIG
        prim = int(np.argmin(np.where(tight, np.abs(f['mu']), np.inf))) if tight.any() else int(np.argmin(np.abs(f['mu'])))
        for k in range(f['K']):
            sel = w[hard == k]
            s_abs = float(np.mod(l.s + f['mu'][k], L))
            dl = cyc(lm.s.values - s_abs, L)
            dl[li] = np.inf if k != prim else dl[li]
            j = int(np.argmin(np.abs(dl)))
            tol = max(KNOWN_TOL, 2 * f['sig'][k])
            if k == prim:
                role = 'primary'
            elif f['sig'][k] > MAX_SIG:
                role = 'broad'      # a wide component acting as background, not a stop place
            elif abs(dl[j]) <= tol:
                role = f'existing_lm@{lm.s[j]:.1f}'
            else:
                real = (abs(f['mu'][k]) >= MIN_SEP and f['sig'][k] <= MAX_SIG and len(sel) >= MIN_EV and
                        sel.bag.nunique() >= MIN_RUNS)
                role = 'secondary' if real else 'weak'
            mid = sel[~(sel['first'] | sel['last'])]
            npass = n_covering(passes, bags, s_abs, L)
            comps.append(dict(tag=tag, lm_s=l.s, lm_cls=l.cls, k=k, role=role, offset=round(float(f['mu'][k]), 3),
                              s=round(s_abs, 3), sigma=round(float(f['sig'][k]), 3), weight=round(float(f['pi'][k]), 3),
                              w_bg=round(float(f['pi0']), 3), n_soft=round(float(R[:, k].sum()), 2), n_ev=len(sel),
                              runs=sel.bag.nunique(), mid_runs=mid.bag.nunique(), n_rtk=int((sel.n_good >= 3).sum()),
                              frac_first_last=round(float((sel['first'] | sel['last']).mean()), 2) if len(sel) else np.nan,
                              n_pass=npass, p_stop=round(mid.bag.nunique() / npass, 3) if npass else np.nan,
                              bags=';'.join(sorted(sel.bag.unique()))))
    return pd.DataFrame(fits), pd.DataFrame(comps)


def dedupe_secondaries(comps):
    sec = comps[comps.role == 'secondary'].sort_values(['n_ev', 'runs'], ascending=False)
    keep = []
    for _, c in sec.iterrows():
        if all(abs(c.s - k.s) > 1.0 for k in keep):
            keep.append(c)
    return pd.DataFrame(keep).sort_values('s') if keep else sec.iloc[0:0]


def mixture_rows(lm, sec):
    base = lm[['s', 'sigma', 'p_stop', 'cls']].copy()
    base['role'] = 'landmark'
    base['parent_s'] = base.s
    extra = pd.DataFrame(dict(s=sec.s.values, sigma=np.maximum(sec.sigma.values, 0.2).round(3),
                              p_stop=sec.p_stop.fillna(0.0).round(3).values, cls='secondary', role='secondary',
                              parent_s=sec.lm_s.values))
    return pd.concat([base, extra], ignore_index=True).sort_values('s').reset_index(drop=True)


def write_lm_file(rows, path, note):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(f'# {note}\n')
        fh.write('# format = maps/landmarks.csv; extra rows (cls=secondary) are mixture components of the place at parent_s,\n')
        fh.write('# see the *_components.csv sidecar; the loader reads s,sigma,p_stop and ignores cls\n')
        fh.write('s,sigma,p_stop,cls\n')
        for r in rows.itertuples(index=False):
            fh.write(f'{r.s:.3f},{r.sigma:.3f},{r.p_stop:.3f},{r.cls}\n')


# ------------------------------------------------------------------------------------------------
# PDA emulation (Estimator::placeUpdate, single-mode filter)
# ------------------------------------------------------------------------------------------------
def pda(s_pred, var_s, P, rows_s, rows_sig, rows_p, L, g=PDA['g'], extra=PDA['extra'], p_random=PDA['p_random'],
        min_prob=PDA['min_prob']):
    """Vectorised over draws: s_pred (N,), var_s (N,) or scalar, P (N,) or scalar.
    Returns s_post, accepted, dominant row index (-1 = 'not a known place'), p_known."""
    s_pred = np.asarray(s_pred, float)
    N = len(s_pred)
    var_s = np.broadcast_to(np.asarray(var_s, float), (N,))
    P = np.broadcast_to(np.asarray(P, float), (N,))
    delta = cyc(rows_s[None, :] - s_pred[:, None], L)
    r = rows_sig[None, :] ** 2 + extra ** 2
    var = var_s[:, None] + r
    inside = delta ** 2 <= g * g * var
    # kMaxCand = 8: the first 8 in s order inside the gate
    inside &= np.cumsum(inside, 1) <= 8
    w = np.where(inside, np.maximum(rows_p, 0.02)[None, :] * np.exp(-0.5 * delta ** 2 / var) / np.sqrt(2 * np.pi * var), 0.0)
    sw = w.sum(1)
    width = 2 * g * np.sqrt(var_s + extra ** 2 + 0.25)
    wr = p_random / np.maximum(width, 1.0)
    pk = np.where(inside.any(1), sw / (sw + wr), 0.0)
    acc = inside.any(1) & (pk >= min_prob)
    beta = w / (sw + wr)[:, None]
    beta0 = wr / (sw + wr)
    rbar = np.where(acc, (beta * r).sum(1) / np.maximum(1 - beta0, 1e-12), 1.0)
    K = P / (P + rbar)
    nu = (beta * delta).sum(1)
    s_post = np.where(acc, s_pred + K * nu, s_pred)
    dom = np.where(beta.max(1) > beta0, beta.argmax(1), -1)
    dom = np.where(inside.any(1), dom, -1)
    return s_post, acc, dom, pk


def truth_label(s_true, rows, L, tol_min=1.0):
    """index of the union row that explains a stop (nearest within max(1 m, 2 sigma)), else -1"""
    d = np.abs(cyc(rows.s.values[None, :] - np.asarray(s_true)[:, None], L))
    tol = np.maximum(tol_min, 2 * rows.sigma.values)[None, :]
    j = np.argmin(np.where(d <= tol, d, np.inf), 1)
    ok = np.take_along_axis(d <= tol, j[:, None], 1)[:, 0]
    return np.where(ok, j, -1)


def main():
    sys.stdout = Tee(HERE / 'fit_mixture_log.txt')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 40)
    import data_io as D
    tr_bags, va_bags = D.split('train'), D.split('val')

    # ---------------- data ----------------
    lm_tr = pd.read_csv(ROOT / 'analysis' / 'validation_maps' / 'landmarks.csv', comment='#')
    chk = select_landmarks(MB / 'map_train' / 'stops.csv')
    assert np.allclose(chk.s.values, lm_tr.s.values, atol=1e-3), 'validation landmarks != map_train/stops.csv rules'
    lm_all = select_landmarks(MB / 'map' / 'stops.csv')
    pkg = ROOT / 'ros2_ws' / 'src' / 'tram_backup_odometry' / 'maps' / 'landmarks.csv'
    if pkg.exists():
        lp = pd.read_csv(pkg, comment='#')
        print(f'package landmarks.csv == map/stops.csv selection: {len(lp) == len(lm_all) and np.allclose(lp.s, lm_all.s, atol=1e-3)}'
              f' ({len(lp)} vs {len(lm_all)})')
    ev_tr_file = pd.read_csv(MB / 'map_train' / 'stops_events.csv')
    ev_all_file = pd.read_csv(MB / 'map' / 'stops_events.csv')
    ev_tr_file = ev_tr_file[ev_tr_file.edge == 'main'].copy()
    ev_all_file = ev_all_file[ev_all_file.edge == 'main'].copy()
    L_tr, passes_tr = load_passes('map_train', tr_bags + va_bags)
    L_all, passes_all = load_passes('map', tr_bags + va_bags)
    print(f'main length: map_train {L_tr:.3f} m, map {L_all:.3f} m; stop events on main: '
          f'train {int((ev_tr_file.split == "train").sum())}, val {int((ev_tr_file.split == "val").sum())}')

    # leak check of the "train-only" validation landmarks (their clusters were built on train+val events)
    stt = pd.read_csv(MB / 'map_train' / 'stops.csv')
    cl_of = dict(zip(stt.s_median.round(3), stt.cluster))
    leak = []
    for _, l in lm_tr.iterrows():
        g = ev_tr_file[ev_tr_file.cluster == cl_of.get(round(l.s, 3), -99)]
        gt = g[g.split == 'train']
        leak.append(dict(s=l.s, cls=l.cls, n_train=len(gt), runs_train=gt.bag.nunique(), n_val=int((g.split == 'val').sum()),
                         med_train=np.median(gt.s) if len(gt) else np.nan))
    leak = pd.DataFrame(leak)
    leak['d_med'] = leak.med_train - leak.s
    bad = leak[leak.runs_train < 3]
    print(f'\n[leak] validation_maps/landmarks.csv comes from map_train/stops.csv, whose clusters include VAL events: '
          f'{len(bad)} of {len(leak)} landmarks have < 3 train runs (would not exist train-only); '
          f'|median shift| > 0.1 m for {int((leak.d_med.abs() > 0.1).sum())}')
    print(bad.round(3).to_string(index=False))
    leak.round(3).to_csv(HERE / 'leak_check_validation_landmarks.csv', index=False)

    # ---------------- step 1: fits ----------------
    ev_tr = ev_tr_file[ev_tr_file.split == 'train']
    ev_va = ev_tr_file[ev_tr_file.split == 'val']
    fits_tr, comps_tr = fit_all(lm_tr, ev_tr, L_tr, passes_tr, tr_bags, 'train')
    fits_all, comps_all = fit_all(lm_all, ev_all_file, L_all, passes_all, tr_bags + va_bags, 'all')
    fits_rtk, comps_rtk = fit_all(lm_tr, ev_tr, L_tr, passes_tr, tr_bags, 'train_rtk', rtk_only=True)
    # teammate's recipe cross-check: sklearn GaussianMixture (no background) BIC choice
    try:
        from sklearn.mixture import GaussianMixture
        agree = 0
        for i, l in lm_tr.iterrows():
            o = cyc(ev_tr.s.values - l.s, L_tr)
            x = o[np.abs(o) <= WIN].reshape(-1, 1)
            if len(x) < 4:
                agree += 1
                continue
            b = [GaussianMixture(K, reg_covar=0.01, n_init=20, random_state=0).fit(x).bic(x) for K in range(1, 4)]
            agree += int(np.argmin(b) + 1 == fits_tr.K_gm[i])
        print(f'\nsklearn GaussianMixture(reg_covar=0.01) BIC choice == own EM (no background): {agree}/{len(lm_tr)} landmarks')
    except Exception as e:  # pragma: no cover
        print('sklearn cross-check skipped:', e)
    for nm, f in (('train', fits_tr), ('all', fits_all), ('train_rtk', fits_rtk)):
        print(f'[{nm}] BIC-selected K, pure GMM: {f.K_gm.value_counts().sort_index().to_dict()}; '
              f'GMM+uniform background: {f.K_bg.value_counts().sort_index().to_dict()}')
    fits = pd.concat([fits_tr, fits_all, fits_rtk], ignore_index=True)
    fits.to_csv(HERE / 'fits_bic.csv', index=False)
    print('\n[TRAIN] per-landmark fits (components as offset/sigma/weight; gm = pure GMM, bg = GMM + uniform background)')
    print(fits_tr[['lm_s', 'lm_cls', 'n', 'runs', 'n_rtk', 'K_gm', 'comp_gm', 'K_bg', 'comp_bg']].to_string(index=False))
    comps = pd.concat([comps_tr, comps_all, comps_rtk], ignore_index=True)
    comps.to_csv(HERE / 'components.csv', index=False)
    cols = ['lm_s', 'lm_cls', 'role', 'offset', 'sigma', 'weight', 'n_ev', 'runs', 'mid_runs', 'n_rtk', 'frac_first_last',
            'n_pass', 'p_stop']
    for nm, c in (('TRAIN', comps_tr), ('ALL', comps_all), ('TRAIN RTK-only', comps_rtk)):
        nonp = c[c.role != 'primary']
        print(f'\n[{nm}] non-primary components of the BIC-selected GMM+background fits '
              f'({(c.role == "secondary").sum()} real secondary, {(c.role == "weak").sum()} weak, '
              f'{(c.role == "broad").sum()} broad (sigma > {MAX_SIG} m), '
              f'{c.role.str.startswith("existing").sum()} = an existing neighbouring landmark):')
        print(nonp[cols].to_string(index=False))
    sec_tr = dedupe_secondaries(comps_tr)
    sec_all = dedupe_secondaries(comps_all)
    sec_rtk = dedupe_secondaries(comps_rtk)
    for nm, c in (('TRAIN', comps_tr), ('ALL', comps_all)):
        rel = c[c.role.isin(['weak', 'broad', 'secondary']) & (c.offset.abs() >= MIN_SEP) & (c.sigma <= 1.5) &
                (c.n_ev >= 2) & (c.runs >= 2)]
        print(f'[{nm}] relaxed rules (sigma <= 1.5 m, >= 2 events from >= 2 runs) would add: '
              + ('; '.join(f'{r.lm_s:.1f}{r.offset:+.2f} (sigma {r.sigma:.2f}, {r.n_ev} ev/{r.runs} runs)'
                           for r in rel.itertuples()) or 'nothing'))
    print(f'\nreal secondary modes (deduplicated): train {len(sec_tr)}, all {len(sec_all)}, train RTK-only {len(sec_rtk)}')

    # focus: s ~ 10960.9
    for nm, evs, L, lms in (('train+val events (map_train s)', ev_tr_file, L_tr, lm_tr),):
        o = cyc(evs.s.values - 10960.857, L)
        w = evs[(o > -12) & (o < 12)].assign(o=o[(o > -12) & (o < 12)]).sort_values('o')
        print(f'\n[focus 10960.857] {nm}: offset, split, bag, dwell, n_good (0 = no RTK fix in the stop), pos_spread')
        print(w[['o', 'split', 'bag', 'dwell', 'n_good', 'pos_spread', 'first', 'last']].round(3).to_string(index=False))
    print(fits[fits.lm_s.between(10960, 10962)].to_string(index=False))

    # ---------------- step 2: val recurrence of train secondary modes ----------------
    rows = []
    known_tr = pd.concat([lm_tr[['s', 'sigma']], sec_tr[['s', 'sigma']]], ignore_index=True)
    for _, c in sec_tr.iterrows():
        tol = max(1.0, 2 * c.sigma)
        ov = cyc(ev_va.s.values - c.lm_s, L_tr)
        win = np.abs(ov) <= WIN
        dv = np.abs(cyc(ev_va.s.values - c.s, L_tr))
        n_in = int((win & (dv <= tol)).sum())
        # background: val events in the window away from every known place (landmarks + train secondaries)
        dk = np.abs(cyc(ev_va.s.values[:, None] - known_tr.s.values[None, :], L_tr))
        near_known = (dk <= np.maximum(1.0, 2 * known_tr.sigma.values)[None, :]).any(1)
        kn_in = np.abs(cyc(known_tr.s.values - c.lm_s, L_tr)) <= WIN
        covered = float(sum(2 * max(1.0, 2 * s) for s in known_tr.sigma.values[kn_in]))
        bg_rate = (win & ~near_known).sum() / max(2 * WIN - covered, 1.0)
        lam = bg_rate * 2 * tol
        from scipy.stats import poisson
        n_prim = int((win & (np.abs(ov) <= max(1.0, 2 * lm_tr.set_index('s').sigma.get(c.lm_s, 0.2)))).sum())
        all_match = sec_all[np.abs(sec_all.s - (c.s + 0.0)) <= 1.5] if len(sec_all) else sec_all
        rows.append(dict(lm_s=c.lm_s, offset=c.offset, s=c.s, sigma=c.sigma, train_n=c.n_ev, train_runs=c.runs,
                         train_share=round(c.n_ev / max(c.n_ev + comps_tr[(comps_tr.lm_s == c.lm_s) & (comps_tr.role == 'primary')].n_ev.sum(), 1), 3),
                         val_n=n_in, val_bags=';'.join(sorted(ev_va[win & (dv <= tol)].bag.unique())),
                         val_n_primary=n_prim, val_share=round(n_in / max(n_in + n_prim, 1), 3), val_n_window=int(win.sum()),
                         null_lambda=round(lam, 3), p_poisson=round(float(poisson.sf(n_in - 1, lam)) if n_in > 0 else 1.0, 4),
                         val_mean_offset=round(float(np.mean(ov[win & (dv <= tol)])), 3) if n_in else np.nan,
                         in_all_fit=len(all_match) > 0,
                         survives_rtk_only=bool(len(sec_rtk) and (np.abs(sec_rtk.s - c.s) <= 1.0).any())))
    rec = pd.DataFrame(rows)
    print('\n[step 2] VAL recurrence of the train secondary modes (tolerance max(1 m, 2 sigma)):')
    print(rec.drop(columns=['val_bags']).to_string(index=False) if len(rec) else '  (none)')
    rec.to_csv(HERE / 'val_recurrence.csv', index=False)
    # modes found only with val data (all fit) but not in train
    new_all = sec_all[[not (np.abs(sec_tr.s.values - s) <= 1.5).any() for s in sec_all.s]] if len(sec_all) else sec_all
    if len(new_all):
        print('\nsecondary modes of the ALL-data fit that the TRAIN fit does not have (support by split):')
        for _, c in new_all.iterrows():
            d_tr = np.abs(cyc(ev_tr.s.values - c.s, L_tr))
            d_va = np.abs(cyc(ev_va.s.values - c.s, L_tr))
            tol = max(1.0, 2 * c.sigma)
            print(f'  s={c.s:.2f} (lm {c.lm_s:.2f} {c.offset:+.2f} m, sigma {c.sigma:.2f}): train {int((d_tr <= tol).sum())} ev '
                  f'/ {ev_tr[d_tr <= tol].bag.nunique()} runs, val {int((d_va <= tol).sum())} ev / {ev_va[d_va <= tol].bag.nunique()} runs')

    # ---------------- step 4: files ----------------
    rows_tr = mixture_rows(lm_tr, sec_tr)
    rows_all = mixture_rows(lm_all, sec_all)
    write_lm_file(rows_tr, HERE / 'landmarks_mixture_train.csv',
                  'stop landmarks + stop-place mixture components, TRAIN events only (validation_maps/landmarks.csv rows unchanged)')
    write_lm_file(rows_all, HERE / 'landmarks_mixture_all.csv',
                  'stop landmarks + stop-place mixture components, ALL events (map/stops.csv landmark rows unchanged)')
    sec_tr.to_csv(HERE / 'landmarks_mixture_train_components.csv', index=False)
    sec_all.to_csv(HERE / 'landmarks_mixture_all_components.csv', index=False)
    print(f'\n[step 4] landmarks_mixture_train.csv: {len(lm_tr)} landmark rows + {len(sec_tr)} secondary rows; '
          f'landmarks_mixture_all.csv: {len(lm_all)} + {len(sec_all)}')
    print(rows_tr[rows_tr.role == 'secondary'].to_string(index=False))
    print(rows_all[rows_all.role == 'secondary'].to_string(index=False))

    # ---------------- step 3: offline benefit on VAL stop events ----------------
    cur = lm_tr[['s', 'sigma', 'p_stop']].reset_index(drop=True)
    mix = rows_tr[['s', 'sigma', 'p_stop', 'role']].reset_index(drop=True)
    # val events that are within the window of any row of the mixture set
    dmin = np.abs(cyc(ev_va.s.values[:, None] - mix.s.values[None, :], L_tr)).min(1)
    V = ev_va[dmin <= WIN].copy()
    s_true = V.s.values
    lab = truth_label(s_true, mix, L_tr)          # which union row explains the stop (-1 = unknown place)
    lab_role = np.where(lab >= 0, mix.role.values[np.maximum(lab, 0)], 'unknown')
    d_near_lm = cyc(s_true[:, None] - cur.s.values[None, :], L_tr)     # stop - landmark
    jn = np.argmin(np.abs(d_near_lm), 1)
    V['d_lm'] = d_near_lm[np.arange(len(V)), jn]
    V['lm_s'] = cur.s.values[jn]
    V['truth'] = lab_role
    band = (np.abs(V.d_lm) >= 1.5) & (np.abs(V.d_lm) <= 4.0)
    expl = band & (V.truth == 'secondary')
    print(f'\n[step 3] VAL stop events (>=5 s, main) within {WIN:.0f} m of a landmark/mixture row: {len(V)}; '
          f'explained by: landmark {int((V.truth == "landmark").sum())}, secondary {int((V.truth == "secondary").sum())}, '
          f'unknown place {int((V.truth == "unknown").sum())}')
    print(f'  in the 1.5-4 m band from the nearest landmark: {int(band.sum())} val events; explained by a TRAIN mixture '
          f'component: {int(expl.sum())}')
    print(V[band][['bag', 's', 'lm_s', 'd_lm', 'truth', 'n_good', 'dwell']].round(3).to_string(index=False))
    # same count with the ALL-data components (in-sample, optimistic) and for train events
    lab_all = truth_label(s_true, rows_all[['s', 'sigma']], L_tr)
    expl_all = band & (lab_all >= 0) & (rows_all.role.values[np.maximum(lab_all, 0)] == 'secondary')
    dtr = cyc(cur.s.values[None, :] - ev_tr.s.values[:, None], L_tr)
    dtr = dtr[np.arange(len(ev_tr)), np.argmin(np.abs(dtr), 1)]
    band_tr = (np.abs(dtr) >= 1.5) & (np.abs(dtr) <= 4.0)
    lab_tr = truth_label(ev_tr.s.values, mix, L_tr)
    print(f'  (in-sample ALL-data components would explain {int(expl_all.sum())} of the {int(band.sum())} val band events; '
          f'TRAIN events in the band: {int(band_tr.sum())}, explained in-sample by train components: '
          f'{int((band_tr & (lab_tr >= 0) & (mix.role.values[np.maximum(lab_tr, 0)] == "secondary")).sum())})')

    # deterministic snap (prediction = truth): anchor = dominant PDA candidate
    def dom_anchor(rows):
        _, acc, dom, pk = pda(s_true, 0.0, 1e-9, rows.s.values, rows.sigma.values, rows.p_stop.values, L_tr)
        anc = np.where(dom >= 0, rows.s.values[np.maximum(dom, 0)], np.nan)
        return cyc(anc - s_true, L_tr), dom
    e1, d1 = dom_anchor(cur)
    e2, d2 = dom_anchor(mix)
    V['snap_err_single'] = e1
    V['snap_err_mix'] = e2
    for nm, msk in (('all', np.ones(len(V), bool)), ('at landmark', V.truth.values == 'landmark'),
                    ('at secondary', V.truth.values == 'secondary'), ('band 1.5-4 m', band.values)):
        a, b = e1[msk], e2[msk]
        print(f'  snap (prediction = truth) {nm:14s} n={msk.sum():3d}: |err| single-comp mean {np.nanmean(np.abs(a)) if np.isfinite(a).any() else np.nan:.3f} '
              f'(no candidate {np.isnan(a).sum()}), mixture mean {np.nanmean(np.abs(b)) if np.isfinite(b).any() else np.nan:.3f} '
              f'(no candidate {np.isnan(b).sum()}); anchor changed {int(np.sum(~np.isclose(np.nan_to_num(a, nan=99), np.nan_to_num(b, nan=99))))}')

    # noisy prediction grid, PDA emulation
    rng = np.random.default_rng(1)
    ND = 400
    grid = []
    true_row_mix = lab                         # index in mix (or -1)
    for sd_true in (0.3, 0.6, 1.0):
        for dist in (0.0, 150.0, 400.0, 1000.0):
            e = rng.normal(0, sd_true, (len(V), ND))
            sp = (s_true[:, None] + e).ravel()
            st_rep = np.repeat(s_true, ND)
            P = sd_true ** 2
            var_s = P + PDA['q'] * dist
            res = {}
            for nm, rows in (('current', cur), ('mixture', mix)):
                post, acc, dom, pk = pda(sp, var_s, P, rows.s.values, rows.sigma.values, rows.p_stop.values, L_tr)
                err = cyc(post - st_rep, L_tr)
                # the dominant row in terms of the union rows: map current rows to mix indices by s
                dom_s = np.where(dom >= 0, rows.s.values[np.maximum(dom, 0)], np.nan)
                tr_s = np.repeat(np.where(true_row_mix >= 0, mix.s.values[np.maximum(true_row_mix, 0)], np.nan), ND)
                wrong = acc & (dom >= 0) & ~(np.abs(np.nan_to_num(dom_s, nan=1e9) - np.nan_to_num(tr_s, nan=-1e9)) < 1e-6)
                res[nm] = (err, acc, wrong)
            for grp, msk in (('all', np.ones(len(V), bool)), ('landmark', V.truth.values == 'landmark'),
                             ('secondary', V.truth.values == 'secondary'), ('unknown', V.truth.values == 'unknown')):
                mm = np.repeat(msk, ND)
                if mm.sum() == 0:
                    continue
                r0 = np.sqrt(np.mean((e.ravel()[mm]) ** 2))
                row = dict(sd_true=sd_true, dist_since_fix=dist, sd_assoc=round(float(np.sqrt(var_s)), 2), group=grp,
                           n_events=int(msk.sum()), prior_rmse=round(float(r0), 3))
                for nm in ('current', 'mixture'):
                    err, acc, wrong = res[nm]
                    row[f'{nm}_rmse'] = round(float(np.sqrt(np.mean(err[mm] ** 2))), 3)
                    row[f'{nm}_p90'] = round(float(np.percentile(np.abs(err[mm]), 90)), 3)
                    row[f'{nm}_acc'] = round(float(acc[mm].mean()), 3)
                    row[f'{nm}_wrong'] = round(float(wrong[mm].mean()), 3)
                grid.append(row)
    grid = pd.DataFrame(grid)
    grid.to_csv(HERE / 'offline_pda_grid.csv', index=False)
    print('\n[step 3] PDA emulation on VAL stop events: prediction = truth + N(0, sd_true^2), filter P = sd_true^2, '
          'association var = P + 0.008 * distance since the last fix; current landmarks vs + train mixture rows')
    print(grid.to_string(index=False))
    V.to_csv(HERE / 'val_events_snap.csv', index=False)

    print('\nfiles:', ', '.join(sorted(p.name for p in HERE.iterdir() if p.is_file())))


if __name__ == '__main__':
    main()
