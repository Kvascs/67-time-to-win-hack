"""Regenerate the analysis figures in plots/ (``python -m anomaly_sim.report_figures``)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .characterize import reference_speed  # noqa: E402
from .constants import CMD, DEFAULT_OUT, FRONT, KMH_PER_MS, PKG_DIR, REAR  # noqa: E402
from .plotting import plot_event  # noqa: E402
from .run import load_pair, load_run  # noqa: E402

PLOTS = PKG_DIR / 'plots'

#: natural episodes (bag, t_rel centre, title) found by ``characterize``
NATURAL = [
    ('30639_50956d6e', 79.5, 'natural: rear traction slip (+38 %)'),
    ('30618_33bec73f', 105.5, 'natural: slip cycles, both bogies (+60..+82 %)'),
    ('30618_2050d396', 401.0, 'natural: rear braking slide, WSP double dip (-26/-55 %)'),
    ('30639_50956d6e', 1037.5, 'natural: both bogies ~locked in braking (-96 %)'),
]


def _natural_panel(ax, bag, tc, title, pad=(6.0, 8.0)):
    run = load_run(bag)
    t0 = run.t_start
    a, b = t0 + tc - pad[0], t0 + tc + pad[1]
    for key, col in ((FRONT, 'tab:blue'), (REAR, 'tab:red')):
        s = run[key]
        m = (s.t_hdr >= a) & (s.t_hdr <= b)
        ax.plot(s.t_hdr[m] - t0, s.val[m, 0] / KMH_PER_MS, '.-', color=col, ms=3, lw=1.2,
                label=key.split('__')[1].split('_')[0])
    grid = np.arange(a, b, 0.05)
    _, vref = reference_speed(run, grid)
    if vref is not None:
        ax.plot(grid - t0, vref, 'k-', lw=1.5, label='GNSS |v| (robust)')
    ax2 = ax.twinx()
    c = run[CMD]
    m = (c.t_hdr >= a) & (c.t_hdr <= b)
    ax2.step(c.t_hdr[m] - t0, c.val[m, 0], where='post', color='m', lw=0.8, alpha=0.6)
    ax2.set_ylim(-16, 16)
    ax2.tick_params(labelsize=7)
    ax.set_title(f'{title}\n{bag}', fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='upper left')


def _find_sim_event(npz_root: Path, scenario: str, pred) -> tuple[Path, dict] | None:
    for f in sorted((npz_root / scenario).glob('*.events.json')):
        doc = json.loads(f.read_text())
        for e in doc['events']:
            if pred(e):
                return f.with_name(f.name.replace('.events.json', '.npz')), e
    return None


def natural_vs_simulated(npz_root: Path = DEFAULT_OUT / 'npz', out: Path = PLOTS / 'natural_vs_simulated.png'):
    def st(e, key='vehicle__rear_bogie_velocity'):
        return e['stats'].get(key, {})
    picks = [
        ('S01_slip_single_bogie', lambda e: e['type'] == 'slip' and e['params']['bogie'] == 'rear'
         and st(e).get('controller') == 'cutoff' and 0.2 < st(e).get('peak_rel', 0) < 0.45, 'simulated: rear slip, cutoff control'),
        ('S02_slip_both_bogies', lambda e: e['type'] == 'slip' and st(e).get('controller') == 'cutoff'
         and st(e).get('peak_rel', 0) > 0.15 and e['params']['both_mode'] == 'simultaneous', 'simulated: both bogies, cutoff'),
        ('S04_slide_braking_wsp', lambda e: e['type'] == 'slide' and st(e).get('controller') == 'cutoff'
         and st(e).get('peak_rel', 0) < -0.25, 'simulated: rear slide, WSP cycling'),
        ('S05_slide_wheel_lock', lambda e: e['type'] == 'slide' and e['params']['bogie'] == 'both'
         and st(e).get('locked_s', 0) > 0.5, 'simulated: both bogies locked'),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(22, 8.5))
    for i, (bag, tc, title) in enumerate(NATURAL):
        _natural_panel(axes[0, i], bag, tc, title)
    for i, (sc, pred, title) in enumerate(picks):
        found = _find_sim_event(npz_root, sc, pred)
        if found is None:
            axes[1, i].set_title(f'{title}: no matching event')
            continue
        path, ev = found
        bad, clean = load_pair(path)
        ev = next(e for e in bad.events if e['id'] == ev['id'])
        p = ev['params']
        plot_event(axes[1, i], bad, clean, ev, pad=(4.0, 6.0),
                   title=f"{title}\n{bad.name} {sc.split('_')[0]} {p.get('model')} "
                         f"peak={st(ev).get('peak_rel', float('nan')):+.2f}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=75)
    plt.close(fig)
    return out


def scenario_gallery(npz_root: Path = DEFAULT_OUT / 'npz', bag: str = '30618_e3d94878',
                     out: Path = PLOTS / 'scenario_gallery.png'):
    """One representative event per scenario."""
    dirs = sorted(d for d in npz_root.iterdir() if d.is_dir() and d.name.startswith('S') and d.name[:3] != 'S00')
    n = len(dirs)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(24, 4.3 * nrow))
    for ax, d in zip(axes.ravel(), dirs):
        path = d / f'{bag}.npz'
        if not path.exists():
            ax.set_visible(False)
            continue
        bad, clean = load_pair(path)
        evs = [e for e in bad.events if e['type'] not in ('gnss_cut',)]
        pri = [e for e in evs if e['type'] in ('slip', 'slide', 'frozen', 'dropout', 'notch_fault', 'stamp_glitch',
                                               'clock_offset', 'zero_stamps')]
        if not pri and evs:
            e0 = next((e for e in evs if e['type'] == 'noise'), evs[0])
            t_mid = e0['t0'] + 30.0 if e0['type'] == 'noise' else bad.t_start + 0.5 * bad.duration
            pri = [dict(e0, t0=t_mid, t1=t_mid + 20.0)]
        if d.name.startswith('S09') or d.name.startswith('S10'):
            outl = [e for e in evs if e['type'] == 'outliers' and FRONT in e['topics']]
            if outl:  # the outlier burst seen against the fastest clean motion
                s = clean[FRONT]
                pri = [max(outl, key=lambda e: float(np.interp(e['t0'], s.t_hdr0, s.val[:, 0])))]
        e = max(pri, key=lambda e: e['t1'] - e['t0']) if d.name[:3] in ('S07', 'S11') else pri[len(pri) // 2]
        try:
            plot_event(ax, bad, clean, e, pad=(6.0, 8.0))
        except Exception as ex:  # pragma: no cover - figure only
            ax.set_title(f'{d.name}: {ex}')
            continue
        ax.set_title(f'{d.name}\n' + ax.get_title(), fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=65)
    plt.close(fig)
    return out


def magnitude_stats(npz_root: Path = DEFAULT_OUT / 'npz', natural_json: Path = DEFAULT_OUT / 'natural' /
                    'natural_events.json', out: Path = PLOTS / 'magnitude_stats.png'):
    rows = []
    for sc in ('S01_slip_single_bogie', 'S02_slip_both_bogies', 'S03_slip_creep_plateau', 'S04_slide_braking_wsp',
               'S05_slide_wheel_lock'):
        for f in sorted((npz_root / sc).glob('*.events.json')):
            for e in json.loads(f.read_text())['events']:
                if e['type'] in ('slip', 'slide'):
                    for k, stt in e['stats'].items():
                        if k.startswith('vehicle') and stt.get('n_rows'):
                            rows.append((sc, e['type'], stt['controller'], stt['peak_rel'], stt['active_s'],
                                         stt.get('locked_s', 0.0)))
    nat = [e for e in json.loads(Path(natural_json).read_text()) if e['type'] == 'wheel_vs_gnss'
           and e.get('gnss_ms', 0) > 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    ax = axes[0]
    for (typ, col) in (('slip', 'tab:red'), ('slide', 'tab:blue')):
        x = [r[3] for r in rows if r[1] == typ]
        ax.hist(x, bins=np.linspace(-1.05, 1.05, 43), color=col, alpha=0.5, label=f'simulated {typ} (n={len(x)})')
    ax.hist([e['rel'] for e in nat], bins=np.linspace(-1.05, 1.05, 43), histtype='step', color='k', lw=2,
            label=f'natural, GNSS-verified (n={len(nat)})')
    ax.axvspan(0.05, 0.30, color='r', alpha=0.07, label='spec slip 5-30 %')
    ax.set_xlabel('peak relative wheel error (wheel - true) / true')
    ax.set_ylabel('events')
    ax.legend(fontsize=7)
    ax.set_title('Peak magnitude: simulated vs natural')
    ax = axes[1]
    for ctrl, col in (('cutoff', 'tab:orange'), ('creep', 'tab:green'), ('none', 'tab:purple')):
        x = [r[4] for r in rows if r[2] == ctrl]
        if x:
            ax.hist(x, bins=np.linspace(0, 12, 25), alpha=0.5, color=col, label=f'{ctrl} (n={len(x)})')
    ax.hist([e['duration'] for e in nat], bins=np.linspace(0, 12, 25), histtype='step', color='k', lw=2,
            label='natural')
    ax.set_xlabel('duration with |error| > max(0.05 m/s, 2 %) [s]')
    ax.legend(fontsize=7)
    ax.set_title('Duration by controller type')
    ax = axes[2]
    x = [r[5] for r in rows if r[5] > 0]
    ax.hist(x, bins=np.linspace(0, 4, 17), color='tab:purple', alpha=0.6)
    ax.set_xlabel('wheel locked (reads 0 while moving) [s]')
    ax.set_title(f'Lock duration (S05, n={len(x)}); spec 0.5-3 s')
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    plt.close(fig)
    return out, rows


def eval_figure(summary_json: Path = DEFAULT_OUT / 'eval' / 'eval_summary.json', out: Path = PLOTS / 'eval_scenarios.png'):
    agg = json.loads(Path(summary_json).read_text())
    scen = sorted({r['scenario'] for r in agg})
    ests = sorted({r['estimator'] for r in agg})
    metrics = [('rmse_rt', 'speed RMSE [m/s]', True), ('p99_err', 'p99 |error| [m/s]', True),
               ('abs_drift_pct', '|distance drift| [%] (median bag)', False)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 7))
    y = np.arange(len(scen))
    for ax, (m, lab, log) in zip(axes, metrics):
        for j, est in enumerate(ests):
            vals = []
            for sc in scen:
                r = next((r for r in agg if r['scenario'] == sc and r['estimator'] == est), None)
                vals.append(r[m] if r else np.nan)
            vals = np.clip(np.array(vals, float), 1e-3, 1e3)
            ax.barh(y + (j - 0.5) * 0.4, vals, height=0.38, label=est)
        ax.set_yticks(y)
        ax.set_yticklabels(scen, fontsize=8)
        ax.invert_yaxis()
        if log:
            ax.set_xscale('log')
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3, axis='x')
        ax.legend(fontsize=8)
    fig.suptitle('Reference estimators on the scenario suite (17 val bags, median over bags; values clipped to 1e3)')
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    plt.close(fig)
    return out


def timing_figure(out: Path = PLOTS / 'natural_timing.png'):
    """Natural timing anomalies: stall + drain, GNSS header-stamp offset segments, clock drift."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.8))
    r = load_run('30618_28538acf')
    f = r[FRONT]
    t0 = r.t_start
    m = (f.t_bag - t0 > 160) & (f.t_bag - t0 < 185)
    axes[0].plot(f.t_bag[m] - t0, f.t_bag[m] - f.t_hdr[m], '.', ms=3)
    axes[0].set_title('30618_28538acf: bus stall ~1 s, then queue drains at 0.9x period', fontsize=9)
    axes[0].set_xlabel('bag time [s]')
    axes[0].set_ylabel('t_bag - header.stamp [s]')
    r2 = load_run('30639_3b3d9eb8')
    from .constants import GNSS_MASTER_VEL
    g = r2[GNSS_MASTER_VEL]
    w = r2[FRONT]
    axes[1].plot(g.t_bag - r2.t_start, g.t_bag - g.t_hdr, '.', ms=1, label='GNSS master vel')
    axes[1].plot(w.t_bag - r2.t_start, w.t_bag - w.t_hdr, '.', ms=1, label='front wheel')
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].legend(fontsize=7)
    axes[1].set_title('30639_3b3d9eb8: GNSS header stamps 1 s ahead for ~4 min', fontsize=9)
    axes[1].set_xlabel('bag time [s]')
    r3 = load_run('30618_e3d94878')
    w3 = r3[FRONT]
    lat = w3.t_bag - w3.t_hdr
    from scipy.ndimage import median_filter
    axes[2].plot(w3.t_bag - r3.t_start, median_filter(lat, 301), '-')
    axes[2].set_title('30618_e3d94878: vehicle clock drift ~55 ppm (running median latency)', fontsize=9)
    axes[2].set_xlabel('bag time [s]')
    axes[2].set_ylabel('median latency [s]')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    plt.close(fig)
    return out


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    print(natural_vs_simulated())
    print(scenario_gallery())
    print(magnitude_stats()[0])
    print(timing_figure())
    if (DEFAULT_OUT / 'eval' / 'eval_summary.json').exists():
        print(eval_figure())


if __name__ == '__main__':
    main()
