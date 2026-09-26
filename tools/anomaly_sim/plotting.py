"""Diagnostic plots of corrupted runs (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .constants import CMD, FRONT, GNSS_MASTER_VEL, GNSS_ROVER_VEL, KMH_PER_MS, REAR  # noqa: E402
from .run import Run  # noqa: E402

EVENT_COLORS = {
    'slip': '#d62728', 'slide': '#1f77b4', 'dropout': '#7f7f7f', 'outliers': '#9467bd', 'frozen': '#8c564b',
    'noise': '#bcbd22', 'stamp_jitter': '#17becf', 'duplicates': '#e377c2', 'reorder': '#ff7f0e',
    'zero_stamps': '#2ca02c', 'stamp_glitch': '#393b79', 'clock_offset': '#637939', 'scale_drift': '#8c6d31',
    'notch_fault': '#843c39', 'gnss_cut': '#000000',
}


TIMING_TYPES = ('stamp_jitter', 'duplicates', 'reorder', 'zero_stamps', 'stamp_glitch', 'clock_offset')


def gnss_speed(run: Run) -> tuple[np.ndarray, np.ndarray]:
    for key in (GNSS_MASTER_VEL, GNSS_ROVER_VEL):
        s = run.streams.get(key)
        if s is not None and len(s) > 10:
            return s.t_hdr, np.hypot(s.val[:, 0], s.val[:, 1])
    return np.array([]), np.array([])


def _gapped(t: np.ndarray, y: np.ndarray, max_gap: float = 0.35):
    """Insert NaNs where consecutive samples are further apart than ``max_gap`` (shows dropouts)."""
    if len(t) < 2:
        return t, y
    order = np.argsort(t, kind='stable')
    t, y = t[order], y[order]
    brk = np.flatnonzero(np.diff(t) > max_gap)
    return np.insert(t, brk + 1, t[brk] + 1e-3), np.insert(y.astype(float), brk + 1, np.nan)


def _is_timing(event: dict) -> bool:
    return event['type'] in TIMING_TYPES or (event['type'] == 'dropout' and event['params'].get('mode') == 'stall')


def plot_event(ax, bad: Run, clean: Run, event: dict, pad: tuple[float, float] = (5.0, 8.0), title: str | None = None):
    """Wheel speeds (clean thin, corrupted bold), GNSS speed and notch around one event.

    Timing events (stall, jitter, glitches...) are shown as latency ``t_bag - t_hdr`` instead.
    """
    t0, t1 = event['t0'] - pad[0], event['t1'] + pad[1]
    tref = clean.t_start
    if _is_timing(event):
        for key, col in ((FRONT, 'tab:blue'), (REAR, 'tab:red'), (CMD, 'm')):
            if key not in event['topics'] and event['type'] != 'dropout':
                continue
            c = clean.streams[key]
            m = (c.t_bag >= t0) & (c.t_bag <= t1)
            ax.plot(c.t_bag[m] - tref, c.t_bag[m] - c.t_hdr[m], '-', color=col, lw=0.8, alpha=0.35)
            b = bad.streams[key]
            m = (b.t_bag >= t0) & (b.t_bag <= t1)
            ax.plot(b.t_bag[m] - tref, np.clip(b.t_bag[m] - b.t_hdr[m], -3, 3), '.', color=col, ms=3,
                    label=f'{key.split("__")[1].split("_")[0]} latency (corrupted)')
        ax.axvspan(event['t0'] - tref, event['t1'] - tref, color=EVENT_COLORS.get(event['type'], 'y'), alpha=0.12)
        ax.set_xlim(t0 - tref, t1 - tref)
        ax.set_xlabel('bag time from run start [s]', fontsize=8)
        ax.set_ylabel('t_bag - header.stamp [s] (clipped +-3)', fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=6)
        ax.tick_params(labelsize=7)
        p = event.get('params', {})
        ax.set_title(title or f"{event['type']} " + ' '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}'
                                                          for k, v in p.items()), fontsize=8)
        return
    c_all = np.concatenate([clean.streams[k].val[(clean.streams[k].t_hdr0 >= t0) & (clean.streams[k].t_hdr0 <= t1), 0]
                            for k in (FRONT, REAR)]) / KMH_PER_MS
    y_hi = (np.nanmax(c_all) if len(c_all) else 10.0) * 1.3 + 2.0
    y_lo = -2.0
    for key, col in ((FRONT, 'tab:blue'), (REAR, 'tab:red')):
        c = clean.streams[key]
        m = (c.t_hdr0 >= t0) & (c.t_hdr0 <= t1)
        ax.plot(*_gapped(c.t_hdr0[m] - tref, c.val[m, 0] / KMH_PER_MS), '-', color=col, lw=0.8, alpha=0.35)
        b = bad.streams[key]
        m = (b.t_hdr0 >= t0) & (b.t_hdr0 <= t1)
        y = b.val[m, 0] / KMH_PER_MS
        tt = b.t_hdr0[m] - tref
        inr = np.isfinite(y) & (y >= y_lo) & (y <= y_hi)
        ax.plot(*_gapped(tt, np.where(inr, y, np.nan)), '.-', color=col, ms=3, lw=1.2,
                label=f'{key.split("__")[1].split("_")[0]} (corrupted)')
        out = ~inr
        if out.any():  # NaN / inf / absurd values drawn as markers on the axis edge
            yy = np.where(np.isnan(y[out]), 0.0, np.clip(y[out], y_lo, y_hi))
            ax.plot(tt[out], yy, 'x' if np.isnan(y[out]).all() else 'v', color=col, ms=7,
                    label=f'{key.split("__")[1].split("_")[0]} NaN/inf/out of range')
    ax.set_ylim(y_lo - 0.3, y_hi + 0.3)
    tg, vg = gnss_speed(clean)
    if len(tg):
        m = (tg >= t0) & (tg <= t1)
        ax.plot(tg[m] - tref, vg[m], 'k-', lw=1.5, label='GNSS |v|')
    ax.axvspan(event['t0'] - tref, event['t1'] - tref, color=EVENT_COLORS.get(event['type'], 'y'), alpha=0.12)
    ax2 = ax.twinx()
    c = clean.streams[CMD]
    m = (c.t_hdr0 >= t0) & (c.t_hdr0 <= t1)
    ax2.step(c.t_hdr0[m] - tref, c.val[m, 0], where='post', color='m', lw=0.8, alpha=0.6)
    ax2.set_ylim(-16, 16)
    ax2.set_ylabel('notch', color='m', fontsize=7)
    ax2.tick_params(labelsize=7)
    ax.set_xlim(t0 - tref, t1 - tref)
    ax.set_xlabel('t from run start [s]', fontsize=8)
    ax.set_ylabel('speed [m/s]', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=6)
    if title is None:
        p = event.get('params', {})
        title = f"{event['type']} {p.get('bogie', '')} {p.get('model', '')} " + \
                ' '.join(f"{k}={v:.2f}" if isinstance(v, float) else f'{k}={v}' for k, v in p.items()
                         if k in ('peak_rel', 'depth_rel', 'lock', 'mode', 'kind'))
    ax.set_title(title, fontsize=8)


def plot_timeline(ax, bad: Run, clean: Run):
    """Whole-run view: speeds plus one lane per anomaly type."""
    tref = clean.t_start
    tg, vg = gnss_speed(clean)
    if len(tg):
        ax.plot(tg - tref, vg, 'k-', lw=0.8, label='GNSS |v| (reference)')
    for key, col in ((FRONT, 'tab:blue'), (REAR, 'tab:red')):
        b = bad.streams[key]
        y = b.val[:, 0] / KMH_PER_MS
        y = np.where(np.isfinite(y) & (np.abs(y) < 40), y, np.nan)
        ax.plot(*_gapped(b.t_hdr0 - tref, y, 1.0), '-', color=col, lw=0.6, alpha=0.8, label=key.split('__')[1])
    types = sorted({e['type'] for e in bad.events})
    ymax = 16.0
    for i, typ in enumerate(types):
        lane = ymax + 1.0 + i * 0.8
        for e in bad.events:
            if e['type'] == typ:
                ax.plot([e['t0'] - tref, max(e['t1'], e['t0'] + 0.5) - tref], [lane, lane], '-',
                        color=EVENT_COLORS.get(typ, 'y'), lw=4, solid_capstyle='butt')
        ax.text(-0.01, lane, typ, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7)
    ax.set_ylim(-1, ymax + 1.5 + 0.8 * len(types))
    ax.set_xlabel('t from run start [s]')
    ax.set_ylabel('speed [m/s]')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=7)


def plot_run(bad: Run, clean: Run, path: str | Path, max_events: int = 6, types: tuple[str, ...] | None = None):
    """Overview timeline + a zoom on up to ``max_events`` value events."""
    skip = ('gnss_cut', 'scale_drift', 'noise', 'stamp_jitter', 'duplicates', 'reorder')
    evs = [e for e in bad.events if e['type'] not in skip and (types is None or e['type'] in types)]
    if not evs:  # whole-run faults: show the first window of them instead
        evs = [dict(e, t1=min(e['t1'], e['t0'] + 20.0)) for e in bad.events
               if e['type'] not in ('gnss_cut',) and (types is None or e['type'] in types)][:1]
    if len(evs) > max_events:
        idx = np.linspace(0, len(evs) - 1, max_events).round().astype(int)
        evs = [evs[i] for i in idx]
    ncol = 3
    nrow = 1 + int(np.ceil(len(evs) / ncol))
    fig = plt.figure(figsize=(18, 4.2 * nrow))
    gs = fig.add_gridspec(nrow, ncol)
    ax = fig.add_subplot(gs[0, :])
    plot_timeline(ax, bad, clean)
    ax.set_title(f"{bad.meta.get('scenario', '')} | {bad.name} | seed {bad.meta.get('seed')}")
    for i, e in enumerate(evs):
        plot_event(fig.add_subplot(gs[1 + i // ncol, i % ncol]), bad, clean, e)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path
