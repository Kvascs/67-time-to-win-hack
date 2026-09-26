"""Step 4. Event level: isolated spikes in the whitened bogie residuals and their counterpart on the other bogie.

Train bags (+ val for the rate/precision check). For every bag:
  z_F, z_R = whitened residual (as in s3) / (1.4826 * MAD over moving samples)
  front event: |z_F| >= ZF, local max of |z_F| within +-3 samples, 3 <= v_gnss <= 20 m/s
  counterpart: max |z_R| with the same sign inside t_F + (L_eff / v_gnss) * [0.85, 1.15]; matched if |z_R| >= ZR
  false-match control: same search centred at 1.6 x the expected lag (no physical counterpart there)
  timing: dt_obs = t_R - t_F (stamps of the two peak samples); rel. error = dt_obs * v_gnss / L_eff - 1
          'centroid' variant: each peak time refined by the |z| centroid of the sample and its two neighbours
          '10 Hz ticks' variant: stamps replaced by the fitted regular 0.1 s grid (removes the ~13 ms stamp jitter)
  position lock: front-bogie position of each matched event = master fix (interpolated) + 9.873 m along the GNSS
          course; share of events that have an event from ANOTHER bag within 3 m, vs the same share for
          random points drawn along the same bags' driven paths (radius 3 m: many fixes are not RTK).
Outputs: s4_events.csv, s4_summary.txt, s4_events.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from bc_common import HERE, KMH, NPZ, K_FIX, glitch_free, load_bag, mono, snap_times, splits

ZF, ZR = 5.0, 4.0
ANT_TO_FRONT = 9.873
R_E = 6378137.0


def whiten(t, x):
    e = np.full(len(x), np.nan)
    span = t[2:] - t[:-2]
    w = (t[1:-1] - t[:-2]) / span
    e[1:-1] = x[1:-1] - ((1 - w) * x[:-2] + w * x[2:])
    e[1:-1][span > 0.3] = np.nan
    return e


def centroid_time(t, z, i):
    if i <= 0 or i >= len(t) - 1:
        return t[i]
    a = np.abs(z[i - 1:i + 2])
    a = np.where(np.isfinite(a), a, 0.0)
    # only the part above the noise floor counts
    a = np.clip(a - 2.0, 0, None)
    if a.sum() <= 0:
        return t[i]
    return float(np.sum(a * t[i - 1:i + 2]) / a.sum())


def per_bag(bag, L_eff):
    d = load_bag(bag)
    t, vg = d["t"], d["vg"]
    ts = snap_times(t)
    f, r = d["F"] / KMH, d["R"] / KMH
    eF, eR = whiten(t, f), whiten(t, r)
    mov = np.isfinite(vg) & (vg > 3) & (vg < 20)
    sF = 1.4826 * np.nanmedian(np.abs(eF[mov]))
    sR = 1.4826 * np.nanmedian(np.abs(eR[mov]))
    zF, zR = eF / sF, eR / sR
    aF = np.where(np.isfinite(zF), np.abs(zF), 0.0)
    # moving time (for rates)
    dt = np.r_[np.diff(t), 0]
    T_mov = float(np.sum(dt[mov & (dt < 0.3)]))
    # fix -> ENU (local tangent plane at the bag's first fix; fine for a few km)
    z = np.load(NPZ / f"{bag}.npz")
    fx = np.asarray(z[K_FIX], float)
    fx = fx[glitch_free(fx)]
    tfx, la, lo = mono(fx[:, 1] - 0.045, fx[:, 2], fx[:, 3])   # fixes lead wheel stamps by ~45 ms
    vm = np.asarray(z["sensing__gnss__master__vel"], float)
    vm = vm[glitch_free(vm)]
    tvv, vx, vy = mono(vm[:, 1], vm[:, 2], vm[:, 3])
    ev = []
    cand = np.flatnonzero(mov & (aF >= ZF))
    for i in cand:
        if aF[i] < aF[max(0, i - 3):i + 4].max():
            continue
        lag = L_eff / vg[i]
        row = {"bag": bag, "t": t[i], "v": vg[i], "zF": zF[i], "lag_exp": lag}
        for tag, c in (("", 1.0), ("_ctrl", 1.6)):
            lo_t, hi_t = t[i] + c * lag * 0.85, t[i] + c * lag * 1.15
            j0, j1 = np.searchsorted(t, lo_t), np.searchsorted(t, hi_t)
            if j1 <= j0 or j1 >= len(t) or np.any(np.diff(t[j0 - 1:j1 + 1]) > 0.3):
                row["zR" + tag] = np.nan
                continue
            seg = np.sign(zF[i]) * zR[j0:j1]
            seg = np.where(np.isfinite(seg), seg, -np.inf)
            k = j0 + int(np.argmax(seg))
            row["zR" + tag] = float(np.sign(zF[i]) * zR[k])
            if tag == "":
                row["dt_obs"] = t[k] - t[i]
                row["dt_cen"] = centroid_time(t, zR, k) - centroid_time(t, zF, i)
                row["dt_snap"] = ts[k] - ts[i]
                row["dtF"] = t[i] - t[i - 1]
                row["dtR"] = t[k] - t[k - 1]
        # front-bogie position
        if len(tfx) > 10 and tfx[0] < t[i] < tfx[-1]:
            lat = np.interp(t[i], tfx, la)
            lon = np.interp(t[i], tfx, lo)
            k2 = np.searchsorted(tvv, t[i])
            if 0 < k2 < len(tvv):
                ex, ny = np.interp(t[i], tvv, vx), np.interp(t[i], tvv, vy)
                h = np.hypot(ex, ny)
                if h > 1:
                    row["lat"], row["lon"] = lat, lon
                    row["ux"], row["uy"] = ex / h, ny / h
        ev.append(row)
    # random reference points along the driven path (moving samples)
    rng = np.random.default_rng(abs(hash(bag)) % 2 ** 32)
    idx = np.flatnonzero(mov)
    ref = []
    if len(idx) and len(tfx) > 10:
        pick = rng.choice(idx, size=min(200, len(idx)), replace=False)
        for i in pick:
            if tfx[0] < t[i] < tfx[-1]:
                ref.append((bag, np.interp(t[i], tfx, la), np.interp(t[i], tfx, lo)))
    return ev, T_mov, ref, (sF, sR)


def mix_fit(e, half=0.15):
    """EM fit of relative lag errors in +-half: w * Normal(mu, sd) (true counterparts) + (1 - w) * Uniform (chance)."""
    e = np.asarray(e, float)
    e = e[np.isfinite(e) & (np.abs(e) < half)]
    w, mu, sd = 0.5, 0.0, 0.05
    for _ in range(500):
        pn = w * np.exp(-0.5 * ((e - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        pu = (1 - w) / (2 * half)
        g = pn / (pn + pu)
        w = g.mean()
        mu = np.sum(g * e) / g.sum()
        sd = max(np.sqrt(np.sum(g * (e - mu) ** 2) / g.sum()), 1e-4)
    return w, mu, sd, len(e)


def to_xy(lat, lon, lat0, lon0):
    return np.c_[np.radians(lon - lon0) * R_E * np.cos(np.radians(lat0)), np.radians(lat - lat0) * R_E]


def lock_share(df_xy, bags, radius=1.5):
    tree = cKDTree(df_xy)
    share = []
    for i, p in enumerate(df_xy):
        nb = tree.query_ball_point(p, radius)
        share.append(any(bags[j] != bags[i] for j in nb))
    return np.mean(share)


def main():
    sp = splits()
    s3 = pd.read_csv(HERE / "s3_per_bag.csv")
    L_eff = float(np.median(s3[s3.split == "train"].L_bag))
    bags = [(b, "train") for b in sp["train"]] + [(b, "val") for b in sp["val"]]
    if len(sys.argv) > 1:
        bags = bags[: int(sys.argv[1])]
    E, T, REF, SC = [], {}, [], []
    for b, split in bags:
        ev, T_mov, ref, sc = per_bag(b, L_eff)
        for e in ev:
            e["split"] = split
        E += ev
        T[b] = (split, T_mov)
        REF += ref
        SC.append((b, *sc))
        print(b, split, len(ev), f"{T_mov:.0f}s", flush=True)
    df = pd.DataFrame(E)
    df.to_csv(HERE / "s4_events.csv", index=False)
    rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))  # noqa: E731
    L = [f"L_eff (from s3, train) = {L_eff:.4f} m; thresholds |z_F| >= {ZF}, counterpart |z_R| >= {ZR} (same sign)"]
    for split in ("train", "val"):
        s = df[df.split == split]
        Tm = sum(v[1] for v in T.values() if v[0] == split)
        if len(s) == 0 or Tm == 0:
            continue
        m = s.zR >= ZR
        mc = s.zR_ctrl >= ZR
        L.append(f"{split}: moving time (3-20 m/s) {Tm / 3600:.2f} h; front events {len(s)} "
                 f"({len(s) / (Tm / 3600):.0f} per h); matched at expected lag {m.mean():.3f} "
                 f"vs control lag {mc.mean():.3f}; matched events per hour {m.sum() / (Tm / 3600):.0f}")
        mm = s[m]
        for col, lab in (("dt_obs", "sample stamps"), ("dt_cen", "centroid"), ("dt_snap", "10 Hz ticks")):
            e = mm[col] / mm.lag_exp - 1
            et = mm[col] - mm.lag_exp
            L.append(f"    timing ({lab:13s}): dt_obs - dt_exp median {et.median() * 1e3:+.1f} ms, robust sd "
                     f"{rs(et) * 1e3:.1f} ms, sd {et.std() * 1e3:.1f} ms; relative: median {e.median():+.4f} "
                     f"robust sd {rs(e):.4f}")
        # theory: uniform sampling phase on both ends
        dts = np.r_[mm.dtF, mm.dtR]
        L.append(f"    sampling-phase floor sqrt(E[dt^2]/6) = {np.sqrt(np.mean(dts ** 2) / 6) * 1e3:.1f} ms "
                 f"(dt of the two peak samples)")
        for lo, hi in ((3, 6), (6, 9), (9, 12), (12, 20)):
            q = mm[(mm.v >= lo) & (mm.v < hi)]
            if len(q) < 10:
                continue
            e = q.dt_obs / q.lag_exp - 1
            L.append(f"    v {lo:2d}-{hi:2d} m/s: matched {len(q):5d} ({(q.zR >= ZR).mean():.2f} of those tested), "
                     f"timing robust sd {rs(q.dt_obs - q.lag_exp) * 1e3:.1f} ms, relative robust sd {rs(e):.4f}, "
                     f"median {e.median():+.4f}")
        for col in ("dt_obs", "dt_snap"):
            w, mu, sd, n = mix_fit(mm[col] / mm.lag_exp - 1)
            rate_true = w * n / Tm
            nneed = (sd / 0.001) ** 2
            L.append(f"    mixture fit ({col}): true counterparts {w * n:.0f} of {n} matched ({rate_true * 3600:.0f} per h), "
                     f"bias {mu * 100:+.2f} %, per-event sd {sd * 100:.2f} % (= {sd * mm.lag_exp.median() * 1e3:.0f} ms at "
                     f"the median lag {mm.lag_exp.median():.2f} s) -> +-0.1 % (1 sigma) needs {nneed:.0f} true events = "
                     f"{nneed / rate_true / 3600:.1f} h of driving, even if every true pair were recognised")
    # position lock (train+val matched events vs random path points)
    mm = df[(df.zR >= ZR) & df.lat.notna()]
    lat0, lon0 = mm.lat.mean(), mm.lon.mean()
    xy = to_xy(mm.lat.values, mm.lon.values, lat0, lon0) + ANT_TO_FRONT * mm[["ux", "uy"]].values
    R = pd.DataFrame(REF, columns=["bag", "lat", "lon"])
    # reference points: same number, drawn from the path points of the same bags
    rng = np.random.default_rng(0)
    Rs = R.sample(n=min(len(R), len(mm)), random_state=1)
    xr = to_xy(Rs.lat.values, Rs.lon.values, lat0, lon0)
    sh_ev = lock_share(xy, mm.bag.values, 3.0)
    sh_rf = lock_share(xr, Rs.bag.values, 3.0)
    L.append(f"position lock: matched events with an event of another bag within 3 m: {sh_ev:.3f} "
             f"(n={len(mm)}); random path points: {sh_rf:.3f} (n={len(Rs)})")
    txt = "\n".join(L)
    print(txt)
    (HERE / "s4_summary.txt").write_text(txt, encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    mm = df[(df.zR >= ZR) & (df.split == "train")]
    ax[0].hist((mm.dt_obs - mm.lag_exp) * 1e3, bins=np.arange(-160, 161, 10), color="k", alpha=0.7)
    ax[0].set_xlabel("observed lag - 7.55/v_gnss (ms), matched train events")
    ax[0].set_ylabel("events")
    ax[0].grid(alpha=0.4)
    ax[1].scatter(xy[:, 0], xy[:, 1], s=2, c="r", label="matched spike events (front bogie position)")
    ax[1].set_aspect("equal")
    ax[1].legend(fontsize=8)
    ax[1].set_title("where the spikes happen")
    fig.tight_layout()
    fig.savefig(HERE / "s4_events.png", dpi=90)


if __name__ == "__main__":
    main()
