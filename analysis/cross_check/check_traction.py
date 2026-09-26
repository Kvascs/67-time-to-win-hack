"""Cross-check 5: open-loop bridging accuracy of the exported traction model (traction_id/traction_model.py).

Independent protocol (val bags, master GNSS vel as truth, header time, 20 Hz grid):
  * the model runs causally along the TRUE speed for the whole bag (lag states warm);
  * every 2 s while moving (v > 1 m/s) the state is copied and the speed is propagated open-loop
    from the true speed with the recorded notch sequence (grade looked up at the predicted position);
  * error at +1/3/5/10 s vs GNSS speed; windows are NOT filtered for automation / emergency braking
    ('all'), plus a 'manual' subset that drops windows where the notch is <= 0 while the tram
    accelerates > 0.3 m/s^2 (handle parked = notch not in control) at any time in the window.
Baselines: hold last speed; model without grade.
"""
import sys

import numpy as np
from pyproj import Transformer

sys.path.insert(0, r"C:\MosTransHack\analysis\cross_check")
sys.path.insert(0, r"C:\MosTransHack\analysis\traction_id")
from xc_common import bag_meta, glitch_free, load, mono  # noqa: E402
from traction_model import TractionModel, Track  # noqa: E402

DT = 0.05
H = (1.0, 3.0, 5.0, 10.0)
TR = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)


def grid(bag):
    d = load(bag)
    vm, fx, c = d["velm"], d["fixm"], d["cmd"]
    gv, gf, gc = glitch_free(vm), glitch_free(fx), glitch_free(c)
    tv, vx, vy = mono(vm[gv, 1], vm[gv, 2], vm[gv, 3])
    tf, la, lo = mono(fx[gf, 1], fx[gf, 2], fx[gf, 3])
    tc, u = mono(c[gc, 1], c[gc, 2])
    t0, t1 = max(tv[0], tc[0], tf[0]) + 1, min(tv[-1], tc[-1], tf[-1]) - 1
    t = np.arange(t0, t1, DT)
    v = np.hypot(np.interp(t, tv, vx), np.interp(t, tv, vy))
    # vel gaps -> invalid
    j = np.clip(np.searchsorted(tv, t), 1, len(tv) - 1)
    valid = (tv[j] - tv[j - 1]) < 0.35
    notch = u[np.clip(np.searchsorted(tc, t, side="right") - 1, 0, len(u) - 1)].astype(int)
    E, N = TR.transform(lo, la)
    x, y = np.interp(t, tf, E), np.interp(t, tf, N)
    return t, v, valid, notch, x, y


def main():
    meta = bag_meta()
    bags = sorted(b for b, m in meta.items() if m["split"] == "val")
    track = Track()
    res = {k: {h: [] for h in H} for k in ("model", "model_nograde", "hold")}
    manual_flag = {h: [] for h in H}
    for b in bags:
        t, v, valid, notch, x, y = grid(b)
        # along-track s and direction on the traction_id centreline, every 1 s then interpolated
        idx = np.arange(0, len(t), 20)
        s_c = np.array([track.project(x[i], y[i]) for i in idx])
        s = np.interp(np.arange(len(t)), idx, s_c)
        ds = np.gradient(s) / DT
        dirn = np.where(ds >= 0, 1.0, -1.0)
        a_ref = np.gradient(np.convolve(v, np.ones(21) / 21, mode="same")) / DT
        m = TractionModel()
        m0 = TractionModel()
        m0.grade = None
        n_h = [int(round(h / DT)) for h in H]
        states = []
        for i in range(len(t)):
            g = m.grade.at(s[i], dirn[i])
            m.step(DT, notch[i], v[i], g)
            m0.step(DT, notch[i], v[i], 0.0)
            if i % 40 == 0 and v[i] > 1.0 and i + n_h[-1] < len(t) and valid[i]:
                states.append((i, list(m.y), list(m0.y)))
        for i, y_m, y_0 in states:
            seg = notch[i + 1:i + 1 + n_h[-1]]
            m.y = list(y_m)
            pv = m.predict_speed(v[i], seg, DT, s0=s[i], direction=dirn[i])
            m0.y = list(y_0)
            pv0 = m0.predict_speed(v[i], seg, DT, s0=None)
            parked = np.any((notch[i:i + n_h[-1]] <= 0) & (a_ref[i:i + n_h[-1]] > 0.3))
            for h, n in zip(H, n_h):
                if not valid[i + n]:
                    continue
                vt = v[i + n]
                res["model"][h].append(pv[n - 1] - vt)
                res["model_nograde"][h].append(pv0[n - 1] - vt)
                res["hold"][h].append(v[i] - vt)
                manual_flag[h].append(not parked)
        print(b, "windows", len(states), flush=True)
    print("\nopen-loop speed error vs horizon (val, RMSE m/s; 'manual' excludes handle-parked windows)")
    for k in res:
        row = []
        for h in H:
            e = np.array(res[k][h])
            mm = np.array(manual_flag[h])
            row.append(f"{h:>4.0f}s all {np.sqrt(np.mean(e**2)):.3f} man {np.sqrt(np.mean(e[mm]**2)):.3f} "
                       f"(p95|e| {np.percentile(np.abs(e[mm]),95):.2f}, bias {np.mean(e[mm]):+.3f})")
        print(f"{k:14s}", " | ".join(row))
    print("n windows", len(res["model"][10.0]), "manual frac", np.mean(manual_flag[10.0]))


if __name__ == "__main__":
    main()
