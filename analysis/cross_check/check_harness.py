"""Cross-check 3: verify the judge-replica harness (tools/harness) against a hand computation.

A trivial estimator publishes on every /vehicle/driver_position_cmd message:
  stamp = cmd header.stamp, v = mean(latest front, latest rear)/3.6 (sample-and-hold, bag order),
  position = (0, 0, 0) (the origin of the frame)
The same numbers are recomputed here from the raw npz with independent code:
  speed: reference = master vel |(vx, vy)| at vel header stamps, nearest output stamp within 0.05 s
  position: reference = master fixes in exact ENU at the first master fix (bag order), error = -p_ref
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\MosTransHack\tools")
sys.path.insert(0, r"C:\MosTransHack\analysis\cross_check")
from xc_common import enu, load, nearest  # noqa: E402


class TrivialEst:
    def __init__(self, frame="enu"):
        self.vf = self.vr = None

    def on_front(self, stamp, v_kmh, t_bag):
        self.vf = v_kmh

    def on_rear(self, stamp, v_kmh, t_bag):
        self.vr = v_kmh

    def on_cmd(self, stamp, notch, t_bag):
        vals = [v for v in (self.vf, self.vr) if v is not None]
        v = (sum(vals) / len(vals) / 3.6) if vals else 0.0
        return (stamp, v, 0.0, 0.0, 0.0)


def by_hand(bag):
    d = load(bag)
    f, r, c, fx, vm = d["front"], d["rear"], d["cmd"], d["fixm"], d["velm"]
    # outputs
    jf = np.searchsorted(f[:, 0], c[:, 0], side="right") - 1
    jr = np.searchsorted(r[:, 0], c[:, 0], side="right") - 1
    vf = np.where(jf >= 0, f[np.clip(jf, 0, None), 2], np.nan)
    vr = np.where(jr >= 0, r[np.clip(jr, 0, None), 2], np.nan)
    vout = np.nanmean(np.stack([vf, vr]), axis=0) / 3.6
    vout[np.isnan(vout)] = 0.0
    ts = np.sort(c[:, 1])
    vs = vout[np.argsort(c[:, 1], kind="stable")]
    # speed
    o = np.argsort(vm[:, 1], kind="stable")
    tv = vm[o, 1]
    vref = np.hypot(vm[o, 2], vm[o, 3])
    idx, dt = nearest(tv, ts)
    ok = dt <= 0.05
    e = vs[idx[ok]] - vref[ok]
    sp = dict(match=ok.mean(), rmse=np.sqrt(np.mean(e * e)), mae=np.mean(np.abs(e)), bias=e.mean())
    # position
    lat0, lon0, alt0 = fx[0, 2], fx[0, 3], fx[0, 4]
    p = enu(fx[:, 2], fx[:, 3], fx[:, 4], lat0, lon0, alt0)
    o = np.argsort(fx[:, 1], kind="stable")
    tp, p = fx[o, 1], p[o]
    idx, dt = nearest(tp, ts)
    ok = dt <= 0.05
    e3 = np.linalg.norm(p[ok], axis=1)
    # own path length: vertices every >= 5 m while GNSS speed > 0.3 m/s
    vi = np.interp(fx[:, 0], vm[:, 0], np.hypot(vm[:, 2], vm[:, 3]))
    pb = enu(fx[:, 2], fx[:, 3], fx[:, 4], lat0, lon0, alt0)
    last = pb[0, :2]
    L = 0.0
    for k in range(1, len(pb)):
        if vi[k] > 0.3:
            dd = np.hypot(*(pb[k, :2] - last))
            if dd >= 5.0:
                L += dd
                last = pb[k, :2]
    pos = dict(match=ok.mean(), rmse3d=np.sqrt(np.mean(e3 ** 2)), max3d=e3.max(),
               zrmse=np.sqrt(np.mean(p[ok, 2] ** 2)), final3d=e3[-1], dist=L, drift=e3[-1] / L * 100)
    return sp, pos


def main():
    from harness.replay import EvalConfig, evaluate_bag
    for bag in ("30618_e3d94878", "30618_defd0170", "30639_3b3d9eb8"):
        res = evaluate_bag(bag, TrivialEst, {}, EvalConfig())
        if "error" in res:
            print(bag, res["error"])
            print(res["traceback"])
            continue
        s = res["speed"]["all"]
        pm = res["pos"]
        sp, pos = by_hand(bag)
        print(f"== {bag}")
        print(f" speed  harness: match {res['speed']['match_rate']:.4f} rmse {s['rmse']:.4f} mae {s['mae']:.4f} bias {s['bias']:+.4f}")
        print(f" speed  by hand: match {sp['match']:.4f} rmse {sp['rmse']:.4f} mae {sp['mae']:.4f} bias {sp['bias']:+.4f}")
        print(f" pos    harness: match {pm['match_rate']:.4f} rmse3d {pm['err3d']['rmse']:.2f} max3d {pm['err3d']['max']:.2f} "
              f"z_rmse {pm['z']['rmse']:.2f} final3d {pm['final']['err3d']:.2f} dist {pm['distance_m']:.1f} drift% {pm['final']['drift_pct_3d']:.3f}")
        print(f" pos    by hand: match {pos['match']:.4f} rmse3d {pos['rmse3d']:.2f} max3d {pos['max3d']:.2f} "
              f"z_rmse {pos['zrmse']:.2f} final3d {pos['final3d']:.2f} dist {pos['dist']:.1f} drift% {pos['drift']:.3f}")
        print(f" rt: rate {res['rt']['rate_hz']:.2f} Hz, stamp_age_med {res['rt']['stamp_age_med_ms']:.1f} ms "
              f"p95 {res['rt']['stamp_age_p95_ms']:.1f} max {res['rt']['stamp_age_max_ms']:.1f}; valid {res['valid']}")


if __name__ == "__main__":
    main()
