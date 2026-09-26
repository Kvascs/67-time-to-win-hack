"""Export estimator-ready parameters derived in this analysis to estimator_params.json
(scale defaults, curve correction table, physical limits, notch envelopes, monitor thresholds, timing facts)."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import anomalies as A  # noqa: E402


def main():
    sc = pd.read_csv(HERE / 'scale_factors.csv')
    sc = sc[sc.dur_s > 300]
    cv = pd.read_csv(HERE / 'curvature_ratio.csv')
    cvm = cv.groupby('k_lo').agg(kappa=('kappa_mean', 'mean'), e_rel=('e_rel_median', 'mean')).reset_index()
    cvm['abs_kappa'] = cvm.kappa.abs()
    curve = cvm.groupby(pd.cut(cvm.abs_kappa, [0, 0.001, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04, 0.07]),
                        observed=True).e_rel.mean()
    steady = pd.read_csv(HERE / 'physics_accel_by_notch_steady.csv')
    by_vd = sc.groupby(['vehicle', 'date']).k_speed_front.agg(['mean', 'min', 'max', 'count']).reset_index()
    out = dict(
        units=dict(wheel_topics='km/h', note='README says m/s; data is km/h'),
        scale=dict(
            k_default_straight=float(sc.k_speed_front.median()),
            k_default_distance=float(sc.k_dist_all_front.median()),
            k_range=[float(sc.k_speed_front.min()), float(sc.k_speed_front.max())],
            front_rear_ratio_median=float((sc.k_speed_rear / sc.k_speed_front).median()),
            by_vehicle_date=[dict(vehicle=str(r.vehicle), date=r.date, k_mean=float(r['mean']), k_min=float(r['min']),
                                  k_max=float(r['max']), n=int(r['count'])) for _, r in by_vd.iterrows()],
            within_bag_segment_sigma_pct=0.08,
            speed_dependence_pct='none beyond +-0.1 % (1..15 m/s, straight track)',
        ),
        curve_correction=dict(
            model='v_true = v_wheel * (1 + min(0.5*|kappa|, 0.0095))',
            table_abs_kappa_upper=[float(i.right) for i in curve.index],
            table_wheel_minus_gnss_rel=[float(x) for x in curve.values],
        ),
        sensor=dict(dead_band_kmh=0.15, noise_sigma_ms=0.006, residual_vs_gnss_rms_ms=0.026,
                    max_identical_repeats_moving=5, signed=True, rollback_min_ms=-0.11),
        timing=dict(wheel_rate_hz_median=9.29, wheel_missing_slots_pct=7.0, header_dt_p1_p99=[0.060, 0.21],
                    front_rear_identical_stamps_pct=99.7, latency_bag_minus_header_ms=[21, 69],
                    startup_burst_msgs_median=24, startup_burst_span_s_max=5.5,
                    all_topic_silences_s=[0.9, 1.2], single_sensor_outage_s_max=73.5,
                    recorder_clock_step_s=1.0, gnss_header_clock_excursion_s=1.0),
        physical_limits=dict(accel_max=1.45, accel_traction_p9999=1.23, decel_service_min=-2.2,
                             decel_service_p001=-1.94, decel_emergency_observed=-4.4, jerk_p01_p999=[-1.21, 1.04],
                             wheel_rate_gate_02s=[-2.8, 2.1]),
        notch_accel_steady=[dict(notch=int(r.notch), p01=float(r.p01), p50=float(r.p50), p999=float(r.p999), n=int(r.n))
                            for _, r in steady.iterrows()],
        notch_ambiguous_values=[0, -8, -9, -10, -11, -12, -13, -14, -15],
        traction_cutoff_landmarks_utm37n=[[400348, 6185516], [403493, 6185843], [399263, 6184978], [401567, 6185381]],
        antenna_baseline=dict(length_m=12.44, rover_ahead_in_travel_direction='71/71 bags'),
        monitor_defaults=asdict(A.MonitorParams()),
    )
    (HERE / 'estimator_params.json').write_text(json.dumps(out, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    print(json.dumps(out['scale'], indent=1)[:800])
    print(out['curve_correction'])


if __name__ == '__main__':
    main()
