#!/usr/bin/env bash
# Regenerates all standard study results (run from C:\MosTransHack\tools):  bash harness/run_all.sh
set -u
R=harness/results
J=${JOBS:-4}
run() { name=$1; shift; python -u -m harness.run_eval --split val --jobs $J --out $R/$name.json "$@" > $R/$name.txt 2>&1; echo "done $name $(date +%T)"; }
B1=harness.baselines:B1
B2=harness.baselines:B2
# --- baselines (judge-like default: master antenna, ENU at first fix, GNSS header time, tol 0.05 s, ref->out)
run b0_val --est harness.baselines:B0
run b1_val --est $B1
run b2_val --est $B2
# --- stamp convention x reference time base (B1, outputs on wheel messages ~10 Hz distinct stamps)
run b1_refhdr_stampbag --est $B1 --time-base header --params '{"stamp_mode":"bag"}'
run b1_refbag_stamphdr --est $B1 --time-base bag    --params '{"stamp_mode":"header"}'
run b1_refbag_stampbag --est $B1 --time-base bag    --params '{"stamp_mode":"bag"}'
run b1_reffix_stamphdr --est $B1 --time-base header_fixed
# --- 40 Hz output (also on cmd messages), with / without state prediction (B2)
run b2_40hz_hdrx_refhdr      --est $B2 --time-base header --params '{"stamp_mode":"hdr_extrap","emit_on_cmd":true}'
run b2_40hz_hdrx_refhdr_pred --est $B2 --time-base header --params '{"stamp_mode":"hdr_extrap","emit_on_cmd":true,"extrapolate":true}'
run b2_40hz_hdrx_refhdr_pred20 --est $B2 --time-base header --params '{"stamp_mode":"hdr_extrap","emit_on_cmd":true,"extrapolate":true,"lead":0.02}'
run b2_40hz_hdrx_reffix_pred20 --est $B2 --time-base header_fixed --params '{"stamp_mode":"hdr_extrap","emit_on_cmd":true,"extrapolate":true,"lead":0.02}'
run b2_40hz_bag_refbag       --est $B2 --time-base bag --params '{"stamp_mode":"bag","emit_on_cmd":true}'
run b2_40hz_bag_refbag_pred  --est $B2 --time-base bag --params '{"stamp_mode":"bag","emit_on_cmd":true,"extrapolate":true}'
run b2_40hz_hdrx_refbag_pred20 --est $B2 --time-base bag --params '{"stamp_mode":"hdr_extrap","emit_on_cmd":true,"extrapolate":true,"lead":0.02}'
run b2_40hz_bag_refhdr       --est $B2 --time-base header --params '{"stamp_mode":"bag","emit_on_cmd":true}'
# --- best-practice timing: trapezoid integration + extrapolation to the stamp + 45 ms position-only lead
run b2_40hz_hdrx_refhdr_trap_pl45 --est $B2 --time-base header --params '{"stamp_mode":"hdr_extrap","emit_on_cmd":true,"extrapolate":true,"integ":"trap","pos_lead":0.045}'
run b2_40hz_bag_refbag_trap_pl45  --est $B2 --time-base bag    --params '{"stamp_mode":"bag","emit_on_cmd":true,"extrapolate":true,"integ":"trap","pos_lead":0.045}'
# --- judge-definition knobs (B1)
run b1_out2ref  --est $B1 --direction out2ref
run b1_cleanref --est $B1 --clean-ref
run b1_utm      --est $B1 --frame utm
run b1_speed3d  --est $B1 --speed-dims 3
# --- robustness: fault injection (B2 @40 Hz, prediction on)
P='{"stamp_mode":"hdr_extrap","emit_on_cmd":true,"extrapolate":true,"lead":0.02}'
run b2_faults_realistic --est $B2 --faults suite:realistic --params "$P"
run b2_faults_basic     --est $B2 --faults suite:basic     --params "$P"
run b2_faults_garbage   --est $B2 --faults suite:garbage   --params "$P"
run b1_faults_realistic --est $B1 --faults suite:realistic --params "$P"
python -m harness.compare --summary-json $R/summary_val.json $R/*.json > /dev/null
echo "all done $(date +%T)"
