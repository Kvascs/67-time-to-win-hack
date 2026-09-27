"""Prepare TRAIN runs like prep_val.py (wheel-stamp table with the estimator's arc and the RTK truth), from the
submitted build's replays (build_core/replay_tmp/bl_so_train): input of the leave-one-out reliability map."""
import sys

sys.argv = sys.argv[:1]  # regress.py (imported by rm_common) reads sys.argv[1] as a speed

import prep_val as P  # noqa: E402

P.M.REPLAY_VAL = P.M.ROOT / 'build_core' / 'replay_tmp' / 'bl_so_train'
P.OUT = P.M.HERE / 'prep_train'
P.OUT.mkdir(exist_ok=True)
P.main()
