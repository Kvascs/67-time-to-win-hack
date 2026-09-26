"""anomaly_sim - realistic anomaly injection for tram wheel-odometry robustness testing.

Typical use (from ``C:\\MosTransHack\\tools``)::

    from anomaly_sim import load_run, load_suite, apply_scenario, save_run, write_bag
    clean = load_run('30618_e3d94878')                 # data/npz/<bag>.npz
    sc = get_scenario(load_suite(), 'S01')
    bad = apply_scenario(clean, sc)                    # deterministic (scenario seed x bag name)
    save_run(bad, 'out/npz/S01/30618_e3d94878.npz')
    write_bag(bad, 'out/bags/S01/30618_e3d94878')      # rosbag2 (Humble layout) for ros2 bag play

CLI: ``python -m anomaly_sim --help``.
"""
__version__ = '1.0.0'

from .constants import Label  # noqa: E402,F401
from .run import Run, Stream, check_npz, load_pair, load_run, save_run  # noqa: E402,F401
from .context import Context  # noqa: E402,F401
from .scenario import Scenario, apply_scenario, generate, get_scenario, load_suite, summarize  # noqa: E402,F401


def write_bag(*args, **kwargs):
    """Lazy import wrapper around :func:`anomaly_sim.bagio.write_bag` (needs ``rosbags``)."""
    from .bagio import write_bag as _wb
    return _wb(*args, **kwargs)
