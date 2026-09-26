import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from anomaly_sim.constants import BAG_DIR, NPZ_DIR  # noqa: E402
from anomaly_sim.run import load_run  # noqa: E402

#: a val bag of vehicle 30618 with GNSS; the crop contains several accelerations and stops
TEST_BAG = '30618_e3d94878'
CROP = (100.0, 420.0)


def _need_data():
    if not (NPZ_DIR / f'{TEST_BAG}.npz').exists():
        pytest.skip('dataset npz not available')


@pytest.fixture(scope='session')
def clean_full():
    _need_data()
    return load_run(TEST_BAG)


@pytest.fixture(scope='session')
def clean(clean_full):
    return clean_full.crop(*CROP)


@pytest.fixture(scope='session')
def src_bag():
    p = BAG_DIR / TEST_BAG
    if not (p / 'metadata.yaml').exists():
        pytest.skip('raw bag not available')
    return p
