import os
import matplotlib

matplotlib.use("Agg")
import pytest


@pytest.fixture(scope="session")
def kwave_bin():
    p = os.environ.get("CHIRPY_KWAVE_BIN")
    if not p:
        pytest.skip("CHIRPY_KWAVE_BIN not set; skipping k-Wave integration tests")
    return p


@pytest.fixture(scope="session")
def tiny_grid():
    from chirpy.geometry import ImageGrid2D

    return ImageGrid2D(nx=64, ny=64, dx=1e-3)


@pytest.fixture(scope="session")
def ring8(tiny_grid):
    from chirpy.geometry import TransducerArray2D

    return TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=8, r=None)


@pytest.fixture(scope="session")
def ring32(tiny_grid):
    from chirpy.geometry import TransducerArray2D

    return TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)


@pytest.fixture(scope="session")
def gaussian_pulse():
    from chirpy.signals import GaussianModulatedPulse

    return GaussianModulatedPulse(f0=3e5, frac_bw=0.75, amp=1.0)


@pytest.fixture(scope="session")
def record_time(tiny_grid):
    c0 = 1500.0
    width = tiny_grid.extent[1] - tiny_grid.extent[0]
    return 1.2 * width / c0


@pytest.fixture(scope="session")
def c0():
    return 1500.0
