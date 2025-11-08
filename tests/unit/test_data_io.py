import numpy as np
from chirpy.data import AcquisitionData
from chirpy.geometry import ImageGrid2D, TransducerArray2D


def test_acq_save_load_roundtrip(tmp_path):
    grid = ImageGrid2D(nx=5, ny=4, dx=1.0)
    arr = np.random.randn(2, 2, 8)
    tarr = TransducerArray2D(
        positions=np.c_[[-0.5, 0.5], [0, 0]], is_tx=[True, True], is_rx=[True, True]
    )
    acq = AcquisitionData(
        array=arr, grid=grid, tx_array=tarr, time=np.arange(8) * 0.1, c0=1500.0
    )
    path = acq.save(tmp_path / "a.npz")
    acq2 = AcquisitionData.load(path)
    np.testing.assert_allclose(acq2.array, arr)
    assert acq2.time.shape == (8,)
    assert acq2.tx_array.n_elements == tarr.n_elements
