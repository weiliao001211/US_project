import numpy as np
from chirpy.processors import AcceptanceMask
from chirpy.data import AcquisitionData


def test_acceptance_mask_circular_distance(ring8, tiny_grid):
    N = ring8.n_elements
    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=ring8)
    AcceptanceMask(delta=2)(acq)
    M = acq.ctx["elem_mask"]
    assert M.shape == (N, N)
    # Row-wise exactly 2*delta+1 False (including self)
    assert np.all((~M).sum(axis=1) == 5)
    # Symmetric in circular sense
    assert np.array_equal(M, M.T)
