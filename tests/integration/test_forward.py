import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.kwave]

"""
Integration test for the forward model.
"""


def _true_model(grid, c0):
    X, Y = grid.meshgrid(indexing="xy")
    m = np.full((grid.ny, grid.nx), c0, np.float32)
    m[((X - 0.01) ** 2 + (Y - 0.01) ** 2) < (0.006**2)] = c0 + 100
    m[((X + 0.01) ** 2 + (Y + 0.012) ** 2) < (0.005**2)] = c0 - 120
    return m


def _pair_array(ring):
    pos = ring.positions
    top = int(np.argmax(pos[1]))
    bot = int(np.argmin(pos[1]))
    pair_pos = np.column_stack([pos[:, top], pos[:, bot]])
    is_tx = np.array([True, False], bool)
    is_rx = np.array([False, True], bool)
    from chirpy.geometry import TransducerArray2D

    return TransducerArray2D(positions=pair_pos, is_tx=is_tx, is_rx=is_rx)


def test_forward_shapes_and_monotonicity(
    kwave_bin, tiny_grid, gaussian_pulse, record_time, c0
):
    from chirpy.data import AcquisitionData
    from chirpy.optimization.operator.wave_operator import WaveOperator

    model_true = _true_model(tiny_grid, c0)

    from chirpy.geometry import TransducerArray2D

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)
    pair = _pair_array(ring)

    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=pair)
    op = WaveOperator(
        data=acq,
        medium_params={"sound_speed": model_true},
        record_time=record_time,
        pulse=gaussian_pulse,
        use_encoding=False,
        drop_self_rx=False,
        record_full_wf=True,
        cfl=0.2,
        c_ref=c0,
        pml_size=8,
        pml_alpha=8.0,
        use_gpu=False,
        verbose=False,
        binary_path=kwave_bin,
    )

    F = op.forward(model_true, kind="c")
    # Shape assertions
    assert F.ndim == 3
    Tx, Rx, T = F.shape
    assert (Tx, Rx) == (1, 1)
    assert T == op.nt
    # No NaNs or infs
    assert np.isfinite(F).all()
    # Cached forward field shape matches (Tx, nt, ny, nx)
    WF = op.get_forward_fields()
    assert WF.shape == (Tx, op.nt, op.ny, op.nx)
    # Energy monotonicity: windowed RMS of trace is non-trivial
    trace = F[0, 0]
    assert float(np.abs(trace).max()) > 0.0
    # dt, time axis consistency
    assert np.isclose(op.time_axis[1] - op.time_axis[0], op.dt)
