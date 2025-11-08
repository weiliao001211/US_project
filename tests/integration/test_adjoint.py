import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.kwave]


def _model_pair(grid, c0):
    X, Y = grid.meshgrid(indexing="xy")
    m_true = np.full((grid.ny, grid.nx), c0, np.float32)
    m_true[(X**2 + (Y - 0.01) ** 2) < 0.006**2] = c0 + 80
    m_bg = np.full_like(m_true, c0)
    return m_true, m_bg


def _pair_array(ring):
    pos = ring.positions
    top = int(np.argmax(pos[1]))
    bot = int(np.argmin(pos[1]))
    pair_pos = np.column_stack([pos[:, top], pos[:, bot]])
    is_tx = np.array([True, False], bool)
    is_rx = np.array([False, True], bool)
    from chirpy.geometry import TransducerArray2D

    return TransducerArray2D(positions=pair_pos, is_tx=is_tx, is_rx=is_rx)


def test_adjoint_backprop_nontrivial_and_real(
    kwave_bin, tiny_grid, gaussian_pulse, record_time, c0
):
    from chirpy.data import AcquisitionData
    from chirpy.optimization.operator.wave_operator import WaveOperator

    from chirpy.geometry import TransducerArray2D

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)
    pair = _pair_array(ring)
    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=pair)

    m_true, m_bg = _model_pair(tiny_grid, c0)

    op = WaveOperator(
        data=acq,
        medium_params={"sound_speed": m_true},
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

    d_true = op.forward(m_true, kind="c")
    # switch medium to background; produce residual
    F_bg = op.forward(m_bg, kind="c")
    r = F_bg - d_true
    assert r.shape == d_true.shape
    lam = op.adjoint(r)
    # Assertions: shape, finiteness, and nonzero content
    assert lam.shape == (op.nt, op.ny, op.nx)
    assert np.isfinite(lam).all()
    assert float(np.linalg.norm(lam)) > 0.0
    # Time reversal sanity: last samples should correlate with earliest forward energy
    f_energy = np.linalg.norm(op.get_forward_fields()[0], axis=(1, 2))
    a_energy = np.linalg.norm(lam, axis=(1, 2))
    # crude check: peaks near opposite ends
    assert np.argmax(f_energy) + np.argmax(a_energy) > op.nt // 2
