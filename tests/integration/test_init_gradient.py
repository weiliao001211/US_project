import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.kwave]


def test_single_shot_gradient_reasonable_range(
    kwave_bin, tiny_grid, gaussian_pulse, record_time, c0
):
    from chirpy.geometry import TransducerArray2D
    from chirpy.data import AcquisitionData
    from chirpy.optimization.operator.wave_operator import WaveOperator
    from chirpy.optimization.gradient.time_grad import AdjointStateGrad

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)

    X, Y = tiny_grid.meshgrid(indexing="xy")
    m_true = np.full((tiny_grid.ny, tiny_grid.nx), c0, np.float32)
    m_true[(X - 0.02) ** 2 + (Y + 0.02) ** 2 < 0.008**2] = c0 + 100
    m_init = np.full_like(m_true, c0, dtype=np.float64)

    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=ring)
    op = WaveOperator(
        data=acq,
        medium_params={"sound_speed": m_true},
        record_time=record_time,
        pulse=gaussian_pulse,
        record_full_wf=True,
        use_encoding=False,
        drop_self_rx=True,
        cfl=0.2,
        c_ref=c0,
        pml_size=8,
        pml_alpha=8.0,
        verbose=False,
        use_gpu=False,
        binary_path=kwave_bin,
    )

    d = op.forward(m_true, kind="c")
    op.set_obs(d)
    ge = AdjointStateGrad(op, K=None, seed=0, use_first_deriv_product=True)
    r = op.forward(m_init.astype(np.float32), kind="c") - d
    g = ge.evaluate(m_init, q=r, kind="c")

    # contract checks
    assert g.shape == (tiny_grid.ny, tiny_grid.nx)
    assert np.isfinite(g).all()
    # non-trivial distribution: both positive and negative values present
    assert (g > 0).any() and (g < 0).any()
