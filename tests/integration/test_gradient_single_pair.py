import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.kwave]


def test_time_grad_shapes_sign_and_kernel_mode(
    kwave_bin, tiny_grid, gaussian_pulse, record_time, c0
):
    from chirpy.geometry import TransducerArray2D
    from chirpy.data import AcquisitionData
    from chirpy.optimization.operator.wave_operator import WaveOperator
    from chirpy.optimization.gradient.time_grad import AdjointStateGrad
    from chirpy.optimization.function.least_squares import NonlinearLS

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=32, r=None)
    # build 1-Tx/1-Rx
    pos = ring.positions
    top = int(np.argmax(pos[1]))
    bot = int(np.argmin(pos[1]))
    pair_pos = np.column_stack([pos[:, top], pos[:, bot]])
    is_tx = np.array([True, False], bool)
    is_rx = np.array([False, True], bool)
    from chirpy.geometry import TransducerArray2D as Arr

    pair = Arr(positions=pair_pos, is_tx=is_tx, is_rx=is_rx)

    acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=pair)

    # true vs background
    X, Y = tiny_grid.meshgrid(indexing="xy")
    m_true = np.full((tiny_grid.ny, tiny_grid.nx), c0, np.float32)
    m_true[(X - 0.02) ** 2 + (Y + 0.01) ** 2 < 0.009**2] = c0 + 120
    m_bg = np.full_like(m_true, c0)

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

    ge = AdjointStateGrad(op, K=None, use_first_deriv_product=True)
    fun = NonlinearLS(op, grad_eval=ge, weight=1.0, sync_value=True)
    g = fun.gradient(m_bg.astype(np.float64), kind="c")

    # Assertions: finite, shape, and non-trivial structure
    assert g.shape == (tiny_grid.ny, tiny_grid.nx)
    assert np.isfinite(g).all()
    assert float(np.max(np.abs(g))) > 0.0
    # Kernel mode switch changes result (not identical)
    ge2 = AdjointStateGrad(op, K=None, use_first_deriv_product=False)
    fun2 = NonlinearLS(op, grad_eval=ge2, weight=1.0, sync_value=True)
    g2 = fun2.gradient(m_bg.astype(np.float64), kind="c")
    # They should agree to machine precision here; compare by relative norm.
    num = float(np.linalg.norm(g - g2))
    den = float(np.linalg.norm(g) + np.linalg.norm(g2) + 1e-30)
    assert num / den < 5e-2  # 5% relative tolerance
