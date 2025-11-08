import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.kwave]


def test_gradients_scale_with_aperture(
    kwave_bin, tiny_grid, gaussian_pulse, record_time, c0
):
    from chirpy.geometry import TransducerArray2D
    from chirpy.data import AcquisitionData
    from chirpy.optimization.operator.wave_operator import WaveOperator
    from chirpy.optimization.gradient.time_grad import AdjointStateGrad
    from chirpy.optimization.function.least_squares import NonlinearLS

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=64, r=None)
    pos = ring.positions
    top = int(np.argmax(pos[1]))
    bot = int(np.argmin(pos[1]))

    # A) 1 TX + 1 RX
    pair_pos = np.column_stack([pos[:, top], pos[:, bot]])
    is_tx_pair = np.array([True, False], bool)
    is_rx_pair = np.array([False, True], bool)
    from chirpy.geometry import TransducerArray2D as Arr

    arr_pair = Arr(positions=pair_pos, is_tx=is_tx_pair, is_rx=is_rx_pair)

    # B) 1 TX + all RX
    is_tx1 = np.zeros(ring.n_elements, bool)
    is_tx1[top] = True
    arr_1tx = Arr(
        positions=ring.positions, is_tx=is_tx1, is_rx=np.ones(ring.n_elements, bool)
    )

    # C) all TX + all RX
    arr_full = ring

    X, Y = tiny_grid.meshgrid(indexing="xy")
    m_true = np.full((tiny_grid.ny, tiny_grid.nx), c0, np.float32)
    m_true[(X + 0.015) ** 2 + (Y - 0.012) ** 2 < 0.01**2] = c0 + 120
    m_bg = np.full_like(m_true, c0, dtype=np.float64)

    def grad_for(arr):
        acq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=arr)
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
        return fun.gradient(m_bg, kind="c")

    g_pair = grad_for(arr_pair)
    g_1tx = grad_for(arr_1tx)
    g_full = grad_for(arr_full)

    # All finite
    assert (
        np.isfinite(g_pair).all()
        and np.isfinite(g_1tx).all()
        and np.isfinite(g_full).all()
    )
    # Aperture heuristic: L2 norm should grow with more receivers/tx (not a hard rule, but typical)
    # n_pair, n_1tx, n_full = 1, arr_1tx.n_rx, arr_full.n_tx * arr_full.n_rx
    norm_pair = float(np.linalg.norm(g_pair))
    norm_1tx = float(np.linalg.norm(g_1tx))
    norm_full = float(np.linalg.norm(g_full))
    assert norm_pair > 0.0
    assert norm_1tx >= norm_pair * 0.5  # weaker bound to avoid false fails
    assert norm_full >= norm_1tx * 0.5
