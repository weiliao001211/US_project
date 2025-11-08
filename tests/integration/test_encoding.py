import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.kwave]


def test_source_encoding_k_sum_vs_single(
    kwave_bin, tiny_grid, gaussian_pulse, record_time, c0
):
    from chirpy.geometry import TransducerArray2D
    from chirpy.data import AcquisitionData
    from chirpy.optimization.gradient.time_grad import AdjointStateGrad
    from chirpy.optimization.function.least_squares import NonlinearLS

    ring = TransducerArray2D.from_ring_array_2D(grid=tiny_grid, n=64, r=None)

    # true model & obs
    X, Y = tiny_grid.meshgrid(indexing="xy")
    m_true = np.full((tiny_grid.ny, tiny_grid.nx), c0, np.float32)
    m_true[(X**2 + Y**2) < 0.01**2] = c0 + 80
    m_bg = np.full_like(m_true, c0, dtype=np.float64)

    # synthesize observation with sequential shots
    acq_seq = AcquisitionData.from_geometry(grid=tiny_grid, tx_array=ring)
    from chirpy.optimization.operator.wave_operator import WaveOperator as WO

    op_true = WO(
        data=acq_seq,
        medium_params={"sound_speed": m_true},
        record_time=record_time,
        pulse=gaussian_pulse,
        record_full_wf=False,
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
    d_obs = op_true.forward(m_true, kind="c")

    # build encoded operator on background with the same geometry & obs
    acq_enc = AcquisitionData(
        array=d_obs, grid=tiny_grid, tx_array=ring, time=op_true.time_axis
    )
    op_enc = WO(
        data=acq_enc,
        medium_params={"sound_speed": m_bg.astype(np.float32)},
        record_time=record_time,
        pulse=gaussian_pulse,
        record_full_wf=True,
        use_encoding=True,  # encoding path
        tau_max=0.0,  # no delays to isolate weight-only behavior
        drop_self_rx=True,  # ignored when encoding
        cfl=0.2,
        c_ref=c0,
        pml_size=8,
        pml_alpha=8.0,
        verbose=False,
        use_gpu=False,
        binary_path=kwave_bin,
    )

    # K=1 vs K>1 : gradient must scale ~ linearly (sum, not average)
    ge1 = AdjointStateGrad(op_enc, K=1, seed=123, use_first_deriv_product=True)
    fun1 = NonlinearLS(op_enc, grad_eval=ge1, weight=1.0, sync_value=True)
    g1 = fun1.gradient(m_bg, kind="c")

    K = 4
    geK = AdjointStateGrad(op_enc, K=K, seed=123, use_first_deriv_product=True)
    funK = NonlinearLS(op_enc, grad_eval=geK, weight=1.0, sync_value=True)
    gK = funK.gradient(m_bg, kind="c")

    # Same seed + no delays ⇒ same random ±1 weights drawn ⇒ K-sum ≈ K × g1
    # Allow small numerical noise.
    np.testing.assert_allclose(gK, K * g1, rtol=1e-3, atol=1e-6)
