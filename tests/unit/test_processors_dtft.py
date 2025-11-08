import numpy as np
from chirpy.processors import DTFT
from chirpy.data import AcquisitionData


def test_dtft_matches_numpy():
    T = 32
    Rx = 3
    Tx = 2
    t = np.arange(T) / 1000.0
    x = np.sin(2 * np.pi * 200 * t)[None, None, :]
    data = np.tile(x, (Tx, Rx, 1))  # (Tx,Rx,T)
    acq = AcquisitionData(array=data, grid=None, tx_array=None, time=t)
    freqs = np.array([200.0])
    DTFT(freqs)(acq)

    dt = float(t[1] - t[0])
    kernel = np.exp(-1j * 2 * np.pi * freqs[:, None] * t[None, :]) * dt  # (F,T)

    # Replicate code path: traces = data.transpose(2,1,0); reshape with Fortran order
    traces = data.transpose(2, 1, 0)  # (T,Rx,Tx)
    ts_matrix = traces.reshape(T, Rx * Tx, order="F")  # (T,Rx*Tx)
    ref = kernel @ ts_matrix  # (F,Rx*Tx)
    ref = ref.reshape(len(freqs), Rx, Tx, order="F")  # (F,Rx,Tx)
    ref = np.transpose(ref, (2, 1, 0))  # (Tx,Rx,F)

    np.testing.assert_allclose(acq.array, ref, rtol=1e-12, atol=1e-12)
