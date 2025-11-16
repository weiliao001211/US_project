import numpy as np


def dtft_wavefield(
    wf_time: np.ndarray,
    dt: float,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Discrete-time Fourier transform of a 4D wavefield.

    Parameters
    ----------
    wf_time : np.ndarray
        Time-domain wavefield with shape (n_tx, n_t, n_y, n_x).
    dt : float
        Time step in seconds.
    freqs : np.ndarray
        1D array of target frequencies in Hz, shape (n_freq,).

    Returns
    -------
    np.ndarray
        Frequency-domain wavefield with shape (n_y, n_x, n_tx, n_freq).
    """
    wf_time = np.asarray(wf_time)
    freqs = np.asarray(freqs, dtype=float)

    if wf_time.ndim != 4:
        raise ValueError(
            f"`wf_time` must be 4-D (n_tx, n_t, n_y, n_x), got {wf_time.shape}"
        )
    if freqs.ndim != 1:
        raise ValueError("`freqs` must be 1-D.")

    n_tx, n_t, n_y, n_x = wf_time.shape
    n_f = freqs.size

    # time axis in seconds
    t = np.arange(n_t, dtype=float) * dt  # (n_t,)

    # DTFT kernel: (n_f, n_t)
    kernel = np.exp(-1j * 2.0 * np.pi * np.outer(freqs, t)) * dt
    kernel = kernel.astype(np.complex128, copy=False)

    # bring time axis first: (n_t, n_y, n_x, n_tx)
    wf_tfirst = np.transpose(wf_time, (1, 2, 3, 0))

    # (n_f, n_t) @ (n_t, n_y, n_x, n_tx) -> (n_y, n_x, n_tx, n_f)
    spec = np.einsum("ft,tyxs->yxsf", kernel, wf_tfirst, optimize=True)

    return spec