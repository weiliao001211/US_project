from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from chirpy.geometry import ImageGrid2D
from chirpy.data.image_data import ImageData


# ======================================================================
# 1. Time-domain resampling (Fourier-based, Kaiser/Blackman window)
# ======================================================================

def _freq_window_1d(n: int, kind: Optional[str] = None, beta: float = 8.0) -> np.ndarray:
    """
    Construct a 1D frequency-domain window.

    Parameters
    ----------
    n : int
        Number of frequency samples.
    kind : {"kaiser", "blackman", None}
        Type of window. If None, returns ones.
    beta : float
        Kaiser window beta parameter.

    Returns
    -------
    w : ndarray, shape (n,)
    """
    if kind is None:
        return np.ones(n, dtype=np.float64)

    kind = kind.lower()
    if kind == "kaiser":
        return np.kaiser(n, beta)
    if kind == "blackman":
        return np.blackman(n)

    raise ValueError(f"Unknown window kind: {kind!r}")


def resample_time_series_fourier(
    data: np.ndarray,
    new_nt: int,
    axis: int = -1,
    window: Optional[str] = "kaiser",
    beta: float = 8.0,
    dt_source: Optional[float] = None,
    f_cut: Optional[float] = None,
) -> np.ndarray:
    """
    Resample real-valued time series along a given axis using Fourier interpolation.

    This is a Python analogue of the k-Wave `filterTimeSeries` + Fourier
    interpolation strategy:
    - Transform to frequency domain with rFFT.
    - Optionally apply:
        * a smooth low-pass window (Kaiser / Blackman),
        * an explicit brick-wall low-pass with cutoff `f_cut`.
    - Truncate or zero-pad the spectrum to match the new length.
    - Transform back with iRFFT.

    Parameters
    ----------
    data : ndarray
        Input array containing time series, e.g. (n_tx, n_rx, nt).
    new_nt : int
        Desired number of time samples along `axis`.
    axis : int, optional
        Axis corresponding to time.
    window : {"kaiser", "blackman", None}, optional
        Frequency-domain window. If None, no smooth windowing is applied.
    beta : float, optional
        Kaiser window beta parameter if `window="kaiser"`.
    dt_source : float, optional
        Time step of the source data. Required if `f_cut` is not None.
    f_cut : float, optional
        Explicit cutoff frequency [Hz]. Components with f > f_cut are
        zeroed in the source spectrum before truncation / padding.

    Returns
    -------
    out : ndarray
        Resampled array with `data.shape[axis] == new_nt`.
    """
    data = np.asarray(data)
    axis = int(axis)

    nt_old = data.shape[axis]
    if nt_old == new_nt and f_cut is None and window is None:
        return data.copy()

    # rFFT in time
    spec = np.fft.rfft(data, axis=axis)
    n_freq_old = spec.shape[axis]

    # optional smooth frequency window
    if window is not None:
        win = _freq_window_1d(n_freq_old, kind=window, beta=beta)
        shape_win = [1] * spec.ndim
        shape_win[axis] = n_freq_old
        spec = spec * win.reshape(shape_win)

    # optional explicit low-pass at f_cut
    if f_cut is not None:
        if dt_source is None:
            raise ValueError("dt_source must be provided when f_cut is not None.")
        freqs = np.fft.rfftfreq(nt_old, d=dt_source)  # shape (n_freq_old,)
        mask = freqs <= float(f_cut)

        shape_mask = [1] * spec.ndim
        shape_mask[axis] = mask.size
        spec = spec * mask.reshape(shape_mask)

    # target spectrum shape
    n_freq_new = new_nt // 2 + 1
    new_spec_shape = list(spec.shape)
    new_spec_shape[axis] = n_freq_new
    spec_new = np.zeros(new_spec_shape, dtype=spec.dtype)

    # copy overlapping low-frequency band
    n_copy = min(n_freq_old, n_freq_new)
    sl_old = [slice(None)] * spec.ndim
    sl_new = [slice(None)] * spec_new.ndim
    sl_old[axis] = slice(0, n_copy)
    sl_new[axis] = slice(0, n_copy)
    spec_new[tuple(sl_new)] = spec[tuple(sl_old)]

    # back to time domain
    out = np.fft.irfft(spec_new, n=new_nt, axis=axis)
    return out


# ======================================================================
# 2. 2D spatial Fourier interpolation (model restriction / prolongation)
# ======================================================================

def _spatial_window_1d(n: int, kind: Optional[str] = None, beta: float = 8.0) -> np.ndarray:
    """
    1D window for spatial frequency smoothing (used to build 2D windows).
    """
    return _freq_window_1d(n, kind=kind, beta=beta)


def fourier_interp2d(
    field: np.ndarray,
    new_shape: Tuple[int, int],
    window: Optional[str] = "blackman",
    beta: float = 8.0,
) -> np.ndarray:
    """
    2D Fourier interpolation between regular grids using zero-padding / truncation.

    This is a simple analogue of k-Wave's `interpftn`:
    - Shift the spectrum so that DC is in the center.
    - Copy the central low-frequency block into the new spectrum.
    - Optionally apply a separable 2D window (Blackman / Kaiser) in the
      frequency domain to smooth the transition.
    - Inverse FFT back to the spatial domain.

    Parameters
    ----------
    field : ndarray, shape (ny, nx)
        Real-valued 2D field on the original grid.
    new_shape : (int, int)
        Target shape (ny_new, nx_new).
    window : {"blackman", "kaiser", None}, optional
        Frequency window kind. If None, no smoothing is applied.
    beta : float, optional
        Kaiser window beta parameter.

    Returns
    -------
    out : ndarray, shape new_shape
        Real-valued interpolated field.
    """
    field = np.asarray(field, dtype=np.float64)
    ny_old, nx_old = field.shape
    ny_new, nx_new = map(int, new_shape)

    # FFT and shift to center
    spec = np.fft.fft2(field)
    spec_shifted = np.fft.fftshift(spec)

    # prepare new spectrum
    spec_new_shifted = np.zeros((ny_new, nx_new), dtype=complex)

    # copy overlapping central block
    ny_min = min(ny_old, ny_new)
    nx_min = min(nx_old, nx_new)

    y0_old = ny_old // 2 - ny_min // 2
    x0_old = nx_old // 2 - nx_min // 2
    y0_new = ny_new // 2 - ny_min // 2
    x0_new = nx_new // 2 - nx_min // 2

    spec_new_shifted[
        y0_new: y0_new + ny_min, x0_new: x0_new + nx_min
    ] = spec_shifted[
        y0_old: y0_old + ny_min, x0_old: x0_old + nx_min
    ]

    # optional frequency smoothing window
    if window is not None:
        wy = _spatial_window_1d(ny_new, kind=window, beta=beta)
        wx = _spatial_window_1d(nx_new, kind=window, beta=beta)
        W = np.outer(wy, wx)
        spec_new_shifted *= W

    # inverse shift + inverse FFT
    spec_new = np.fft.ifftshift(spec_new_shifted)
    out = np.fft.ifft2(spec_new)

    # return real part (imaginary residuals are numerical noise)
    return out.real.astype(np.float64)


def restrict_model_fourier(
    model_fine: np.ndarray,
    grid_fine: ImageGrid2D,
    grid_coarse: ImageGrid2D,
    window: Optional[str] = "blackman",
    beta: float = 8.0,
) -> np.ndarray:
    """
    Restrict a 2D model field from a fine grid to a coarse grid using Fourier interpolation.
    """
    _ = grid_fine
    ny_c, nx_c = grid_coarse.ny, grid_coarse.nx
    return fourier_interp2d(model_fine, (ny_c, nx_c), window=window, beta=beta)


def prolong_model_fourier(
    model_coarse: np.ndarray,
    grid_coarse: ImageGrid2D,
    grid_fine: ImageGrid2D,
    window: Optional[str] = "blackman",
    beta: float = 8.0,
) -> np.ndarray:
    """
    Prolong a 2D model field from a coarse grid to a fine grid using Fourier interpolation.
    """
    _ = grid_coarse
    ny_f, nx_f = grid_fine.ny, grid_fine.nx
    return fourier_interp2d(model_coarse, (ny_f, nx_f), window=window, beta=beta)


# ======================================================================
# 3. Image-space restriction / prolongation (using ImageData)
# ======================================================================

def restrict_model_image(
    model_fine: np.ndarray,
    grid_fine: ImageGrid2D,
    grid_coarse: ImageGrid2D,
) -> np.ndarray:
    """
    Convenience wrapper: use `ImageData.downsample_to` for spatial restriction.
    """
    img = ImageData(model_fine, grid=grid_fine)
    return img.downsample_to(new_grid=grid_coarse).array.astype(np.float32)


def prolong_model_image(
    model_coarse: np.ndarray,
    grid_coarse: ImageGrid2D,
    grid_fine: ImageGrid2D,
) -> np.ndarray:
    """
    Convenience wrapper: upsample a coarse model to a fine grid using ImageData.
    """
    img = ImageData(model_coarse, grid=grid_coarse)
    return img.downsample_to(new_grid=grid_fine).array.astype(np.float32)


# ======================================================================
# 4. Helpers for multigrid time/pulse handling
# ======================================================================

def resample_observations_fourier(
    d_obs: np.ndarray,
    nt_target: int,
    dt_source: float,
    f_cut: Optional[float] = None,
    window: str = "kaiser",
    beta: float = 8.0,
) -> np.ndarray:
    """
    Resample observations from a fine time grid to match a target operator Nt,
    with an explicit low-pass cutoff f_cut.
    """
    return resample_time_series_fourier(
        d_obs,
        new_nt=nt_target,
        axis=-1,
        window=window,
        beta=beta,
        dt_source=dt_source,
        f_cut=f_cut,
    )


def resample_pulse_fourier(
    pulse_fine: np.ndarray,
    nt_target: int,
    dt_source: float,
    f_cut: Optional[float] = None,
    window: str = "kaiser",
    beta: float = 8.0,
) -> np.ndarray:
    """
    Resample a 1D pulse waveform from a fine time grid to a target Nt,
    with an explicit low-pass cutoff f_cut.
    """
    pulse_fine = np.asarray(pulse_fine, dtype=np.float64).reshape(1, 1, -1)
    pulse_coarse = resample_time_series_fourier(
        pulse_fine,
        new_nt=nt_target,
        axis=-1,
        window=window,
        beta=beta,
        dt_source=dt_source,
        f_cut=f_cut,
    )
    return pulse_coarse.reshape(-1).astype(np.float64)