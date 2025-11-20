"""
Utilities for animating breast CT waveform inversion results.

This module provides a single main entry point,
:func:`animate_breast_inversion`, which:

* loads a k-Wave-based inversion result (MAT v7),
* loads the corresponding ground-truth dataset (MAT v7.3),
* constructs a 2×3 matplotlib figure (vel/atten/gradient/search),
* animates all iterations in time.

It is written so that *importing* the module has no side effects:
no files are opened, no figures are created, and nothing is plotted
until you explicitly call :func:`animate_breast_inversion` or run the
module as a script::

    python -m chirpy.utils.animate_results

"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py


__all__ = ["animate_breast_inversion", "main"]


# -----------------------------------------------------------------------------#
# Internal helpers
# -----------------------------------------------------------------------------#


def _default_paths() -> Tuple[Path, Path]:
    """
    Return default paths for the inversion results and original dataset.

    The defaults mirror the original relative paths used in the prototype
    script, but expressed in a more robust way (relative to this file).

    Returns
    -------
    res_path : Path
        Path to ``kWave_BreastCT_WaveformInversionResults_cpu.mat``.
    orig_path : Path
        Path to ``kWave_BreastCT.mat``.
    """
    here = Path(__file__).resolve()
    root = here.parents[2]  # .../src/chirpy/utils → ... (repo root)

    res_path = root / "Results" / "kWave_BreastCT_WaveformInversionResults_cpu.mat"
    orig_path = root / "SampleData" / "kWave_BreastCT.mat"
    return res_path, orig_path


def _load_inversion_results(res_path: Path):
    """
    Load inversion results (MAT v7) containing per-iteration estimates.

    Parameters
    ----------
    res_path
        Path to the MAT v7 results file.

    Returns
    -------
    xi : np.ndarray
        X coordinates (Nx,).
    yi : np.ndarray
        Y coordinates (Ny,).
    VEL : np.ndarray
        Estimated velocity, shape (Ny, Nx, n_iter).
    ATT : np.ndarray
        Estimated attenuation (k-Wave units), shape (Ny, Nx, n_iter).
    GRAD : np.ndarray
        Gradient images, shape (Ny, Nx, n_iter).
    SRCH : np.ndarray
        Search directions, shape (Ny, Nx, n_iter).
    """
    res_path = Path(res_path)
    if not res_path.is_file():
        raise FileNotFoundError(f"Results file not found: {res_path}")

    res = loadmat(res_path)

    xi = res["xi"].squeeze()  # (Nx,)
    yi = res["yi"].squeeze()  # (Ny,)
    VEL = res["VEL_ESTIM_ITER"]  # (Ny, Nx, Niter)
    ATT = res["ATTEN_ESTIM_ITER"]  # (Ny, Nx, Niter)
    GRAD = res["GRAD_IMG_ITER"]  # (Ny, Nx, Niter)
    SRCH = res["SEARCH_DIR_ITER"]  # (Ny, Nx, Niter)

    return xi, yi, VEL, ATT, GRAD, SRCH


def _load_ground_truth(orig_path: Path):
    """
    Load original (ground-truth) dataset (MAT v7.3 via h5py).

    Parameters
    ----------
    orig_path
        Path to the MAT v7.3 file.

    Returns
    -------
    C : np.ndarray
        True velocity (Ny, Nx).
    atten0 : np.ndarray
        True attenuation (Ny, Nx).
    xi0 : np.ndarray
        Original x-axis (Nx,).
    yi0 : np.ndarray
        Original y-axis (Ny,).
    """
    orig_path = Path(orig_path)
    if not orig_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {orig_path}")

    with h5py.File(orig_path, "r") as f0:

        def _load(key: str) -> np.ndarray:
            arr = np.array(f0[key])
            # Transpose if multi-dimensional to match MATLAB layout
            return arr.T if arr.ndim > 1 else arr

        C = _load("C")
        atten0 = _load("atten")
        xi0 = _load("xi_orig")
        yi0 = _load("yi_orig")

    return C, atten0, xi0, yi0


def _compute_display_ranges(
    VEL: np.ndarray,
    ATT: np.ndarray,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Compute display ranges and convert attenuation to visualization units.

    Parameters
    ----------
    VEL
        Velocity estimate array (Ny, Nx, Niter).
    ATT
        Attenuation estimate array (Ny, Nx, Niter) in k-Wave units.

    Returns
    -------
    vmin_vel, vmax_vel : float
        Global min/max for velocity.
    vmin_att, vmax_att : float
        Global min/max for attenuation (dB/(MHz·mm)).
    ATT_vis : np.ndarray
        Converted attenuation, same shape as ATT.
    """
    vmin_vel, vmax_vel = float(VEL.min()), float(VEL.max())

    # convert attenuation to display units (dB/(MHz·mm))
    Np2dB = 20.0 / np.log(10.0)
    slow2atten = 1.0e6 / 1.0e2
    ATT_vis = Np2dB * slow2atten * ATT
    vmin_att, vmax_att = float(ATT_vis.min()), float(ATT_vis.max())

    return vmin_vel, vmax_vel, vmin_att, vmax_att, ATT_vis


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#


def animate_breast_inversion(
    res_path: Optional[Path | str] = None,
    orig_path: Optional[Path | str] = None,
    *,
    pause: float = 2.0,
    block: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Animate breast CT inversion results over iterations.

    Parameters
    ----------
    res_path
        Path to the inversion-results MAT file
        (default resolves to ``Results/kWave_BreastCT_WaveformInversionResults_cpu.mat``
        relative to the repository root).
    orig_path
        Path to the original sample data MAT file
        (default resolves to ``SampleData/kWave_BreastCT.mat`` relative
        to the repository root).
    pause
        Time in seconds to pause between frames.
    block
        If True, call ``plt.show(block=True)`` at the end. If False, the
        figure is left interactive and control returns immediately.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : np.ndarray
        The 2×3 array of axes.

    Raises
    ------
    FileNotFoundError
        If either of the MAT files cannot be found.
    """
    if res_path is None or orig_path is None:
        def_res, def_orig = _default_paths()
        if res_path is None:
            res_path = def_res
        if orig_path is None:
            orig_path = def_orig

    res_path = Path(res_path)
    orig_path = Path(orig_path)

    # --- Load data --------------------------------------------------
    xi, yi, VEL, ATT, GRAD, SRCH = _load_inversion_results(res_path)
    C, atten0, xi0, yi0 = _load_ground_truth(orig_path)
    vmin_vel, vmax_vel, vmin_att, vmax_att, ATT_vis = _compute_display_ranges(VEL, ATT)

    n_iter = VEL.shape[2]

    # --- Create plotting canvas ------------------------------------
    plt.ion()
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # top-left: Estimated Velocity
    im_vel = ax[0, 0].imshow(
        VEL[:, :, 0],
        extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        vmin=vmin_vel,
        vmax=vmax_vel,
        cmap="gray",
        aspect="auto",
    )
    ax[0, 0].set_title("Estimated Velocity")

    # top-middle: True Velocity
    ax[0, 1].imshow(
        C,
        extent=[xi0.min(), xi0.max(), yi0.max(), yi0.min()],
        vmin=vmin_vel,
        vmax=vmax_vel,
        cmap="gray",
        aspect="auto",
    )
    ax[0, 1].set_title("True Velocity")

    # top-right: Search Direction
    im_s = ax[0, 2].imshow(
        SRCH[:, :, 0],
        extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        cmap="gray",
        aspect="auto",
    )
    ax[0, 2].set_title("Search Direction")

    # bottom-left: Estimated Attenuation
    im_att = ax[1, 0].imshow(
        ATT_vis[:, :, 0],
        extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        vmin=vmin_att,
        vmax=vmax_att,
        cmap="gray",
        aspect="auto",
    )
    ax[1, 0].set_title("Estimated Attenuation")

    # bottom-middle: True Attenuation
    ax[1, 1].imshow(
        atten0,
        extent=[xi0.min(), xi0.max(), yi0.max(), yi0.min()],
        vmin=vmin_att,
        vmax=vmax_att,
        cmap="gray",
        aspect="auto",
    )
    ax[1, 1].set_title("True Attenuation")

    # bottom-right: Gradient
    im_g = ax[1, 2].imshow(
        -GRAD[:, :, 0],  # invert sign to match MATLAB convention
        extent=[xi.min(), xi.max(), yi.max(), yi.min()],
        cmap="gray",
        aspect="auto",
    )
    ax[1, 2].set_title("Gradient")

    for a in ax.ravel():
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")

    plt.tight_layout()
    plt.pause(0.1)

    # --- Animated update loop --------------------------------------
    try:
        for k in range(n_iter):
            # Estimated velocity
            im_vel.set_data(VEL[:, :, k])
            ax[0, 0].set_title(f"Estimated Velocity    (Iter {k + 1}/{n_iter})")

            # Search direction (dynamic colour scaling)
            sd = SRCH[:, :, k]
            im_s.set_data(sd)
            im_s.set_clim(sd.min(), sd.max())
            ax[0, 2].set_title(f"Search Direction      (Iter {k + 1}/{n_iter})")

            # Attenuation
            im_att.set_data(ATT_vis[:, :, k])
            ax[1, 0].set_title(f"Estimated Attenuation (Iter {k + 1}/{n_iter})")

            # Gradient (dynamic colour scaling)
            g = -GRAD[:, :, k]
            im_g.set_data(g)
            im_g.set_clim(g.min(), g.max())
            ax[1, 2].set_title(f"Gradient              (Iter {k + 1}/{n_iter})")

            fig.canvas.draw_idle()
            plt.pause(pause)
    except KeyboardInterrupt:
        # Allow user to stop the animation gracefully
        pass
    finally:
        if block:
            plt.ioff()
            plt.show()
        else:
            # leave interactive mode on and return immediately
            plt.draw()

    return fig, ax


# -----------------------------------------------------------------------------#
# CLI entry point
# -----------------------------------------------------------------------------#


def main(argv: Optional[list[str]] = None) -> None:
    """
    Command-line entry point.

    Examples
    --------
    Use default paths (relative to the repo root)::

        python -m chirpy.utils.animate_results

    Override paths explicitly::

        python -m chirpy.utils.animate_results \\
            --results /path/to/kWave_BreastCT_WaveformInversionResults_cpu.mat \\
            --original /path/to/kWave_BreastCT.mat
    """
    import argparse

    def_res, def_orig = _default_paths()

    parser = argparse.ArgumentParser(
        description="Animate breast CT inversion results over iterations."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=def_res,
        help=f"Path to inversion results MAT file (default: {def_res})",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=def_orig,
        help=f"Path to original sample data MAT file (default: {def_orig})",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Pause (seconds) between frames (default: 2.0).",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="Do not block at the end (useful in notebooks).",
    )

    args = parser.parse_args(argv)

    animate_breast_inversion(
        res_path=args.results,
        orig_path=args.original,
        pause=args.pause,
        block=not args.no_block,
    )


if __name__ == "__main__":
    main()
