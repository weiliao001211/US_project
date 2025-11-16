from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path

from chirpy.geometry import ImageGrid2D, TransducerArray2D, GeometryConfigurator
from chirpy.data import AcquisitionData, ImageData
from chirpy.optimization.operator import WaveOperator
from chirpy.optimization.gradient import AdjointStateGrad
from chirpy.optimization.function import NonlinearLS
from chirpy.optimization.algorithm import GD, CG_Time, SGD
from chirpy.utils.visualizer_multi_mode import Visualizer
from chirpy.signals import GaussianModulatedPulse
from chirpy.utils.progress import Progress, ProgressConfig

from chirpy.utils.multigrid_tools import (
    restrict_model_image,
    resample_observations_to_operator_time,
    resample_pulse_to_operator_time,
)

"""
Breast phantom inversion (time domain), coarse-grid stage (2 mm).

This script implements a coarse-grid FWI stage using observations
simulated on a fine grid (0.5 mm, 480x480). The workflow is:

    (1) Load the ground-truth sound speed C_true (fine resolution).
    (2) Define a fine imaging grid (Nx_f=480, dx_f=0.5 mm) and
        a coarse imaging grid (Nx_c=120, dx_c=2.0 mm) with the
        same physical extent.
    (3) Load precomputed fine-grid observations d_obs (time-domain).
    (4) Build a temporary coarse WaveOperator to obtain dt_coarse and Nt_coarse.
    (5) Resample d_obs (and the source pulse) from the fine time grid
        to the coarse time grid using Fourier interpolation with a
        Kaiser window (cf. k-Wave filterTimeSeries).
    (6) Build the final coarse WaveOperator using the resampled data
        and the resampled pulse, with source encoding + SGD optimization.
    (7) Run FWI on the coarse grid (2 mm), starting from a homogeneous
        background model, and save the results and visualizations.

This is only the coarse-grid stage; it does not yet propagate the
coarse reconstruction back to the fine grid.
"""

# --------------------------- Configuration --------------------------- #

ROOT_DIR = Path.cwd()
DATA_DIR = Path(ROOT_DIR / "data")
SAVE_DIR = Path(ROOT_DIR / "outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
KWAVE_DIR = None

progress = Progress(ProgressConfig(enabled=True, backend="tqdm", ncols=90))

# Inversion controls
USE_ENCODING = True
K = 1
TAU_MAX = 0.0
DROP_SELF_RX = True
NORMALIZE = True
N_ITER = 80
ALGO = "SGD"  # {"GD", "CG_Time", "SGD"}
ETA0 = 6.0e-1
PLOT_TIMELINE = False

# Fine grid / physics (matches simulation script)
Nx_f = Ny_f = 480
dx_f = dy_f = 0.5e-3          # 0.5 mm
f0_fine = 1.0e6               # 1 MHz in the forward simulation
n_tx = 512
radius = 110e-3

# Coarse grid (2 mm)
Nx_c = Ny_c = 280
dx_c = dy_c = 1.0e-3          # 1.0 mm

c0 = 1500.0                   # initial homogeneous background
use_gpu = False


def compute_record_time(grid: ImageGrid2D, c_min: float, pad: float = 1.3) -> float:
    Lx = grid.extent[1] - grid.extent[0]
    return float(pad * Lx / c_min)


def sgd_schedule(k: int, lr0: float) -> float:
    """
    Simple polynomial decay schedule:
        lr_k = lr0 / (1 + gamma * k)^p
    """
    gamma = 0.03
    p = 2.0
    return lr0 / ((1.0 + gamma * k) ** p)


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Load fine C_true and construct fine & coarse grids
    # ------------------------------------------------------------------
    model_raw = loadmat(DATA_DIR / "C_true.mat")["C_true"]

    grid_fine = ImageGrid2D(nx=Nx_f, ny=Ny_f, dx=dx_f)
    grid_coarse = ImageGrid2D(nx=Nx_c, ny=Ny_c, dx=dx_c)

    c_true_fine = ImageData(model_raw).downsample_to(new_grid=grid_fine).array
    # c_true_coarse = restrict_model_image(
    #     c_true_fine, grid_fine=grid_fine, grid_coarse=grid_coarse
    # ).astype(np.float64)
    c_true_coarse = ImageData(model_raw).downsample_to(new_grid=grid_coarse).array


    c_min = float(c_true_fine.min())
    c_ref = float(c_true_fine.max())
    record_time = compute_record_time(grid_fine, c_min)

    # ------------------------------------------------------------------
    # 2) Load fine-grid observations and time axis
    # ------------------------------------------------------------------
    obs_path = (
        DATA_DIR / f"d_obs_{Ny_f}x{Nx_f}_{dx_f * 1e3:.1f}mm_{f0_fine / 1e6:.1f}MHz_{n_tx}.npz"
    )
    dat = np.load(obs_path, allow_pickle=True)
    d_obs_fine = dat["array"]      # shape: (n_tx, n_rx, nt_fine)
    t_vec_fine = dat["time"]       # shape: (nt_fine,)
    dt_fine = float(t_vec_fine[1] - t_vec_fine[0])
    nt_fine = t_vec_fine.size

    # ------------------------------------------------------------------
    # 3) Build a temporary coarse operator to probe dt_coarse, Nt_coarse
    # ------------------------------------------------------------------
    tx_array_coarse = TransducerArray2D.from_ring_array_2D(
        r=radius, grid=grid_coarse, n=n_tx
    )
    geom_coarse = GeometryConfigurator(grid_coarse, tx_array_coarse)
    geom_coarse.configure_acceptance(delta=0)

    # medium for probing (sound_speed will be updated later anyway)
    medium_probe = {
        "density": np.full((Ny_c, Nx_c), 1000.0, np.float32),
        "alpha_coeff": np.zeros((Ny_c, Nx_c), np.float32),
        "alpha_power": 1.01,
        "alpha_mode": "no_dispersion",
    }
    pulse_probe = GaussianModulatedPulse(f0=f0_fine, frac_bw=0.75, amp=1.0)

    op_probe = WaveOperator(
        geom_config=geom_coarse,
        medium_params=medium_probe,
        record_time=record_time,
        record_full_wf=False,
        use_encoding=USE_ENCODING,
        tau_max=TAU_MAX,
        pulse=pulse_probe,
        c_ref=c_ref,
        use_gpu=use_gpu,
        progress=progress,
        binary_path=KWAVE_DIR,
    )

    dt_coarse = float(op_probe.dt)
    nt_coarse = int(op_probe.nt)
    time_axis_coarse = np.linspace(
        0.0, dt_coarse * nt_coarse, nt_coarse, endpoint=False
    )

    print(f"[info] fine: nt={nt_fine}, dt={dt_fine:.3e}")
    print(f"[info] coarse: nt={nt_coarse}, dt={dt_coarse:.3e}")

    # ------------------------------------------------------------------
    # 4) Resample observations and pulse to coarse time grid
    # ------------------------------------------------------------------
    d_obs_coarse = resample_observations_to_operator_time(
        d_obs_fine, nt_target=nt_coarse, window="kaiser", beta=8.0
    )

    # reconstruct fine pulse on fine time grid
    pulse_fine = GaussianModulatedPulse(f0=f0_fine, frac_bw=0.75, amp=1.0)
    pulse_wave_fine = pulse_fine.sample(dt_fine, nt_fine)

    # project pulse to coarse Nt (this will implicitly low-pass)
    pulse_wave_coarse = resample_pulse_to_operator_time(
        pulse_wave_fine, nt_target=nt_coarse, window="kaiser", beta=8.0
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # 5) Build the final coarse operator with resampled data
    # ------------------------------------------------------------------
    acq_coarse = AcquisitionData(
        array=d_obs_coarse,
        tx_array=tx_array_coarse,
        grid=grid_coarse,
        time=time_axis_coarse,
    )
    geom_coarse_final = GeometryConfigurator(grid_coarse, tx_array_coarse)
    geom_coarse_final.configure_acceptance(delta=0)

    medium_inv = {
        "density": np.full((Ny_c, Nx_c), 1000.0, np.float32),
        "alpha_coeff": np.zeros((Ny_c, Nx_c), np.float32),
        "alpha_power": 1.01,
        "alpha_mode": "no_dispersion",
    }
    # initial pulse object (will be overridden by pulse_wave_coarse)
    pulse_dummy = GaussianModulatedPulse(f0=f0_fine, frac_bw=0.75, amp=1.0)

    op = WaveOperator(
        data=acq_coarse,
        geom_config=geom_coarse_final,
        medium_params=medium_inv,
        record_time=record_time,
        record_full_wf=True,
        use_encoding=USE_ENCODING,
        tau_max=TAU_MAX,
        # drop_self_rx=DROP_SELF_RX,  # optional
        pulse=pulse_dummy,
        c_ref=c_ref,
        use_gpu=use_gpu,
        progress=progress,
        binary_path=KWAVE_DIR,
    )

    # override the internal pulse with the resampled / filtered version
    if pulse_wave_coarse.shape[0] != op.pulse.shape[1]:
        raise RuntimeError(
            f"pulse length mismatch: got {pulse_wave_coarse.shape[0]}, "
            f"operator.nt={op.pulse.shape[1]}"
        )
    op.pulse[0, :] = pulse_wave_coarse

    # ------------------------------------------------------------------
    # 6) Gradient, objective, and solver
    # ------------------------------------------------------------------
    grad = AdjointStateGrad(op, K=(K if (USE_ENCODING and K > 1) else None), seed=0)
    f_ls = NonlinearLS(op, grad_eval=grad, normalize=NORMALIZE)

    # initial model on the coarse grid
    m0 = ImageData(np.full((Ny_c, Nx_c), c0, np.float64))

    viz = Visualizer(
        xi=grid_coarse.xi,
        yi=grid_coarse.yi,
        C_true=c_true_coarse,
        atten_true=np.zeros_like(c_true_coarse),
        mode="vel",
    )

    if ALGO == "CG_Time":
        solver = CG_Time(viz=viz, progress=progress)
    elif ALGO == "SGD":
        solver = SGD(
            lr=50 * ETA0,
            schedule_fn=sgd_schedule,
            momentum=0.9,
            viz=viz,
            progress=progress,
        )
    else:
        solver = GD(
            lr=50 * ETA0,
            backtrack=False,
            max_bt=12,
            schedule_fn=lambda k, lr0: lr0,
            viz=viz,
            progress=progress,
        )

    solver.solve(fun=f_ls, m0=m0, kind="c", n_iter=N_ITER)
    rec_path = SAVE_DIR / f"record_{ALGO}_breast_coarse_{n_tx}.npz"
    solver.save_record(rec_path)
    print(f"[ok] Saved coarse-grid record → {rec_path}")

    # ------------------------------------------------------------------
    # 7) Final visualization on coarse grid
    # ------------------------------------------------------------------
    rec = solver.get_record()
    vel_all = np.array(rec["vel"], np.float64)
    grad_all = rec["grad"].real

    tx_x, tx_y = op.tx_pos
    extent = (
        -Nx_c / 2 * dx_c,
        Nx_c / 2 * dx_c,
        -Ny_c / 2 * dy_c,
        Ny_c / 2 * dy_c,
    )

    vmin_c, vmax_c = c_true_coarse.min(), c_true_coarse.max()
    g_abs = np.percentile(np.abs(grad_all), 99)
    vmin_g, vmax_g = -g_abs, g_abs

    fig, ax = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    for a, (arr, title, vmin, vmax) in zip(
        ax,
        [
            (c_true_coarse, "True c (coarse)", vmin_c, vmax_c),
            (vel_all[..., -1], f"Final recon @ {N_ITER}", vmin_c, vmax_c),
            (grad_all[..., -1], "Final gradient", vmin_g, vmax_g),
        ],
    ):
        im = a.imshow(
            arr,
            extent=extent,
            origin="lower",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )
        a.scatter(tx_x, tx_y, marker="*", s=30)
        a.set_title(title)
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
        fig.colorbar(im, ax=a, fraction=0.046)

    out_img = SAVE_DIR / "coarse_delta_c_grad.png"
    plt.savefig(out_img, dpi=220)
    print(f"[ok] {out_img}")


if __name__ == "__main__":
    main()