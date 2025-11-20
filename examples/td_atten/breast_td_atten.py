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
from chirpy.utils.paths import detect_root
from chirpy.utils.progress import Progress, ProgressConfig

"""
Breast phantom inversion (time domain) using precomputed observations.

Process:
1) Load C_true (for visualization/record-time) and downsample to working grid.
2) Build the same ring array and load the saved observations d_obs,time (from simulation step).
3) Configure WaveOperator (encoding on/off), adjoint gradient, and NonlinearLS.
4) Optimize from homogeneous initial model for N_ITER iterations.
5) Save record and plot final c and gradient (timeline optional).
"""

# --------------------------- Configuration --------------------------- #
# ROOT_DIR = detect_root()
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
N_ITER = 100
ALGO = "SGD"  # {"GD","CG_Time","SGD"}
ETA0 = 6.0e-1
PLOT_TIMELINE = True

# Grid / physics
Nx = Ny = 240
dx = dy = 1.0e-3
f0 = 0.3e6
c0 = 1500.0
n_tx = 512
radius = 110e-3

alpha0 = 0.0

use_gpu = False
use_tqdm = True


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
    # 1) Load & downsample C_true to get c_ref/record_time
    c_raw = loadmat(DATA_DIR / "C_true.mat")["C_true"]
    grid = ImageGrid2D(nx=Nx, ny=Ny, dx=dx)

    alpha_raw = loadmat(DATA_DIR / "alpha_true.mat")["alpha_true"]

    c_true = ImageData(c_raw).downsample_to(new_grid=grid).array
    alpha_true = ImageData(alpha_raw).downsample_to(new_grid=grid).array

    # ImageData(alpha_raw).downsample_to(new_grid=grid).show()
    # exit()

    c_ref = float(np.max(c_true))
    c_min = float(np.min(c_true))
    record_time = compute_record_time(grid, c_min)

    # 2) Ring array and load observations from prior simulation
    tx_array = TransducerArray2D.from_ring_array_2D(r=radius, grid=grid, n=n_tx)
    obs_path = (
            DATA_DIR / f"d_obs_with_atten_{Ny}x{Nx}_{dx * 1e3:.1f}mm_{f0 / 1e6:.1f}MHz_{n_tx}.npz"
    )
    dat = np.load(obs_path, allow_pickle=True)
    d_obs, t_vec = dat["array"], dat["time"]

    acq_inv = AcquisitionData(array=d_obs, tx_array=tx_array, grid=grid, time=t_vec)
    geom = GeometryConfigurator(grid, tx_array)

    # 3) Medium (fixed for inversion), operator, gradient, LS
    medium = {
        "sound_speed": np.full((Ny, Nx), c0, np.float32),
        "density": np.full((Ny, Nx), 1000.0, np.float32),
        "alpha_coeff": np.full((Ny, Nx), alpha_true, np.float32),
        "alpha_power": 1.01,
        "alpha_mode": "no_dispersion",
    }

    pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)
    op = WaveOperator(
        data=acq_inv,
        geom_config=geom,
        medium_params=medium,
        record_time=record_time,
        record_full_wf=True,
        use_encoding=USE_ENCODING,
        pulse=pulse,
        c_ref=c_ref,
        use_gpu=use_gpu,
        progress=progress,
        binary_path=KWAVE_DIR,
    )
    grad = AdjointStateGrad(op, K=(K if (USE_ENCODING and K > 1) else None), seed=0)
    f_ls = NonlinearLS(op, grad_eval=grad, normalize=NORMALIZE)


    # 4) Optimize
    m0 = ImageData(np.full((Ny, Nx), alpha0, np.float32))
    # m0 = ImageData(np.full((Ny, Nx), c0, np.float32))

    # r = f_ls.residual(m=m0.array, kind="alpha")
    # print(r)
    # AcquisitionData(array=r, time=t_vec).show_trace(tx=0, rx=1)
    # exit()

    viz = Visualizer(
        xi=grid.xi,
        yi=grid.yi,
        C_true=c_true,
        atten_true=alpha_true,
        mode="both"
    )

    solver = SGD(
        lr=50 * ETA0,
        schedule_fn=sgd_schedule,
        momentum=0.9,
        viz=viz,
        progress=progress,
    )

    solver.solve(fun=f_ls, m0=m0, kind="alpha", n_iter=N_ITER)

    rec_path = SAVE_DIR / f"record_{ALGO}_breast_{n_tx}.npz"
    solver.save_record(rec_path)
    print(f"[ok] Saved record → {rec_path}")

    # 5) Final visualization (+ optional timeline)
    rec = solver.get_record()
    vel_all = np.array(rec["vel"], np.float64)
    grad_all = rec["grad"].real
    tx_x, tx_y = op.tx_pos
    extent = (-Nx / 2 * dx, Nx / 2 * dx, -Ny / 2 * dy, Ny / 2 * dy)

    vmin_c, vmax_c = c_true.min(), c_true.max()
    g_abs = np.percentile(np.abs(grad_all), 99)
    vmin_g, vmax_g = -g_abs, g_abs

    fig, ax = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    for a, (arr, title, vmin, vmax) in zip(
            ax,
            [
                (c_true, "True c", vmin_c, vmax_c),
                (vel_all[..., -1], f"Final recon @ {N_ITER}", vmin_c, vmax_c),
                (grad_all[..., -1], "Final gradient", vmin_g, vmax_g),
            ],
    ):
        im = a.imshow(
            arr, extent=extent, origin="lower", cmap="seismic", vmin=vmin, vmax=vmax
        )
        a.scatter(tx_x, tx_y, marker="*", s=30)
        a.set_title(title)
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
        fig.colorbar(im, ax=a, fraction=0.046)
    out1 = SAVE_DIR / "final_delta_c_grad.png"
    plt.savefig(out1, dpi=220)
    print(f"[ok] {out1}")

    if PLOT_TIMELINE:
        mis = rec["misfit"][1, :]
        dC = vel_all.real - c0
        v_abs = 200
        vmin_r, vmax_r = -v_abs, v_abs
        n_rows = 1 + N_ITER
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(10, 3.2 * n_rows), constrained_layout=True
        )
        axes = np.atleast_2d(axes)

        im = axes[0, 0].imshow(
            c_true - c0,
            extent=extent,
            origin="lower",
            cmap="seismic",
            vmin=vmin_r,
            vmax=vmax_r,
        )
        axes[0, 0].set_title("True Δc")
        axes[0, 0].scatter(tx_x, tx_y, marker="*", s=30)
        fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

        im = axes[0, 1].imshow(
            grad_all[..., 0],
            extent=extent,
            origin="lower",
            cmap="seismic",
            vmin=vmin_g,
            vmax=vmax_g,
        )
        axes[0, 1].set_title("Initial gradient")
        axes[0, 1].scatter(tx_x, tx_y, marker="*", s=30)
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

        for i in range(1, n_rows):
            im0 = axes[i, 0].imshow(
                dC[..., i - 1],
                extent=extent,
                origin="lower",
                cmap="seismic",
                vmin=vmin_r,
                vmax=vmax_r,
            )
            axes[i, 0].set_title(f"Δc @ iter {i}\nmisfit={mis[i - 1]:.3e}")
            axes[i, 0].scatter(tx_x, tx_y, marker="*", s=30)
            fig.colorbar(im0, ax=axes[i, 0], fraction=0.046)

            im1 = axes[i, 1].imshow(
                grad_all[..., i - 1],
                extent=extent,
                origin="lower",
                cmap="seismic",
                vmin=vmin_g,
                vmax=vmax_g,
            )
            axes[i, 1].set_title(f"Grad @ iter {i}")
            axes[i, 1].scatter(tx_x, tx_y, marker="*", s=30)
            fig.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        out2 = SAVE_DIR / "timeline_delta_c_grad.png"
        plt.savefig(out2, dpi=220)
        print(f"[ok] {out2}")




if __name__ == "__main__":
    main()
