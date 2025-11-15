import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from chirpy.geometry import TransducerArray2D, ImageGrid2D, GeometryConfigurator
from chirpy.data import AcquisitionData, ImageData
from chirpy.optimization.operator.wave_operator import WaveOperator
from chirpy.optimization.gradient.time_grad import AdjointStateGrad
from chirpy.optimization.function.least_squares import NonlinearLS
from chirpy.optimization.algorithm import GD, CG_Time, SGD
from chirpy.utils.visualizer_multi_mode import Visualizer
from chirpy.signals import GaussianModulatedPulse
from chirpy.utils.paths import detect_root
from chirpy.utils.progress import Progress, ProgressConfig

"""
Two-circle time-domain inversion demo.

Process:
1) Build ImageGrid2D and a synthetic "true" c(x,y) with two circular inclusions.
2) Construct a ring array and AcquisitionData; simulate d_obs (no encoding).
3) Build inversion operator (with or without source encoding), define LS + adjoint gradient.
4) Run optimization (GD or CG_Time) for N_ITER iterations from a homogeneous initial model.
5) Save solver record and plot final maps plus (optionally) a timeline grid.
"""

# --------------------------- Configuration --------------------------- #
# ROOT_DIR = detect_root()
ROOT_DIR = Path.cwd()
SAVE_DIR = Path(ROOT_DIR / "outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

progress = Progress(ProgressConfig(enabled=True, backend="tqdm", ncols=90))

# Grid / physics
Nx = Ny = 128
dx = dy = 5.0e-4
c0 = 1500.0
n_tx = 64
record_time = 1.2 * Nx * dx / c0

# Inversion
USE_ENCODING = True  # source encoding on/off
K = 5  # random enc. averages when encoding
TAU_MAX = 0.0  # random delay bound
DROP_SELF_RX = True  # used if not encoding
N_ITER = 10
NORMALIZE = True
ALGO = "SGD"  # {"GD","CG_Time"}
ETA0 = 6.0e-1

# Exec
use_gpu = False
use_tqdm = True
PLOT_TIMELINE = True  # set False to skip big grid of images


def make_true_model(grid: ImageGrid2D) -> np.ndarray:
    X, Y = grid.meshgrid()
    m = np.full((grid.ny, grid.nx), c0, np.float64)
    m[((X - 0.01) ** 2 + (Y - 0.01) ** 2) < 0.006**2] = c0 + 100
    m[(X**2 + (Y + 0.01) ** 2) < 0.005**2] = c0 - 100
    return m


def main() -> None:
    # 1) Grid and true model
    grid = ImageGrid2D(dx=dx, nx=Nx, ny=Ny)
    c_true = make_true_model(grid)
    tx_array = TransducerArray2D.from_ring_array_2D(
        grid=grid, r=(min(Nx, Ny) // 2 - 2) * dx, n=n_tx
    )
    acq_geom = AcquisitionData.from_geometry(grid=grid, tx_array=tx_array)
    geom = GeometryConfigurator(grid, tx_array)
    geom.configure_acceptance(delta=0)

    # 2) Forward data (no encoding)
    pulse = GaussianModulatedPulse(f0=5e5, frac_bw=0.75, amp=1.0)
    op_true = WaveOperator(
        data=acq_geom,
        geom_config=geom,
        medium_params={"sound_speed": c_true},
        record_time=record_time,
        use_encoding=False,
        record_full_wf=False,
        pml_size=10,
        cfl=0.2,
        # drop_self_rx=True,
        pulse=pulse,
        use_gpu=use_gpu,
        progress=progress,
    )
    acq_sim = op_true.simulate()
    obs_path = SAVE_DIR / f"acq_sim_ring_full_{n_tx}.npz"
    acq_sim.save(obs_path)
    dat = np.load(obs_path, allow_pickle=True)
    d_obs, t_vec = dat["array"], dat["time"]

    # 3) Inversion operator, LS, gradient
    acq_inv = AcquisitionData(
        array=d_obs, tx_array=acq_geom.tx_array, grid=grid, time=t_vec
    )

    op_inv = WaveOperator(
        data=acq_inv,
        geom_config=geom,
        medium_params={"sound_speed": c0},
        record_time=record_time,
        use_encoding=USE_ENCODING,
        tau_max=(TAU_MAX if USE_ENCODING else 0.0),
        record_full_wf=True,
        pml_size=10,
        cfl=0.2,
        # drop_self_rx=bool(DROP_SELF_RX and not USE_ENCODING),
        pulse=pulse,
        use_gpu=use_gpu,
        progress=progress,
    )
    grad = AdjointStateGrad(op_inv, K=(K if (USE_ENCODING and K > 1) else None), seed=0)
    f_ls = NonlinearLS(op_inv, grad_eval=grad, weight=1.0, normalize=NORMALIZE)

    # 4) Initial model & optimizer
    m0 = ImageData(np.full((Ny, Nx), c0, np.float64))
    viz = Visualizer(
        xi=grid.xi,
        yi=grid.yi,
        C_true=c_true,
        atten_true=np.zeros_like(c_true),
        mode="vel",
    )
    if ALGO == "GD":
        solver = GD(
            lr=50 * ETA0,
            backtrack=False,
            max_bt=12,
            schedule_fn=lambda k, lr: lr,
            viz=viz,
            progress=progress,
        )
    elif ALGO == "SGD":
        def sgd_schedule(k: int, lr0: float) -> float:
            return lr0 * (0.95**(k // 5))

        solver = SGD(
            lr=50 * ETA0,
            schedule_fn=sgd_schedule,
            momentum=0.9,
            viz=viz,
            progress=progress,
        )
    else:
        solver = CG_Time(viz=viz, progress=progress)

    solver.solve(kind="c", fun=f_ls, m0=m0, n_iter=N_ITER)
    rec_path = SAVE_DIR / f"record_{'ENC' if USE_ENCODING else 'NOENC'}.npz"
    solver.save_record(rec_path)
    print(f"[ok] Saved record → {rec_path}")

    # 5) Visualization (final + optional timeline)
    rec = solver.get_record()
    vel_all = np.array(rec["vel"], dtype=np.float64)
    grad_all = np.array(rec["grad"], dtype=np.float64).real
    tx_x, tx_y = op_inv.tx_pos
    extent = (-Nx / 2 * dx, Nx / 2 * dx, -Ny / 2 * dy, Ny / 2 * dy)

    vmin_c, vmax_c = c_true.min(), c_true.max()
    g_abs = np.percentile(np.abs(grad_all), 99)
    vmin_g, vmax_g = -g_abs, g_abs

    fig, ax = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    for a, (arr, title, vmin, vmax) in zip(
        ax,
        [
            (c_true, "True c", vmin_c, vmax_c),
            (vel_all[..., -1], f"Recon it {N_ITER}", vmin_c, vmax_c),
            (grad_all[..., -1], f"Grad it {N_ITER}", vmin_g, vmax_g),
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
    out1 = SAVE_DIR / "final.png"
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
            axes[i, 0].set_title(f"Δc @ iter {i}\nmisfit={mis[i - 1]:.2e}")
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

        out2 = SAVE_DIR / "timeline.png"
        plt.savefig(out2, dpi=220)
        print(f"[ok] {out2}")


if __name__ == "__main__":
    main()
