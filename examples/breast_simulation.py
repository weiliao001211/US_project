import numpy as np
from scipy.io import loadmat
from pathlib import Path

from chirpy.geometry import ImageGrid2D, TransducerArray2D
from chirpy.data import AcquisitionData
from chirpy.data.image_data import ImageData
from chirpy.optimization.operator import WaveOperator
from chirpy.signals import GaussianModulatedPulse
from chirpy.utils.paths import detect_root
from chirpy.utils.progress import Progress, ProgressConfig

"""
Breast phantom simulation (time domain).

Process:
1) Load ground-truth speed map C_true (MATLAB .mat).
2) Downsample to the working ImageGrid2D (Nx, Ny, dx, dy).
3) Build a ring TransducerArray2D and AcquisitionData on that grid.
4) Configure WaveOperator with medium parameters and a Gaussian-modulated pulse.
5) Run forward simulation to synthesize observations; save to outputs/.
"""

# --------------------------- Configuration --------------------------- #
ROOT_DIR = detect_root()
DATA_DIR = Path(ROOT_DIR / "data")
SAVE_DIR = Path(ROOT_DIR / "outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

progress = Progress(ProgressConfig(enabled=True, backend="tqdm", ncols=90))

# Grid / physics
Nx = Ny = 240
dx = dy = 1.0e-3
c0_ref = 1500.0
f0 = 0.3e6
use_gpu = False
use_tqdm = True

# Acquisition
n_tx = 512
radius = 110e-3  # 110 mm


# --------------------------- Utilities --------------------------- #
def compute_record_time(grid: ImageGrid2D, c_min: float, pad: float = 1.3) -> float:
    Lx = grid.extent[1] - grid.extent[0]
    return float(pad * Lx / c_min)


# --------------------------- Main flow --------------------------- #
def main() -> None:
    # 1) Load & downsample ground-truth sound speed
    mat = loadmat(DATA_DIR / "C_true.mat")
    model_raw = mat["C_true"]  # expected 2D array
    img_grid = ImageGrid2D(nx=Nx, ny=Ny, dx=dx)

    img_true = ImageData(model_raw).downsample_to(new_grid=img_grid)
    c_true = img_true.array.astype(np.float32)

    # 2) Record time and reference speed
    c_min = float(c_true.min())
    c_ref = float(c_true.max())
    record_time = compute_record_time(img_grid, c_min)

    # 3) Ring array + acquisition container
    tx_array = TransducerArray2D.from_ring_array_2D(r=radius, grid=img_grid, n=n_tx)
    acq_geom = AcquisitionData.from_geometry(tx_array=tx_array, grid=img_grid)

    # Optional quick view
    # ImageData(array=c_true, tx_array=tx_array, grid=img_grid).show()

    # 4) Medium + pulse + operator
    medium = {
        "sound_speed": c_true,
        "density": np.full_like(c_true, 1000.0, dtype=np.float32),
        "alpha_coeff": np.zeros_like(c_true, dtype=np.float32),
        "alpha_power": 1.01,
        "alpha_mode": "no_dispersion",
    }
    pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)

    op = WaveOperator(
        data=acq_geom,
        medium_params=medium,
        record_time=record_time,
        record_full_wf=False,
        use_encoding=False,
        drop_self_rx=True,
        pulse=pulse,
        c_ref=c_ref,
        use_gpu=use_gpu,
        progress=progress,
    )

    # 5) Simulate and save
    out = op.simulate()
    out_path = (
        SAVE_DIR / f"d_obs_{Ny}x{Nx}_{dx * 1e3:.0f}mm_{f0 / 1e6:.1f}MHz_{n_tx}.npz"
    )
    out.save(out_path)
    print(f"[ok] Saved observations → {out_path}")


if __name__ == "__main__":
    main()
