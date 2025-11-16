import numpy as np
from scipy.io import loadmat
from pathlib import Path

from chirpy.geometry import ImageGrid2D, TransducerArray2D, GeometryConfigurator
from chirpy.data import AcquisitionData
from chirpy.data.image_data import ImageData
from chirpy.optimization.operator import WaveOperator
from chirpy.signals import GaussianModulatedPulse
from chirpy.utils.paths import detect_root
from chirpy.utils.progress import Progress, ProgressConfig

# --------------------------- Configuration --------------------------- #
# ROOT_DIR = detect_root()
ROOT_DIR = Path.cwd()
DATA_DIR = Path(ROOT_DIR / "data")
SAVE_DIR = Path(ROOT_DIR / "outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
KWAVE_DIR = None

progress = Progress(ProgressConfig(enabled=True, backend="tqdm", ncols=90))

# Grid / physics
Nx = Ny = 480
dx = dy = 0.5e-3
c0_ref = 1500.0

f0 = 0.3e6
use_gpu = False
use_tqdm = True

# Acquisition
n_tx = 512
radius = 110e-3


# --------------------------- Utilities --------------------------- #
def compute_record_time(grid: ImageGrid2D, c_min: float, pad: float = 1.3) -> float:
    Lx = grid.extent[1] - grid.extent[0]
    return float(pad * Lx / c_min)


# --------------------------- Main flow --------------------------- #
def main() -> None:
    # 1) Load & downsample ground-truth sound speed
    mat = loadmat(DATA_DIR / "alpha_true.mat")
    model_raw = mat["alpha_true"]  # expected 2D array
    img_grid = ImageGrid2D(nx=Nx, ny=Ny, dx=dx)

    img_true = ImageData(model_raw).downsample_to(new_grid=img_grid)
    a_true = img_true.array.astype(np.float32)

    # 2) Record time and reference speed
    a_min = float(a_true.min())
    c_ref = float(a_true.max())
    record_time = compute_record_time(img_grid, a_min)

    # 3) Ring array + acquisition container
    tx_array = TransducerArray2D.from_ring_array_2D(r=radius, grid=img_grid, n=n_tx)
    acq_geom = AcquisitionData.from_geometry(tx_array=tx_array, grid=img_grid)
    geom = GeometryConfigurator(img_grid, tx_array)
    geom.configure_acceptance(delta=0)

    # Optional quick view
    ImageData(array=a_true, tx_array=tx_array, grid=img_grid).show(cmap="viridis")


if __name__ == "__main__":
    main()
