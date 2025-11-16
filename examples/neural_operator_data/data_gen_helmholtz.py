"""
This script generates training data for neural operator models using
frequency-domain Helmholtz simulations. The pipeline includes:

    (1) Load a 3D ultrasound breast phantom and extract a 2D slice.
    (2) Pad the slice to a square shape and downsample it to a target grid.
    (3) Build a circular 2D transducer array and configure the acquisition geometry.
    (4) Loop over a set of frequencies and, for each frequency:
            • Solve the Helmholtz equation in a homogeneous background
              to obtain incident fields.
            • Solve the Helmholtz equation in the heterogeneous medium
              to obtain total fields.
            • Form scattered fields = total - incident.
            • Save incident and scattered fields for that frequency.
    (5) Stack the per-frequency fields into tensors of shape:
            • Incident fields:  (Ny, Nx, N_tx, N_freqs)
            • Scattered fields: (Ny, Nx, N_tx, N_freqs)
    (6) Package all relevant tensors and geometry metadata into a
        single NPZ file for neural operator training, including:
            • Incident fields (frequency domain)   # (Ny, Nx, N_tx, N_freqs)
            • Scattered fields (frequency domain)  # (Ny, Nx, N_tx, N_freqs)
            • Frequencies
            • Grid metadata
            • Transducer geometry
            • True & homogeneous sound speed maps

Outputs
    - Per-frequency incident fields:
        outputs/neural_operator_data_Helmholtz_gen/incident_fields/*.npz
    - Per-frequency scattered fields:
        outputs/neural_operator_data_Helmholtz_gen/scattered_fields/*.npz
    - Packed neural-operator dataset:
        outputs/neural_operator_data_Helmholtz_gen/neural_operator_data/neural_operator_training_data.npz
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np

from chirpy.geometry import ImageGrid2D, TransducerArray2D, GeometryConfigurator
from chirpy.data.image_data import ImageData

from chirpy.optimization.operator.helmholtz import HelmholtzOperator
from chirpy.utils.progress import Progress, ProgressConfig

DAT_PATH = Path("data/NumerialBreastPhantoms/Neg_07_Left/MergedPhantom.DAT")

ROOT_DIR = Path.cwd()
DATA_PATH = Path(ROOT_DIR / DAT_PATH)

# -------------------- Configuration --------------------
SAVE_DIR = Path("outputs/neural_operator_data_Helmholtz_gen")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

use_gpu = True

# Phantom source
KWAVE_DIR = None  # kept for symmetry with time-domain script, not used here
NX, NY, NZ = 616, 485, 719  # raw 3D (x,y,z) used to extract a single x-slice
SLICE_AXIS_X_INDEX = NX // 2  # take the middle X-plane

# Acoustic mapping
SOUND_SPEED_MAP = {
    0: 1500.0,  # background / water
    2: 1540.0,  # fibroglandular
    3: 1450.0,  # fat
    4: 1555.0,  # skin
    5: 1548.0,  # vessel
}
DENSITY_MAP = {0: 1000.0, 2: 1040.0, 3: 911.0, 4: 1100.0, 5: 945.0}

# Grids
nx_grid = ny_grid = 240
dx_grid = dy_grid = 1e-3  # 1 mm spacing
xmax = 120e-3  # 240*240 mm field with nx=480/dx=0.5mm(fine) or nx=240/dx=1mm(coarse)

# Array & physics
N_TX = 512
downsample_step_tx = 1  # select every nth TX element
RADIUS_M = 110e-3  # 110 mm radius of the ring array
F0 = 1e6  # kept for symmetry, not directly used
C0_BG = 1500.0  # background sound speed

# Frequencies (Hz)
freqs = np.arange(0.3, 1.35, 0.05) * 1e6  # (N_freqs,)

# Progress
progress = Progress(ProgressConfig(enabled=True, backend="tqdm", ncols=90))

# Neural operator packaging
PACK_TENSORS = True

# Filenames / directories
INC_DIR = SAVE_DIR / "incident_fields"
SCAT_DIR = SAVE_DIR / "scattered_fields"
INC_DIR.mkdir(exist_ok=True, parents=True)
SCAT_DIR.mkdir(exist_ok=True, parents=True)
NEURAL_DIR = SAVE_DIR / "neural_operator_data"
NEURAL_DIR.mkdir(exist_ok=True, parents=True)
NEURAL_NPZ = NEURAL_DIR / "neural_operator_training_data.npz"


# -------------------- Utilities --------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_labels_slice(
        dat_path: Path, shape_xyz: tuple[int, int, int], x_index: int
) -> np.ndarray:
    """Load 3D uint8 labels in Fortran order and return one x-slice as (Ny, Nz)."""
    NX_, NY_, NZ_ = shape_xyz
    with open(dat_path, "rb") as f:
        raw = np.fromfile(f, dtype=np.uint8)
    raw = raw.reshape((NX_, NY_, NZ_), order="F")
    return raw[x_index, :, :]  # (Ny, Nz)


def labels_to_maps(labels_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map integer labels → sound speed & density arrays."""
    ss = np.zeros_like(labels_2d, dtype=np.float32)
    rho = np.zeros_like(labels_2d, dtype=np.float32)
    for lab, c in SOUND_SPEED_MAP.items():
        mask = labels_2d == lab
        ss[mask] = float(c)
        rho[mask] = float(DENSITY_MAP.get(lab, 1000.0))
    ss[ss <= 0] = SOUND_SPEED_MAP[3]  # default unmapped to fat
    return ss, rho


def pad_to_square(arr: np.ndarray, target: int, bg: float = 1500.0) -> np.ndarray:
    """
    Center-pad a 2D array to (target, target) using a constant background value.
    """
    ny, nx = arr.shape
    if target < max(ny, nx):
        raise ValueError(f"target={target} < original size {(ny, nx)}")

    pad_y = target - ny
    pad_x = target - nx

    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left

    arr_pad = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=float(bg),
    )
    return arr_pad


def main() -> None:
    # (1) Load & map phantom labels
    log("Loading phantom DAT and building sound-speed slice...")
    labels = load_labels_slice(DAT_PATH, (NX, NY, NZ), SLICE_AXIS_X_INDEX)  # (Ny, Nz)
    c_raw, _rho_raw = labels_to_maps(labels)  # (Ny, Nz)
    # ImageData(c_raw).show(cmap="viridis")

    # (2) Pad to square & downsample to target grid
    log("Padding to square 1024x1024")
    c_pad = pad_to_square(c_raw, target=1024, bg=1500.0)
    # ImageData(c_pad).show(cmap="viridis")

    log("Building target grid and downsampling true sound-speed map")
    grid = ImageGrid2D(nx=nx_grid, dx=dx_grid)
    c_true = ImageData(c_pad).downsample_to(new_grid=grid)
    # c_true.show(cmap="viridis")

    # (3) Build ring array and geometry configurator
    log("Building 512-element ring array and configuring geometry...")
    tx_array = TransducerArray2D.from_ring_array_2D(r=RADIUS_M, grid=grid, n=N_TX)
    geom_config = GeometryConfigurator(grid, tx_array)
    geom_config.configure_acceptance(delta=0)
    geom_config.select_tx(step=downsample_step_tx)  # select every 50th TX element

    log(f"Frequencies (Hz): {freqs}")

    # Slowness fields (frequency-independent, no attenuation here)
    slow_hom = np.full((grid.ny, grid.nx), (1.0 / C0_BG), dtype=np.complex128)
    slow_het = (1.0 / c_true.array).astype(np.complex128)

    # Number of selected transmitters
    n_tx = geom_config.tx_keep.size
    n_y, n_x = grid.ny, grid.nx
    n_freqs = freqs.size

    # -------------------- Per-frequency simulations + save --------------------
    for k, f in enumerate(freqs):
        f_mhz = f * 1e-6
        log(f"Solving Helmholtz at f = {f_mhz:.3f} MHz ({k + 1}/{n_freqs})")

        op = HelmholtzOperator(
            geom_config=geom_config,
            freq=f,
            sign_conv=-1,
            pml_alpha=10.0,
            pml_size=9.0e-3,
            use_gpu=use_gpu,
            progress=progress,
        )

        # Incident fields in homogeneous background
        _ = op.forward(slow_hom)
        incident_fields = op._cache.WF.copy()  # (ny, nx, n_tx)

        # Total fields in heterogeneous medium
        _ = op.forward(slow_het)
        total_fields = op._cache.WF.copy()  # (ny, nx, n_tx)

        scattered_fields = total_fields - incident_fields  # (ny, nx, n_tx)

        # Save per-frequency incident fields
        inc_path = INC_DIR / (
            f"inc_fields_fd_{nx_grid}x{ny_grid}_{dx_grid * 1e3:.1f}mm_"
            f"{n_tx}_f{f_mhz:.3f}MHz.npz"
        )
        np.savez(
            inc_path,
            incident_fields=incident_fields,
            freqs=np.array([f], dtype=float),
        )
        log(f"Saved incident fields for f={f_mhz:.3f} MHz to: {inc_path}")

        # Save per-frequency scattered fields
        scat_path = SCAT_DIR / (
            f"scat_fields_fd_{nx_grid}x{ny_grid}_{dx_grid * 1e3:.1f}mm_"
            f"{n_tx}_f{f_mhz:.3f}MHz.npz"
        )
        np.savez(
            scat_path,
            scattered_fields=scattered_fields,
            freqs=np.array([f], dtype=float),
        )
        log(f"Saved scattered fields for f={f_mhz:.3f} MHz to: {scat_path}")

        # quick view
        # i = 0
        # ImageData(incident_fields[:, :, i], grid=grid).show(cmap="viridis", title="Incident Field")
        # ImageData(total_fields[:, :, i], grid=grid).show(cmap="viridis", title="Total Field")
        # ImageData(scattered_fields[:, :, i], grid=grid).show(cmap="viridis", title="Scattered Field")

    # -------------------- Pack full neural-operator dataset --------------------
    if PACK_TENSORS:
        log("Loading per-frequency files and packing full tensors...")

        incident_fields_fd = np.zeros(
            (n_y, n_x, n_tx, n_freqs), dtype=np.complex128
        )
        scattered_fields_fd = np.zeros_like(incident_fields_fd)

        for k, f in enumerate(freqs):
            f_mhz = f * 1e-6

            inc_path = INC_DIR / (
                f"inc_fields_fd_{nx_grid}x{ny_grid}_{dx_grid * 1e3:.1f}mm_"
                f"{n_tx}_f{f_mhz:.3f}MHz.npz"
            )
            scat_path = SCAT_DIR / (
                f"scat_fields_fd_{nx_grid}x{ny_grid}_{dx_grid * 1e3:.1f}mm_"
                f"{n_tx}_f{f_mhz:.3f}MHz.npz"
            )

            if not inc_path.exists():
                raise FileNotFoundError(f"Missing incident file: {inc_path}")
            if not scat_path.exists():
                raise FileNotFoundError(f"Missing scattered file: {scat_path}")

            inc_npz = np.load(inc_path)
            scat_npz = np.load(scat_path)

            inc_f = inc_npz["incident_fields"]  # (ny, nx, n_tx)
            scat_f = scat_npz["scattered_fields"]  # (ny, nx, n_tx)

            if inc_f.shape != (n_y, n_x, n_tx):
                raise RuntimeError(
                    f"Packed incident shape mismatch at f={f}: "
                    f"got {inc_f.shape}, expected {(n_y, n_x, n_tx)}"
                )
            if scat_f.shape != (n_y, n_x, n_tx):
                raise RuntimeError(
                    f"Packed scattered shape mismatch at f={f}: "
                    f"got {scat_f.shape}, expected {(n_y, n_x, n_tx)}"
                )

            incident_fields_fd[:, :, :, k] = inc_f
            scattered_fields_fd[:, :, :, k] = scat_f

        log("Packing full neural-operator dataset...")

        tx_x_idx = geom_config.elem_x_idx[geom_config.tx_keep]
        tx_y_idx = geom_config.elem_y_idx[geom_config.tx_keep]
        tx_positions = tx_array.positions[:, geom_config.tx_keep]  # (2, n_tx)

        np.savez(
            NEURAL_NPZ,
            incident_fields_fd=incident_fields_fd,  # (Ny, Nx, N_tx, N_freqs)
            scattered_fields_fd=scattered_fields_fd,  # (Ny, Nx, N_tx, N_freqs)
            freqs=freqs,

            # geometry
            grid_nx=grid.nx,
            grid_ny=grid.ny,
            grid_dx=grid.spacing[0],
            grid_dy=grid.spacing[1],
            grid_extent=np.array(grid.extent),

            # tx-rx metadata
            tx_indices=geom_config.tx_keep,
            rx_indices=geom_config.rx_keep,
            tx_positions=tx_positions,
            tx_grid_x=tx_x_idx,
            tx_grid_y=tx_y_idx,

            # medium
            c_true=c_true.array,
            c_hom=np.full_like(c_true.array, C0_BG, dtype=np.float32),
        )

        log(f"Saved neural-operator dataset to: {NEURAL_NPZ}")


if __name__ == "__main__":
    main()
