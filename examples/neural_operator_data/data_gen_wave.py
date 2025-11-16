"""
This script generates training data for neural operator models using
k-Wave time-domain simulations. The pipeline includes:

    (1) Load a 3D ultrasound breast phantom and extract a 2D slice.
    (2) Pad the slice to a square shape and downsample it to a target grid.
    (3) Build a circular 2D transducer array and configure the acquisition geometry.
    (4) Compute record time from minimum sound speed and grid width.
    (5) Run time-domain k-Wave forward simulations:
            • Incident fields (homogeneous medium)
            • Total fields (heterogeneous medium)
    (6) Transform wavefields to the frequency domain via DTFT.
    (7) Compute scattered fields = total - incident.
    (8) Package all relevant tensors and geometry metadata into a
        single NPZ file for neural operator training, including:
            • Incident fields (frequency domain)  # (Ny, Nx, N_tx, N_freqs)
            • Scattered fields (frequency domain) # (Ny, Nx, N_tx, N_freqs)
            • Frequencies
            • Grid metadata
            • Transducer geometry
            • True & homogeneous sound speed maps
            • Time-domain parameters

Outputs
    - Frequency-domain incident fields:
        outputs/neural_operator_data_wave_gen/incident_fields/*.npz
    - Frequency-domain scattered fields:
        outputs/neural_operator_data_wave_gen/scattered_fields/*.npz
    - Packed neural-operator dataset:
        outputs/neural_operator_data_wave_gen/neural_operator_data/neural_operator_training_data.npz
"""


from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np

from chirpy.geometry import ImageGrid2D, TransducerArray2D, GeometryConfigurator
from chirpy.data.image_data import ImageData

from chirpy.optimization.operator import WaveOperator
from chirpy.utils.progress import Progress, ProgressConfig
from chirpy.signals import GaussianModulatedPulse

from chirpy.utils.dtft_wavefield import dtft_wavefield

DAT_PATH = Path("data/NumerialBreastPhantoms/Neg_07_Left/MergedPhantom.DAT")

ROOT_DIR = Path.cwd()
DATA_PATH = Path(ROOT_DIR / DAT_PATH)

# -------------------- Configuration --------------------
SAVE_DIR = Path("outputs/neural_operator_data_wave_gen")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

use_gpu = False

# Phantom source
KWAVE_DIR = None
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
downsample_step_tx = 50 # select every 50th TX element
RADIUS_M = 110e-3 # 110 mm radius of the ring array
F0 = 1e6 # 1 MHz center frequency of pulse
C0_BG = 1500.0 # background sound speed

# Time record policy
RECORD_PAD = 1.3

# Frequencies
freqs = np.arange(0.3, 1.35, 0.05) * 1e6

# Progress
progress = Progress(ProgressConfig(enabled=True, backend="tqdm", ncols=90))

# Neural operator packaging
PACK_TENSORS = True

# Filenames
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
    Center-pad a 2D array to (target, target) using constant background value.
    """
    ny, nx = arr.shape
    if target < max(ny, nx):
        raise ValueError(f"target={target} < original size {(ny, nx)}")

    # compute padding sizes
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


def compute_record_time(
        grid: ImageGrid2D, c_min: float, pad: float = RECORD_PAD
) -> float:
    extent = grid.extent
    width = extent[1] - extent[0]
    return float(pad * width / c_min)


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

    # (3) Build ring array, record time, reference speed, geometry configurator
    log("Building 512-elt ring and computing record time...")
    tx_array = TransducerArray2D.from_ring_array_2D(r=RADIUS_M, grid=grid, n=N_TX)
    # quick view
    # geom_view = ImageData(array=c_true.array, grid=grid, tx_array=tx_array)
    # geom_view.show(cmap="viridis")

    c_ref = float(np.max(c_true.array))
    c_min = float(np.min(c_true.array))
    record_time = compute_record_time(grid, c_min)
    print(f"Reference sound speed: {c_ref} m/s")
    print(f"Minimum sound speed: {c_min} m/s")
    print(f"Record time: {record_time * 1e6:.1f} us")

    # Downsample transmitters
    geom_config = GeometryConfigurator(grid, tx_array)
    geom_config.configure_acceptance(delta=0)
    geom_config.select_tx(step=downsample_step_tx)  # select every 50th TX element

    print("Freqs: ", freqs)

    # (4) Medium + pulse + operator
    medium = {
        "density": np.full_like(c_true.array, 1000.0, dtype=np.float32),
        "alpha_coeff": np.zeros_like(c_true.array, dtype=np.float32),
        "alpha_power": 1.01,
        "alpha_mode": "no_dispersion",
    }
    pulse = GaussianModulatedPulse(f0=F0, frac_bw=0.75, amp=1.0)
    op = WaveOperator(
        geom_config=geom_config,
        medium_params=medium,
        record_time=record_time,
        record_full_wf=True,
        pulse=pulse,
        c_ref=c_ref,
        use_gpu=use_gpu,
        binary_path=KWAVE_DIR,
        progress=progress,
        # verbose=True,
    )

    time_axis = op.time_axis  # (Nt,)

    # (5) Simulate incident fields (homogeneous c0)
    c_hom = np.full_like(c_true.array, C0_BG, dtype=np.float32)
    _ = op.forward(model=c_hom, kind='c')
    incident_fields_td = op.get_forward_fields()  # (Ntx, Nt, Ny, Nx)

    log("Transforming incident fields to frequency-domain...")
    incident_fields_fd = dtft_wavefield(
        incident_fields_td,
        dt=op.dt,
        freqs=freqs,
    )  # (Ny, Nx, Ntx, Nfreqs)

    # quick view
    # ImageData(incident_fields_fd[:, :, 0, 0], grid=grid).show(cmap="viridis", title="Incident field")

    # save incident fields (frequency-domain)
    log("Saving incident fields (frequency-domain)...")
    n_tx = incident_fields_fd.shape[2]
    out_path = INC_DIR / f"inc_fields_fd_{nx_grid}x{ny_grid}_{dx_grid * 1e3:.1f}mm_{n_tx}.npz"

    np.savez(
        out_path,
        incident_fields=incident_fields_fd,
        freqs=freqs,
    )
    log(f"Saved incident fields to: {out_path}")

    # (6) Simulate scattered fields
    log("Simulating scattered fields...")
    c_het = c_true.array
    _ = op.forward(model=c_het, kind='c')
    total_fields_td = op.get_forward_fields()  # (Ntx, Nt, Ny, Nx)

    log("Transforming total fields to frequency-domain...")
    total_fields_fd = dtft_wavefield(
        total_fields_td,
        dt=op.dt,
        freqs=freqs,
    ) # (Ny, Nx, Ntx, Nfreqs)
    scattered_fields_fd = total_fields_fd - incident_fields_fd # (Ny, Nx, Ntx, Nfreqs)

    # quick view
    # ImageData(total_fields_fd[:, :, 0, 0], grid=grid).show(cmap="viridis", title="Total field")
    # ImageData(scattered_fields_fd[:, :, 0, 0], grid=grid).show(cmap="viridis", title="Scattered field")

    # save scattered fields (frequency-domain)
    log("Saving scattered fields (frequency-domain)...")
    n_tx = scattered_fields_fd.shape[2]
    out_path = SCAT_DIR / f"scat_fields_fd_{nx_grid}x{ny_grid}_{dx_grid * 1e3:.1f}mm_{n_tx}.npz"
    np.savez(
        out_path,
        scattered_fields=scattered_fields_fd,
        freqs=freqs,
    )

    # (7) Pack full neural-operator dataset
    log("Packing full neural-operator dataset...")

    tx_x_idx = geom_config.elem_x_idx[geom_config.tx_keep]
    tx_y_idx = geom_config.elem_y_idx[geom_config.tx_keep]

    tx_positions = tx_array.positions[:, geom_config.tx_keep]  # (2, n_tx)

    np.savez(
        NEURAL_NPZ,
        incident_fields_fd=incident_fields_fd, # (Ny, Nx, N_tx, N_freqs)
        scattered_fields_fd=scattered_fields_fd, # (Ny, Nx, N_tx, N_freqs)
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
        c_hom=c_hom,

        # time-domain info
        dt=op.dt,
        time_axis=time_axis,
    )

    log(f"Saved neural-operator dataset to: {NEURAL_NPZ}")


if __name__ == "__main__":
    main()
