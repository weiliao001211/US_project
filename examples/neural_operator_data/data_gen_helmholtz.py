from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np

# Progress bar (works in terminals & notebooks)
from tqdm.auto import tqdm

# --- chirpy imports (UFWI -> chirpy) ---
from chirpy.geometry import ImageGrid2D, TransducerArray2D
from chirpy.data import AcquisitionData
from chirpy.data.image_data import ImageData
from chirpy.signals import GaussianModulatedPulse

from chirpy.processors import (
    GaussianTimeWindow,
    DTFT,
    PhaseScreenCorrection,
    DownSample,
    AcceptanceMask,
    MagnitudeOutlierFilter,
    Pipeline,
)

from chirpy.optimization.operator.helmholtz import HelmholtzOperator

from scipy.io import savemat

"""


Outputs (unchanged):
- outputs/d_obs_180x180_1mm_0p3MHz_new_368.npz
- outputs/incident_fields/incident_fields_freq_XX_f_YYY.npy
- outputs/scattered_fields/scattered_fields_freq_XX_f_YYY.npy
- outputs/kWave_BreastCT_WaveformInversionResults.mat
- outputs/neural_operator_data/neural_operator_training_data.npz

Note on memory: PACK_TENSORS=True builds large 4D tensors; set False to skip.
"""

# -------------------- Configuration (kept consistent with your script) --------------------
SAVE_DIR = Path("outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Phantom source
KWAVE_DIR = None
DAT_PATH = Path("../../data/NumericalBreastPhantoms/Neg_07_Left/MergedPhantom.DAT")
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
CORE_N = 120
PAD_TO = 180
DX = DY = 1.0e-3  # 1 mm spacing

# Array & physics
N_TX = 512
RADIUS_M = 110e-3
F0 = 1e6
C0_BG = 1500.0

# Time record policy
RECORD_PAD = 1.3

# Frequencies
F_SOS = np.arange(0.25, 0.35, 0.02) * 1e6

# Neural operator packaging
PACK_TENSORS = True
use_gpu = False

# Filenames
OBS_NAME = SAVE_DIR / f"d_obs_{PAD_TO}x{PAD_TO}_{int(DX * 1e3)}mm_0p3MHz_new_{N_TX}.npz"
INC_DIR = SAVE_DIR / "incident_fields"
SCAT_DIR = SAVE_DIR / "scattered_fields"
INC_DIR.mkdir(exist_ok=True, parents=True)
SCAT_DIR.mkdir(exist_ok=True, parents=True)
NEURAL_DIR = SAVE_DIR / "neural_operator_data"
NEURAL_DIR.mkdir(exist_ok=True, parents=True)

MAT_RESULTS = SAVE_DIR / "kWave_BreastCT_WaveformInversionResults.mat"
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


def main() -> None:
    # (1) Load & map phantom labels
    log("Loading phantom DAT and building sound-speed slice...")
    labels = load_labels_slice(DAT_PATH, (NX, NY, NZ), SLICE_AXIS_X_INDEX)  # (Ny, Nz)
    ss_raw, _rho_raw = labels_to_maps(labels)  # (Ny, Nz)


if __name__ == "__main__":
    main()
