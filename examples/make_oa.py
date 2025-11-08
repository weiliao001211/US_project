from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import h5py
import numpy as np
from scipy.ndimage import zoom
from chirpy.utils.paths import detect_root

from chirpy.geometry import ImageGrid2D, TransducerArray2D
from chirpy.data import AcquisitionData
from chirpy.optimization.operator.wave_operator import WaveOperator
from chirpy.signals import GaussianModulatedPulse
import hdf5storage


# ======= CONFIG =======
# Set these two paths:
ROOT_DIR = detect_root()
DATA_DIR = Path(ROOT_DIR / "data")
H5_PATH = Path("NumericalBreastPhantoms-selected/hdf5/Neg_35_Left.h5")
OUT_MAT = Path(DATA_DIR / "kWave_BreastCT.mat")

# Reasonable defaults that won't explode runtime
AXIS_ORDER = "ZYX"  # order of axes in the HDF5 dataset
SLICE_AXIS = 0  # 0=Z, 1=Y, 2=X
SLICE_POLICY = "middle"  # "middle" or "max_variance"
SLICE_IDX = None  # override index (int) or None to use policy
TARGET_SHAPE_2D = (
    192,
    192,
)  # resample 2-D slice to this (H,W). Set None to keep native
N_TX = 128  # will be deduped after snapping to grid
F0_HZ = 0.3e6  # carrier frequency (Hz)
RING_MARGIN = 0.45  # fraction of inscribed radius (0..1)

# OA-BREAST labels: {0,2,3,4,5} = {bg, fibro, fat, skin, vessel}
VALID_LABELS = {0, 2, 3, 4, 5}
DEFAULT_SOS = {0: 1500.0, 2: 1515.0, 3: 1470.0, 4: 1650.0, 5: 1584.0}  # m/s
DEFAULT_MUA = {0: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 5: 2.00}  # 1/m

# >>>>> FIX: attenuation map expected by FD script <<<<<
# Per-label attenuation slope (dB/cm/MHz) → converted to Np/(Hz·m)
DEFAULT_ATTEN_DB_CM_MHZ = {
    0: 0.60,  # background ~ fat
    2: 0.80,  # fibro
    3: 0.60,  # fat
    4: 1.00,  # skin
    5: 0.30,  # vessel/blood
}
_DB_CM_MHZ_TO_NP_PER_HZ_M = 1.151292546e-5


def map_labels_to_atten_slope(labels: np.ndarray) -> np.ndarray:
    """Return α (Np/(Hz·m)) as a piecewise-constant map from labels."""
    a = np.empty(labels.shape, dtype=np.float32)
    for lab in np.unique(labels):
        lab_i = int(lab)
        a_db = DEFAULT_ATTEN_DB_CM_MHZ.get(lab_i, DEFAULT_ATTEN_DB_CM_MHZ[0])
        a[labels == lab_i] = a_db * _DB_CM_MHZ_TO_NP_PER_HZ_M
    return a


# ======================


# ------------------------------- HDF5 helpers ------------------------------- #
def _first_dataset(f: h5py.File) -> np.ndarray:
    for _, obj in f.items():
        if isinstance(obj, h5py.Dataset):
            return np.array(obj)
        if isinstance(obj, h5py.Group):
            for __, oo in obj.items():
                if isinstance(oo, h5py.Dataset):
                    return np.array(oo)
    raise ValueError("No datasets found in HDF5.")


def _find_labels(f: h5py.File) -> np.ndarray:
    for key in ("MergedPhantom", "merged_phantom", "phantom", "labels", "tissueType"):
        if key in f and isinstance(f[key], h5py.Dataset):
            return np.array(f[key])
    return _first_dataset(f)


def _ensure_axis_order_zyx(arr: np.ndarray, axis_order: str) -> np.ndarray:
    ao = axis_order.upper()
    if ao == "ZYX":
        return arr
    if ao == "XYZ":
        return np.transpose(arr, (2, 1, 0))
    if ao == "YZX":
        return np.transpose(arr, (1, 0, 2))
    raise ValueError(f"Unsupported axis_order={axis_order}.")


def _check_labels(lbl: np.ndarray):
    u = set(np.unique(lbl).tolist())
    if not u.issubset(VALID_LABELS):
        raise ValueError(
            f"Unexpected labels {sorted(u)}; expected subset of {sorted(VALID_LABELS)}."
        )


def _slice_axis_and_spacing(
    labels_zyx: np.ndarray,
    spacing_zyx_mm: Tuple[float, float, float],
    slice_axis: int,
    slice_idx: Optional[int],
    policy: Literal["middle", "max_variance", None],
) -> tuple[np.ndarray, int, Tuple[float, float]]:
    Z, Y, X = labels_zyx.shape
    if slice_idx is None:
        if policy == "max_variance":
            stats = (
                np.var(labels_zyx, axis=(1, 2))
                if slice_axis == 0
                else np.var(labels_zyx, axis=(0, 2))
                if slice_axis == 1
                else np.var(labels_zyx, axis=(0, 1))
            )
            slice_idx = int(np.argmax(stats))
        else:
            slice_idx = [Z // 2, Y // 2, X // 2][slice_axis]
    slc = [slice(None)] * 3
    slc[slice_axis] = slice_idx
    lbl2d = labels_zyx[tuple(slc)]
    sp = list(spacing_zyx_mm)
    spacing2d_mm = (
        tuple(sp[1:])
        if slice_axis == 0
        else (sp[0], sp[2])
        if slice_axis == 1
        else tuple(sp[:2])
    )
    return lbl2d, slice_idx, spacing2d_mm  # (H,W), (dy_mm, dx_mm)


def _resample_labels_2d(
    lbl2d: np.ndarray, target_shape_hw: Tuple[int, int]
) -> tuple[np.ndarray, Tuple[float, float]]:
    H, W = map(int, target_shape_hw)
    srcH, srcW = lbl2d.shape
    zf = (H / srcH, W / srcW)
    try:
        lbl2d_r = zoom(lbl2d, zf, order=0, grid_mode=True)
    except TypeError:
        lbl2d_r = zoom(lbl2d, zf, order=0)
    return lbl2d_r, (srcH / H, srcW / W)


# ------------------------------- label → fields ------------------------------- #
def map_labels_to_fields(
    labels: np.ndarray,
    label_to_sos: Dict[int, float],
    label_to_mua: Dict[int, float],
    gruneisen: float,
    normalize_p0: bool,
) -> tuple[np.ndarray, np.ndarray]:
    c = np.empty(labels.shape, dtype=np.float32)
    mua = np.empty(labels.shape, dtype=np.float32)
    for lab in np.unique(labels):
        lab = int(lab)
        c[labels == lab] = label_to_sos.get(lab, label_to_sos[0])
        mua[labels == lab] = label_to_mua.get(lab, label_to_mua[0])
    p0 = gruneisen * mua
    if normalize_p0 and p0.max() > 0:
        p0 = p0 / p0.max()
    return p0, c


# ------------------------------- simulation utils ------------------------------- #
def sanitize_c(c_in: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (c32, c_min_eff, c_ref_eff). Clamp air/background; fix mm/s if needed."""
    c = np.asarray(c_in, float).copy()
    if np.median(c) > 3000.0:  # likely mm/s
        c *= 1e-3
    mask = c > 1200.0
    if not np.any(mask):
        mask = c > np.percentile(c, 5.0)
    c[~mask] = 1500.0
    c_min_eff = max(1300.0, float(np.percentile(c[mask], 1.0)))
    c_ref_eff = float(np.median(c[mask]))
    return c.astype(np.float32), c_min_eff, c_ref_eff


def dedupe_ring_on_grid(
    ring: TransducerArray2D, grid: ImageGrid2D
) -> TransducerArray2D:
    snapped = ring.attach_to_grid(grid)
    seen, keep = set(), []
    for i in range(snapped.n_elements):
        x, y = float(snapped.positions[0, i]), float(snapped.positions[1, i])
        ix, iy = grid.coord2index(x, y)
        key = (ix, iy)
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    if len(keep) == snapped.n_elements:
        return snapped
    P = snapped.positions[:, keep]
    flags = np.ones(P.shape[1], dtype=bool)
    return TransducerArray2D(positions=P, is_tx=flags, is_rx=flags)


# ------------------------------- main builder ------------------------------- #
def build_kWave_BreastCT_from_OA(
    h5_path: Path,
    *,
    axis_order: str = AXIS_ORDER,
    source_spacing_mm: Optional[Tuple[float, float, float]] = None,  # (dz,dy,dx)
    slice_axis: int = SLICE_AXIS,
    slice_policy: Literal["middle", "max_variance", None] = SLICE_POLICY,
    slice_idx: Optional[int] = SLICE_IDX,
    target_shape_2d: Optional[Tuple[int, int]] = TARGET_SHAPE_2D,
    n_tx: int = N_TX,
    f0: float = F0_HZ,
    ring_margin: float = RING_MARGIN,
) -> dict:
    # 1) Load labels volume
    with h5py.File(h5_path, "r") as f:
        labels_zyx = _ensure_axis_order_zyx(_find_labels(f), axis_order)
    labels_zyx = labels_zyx.astype(np.uint8, copy=False)
    _check_labels(labels_zyx)

    # 2) Spacing defaults (mm)
    spacing_zyx_mm = (
        source_spacing_mm if source_spacing_mm is not None else (1.0, 1.0, 1.0)
    )

    # 3) Slice and (optionally) resample to target 2-D shape
    lbl2d, chosen_idx, (dy_mm, dx_mm) = _slice_axis_and_spacing(
        labels_zyx, spacing_zyx_mm, slice_axis, slice_idx, slice_policy
    )
    if target_shape_2d is not None:
        lbl2d, (sH, sW) = _resample_labels_2d(lbl2d, target_shape_2d)
        dy_mm *= sH
        dx_mm *= sW

    # 4) Map labels → fields
    _p0, c_map = map_labels_to_fields(
        lbl2d, DEFAULT_SOS, DEFAULT_MUA, gruneisen=0.2, normalize_p0=False
    )
    atten = map_labels_to_atten_slope(lbl2d)  # <<< FIX: provide attenuation (Np/(Hz·m))

    # 5) Image grid (meters)
    dx = float(dx_mm) * 1e-3
    dy = float(dy_mm) * 1e-3
    Ny, Nx = c_map.shape
    img_grid = ImageGrid2D(nx=Nx, ny=Ny, dx=dx, dy=dy)

    # 6) Sanitize sound speed + timing
    c32, c_min_eff, c_ref_eff = sanitize_c(c_map)
    xmin, xmax, ymin, ymax = img_grid.extent
    width = xmax - xmin
    record_time = 1.2 * width / c_min_eff

    # 7) Ring (snap + dedupe)
    radius = ring_margin * 0.5 * min(xmax - xmin, ymax - ymin)
    ring = TransducerArray2D.from_ring_array_2D(grid=img_grid, n=n_tx, r=radius)
    ring = dedupe_ring_on_grid(ring, img_grid)

    # 8) Simulate sequential shots (no full wavefields)
    acq_geom = AcquisitionData.from_geometry(grid=img_grid, tx_array=ring, c0=c_ref_eff)
    pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)
    op = WaveOperator(
        data=acq_geom,
        medium_params={"sound_speed": c32},
        record_time=record_time,
        record_full_wf=False,
        use_encoding=False,
        drop_self_rx=True,
        pulse=pulse,
        c_ref=c_ref_eff,
        use_gpu=True,
        verbose=False,
    )
    acq = op.simulate()  # element order (Tx, Rx, T)

    # 9) Build .mat payload expected by breast_frequency_domain.py
    raw = {
        "transducerPositionsXY": ring.positions.astype(np.float32),  # (2, Nuniq)
        "full_dataset": np.transpose(acq.array, (2, 1, 0)).astype(
            np.float64
        ),  # (T, Rx, Tx)
        "time": acq.time.astype(np.float64),
        "C": c32.astype(np.float32),
        "atten": atten.astype(np.float32),  # <<< FIX: include attenuation
        "dx": np.array(dx, np.float64),
        "dy": np.array(dy, np.float64),
        "slice_index": np.array(chosen_idx, np.int32),
    }
    return raw


raw = build_kWave_BreastCT_from_OA(h5_path=H5_PATH)

# Overwrite any existing non-HDF5 .mat to avoid signature mismatch
try:
    if OUT_MAT.exists():
        OUT_MAT.unlink()
except Exception as e:
    raise RuntimeError(f"Cannot remove existing file: {OUT_MAT} — {e}")

# Save as MATLAB v7.3 (HDF5) .mat so your loader (h5py) can open it
hdf5storage.savemat(
    str(OUT_MAT),
    raw,
    format="7.3",  # HDF5-based .mat
    oned_as="row",  # typical MATLAB row vectors
    store_python_metadata=False,
)

print(
    f"[ok] wrote {OUT_MAT} | Tx={raw['full_dataset'].shape[2]}, Rx={raw['full_dataset'].shape[1]}, T={raw['full_dataset'].shape[0]}"
)
