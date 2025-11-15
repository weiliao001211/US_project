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

from chirpy.optimization.function.least_squares import NonlinearLS
from chirpy.optimization.algorithm.cg import CG
from chirpy.optimization.operator.helmholtz import HelmholtzOperator
from chirpy.optimization.gradient.adjoint_helmholtz import HelmholtzAdjointGrad

from scipy.io import savemat

"""
Chirpy-based refactor of Leili's work, with tqdm progress bars.

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
DAT_PATH = Path("NumericalBreastPhantoms-selected/Neg_07_Left/MergedPhantom.DAT")
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
N_TX = 368
RADIUS_M = 70e-3
F0 = 0.3e6
C0_BG = 1500.0
DROP_SELF_RX_SIM = False

# Time record policy
RECORD_PAD = 1.3

# Frequencies
F_SOS = np.arange(0.25, 0.35, 0.02) * 1e6
F_ATT = np.arange(0.255, 0.355, 0.02) * 1e6
FREQS = np.concatenate([F_SOS, F_ATT])

# Iterations per frequency
NITER_SOS_PER_FREQ = np.array([3] * F_SOS.size + [3] * F_ATT.size, dtype=int)
NITER_ATT_PER_FREQ = np.array([0] * F_SOS.size + [3] * F_ATT.size, dtype=int)

# Preprocessing
ACCEPTANCE_DELTA = 63
OUTLIER_THRESH = 0.99

# Neural operator packaging
PACK_TENSORS = True
use_gpu = False

# Filenames
OBS_NAME = SAVE_DIR / f"d_obs_{PAD_TO}x{PAD_TO}_{int(DX*1e3)}mm_0p3MHz_new_{N_TX}.npz"
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


def compute_record_time(
    grid: ImageGrid2D, c_min: float, pad: float = RECORD_PAD
) -> float:
    extent = grid.extent
    width = extent[1] - extent[0]
    return float(pad * width / c_min)


# -------------------- Main flow --------------------
def main() -> None:
    log("=== Refactor start (chirpy) ===")

    # (1) Load & map phantom labels
    log("Loading phantom DAT and building sound-speed slice...")
    labels = load_labels_slice(DAT_PATH, (NX, NY, NZ), SLICE_AXIS_X_INDEX)  # (Ny, Nz)
    ss_raw, _rho_raw = labels_to_maps(labels)  # (Ny, Nz)

    # (2) Downsample to 120×120 then pad to 180×180
    log("Downsampling to 120×120 and padding to 180×180...")
    grid_core = ImageGrid2D(nx=CORE_N, ny=CORE_N, dx=DX, dy=DY)
    c_core = ImageData(ss_raw).downsample_to(new_grid=grid_core).array
    pad_w = (PAD_TO - CORE_N) // 2
    c_true = np.pad(
        c_core, pad_w, mode="constant", constant_values=SOUND_SPEED_MAP[0]
    )  # (180,180)
    grid_pad = ImageGrid2D(nx=PAD_TO, ny=PAD_TO, dx=DX, dy=DY)

    # (3) Ring array & record time
    log("Building 368-elt ring and computing record time...")
    tx_array = TransducerArray2D.from_ring_array_2D(r=RADIUS_M, grid=grid_pad, n=N_TX)
    c_ref = float(np.max(c_true))
    c_min = float(np.min(c_true))
    record_time = compute_record_time(grid_pad, c_min)
    log(f"record_time = {record_time*1e3:.2f} ms")

    # (4) Acquisition container
    acq_geom = AcquisitionData.from_geometry(tx_array=tx_array, grid=grid_pad, c0=C0_BG)

    # (5) Forward simulation (time domain) — load if exists
    pulse = GaussianModulatedPulse(f0=F0, frac_bw=0.75, amp=1.0)
    if OBS_NAME.exists():
        log(f"Found existing observations → {OBS_NAME}")
        dat = np.load(OBS_NAME, allow_pickle=True)
        acq_for_freq = AcquisitionData(
            array=dat["array"],
            time=dat["time"],
            tx_array=tx_array,
            grid=grid_pad,
            c0=C0_BG,
        )
    else:
        log("Simulating time-domain data with chirpy.WaveOperator (no encoding) ...")
        from chirpy.optimization.operator.wave_operator import WaveOperator

        medium = {
            "sound_speed": c_true.astype(np.float32),
            "density": np.full_like(c_true, 1000.0, dtype=np.float32),
            "alpha_coeff": np.zeros_like(c_true, dtype=np.float32),
            "alpha_power": 1.01,
            "alpha_mode": "no_dispersion",
        }
        op_true = WaveOperator(
            data=acq_geom,
            medium_params=medium,
            record_time=record_time,
            record_full_wf=False,
            use_encoding=False,
            drop_self_rx=DROP_SELF_RX_SIM,
            pulse=pulse,
            c_ref=c_ref,
            use_gpu=use_gpu,
            binary_path=KWAVE_DIR,
        )
        acq_sim = op_true.simulate()
        acq_sim.save(OBS_NAME)
        log(f"[ok] Saved observations → {OBS_NAME}")
        acq_for_freq = acq_sim

    # (6) Frequency-domain preprocessing
    log(
        "Running preprocessing pipeline (Gaussian window → DTFT → phase screen → mask → outlier filter)..."
    )
    pipe = Pipeline(
        stages=[
            GaussianTimeWindow(),
            DTFT(FREQS),
            PhaseScreenCorrection(grid_pad),
            DownSample(step=1),
            AcceptanceMask(delta=ACCEPTANCE_DELTA),
            MagnitudeOutlierFilter(threshold=OUTLIER_THRESH),
        ],
        verbose=True,
    )
    acq_fd = pipe(acq_for_freq)  # (Tx,Rx,Nfreq)

    Tx, Rx, Nfreq = acq_fd.array.shape
    assert Nfreq == FREQS.size, "Frequency count mismatch after preprocessing."

    # (7) Prepare inversion scaffolding
    log("Setting up per-frequency Helmholtz operator, gradient, and CG solver...")
    cg = CG(c1=1e-4, shrink=0.5, max_ls=20)
    slow = ImageData(
        array=np.full(
            (grid_pad.ny, grid_pad.nx),
            (1.0 / 1480.0) + 1j * (0.0 / (2.0 * np.pi)),
            dtype=np.complex128,
        ),
        grid=grid_pad,
    )

    # Optional: allocate neural-operator tensors up front
    if PACK_TENSORS:
        ny, nx = c_true.shape
        ui_tensor = np.zeros((Nfreq, ny, nx, N_TX), dtype=np.complex128)
        us_tensor = np.zeros((Nfreq, ny, nx, N_TX), dtype=np.complex128)
        c_xy_tensor = (
            np.broadcast_to(c_true[np.newaxis, ...], (Nfreq, ny, nx))
            .astype(np.float64)
            .copy()
        )
        frequency_array = FREQS.astype(np.float64)

    # -------------------- Progress bar for frequency loop --------------------
    freq_bar = tqdm(range(Nfreq), desc="Frequencies", unit="freq")
    for k in freq_bar:
        f_mhz = FREQS[k] / 1e6
        freq_bar.set_postfix_str(f"f={f_mhz:.3f} MHz")

        op = HelmholtzOperator(
            acq_fd,
            k,
            sign_conv=-1,
            pml_alpha=10.0,
            pml_size=9.0e-3,
            use_gpu=use_gpu,
        )
        grad = HelmholtzAdjointGrad(
            op,
            deriv_fn=lambda m, o: 8
            * np.pi**2
            * o.get_field("freq") ** 2
            * (o.get_field("PML") / o.get_field("V")),
        )
        fun = NonlinearLS(op, grad_eval=grad)

        # ---- Incident fields (homogeneous c0) ----
        slow_inc = np.full(
            (grid_pad.ny, grid_pad.nx), (1.0 / C0_BG) + 0j, dtype=np.complex128
        )
        _ = op.forward(slow_inc)  # caches wavefield for all sources
        incident_fields = op._cache.WF.copy()  # (ny, nx, n_tx)

        # Save incident fields
        inc_path = INC_DIR / f"incident_fields_freq_{k:02d}_f_{f_mhz:.3f}MHz.npy"
        np.save(inc_path, incident_fields)

        # ---- Total fields in heterogeneous medium → scattered = total - incident ----
        slow_het = (1.0 / c_true).astype(np.complex128)  # no attenuation
        _ = op.forward(slow_het)
        total_fields = op._cache.WF.copy()
        scattered_fields = total_fields - incident_fields

        # Save scattered fields
        scat_path = SCAT_DIR / f"scattered_fields_freq_{k:02d}_f_{f_mhz:.3f}MHz.npy"
        np.save(scat_path, scattered_fields)

        # ---- Pack tensors (optional) ----
        if PACK_TENSORS:
            ui_tensor[k, ...] = incident_fields
            us_tensor[k, ...] = scattered_fields

        # ---- Inversion: SoS-only then attenuation-only ----
        ns = int(NITER_SOS_PER_FREQ[k])
        na = int(NITER_ATT_PER_FREQ[k])

        # Small inner bars to show CG progress per frequency (optional, cheap)
        if ns > 0:
            for _ in tqdm(
                range(ns), leave=False, desc=f"CG real @ {f_mhz:.3f}MHz", unit="it"
            ):
                cg.solve(
                    fun, slow, n_iter=1, mode="real", viz=None, do_print_time=False
                )
        if na > 0:
            for _ in tqdm(
                range(na), leave=False, desc=f"CG imag @ {f_mhz:.3f}MHz", unit="it"
            ):
                cg.solve(
                    fun, slow, n_iter=1, mode="imag", viz=None, do_print_time=False
                )

    # (9) Save inversion recorder snapshots (.mat)
    rec = cg.get_record()
    log("Saving inversion snapshots to MATLAB .mat ...")
    savemat(
        MAT_RESULTS,
        {
            "xi": grid_pad.xi,
            "yi": grid_pad.yi,
            "fDATA": FREQS.reshape(1, -1),
            "niterAttenPerFreq": NITER_ATT_PER_FREQ.reshape(1, -1),
            "niterSoSPerFreq": NITER_SOS_PER_FREQ.reshape(1, -1),
            "VEL_ESTIM_ITER": rec["vel"],
            "ATTEN_ESTIM_ITER": rec["atten"],
            "GRAD_IMG_ITER": rec["grad"],
            "SEARCH_DIR_ITER": rec["search"],
        },
        do_compression=True,
    )
    log(f"[ok] Saved → {MAT_RESULTS}")

    # (10) Save neural operator training tensors (optional)
    if PACK_TENSORS:
        log("Saving consolidated neural-operator training tensors (.npz) ...")
        np.savez_compressed(
            NEURAL_NPZ,
            ui_tensor=ui_tensor,
            us_tensor=us_tensor,
            c_xy_tensor=c_xy_tensor,
            frequency_array=frequency_array,
            grid_spacing=grid_pad.dx,
            grid_extent=grid_pad.extent,
            n_frequencies=Nfreq,
            n_transmitters=N_TX,
            processed_freq_indices=list(range(Nfreq)),
            description="Neural operator training data: tensor format",
        )
        size_mb = NEURAL_NPZ.stat().st_size / 1024**2
        log(f"[ok] Saved → {NEURAL_NPZ} ({size_mb:.1f} MB)")

    log("=== Done ===")


if __name__ == "__main__":
    main()
