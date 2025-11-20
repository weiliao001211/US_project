#!/usr/bin/env python3
"""
Comparison script: Original Helmholtz (CuPy/SciPy) vs JAX Helmholtz.

    - If CuPy is installed → original uses GPU.
    - If CuPy is NOT installed → original uses SciPy (CPU).
    - JAX uses whatever backend it is already configured to use.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# Working directory
# -----------------------------------------------------------------------------
WORKDIR = Path(".")
if WORKDIR.exists():
    os.chdir(WORKDIR)

# -----------------------------------------------------------------------------
# Check CuPy availability
# -----------------------------------------------------------------------------
try:
    import cupy as cp

    GPU_AVAILABLE_CUPY = True
    print("✓ CuPy found → Original operator will use GPU")
except Exception:
    GPU_AVAILABLE_CUPY = False
    print("✗ CuPy not found → Original operator will use SciPy (CPU)")

# -----------------------------------------------------------------------------
# Check JAX backend
# -----------------------------------------------------------------------------
import jax

jax.config.update("jax_enable_x64", True)  # match SciPy numerics

devices = jax.devices()
if not devices:
    print("✗ JAX has no available devices — cannot continue")
    sys.exit(1)

JAX_PLATFORM = devices[0].platform.upper()
print(f"✓ JAX running on: {JAX_PLATFORM}")

ORIGINAL_USE_GPU = GPU_AVAILABLE_CUPY
ORIGINAL_LABEL = "Original (CuPy GPU)" if ORIGINAL_USE_GPU else "Original (SciPy CPU)"
JAX_LABEL = f"JAX ({JAX_PLATFORM})"

# -----------------------------------------------------------------------------
# Imports from chirpy
# -----------------------------------------------------------------------------
print("\nImporting chirpy modules...")

from chirpy.geometry import ImageGrid2D, TransducerArray2D
from chirpy.data import AcquisitionData
from chirpy.optimization.operator.helmholtz import HelmholtzOperator
from chirpy.optimization.operator.helmholtz_jax import HelmholtzSolverJAX


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        return cp.asnumpy(x)
    except Exception:
        return np.asarray(x)


def get_tx_xy(tx_array, grid):
    for name in ("positions", "xy", "coords", "centers", "elements_xy"):
        if hasattr(tx_array, name):
            arr = np.asarray(getattr(tx_array, name))
            if arr.ndim == 2:
                if arr.shape[1] == 2:
                    return arr
                if arr.shape[0] == 2:
                    return arr.T
    raise RuntimeError("Cannot extract TX positions")


def nearest_idx(axis_vals, coords):
    j = np.searchsorted(axis_vals, coords)
    j = np.clip(j, 1, len(axis_vals) - 1)
    left = axis_vals[j - 1]
    right = axis_vals[j]
    return np.where(np.abs(coords - left) <= np.abs(coords - right), j - 1, j).astype(
        np.int32
    )


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PAD_TO = 180
DX = DY = 1e-3
N_TX = 368
RADIUS_M = 70e-3
C0_BG = 1500.0

FREQS = np.arange(0.25e6, 0.35e6, 0.02e6)
N_FREQ_TEST = len(FREQS)

PML_ALPHA = 10.0
PML_SIZE = 9e-3
SIGN_CONV = -1

print("\n" + "=" * 70)
print("TEST CONFIGURATION")
print("=" * 70)
print(f"• Grid: {PAD_TO}x{PAD_TO}")
print(f"• dx = dy = {DX*1e3:.1f} mm")
print(f"• TX count: {N_TX}")
print(f"• Frequencies: {FREQS/1e6} MHz")
print(f"• Original: {ORIGINAL_LABEL}")
print(f"• JAX: {JAX_LABEL}")

# -----------------------------------------------------------------------------
# Synthetic Data
# -----------------------------------------------------------------------------
print("\nCreating synthetic test data...")

grid_pad = ImageGrid2D(nx=PAD_TO, ny=PAD_TO, dx=DX, dy=DY)
tx_array = TransducerArray2D.from_ring_array_2D(r=RADIUS_M, grid=grid_pad, n=N_TX)

# Heterogeneous speed map
c_true = np.ones((PAD_TO, PAD_TO)) * C0_BG
y, x = np.ogrid[:PAD_TO, :PAD_TO]
center = PAD_TO // 2
for dx0, dy0 in [(-20, -20), (20, 20), (-20, 20), (20, -20)]:
    mask = (x - (center + dx0)) ** 2 + (y - (center + dy0)) ** 2 < 15**2
    c_true[mask] = 1540.0

atten_zeros = np.zeros_like(c_true)

dummy_array = np.zeros((N_TX, N_TX, N_FREQ_TEST), dtype=np.complex128)
acq_data = AcquisitionData(
    array=dummy_array,
    tx_array=tx_array,
    grid=grid_pad,
    freqs=FREQS,
    c0=C0_BG,
)

tx_xy = get_tx_xy(tx_array, grid_pad)
x_idx = nearest_idx(grid_pad.xi, tx_xy[:, 0])
y_idx = nearest_idx(grid_pad.yi, tx_xy[:, 1])

acq_data.ctx = {
    "elem_mask": np.ones((N_TX, N_TX), bool),
    "x_idx": x_idx,
    "y_idx": y_idx,
    "grid_lin_idx": np.arange(N_TX, dtype=np.int32),
}

SRC = np.zeros((grid_pad.ny, grid_pad.nx, N_TX), np.complex128)
for s, (ix, iy) in enumerate(zip(x_idx, y_idx)):
    SRC[iy, ix, s] = 1.0


# -----------------------------------------------------------------------------
# Original operator runner
# -----------------------------------------------------------------------------
def run_original(f_idx, slow):
    t0 = time.time()
    op = HelmholtzOperator(
        acq_data,
        f_idx,
        sign_conv=SIGN_CONV,
        pml_alpha=PML_ALPHA,
        pml_size=PML_SIZE,
        use_gpu=ORIGINAL_USE_GPU,
    )
    if ORIGINAL_USE_GPU:
        cp.cuda.Stream.null.synchronize()
    t_init = time.time() - t0

    t1 = time.time()
    _ = op.forward(slow)
    if ORIGINAL_USE_GPU:
        cp.cuda.Stream.null.synchronize()
    wf = to_numpy(op._cache.WF)
    t_solve = time.time() - t1

    return t_init, t_solve, wf


# -----------------------------------------------------------------------------
# JAX operator runner
# -----------------------------------------------------------------------------
jax_cache = {}


def run_jax(f_idx, vel, atten, src):
    freq = FREQS[f_idx]
    key = (freq, hash(vel.tobytes()))

    t0 = time.time()
    if key not in jax_cache:
        op = HelmholtzSolverJAX(
            x=grid_pad.xi,
            y=grid_pad.yi,
            vel=vel,
            atten=atten,
            f=freq,
            signConvention=SIGN_CONV,
            a0=PML_ALPHA,
            L_PML=PML_SIZE,
        )
        jax_cache[key] = op
    else:
        op = jax_cache[key]
    t_init = time.time() - t0

    t1 = time.time()
    wf, _ = op.solve(src, adjoint=False)
    t_solve = time.time() - t1

    return t_init, t_solve, wf


# -----------------------------------------------------------------------------
# Run tests
# -----------------------------------------------------------------------------
print("\nRunning benchmarks...\n")

results_orig = []
results_jax = []

slow_inc = np.full_like(c_true, 1.0 / C0_BG, dtype=np.complex128)
slow_het = (1.0 / c_true).astype(np.complex128)
vel_inc = np.full_like(c_true, C0_BG)
vel_het = c_true

for f_idx, freq in enumerate(FREQS):
    print(f"\n--- Frequency {freq/1e6:.3f} MHz ---")

    # Original:
    ti, ts, w_inc = run_original(f_idx, slow_inc)
    _, ts2, w_het = run_original(f_idx, slow_het)
    orig_total = ti + ts + ts2

    results_orig.append(
        {
            "incident": w_inc,
            "scattered": w_het - w_inc,
            "t_total": orig_total,
            "t_init": ti,
            "t_incident": ts,
            "t_hetero": ts2,
        }
    )

    # JAX:
    ti_j, ts_j, j_inc = run_jax(f_idx, vel_inc, atten_zeros, SRC)
    _, ts_j2, j_het = run_jax(f_idx, vel_het, atten_zeros, SRC)
    jax_total = ti_j + ts_j + ts_j2

    results_jax.append(
        {
            "incident": j_inc,
            "scattered": j_het - j_inc,
            "t_total": jax_total,
            "t_init": ti_j,
            "t_incident": ts_j,
            "t_hetero": ts_j2,
        }
    )

    print(f"  {ORIGINAL_LABEL} total: {orig_total:.3f}s")
    print(f"  {JAX_LABEL} total:      {jax_total:.3f}s")

    diff_inc = np.max(
        np.abs(results_orig[-1]["incident"] - results_jax[-1]["incident"])
    )
    diff_scat = np.max(
        np.abs(results_orig[-1]["scattered"] - results_jax[-1]["scattered"])
    )
    print(f"  max|incident diff|  = {diff_inc:.2e}")
    print(f"  max|scattered diff| = {diff_scat:.2e}")

# -----------------------------------------------------------------------------
# Accuracy summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ACCURACY SUMMARY")
print("=" * 70)

all_ok = True
rtol = 1e-6
atol = 1e-6

for f_idx, freq in enumerate(FREQS):
    inc_o = results_orig[f_idx]["incident"]
    inc_j = results_jax[f_idx]["incident"]
    sc_o = results_orig[f_idx]["scattered"]
    sc_j = results_jax[f_idx]["scattered"]

    def relerr(a, b):
        num = np.max(np.abs(a - b))
        den = np.max(np.abs(b)) + 1e-15
        return num / den, num

    r_inc, a_inc = relerr(inc_o, inc_j)
    r_sc, a_sc = relerr(sc_o, sc_j)

    passed = (a_inc < atol or r_inc < rtol) and (a_sc < atol or r_sc < rtol)
    all_ok = all_ok and passed

    status = "PASS" if passed else "FAIL"
    print(f"\nFreq {freq/1e6:.3f} MHz: {status}")
    print(f"  Incident rel={r_inc:.2e}, abs={a_inc:.2e}")
    print(f"  Scatter  rel={r_sc:.2e}, abs={a_sc:.2e}")

print("\nOverall:", "✓ ALL PASSED" if all_ok else "✗ SOME FAILED")

# -----------------------------------------------------------------------------
# Performance summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)

orig_times = [r["t_total"] for r in results_orig]
jax_times = [r["t_total"] for r in results_jax]

print(f"\nOriginal total: {sum(orig_times):.3f}s")
print(f"JAX total:      {sum(jax_times):.3f}s")
print(f"Speedup:        {(sum(orig_times)/(sum(jax_times)+1e-15)):.2f}x")
print("=" * 70)
