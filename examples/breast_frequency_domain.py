from pathlib import Path
import numpy as np
from scipy.io import savemat

from chirpy.io import load_mat
from chirpy.geometry import TransducerArray2D, ImageGrid2D
from chirpy.data import AcquisitionData, ImageData
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
from chirpy.utils.visualizer_multi_mode import Visualizer
from chirpy.utils.paths import detect_root
from chirpy.utils.progress import Progress, ProgressConfig

"""
Breast inversion (frequency domain, Helmholtz).

Process:
1) Load raw k-Wave data + transducer positions → AcquisitionData on ImageGrid2D.
2) Preprocess via Pipeline: Gaussian window → DTFT(freqs) → phase-screen
   correction → downsample → acceptance mask → magnitude outlier filter.
3) For each frequency, build HelmholtzOperator + adjoint gradient; run CG with
   SoS-only iterations first (real part), then attenuation-only (imag part).
4) Save recorder snapshots to a MATLAB .mat file for downstream analysis.
"""

# --------------------------- Configuration --------------------------- #
ROOT_DIR = detect_root()
DATA_DIR = Path(ROOT_DIR / "data")
SAVE_DIR = Path(ROOT_DIR / "outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

RAW_MAT = Path(DATA_DIR / "kWave_BreastCT.mat")

progress = Progress(ProgressConfig(enabled=True, backend="tqdm", ncols=90))

dxi = 0.6e-3
xmax = 120e-3
c0 = 1540.0

f_sos = np.arange(0.3, 1.3, 0.05) * 1e6
f_att = np.arange(0.325, 1.325, 0.05) * 1e6
freqs = np.concatenate([f_sos, f_att])
use_gpu = False


def main() -> None:
    # 1) Load raw data
    raw = load_mat(RAW_MAT)
    pos = raw["transducerPositionsXY"]  # (2, N)
    N = pos.shape[1]
    tx_array = TransducerArray2D(
        positions=pos.astype(np.float32),
        is_tx=np.ones(N, dtype=bool),
        is_rx=np.ones(N, dtype=bool),
    )
    grid = ImageGrid2D(dx=dxi, xmax=xmax)

    acq = AcquisitionData(
        array=raw["full_dataset"].transpose(2, 1, 0),  # (Tx,Rx,T)
        time=raw["time"],
        tx_array=tx_array,
        grid=grid,
        c0=c0,
    )

    # 2) Preprocess pipeline → (Tx,Rx,Nfreq)
    pipe = Pipeline(
        stages=[
            GaussianTimeWindow(),
            DTFT(freqs),
            PhaseScreenCorrection(grid),
            DownSample(step=1),
            AcceptanceMask(delta=63),
            MagnitudeOutlierFilter(threshold=0.99),
        ],
        verbose=True,
    )
    acq = pipe(acq)

    Tx, Rx, Nf = acq.array.shape
    n_sos, n_att = f_sos.size, f_att.size
    assert n_sos + n_att == Nf
    niterSoSPerFreq = np.array([3] * n_sos + [3] * n_att)
    niterAttenPerFreq = np.array([0] * n_sos + [3] * n_att)

    # 3) Initial complex slowness and visualizer
    SLOW_INIT = (1.0 / 1480.0) + 1j * (0.0 / (2.0 * np.pi))
    slow = ImageData(
        array=np.full((grid.ny, grid.nx), SLOW_INIT, np.complex128), grid=grid
    )

    viz = Visualizer(
        xi=grid.xi,
        yi=grid.yi,
        C_true=raw["C"],
        atten_true=raw["atten"],
        mode="both",
        baseline=1500,
        sign_conv=-1,
        atten_unit="Np/(Hz·m)",
    )

    # 4) Per-frequency CG, SoS then attenuation
    cg = CG(c1=1e-4, shrink=0.5, max_ls=20)
    for k in range(Nf):
        print(f"\n=== freq {k}/{Nf - 1}: {freqs[k] / 1e6:.3f} MHz ===")
        op = HelmholtzOperator(
            acq,
            k,
            sign_conv=-1,
            pml_alpha=10.0,
            pml_size=9.0e-3,
            use_gpu=use_gpu,
            progress=progress,
        )
        grad = HelmholtzAdjointGrad(
            op,
            deriv_fn=lambda m, o: 8
            * np.pi**2
            * o.get_field("freq") ** 2
            * (o.get_field("PML") / o.get_field("V")),
        )
        fun = NonlinearLS(op, grad_eval=grad)

        ns, na = int(niterSoSPerFreq[k]), int(niterAttenPerFreq[k])
        if ns > 0:
            cg.solve(fun, slow, n_iter=ns, mode="real", viz=viz, do_print_time=True)
        if na > 0:
            cg.solve(fun, slow, n_iter=na, mode="imag", viz=viz, do_print_time=True)

    rec = cg.get_record()

    # 5) Save to .mat
    savemat(
        Path(SAVE_DIR / "kWave_BreastCT_WaveformInversionResults.mat"),
        {
            "xi": grid.xi,
            "yi": grid.yi,
            "fDATA": freqs.reshape(1, -1),
            "niterAttenPerFreq": niterAttenPerFreq.reshape(1, -1),
            "niterSoSPerFreq": niterSoSPerFreq.reshape(1, -1),
            "VEL_ESTIM_ITER": rec["vel"],
            "ATTEN_ESTIM_ITER": rec["atten"],
            "GRAD_IMG_ITER": rec["grad"],
            "SEARCH_DIR_ITER": rec["search"],
        },
        do_compression=True,
    )
    print("[ok] Saved Results/*.mat")


if __name__ == "__main__":
    main()
