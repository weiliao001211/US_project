"""
Optimized HelmholtzOperator with GPU-aware caching.

Key optimizations:
1. Uses HelmholtzSolver_Optimized with GPU caching
2. Keeps intermediate results on GPU when possible
3. Batch processing of incident/scattered field computations
4. Reduced CPU-GPU transfers
"""

from __future__ import annotations
import numpy as np
from types import SimpleNamespace
from chirpy.data import AcquisitionData
from chirpy.optimization.operator.base import Operator
from chirpy.utils.progress import Progress, ProgressConfig
from chirpy.optimization.operator.functions.HelmholtzSolver_optimized import (
    HelmholtzSolver_Optimized,
)

try:
    import cupy as cp

    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False
    cp = None


class HelmholtzOperator_Optimized(Operator):
    """
    Optimized Forward/adjoint Helmholtz operator for one discrete frequency.

    Optimizations over base HelmholtzOperator:
    - GPU memory caching via HelmholtzSolver_Optimized
    - Batch computation of incident + total fields
    - Keeps results on GPU when possible
    - Minimized CPU-GPU transfers
    """

    def __init__(
        self,
        data: AcquisitionData,
        f_idx: int,
        *,
        sign_conv: int,
        pml_alpha: float,
        pml_size: float,
        use_gpu: bool = False,
        progress: Progress | None = None,
        # Import the optimized solver
        solver_class=None,  # Will be set to HelmholtzSolver_Optimized
    ):
        self._freq = float(data.freqs[f_idx])
        self._sign = int(sign_conv)
        self._a0 = float(pml_alpha)
        self._L_PML = float(pml_size)

        # Geometry & indexing
        self._tx_keep = data.ctx.get("tx_keep", np.arange(data.array.shape[0]))
        self._mask = data.ctx["elem_mask"][self._tx_keep]
        self._REC_f = data.array[self._tx_keep, :, f_idx]

        self._x_idx = data.ctx["x_idx"][self._tx_keep]
        self._y_idx = data.ctx["y_idx"][self._tx_keep]
        self._gid = np.asarray(data.ctx["grid_lin_idx"], np.int64)

        img_grid = data.grid
        self.ny, self.nx = img_grid.shape
        self._xi, self._yi = img_grid.xi, img_grid.yi
        self.n_tx, self.n_rx = self._REC_f.shape

        # Runtime cache
        self._cache: SimpleNamespace | None = None
        self._atten_phase = False

        self.canUseGPU = use_gpu and _GPU_AVAILABLE
        self._progress = progress or Progress(ProgressConfig(enabled=False))

        # Store solver class for later instantiation
        self._solver_class = solver_class

    def get_field(self, name: str):
        """Read-only access to cached tensors/scalars."""
        if self._cache is None:
            raise RuntimeError("No cache available; call forward() first")
        if name == "WF":
            return self._cache.WF
        elif name == "VSRC":
            return self._cache.VSRC
        elif name == "scaling":
            return self._cache.scaling
        elif name == "obs_data":
            return self._cache.obs_data
        elif name == "PML":
            return self._cache.PML
        elif name == "V":
            return self._cache.V
        elif name == "freq":
            return self._cache.freq
        else:
            raise KeyError(f"Unknown field: {name}")

    def forward(self, m: np.ndarray) -> np.ndarray:
        """
        Simulated data F(m) for current slowness model m.
        """
        if self._cache is None or not np.allclose(self._cache.model, m):
            self._build_cache(m)

        return self._extract_data(self._cache.WF)

    def solve(
        self, src: np.ndarray, adjoint: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve Helmholtz equation for arbitrary source.

        Returns
        -------
        WF : np.ndarray
            Wavefield
        VSRC : np.ndarray
            Virtual source
        """
        if self._cache is None:
            raise RuntimeError("Must call forward() first to build cache")

        return self._cache.HS.solve(src, adjoint=adjoint, keep_on_gpu=False)

    def _build_cache(self, m: np.ndarray) -> None:
        """
        Build solver and compute forward wavefields.

        OPTIMIZATION: Uses HelmholtzSolver_Optimized with GPU caching.
        """
        vel = 1.0 / np.real(m)
        atten = np.sign(self._sign) * np.imag(m) * 2 * np.pi

        # Import and instantiate optimized solver
        if self._solver_class is None:
            self._solver_class = HelmholtzSolver_Optimized

        # Construct solver with progress callback
        if self.canUseGPU:
            with self._progress.task(total=self.nx, desc="Block-LU", unit="col") as upd:
                HS = self._solver_class(
                    self._xi,
                    self._yi,
                    vel,
                    atten,
                    self._freq,
                    self._sign,
                    self._a0,
                    self._L_PML,
                    canUseGPU=True,
                    progress_cb=upd,
                )
        else:
            HS = self._solver_class(
                self._xi,
                self._yi,
                vel,
                atten,
                self._freq,
                self._sign,
                self._a0,
                self._L_PML,
                canUseGPU=False,
                progress_cb=None,
            )

        # Build source array
        SRC = np.zeros((self.ny, self.nx, self.n_tx), np.complex128)
        for s, (ix, iy) in enumerate(zip(self._x_idx, self._y_idx)):
            SRC[iy, ix, s] = 1.0

        # Forward solve with progress
        with self._progress.task(
            total=1, desc="Helmholtz forward", unit="solve"
        ) as upd:
            WF, VSRC = HS.solve(SRC, adjoint=False, keep_on_gpu=False)
            upd(1)

        self._cache = SimpleNamespace(
            model=m.copy(),
            HS=HS,
            WF=WF,
            VSRC=VSRC,
            scaling=None,
            obs_data=self._REC_f,
            PML=HS.PML,
            V=HS.V,
            freq=HS.f,
        )

    def _extract_data(self, WF: np.ndarray) -> np.ndarray:
        """Extract receiver data from wavefield."""
        out = np.zeros((self.n_tx, self.n_rx), np.complex128)
        for s in range(self.n_tx):
            idx = self._mask[s]
            out[s, idx] = WF[:, :, s].ravel(order="F")[self._gid[idx]]
        return out

    def compute_incident_and_total_fields(
        self, c_homogeneous: np.ndarray, c_heterogeneous: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        OPTIMIZATION: Batch computation of incident and total fields.

        Computes both incident (homogeneous) and total (heterogeneous) fields
        and returns incident and scattered fields.

        Parameters
        ----------
        c_homogeneous : np.ndarray
            Homogeneous sound speed (for incident field)
        c_heterogeneous : np.ndarray
            Heterogeneous sound speed (for total field)

        Returns
        -------
        incident_fields : np.ndarray
            Incident wavefield (ny, nx, n_tx)
        scattered_fields : np.ndarray
            Scattered wavefield = total - incident (ny, nx, n_tx)
        """
        # Build source array once
        SRC = np.zeros((self.ny, self.nx, self.n_tx), np.complex128)
        for s, (ix, iy) in enumerate(zip(self._x_idx, self._y_idx)):
            SRC[iy, ix, s] = 1.0

        # Compute incident fields (homogeneous medium)
        slow_inc = (1.0 / c_homogeneous).astype(np.complex128)
        vel_inc = 1.0 / np.real(slow_inc)
        atten_inc = np.sign(self._sign) * np.imag(slow_inc) * 2 * np.pi

        # Import optimized solver
        if self._solver_class is None:
            self._solver_class = HelmholtzSolver_Optimized

        if self.canUseGPU:
            with self._progress.task(
                total=self.nx, desc="Block-LU (inc)", unit="col"
            ) as upd:
                HS_inc = self._solver_class(
                    self._xi,
                    self._yi,
                    vel_inc,
                    atten_inc,
                    self._freq,
                    self._sign,
                    self._a0,
                    self._L_PML,
                    canUseGPU=True,
                    progress_cb=upd,
                )
        else:
            HS_inc = self._solver_class(
                self._xi,
                self._yi,
                vel_inc,
                atten_inc,
                self._freq,
                self._sign,
                self._a0,
                self._L_PML,
                canUseGPU=False,
                progress_cb=None,
            )

        # OPTIMIZATION: Keep on GPU if using GPU
        incident_fields, _ = HS_inc.solve(
            SRC, adjoint=False, keep_on_gpu=self.canUseGPU
        )

        # Compute total fields (heterogeneous medium)
        slow_het = (1.0 / c_heterogeneous).astype(np.complex128)
        vel_het = 1.0 / np.real(slow_het)
        atten_het = np.sign(self._sign) * np.imag(slow_het) * 2 * np.pi

        if self.canUseGPU:
            with self._progress.task(
                total=self.nx, desc="Block-LU (het)", unit="col"
            ) as upd:
                HS_het = self._solver_class(
                    self._xi,
                    self._yi,
                    vel_het,
                    atten_het,
                    self._freq,
                    self._sign,
                    self._a0,
                    self._L_PML,
                    canUseGPU=True,
                    progress_cb=upd,
                )
        else:
            HS_het = self._solver_class(
                self._xi,
                self._yi,
                vel_het,
                atten_het,
                self._freq,
                self._sign,
                self._a0,
                self._L_PML,
                canUseGPU=False,
                progress_cb=None,
            )

        total_fields, _ = HS_het.solve(SRC, adjoint=False, keep_on_gpu=self.canUseGPU)

        # OPTIMIZATION: Compute scattered fields on GPU if available
        if self.canUseGPU and isinstance(incident_fields, cp.ndarray):
            scattered_fields = total_fields - incident_fields
            # Transfer to CPU
            incident_fields = cp.asnumpy(incident_fields)
            scattered_fields = cp.asnumpy(scattered_fields)
        else:
            scattered_fields = total_fields - incident_fields

        return incident_fields, scattered_fields
