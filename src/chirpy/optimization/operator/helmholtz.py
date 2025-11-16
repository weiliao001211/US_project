"""
UFWI.optimization.operator.helmholtz
==========================================

Single-frequency operator that wraps an internal
:pyclass:`functions.HelmholtzSolver`.

External code should **only** interact with the Helmholtz solver via the following interfaces:

* ``forward(m)`` - Produces simulated complex Tx x Rx data F(m)
* ``solve(src, adjoint=False)`` - Solves forward or adjoint wave equation for arbitrary source
* ``get_field(key)`` - Read-only access to cache:
  ``WF`` ``VSRC`` ``scaling`` ``obs_data``
  ``PML`` ``V`` ``freq``

Deprecated interfaces ``residual`` / ``backproj`` have been removed;
their functionality is now part of
:pyclass:`NonlinearLS` and :pyclass:`HelmholtzAdjointGrad`.
"""

from __future__ import annotations
import numpy as np
from types import SimpleNamespace
from chirpy.data import AcquisitionData
from chirpy.optimization.operator.base import Operator
from chirpy.optimization.operator.functions.HelmholtzSolver import (
    HelmholtzSolver,
)  # internal only
from chirpy.utils.progress import Progress, ProgressConfig
from chirpy.geometry import GeometryConfigurator


class HelmholtzOperator(Operator):
    """Forward/adjoint Helmholtz operator for one discrete frequency."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        data: AcquisitionData | None = None,
        geom_config: GeometryConfigurator | None = None,
        f_idx: int | None = None,
        *,
        freq: float | None = None,
        sign_conv: int,
        pml_alpha: float,
        pml_size: float,
        use_gpu: bool = False,
        progress: Progress | None = None,
    ):

        if geom_config is None:
            geom = GeometryConfigurator(data.grid, data.tx_array)
        else:
            geom = geom_config

        self._geom = geom

        if freq is not None:
            self._freq = float(freq)
        else:
            if f_idx is None or data is None or not hasattr(data, "freqs"):
                raise ValueError("Either `freq` must be provided, or `data` with `freqs` and `f_idx`.")
            self._freq = float(data.freqs[f_idx])

        self._sign = int(sign_conv)
        self._a0 = float(pml_alpha)
        self._L_PML = float(pml_size)

        # --- geometry / indexing via GeometryConfigurator ------------- #
        # Element indices after TX/RX selection
        tx_keep = self._geom.get_tx_elem_indices()          # (Tx,)
        rx_lin_idx = self._geom.get_rx_lin_idx()            # (Rx,)
        mask = self._geom.get_elem_mask()                   # (Tx, Rx)

        tx_roles = self._geom.get_tx_role_indices()  # role indices for array axis 0
        rx_roles = self._geom.get_rx_role_indices()  # role indices for array axis 1

        # Grid indices for each active transmitter
        self._x_idx, self._y_idx = self._geom.get_tx_grid_indices()

        # Observation data (optional)
        if data is not None and data.array is not None and f_idx is not None:
            rec_f = self._resolve_observed_data(
                data.array, f_idx, tx_roles, rx_roles
            )
        else:
            rec_f = None

        # --- store per-shot metadata ---------------------------------- #
        self._tx_keep = tx_keep
        self._mask = mask  # (Tx, Rx)
        self._gid = rx_lin_idx  # (Rx,)
        self._REC_f = rec_f  # optional observed data

        # --- geometry & indexing (grid coordinates) ------------------- #
        img_grid = self._geom.grid
        self.ny, self.nx = img_grid.shape
        self._xi, self._yi = img_grid.xi, img_grid.yi
        self.n_tx, self.n_rx = mask.shape

        # runtime cache
        self._cache: SimpleNamespace | None = None
        self._atten_phase = False  # imag-stage flag, set by CG_Time

        self.canUseGPU = use_gpu
        self._progress = progress or Progress(ProgressConfig(enabled=False))

    # ------------------------------------------------------------------ #
    # public helpers
    # ------------------------------------------------------------------ #
    @property
    def frequency(self) -> float:
        return self._freq

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._freq = float(value)
        self._cache = None

    def get_field(self, name: str):
        """
        Read-only access to cached tensors/scalars.

        =============== =============================
        key             description
        --------------- -----------------------------
        WF              forward wavefield (ny,nx,Tx)
        VSRC            virtual source basis
        scaling         scaling coefficients to simData
        obs_data        measured data d (Tx,Rx)
        PML, V, freq    solver internals
        =============== =============================
        """
        if self._cache is None:
            raise RuntimeError("forward() must be called first")
        try:
            return getattr(self._cache, name)
        except AttributeError as exc:
            raise KeyError(name) from exc

    def _solve(self, src: np.ndarray, *, adjoint: bool = False):
        """Delegates to the private :pyclass:`HelmholtzSolver`."""
        if self._cache is None:
            raise RuntimeError("forward() must be called first")
        return self._cache.HS.solve(src, adjoint=adjoint)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # forward modelling
    # ------------------------------------------------------------------ #
    def forward(self, m: np.ndarray, kind=None) -> np.ndarray:
        """Return simulated Tx×Rx frequency-domain data F(m)."""
        if self._cache is None or not np.array_equal(m, self._cache.model):
            self._build_cache(m)
        return self._extract_data(self._cache.WF)

    def adjoint(self, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the adjoint Helmholtz equation H^H w = src for arbitrary `src`.

        Parameters
        ----------
        src : np.ndarray
            Shape (ny, nx, K). Arbitrary adjoint source(s).

        Returns
        -------
        ADJ_WF : np.ndarray
            Adjoint wavefield(s), shape (ny, nx, K).
        VSRC   : np.ndarray
            Virtual-source factor applied to ADJ_WF (same shape).
        """
        if self._cache is None:
            raise RuntimeError("forward() must be called first")

        with self._progress.task(
            total=1, desc="Helmholtz adjoint", unit="solve"
        ) as upd:
            ADJ_WF, VSRC = self._solve(src, adjoint=True)
            upd(1)

        # cache for diagnostics if desired
        self._cache.ADJ_WF = ADJ_WF  # type: ignore[attr-defined]
        return ADJ_WF, VSRC

    # ------------------------------------------------------------------ #
    # internal
    # ------------------------------------------------------------------ #
    def _build_cache(self, m: np.ndarray) -> None:
        vel = 1.0 / np.real(m)
        atten = np.sign(self._sign) * np.imag(m) * 2 * np.pi

        # Construct solver; on GPU expose block-LU column progress via callback
        if self.canUseGPU:
            # one tick per x-column (Nx); HelmholtzSolver will call the callback
            with self._progress.task(total=self.nx, desc="Block-LU", unit="col") as upd:
                HS = HelmholtzSolver(
                    self._xi,
                    self._yi,
                    vel,
                    atten,
                    self._freq,
                    self._sign,
                    self._a0,
                    self._L_PML,
                    canUseGPU=True,
                    progress_cb=upd,  # forwarded into _compute_block_lu_gpu
                )
        else:
            HS = HelmholtzSolver(
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

        SRC = np.zeros((self.ny, self.nx, self.n_tx), np.complex128)
        for s, (ix, iy) in enumerate(zip(self._x_idx, self._y_idx)):
            SRC[iy, ix, s] = 1.0

        # Forward solve progress (coarse)
        with self._progress.task(
            total=1, desc="Helmholtz forward", unit="solve"
        ) as upd:
            WF, VSRC = HS.solve(SRC, adjoint=False)
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
        out = np.zeros((self.n_tx, self.n_rx), np.complex128)
        for s in range(self.n_tx):
            idx = self._mask[s]
            out[s, idx] = WF[:, :, s].ravel(order="F")[self._gid[idx]]
        return out

    # ------------------------------------------------------------------ #
    # metadata helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_observed_data(
            array: np.ndarray | None,
            f_idx: int,
            tx_idx: np.ndarray,
            rx_idx: np.ndarray,
    ) -> np.ndarray | None:
        if array is None:
            return None

        arr = np.asarray(array)
        if arr.size == 0:
            return None

        if arr.ndim == 3:
            arr = arr[..., f_idx]
        elif arr.ndim != 2:
            raise ValueError("Acquisition array must be 2-D or 3-D")

        if arr.shape[0] <= tx_idx.max() or arr.shape[1] <= rx_idx.max():
            raise ValueError("Tx/Rx indices out of range for acquisition array shape.")

        arr = arr[np.ix_(tx_idx, rx_idx)]
        return arr.astype(np.complex128, copy=False)