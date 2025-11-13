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


class HelmholtzOperator(Operator):
    """Forward/adjoint Helmholtz operator for one discrete frequency."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        data: AcquisitionData,
        f_idx: int | None = None,
        *,
        freq: float | None = None,
        sign_conv: int,
        pml_alpha: float,
        pml_size: float,
        use_gpu: bool = False,
        progress: Progress | None = None,
    ):

        if freq is not None:
            self._freq = float(freq)
        else:
            if f_idx is None:
                raise ValueError("Either `f_idx` or `freq` must be provided.")
            try:
                self._freq = float(data.freqs[f_idx])
            except AttributeError:
                raise ValueError(
                    "AcquisitionData has no `freqs`; please pass `freq=` explicitly."
                )

        self._sign = int(sign_conv)
        self._a0 = float(pml_alpha)
        self._L_PML = float(pml_size)

        # --- gather geometry / indexing metadata ---------------------- #
        tx_array = getattr(data, "tx_array", None)
        grid = getattr(data, "grid", None)

        x_idx_full, y_idx_full, lin_idx_full = self._resolve_element_indices(
            data, tx_array, grid
        )

        tx_keep = self._resolve_tx_keep(data, tx_array)
        mask, rx_lin_idx = self._resolve_mask_and_gid(
            data, tx_array, tx_keep, lin_idx_full
        )

        if data.array is not None and f_idx is not None:
            rec_f = self._resolve_observed_data(data.array, f_idx, tx_keep, mask.shape[1])
        else:
            rec_f = None

        # --- store per-shot metadata ---------------------------------- #
        self._tx_keep = tx_keep
        self._mask = mask  # (Tx, Rx)
        self._REC_f = rec_f  # optional observed data

        # --- geometry & indexing -------------------------------------- #
        self._x_idx = x_idx_full[tx_keep]
        self._y_idx = y_idx_full[tx_keep]
        self._gid = rx_lin_idx  # (Rx,)

        img_grid = data.grid
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
    def _resolve_element_indices(
        self, data, tx_array, grid
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        computed = None
        if tx_array is not None and grid is not None:
            attached = tx_array.attach_to_grid(grid)
            coords = np.asarray(attached.positions, float)
            if coords.ndim != 2:
                raise ValueError("transducer positions must be 2-D")
            n_elem = coords.shape[1]
            x_idx = np.empty(n_elem, dtype=np.int64)
            y_idx = np.empty(n_elem, dtype=np.int64)
            for i, (x, y) in enumerate(coords.T):
                ix, iy = grid.coord2index(float(x), float(y))
                x_idx[i] = ix
                y_idx[i] = iy
            gid = np.ravel_multi_index(
                (y_idx, x_idx), (grid.ny, grid.nx), order="F"
            ).astype(np.int64)
            computed = (x_idx, y_idx, gid)

        def _coerce(values, fallback_idx):
            if values is None:
                if computed is None:
                    raise ValueError(
                        "HelmholtzOperator requires geometry metadata (x_idx/y_idx/grid indices)"
                    )
                return computed[fallback_idx]
            arr = np.asarray(values)
            if arr.ndim != 1:
                raise ValueError("Geometry metadata arrays must be 1-D")
            return arr.astype(np.int64, copy=False)

        x_idx_full = _coerce(data.ctx.get("x_idx"), 0)
        y_idx_full = _coerce(data.ctx.get("y_idx"), 1)
        lin_idx_full = _coerce(data.ctx.get("grid_lin_idx"), 2)
        return x_idx_full, y_idx_full, lin_idx_full

    @staticmethod
    def _resolve_tx_keep(data, tx_array) -> np.ndarray:
        tx_keep = data.ctx.get("tx_keep")
        if tx_keep is not None:
            tx_keep = np.asarray(tx_keep, dtype=np.int64)
        else:
            if tx_array is not None:
                if hasattr(tx_array, "is_tx") and np.any(tx_array.is_tx):
                    tx_keep = np.nonzero(tx_array.is_tx)[0].astype(np.int64)
                else:
                    raise ValueError(
                        "No active transmitters found in tx_array. "
                        "Please define at least one transmitter (is_tx=True)."
                    )
            else:
                raise ValueError(
                    "tx_array is None — cannot resolve transmitters. "
                    "Please provide a valid TransducerArray2D with at least one transmitter."
                )

        if tx_keep.ndim != 1 or tx_keep.size == 0:
            raise ValueError(
                "tx_keep must be a non-empty 1-D array of transmitter indices."
            )
        return tx_keep

    def _resolve_mask_and_gid(
        self,
        data,
        tx_array,
        tx_keep: np.ndarray,
        lin_idx_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        elem_mask = data.ctx.get("elem_mask")
        if elem_mask is not None:
            elem_mask = np.asarray(elem_mask, dtype=bool)
            if elem_mask.ndim != 2:
                raise ValueError("elem_mask must be 2-D")
            tx_rx_mask = elem_mask[tx_keep]
            lin_idx = np.asarray(lin_idx_full, dtype=np.int64)

            n_rx = tx_rx_mask.shape[1]
            if lin_idx.shape[0] < tx_rx_mask.shape[1]:
                raise ValueError("grid_lin_idx shorter than receiver dimension")

            rx_lin_idx = lin_idx[:n_rx]
            return tx_rx_mask, rx_lin_idx

        if tx_array is None:
            raise ValueError(
                "elem_mask missing; transducer geometry is required to infer defaults"
            )

        rx_indices = np.nonzero(tx_array.is_rx)[0].astype(np.int64)
        if rx_indices.size == 0:
            raise ValueError("Transducer array does not define any receiver elements")

        tx_rx_mask = np.ones((tx_keep.size, rx_indices.size), dtype=bool)
        rx_lookup = {elem_idx: idx for idx, elem_idx in enumerate(rx_indices)}
        for row, elem_idx in enumerate(tx_keep):
            col = rx_lookup.get(int(elem_idx))
            if col is not None:
                tx_rx_mask[row, col] = False

        rx_lin_idx = np.asarray(lin_idx_full, dtype=np.int64)[rx_indices]
        return tx_rx_mask, rx_lin_idx

    @staticmethod
    def _resolve_observed_data(
        array: np.ndarray | None, f_idx: int, tx_keep: np.ndarray, n_rx: int
    ) -> np.ndarray | None:
        n_tx = tx_keep.size
        if array is None:
            return None

        arr = np.asarray(array)
        if arr.size == 0:
            return None

        if arr.ndim == 3:
            arr = arr[..., f_idx]
        elif arr.ndim != 2:
            raise ValueError("Acquisition array must be 2-D or 3-D")

        if arr.shape[1] < n_rx:
            raise ValueError("Acquisition array has fewer receivers than required")
        if arr.shape[1] > n_rx:
            arr = arr[:, :n_rx]

        # Choose transmitter rows
        if arr.shape[0] >= n_tx and tx_keep.size and tx_keep.max() < arr.shape[0]:
            arr = arr[tx_keep]
        elif arr.shape[0] != n_tx:
            if arr.shape[0] < n_tx:
                raise ValueError(
                    "Acquisition array has fewer transmitters than required"
                )
            arr = arr[:n_tx]

        return arr.astype(np.complex128, copy=False)
