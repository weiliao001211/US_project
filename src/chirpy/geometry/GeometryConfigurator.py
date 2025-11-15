from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from chirpy.geometry.image_grid_2D import ImageGrid2D
from chirpy.geometry.transducer_array_2D import TransducerArray2D


class GeometryConfigurator:
    """
    Helper that centralises TX/RX geometry management for operators.

    Responsibilities
    ----------------
    - Map continuous transducer positions onto the discrete ImageGrid2D.
    - Track the full set of TX / RX elements and the current active subsets.
    - Build element-level TX×RX acceptance masks (optionally using external masks).
    - Provide grid / linear indices and k-Wave-style masks that can be reused
      by both time-domain and frequency-domain operators.
    """

    def __init__(self, grid: ImageGrid2D, tx_array: TransducerArray2D) -> None:
        self.grid = grid
        self.tx_array = tx_array

        # element-level metadata
        self.n_elem: int = 0
        self.elem_x_idx: np.ndarray
        self.elem_y_idx: np.ndarray
        self.elem_lin_idx: np.ndarray

        # full roles (element indices)
        self.tx_all: np.ndarray
        self.rx_all: np.ndarray

        # active subsets (element indices)
        self.tx_keep: np.ndarray
        self.rx_keep: np.ndarray

        # TX×RX acceptance mask (TX along rows, RX along columns)
        self.elem_mask: np.ndarray
        # linear grid indices for active RX
        self.rx_lin_idx: np.ndarray

        # configuration for acceptance
        self.elem_mask_external: Optional[np.ndarray] = None
        self.delta: int = 0

        # element index → role index lookup
        self._tx_all_lookup: np.ndarray
        self._rx_all_lookup: np.ndarray

        # initialise internal state
        self._init_element_indices()
        self._init_base_roles()
        self.tx_keep = self.tx_all.copy()
        self.rx_keep = self.rx_all.copy()
        self._rebuild_acceptance_mask()

    # ------------------------------------------------------------------
    # initialisation helpers
    # ------------------------------------------------------------------
    def _init_element_indices(self) -> None:
        positions = np.asarray(self.tx_array.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[0] != 2:
            raise ValueError("transducer positions must have shape (2, N)")

        self.n_elem = positions.shape[1]
        x_idx = np.empty(self.n_elem, dtype=np.int64)
        y_idx = np.empty(self.n_elem, dtype=np.int64)

        for i, (x, y) in enumerate(positions.T):
            ix, iy = self.grid.coord2index(float(x), float(y))
            x_idx[i] = ix
            y_idx[i] = iy

        lin_idx = np.ravel_multi_index(
            (y_idx, x_idx), (self.grid.ny, self.grid.nx), order="F"
        ).astype(np.int64)

        self.elem_x_idx = x_idx
        self.elem_y_idx = y_idx
        self.elem_lin_idx = lin_idx

    def _init_base_roles(self) -> None:
        tx_all = np.nonzero(self.tx_array.is_tx)[0].astype(np.int64)
        rx_all = np.nonzero(self.tx_array.is_rx)[0].astype(np.int64)

        if tx_all.size == 0:
            raise ValueError("transducer array does not define any transmitters")
        if rx_all.size == 0:
            raise ValueError("transducer array does not define any receivers")

        self.tx_all = tx_all
        self.rx_all = rx_all

        # element index → role index lookup
        self._tx_all_lookup = np.full(self.n_elem, -1, dtype=np.int64)
        self._tx_all_lookup[tx_all] = np.arange(tx_all.size, dtype=np.int64)

        self._rx_all_lookup = np.full(self.n_elem, -1, dtype=np.int64)
        self._rx_all_lookup[rx_all] = np.arange(rx_all.size, dtype=np.int64)

    # ------------------------------------------------------------------
    # acceptance mask (TX×RX)
    # ------------------------------------------------------------------
    def _rebuild_acceptance_mask(self) -> None:
        if self.tx_keep.size == 0:
            raise ValueError("at least one transmitter must remain active")
        if self.rx_keep.size == 0:
            raise ValueError("at least one receiver must remain active")

        # linear indices for active receivers
        self.rx_lin_idx = self.elem_lin_idx[self.rx_keep]

        # external mask path
        if self.elem_mask_external is not None:
            mask_full = np.asarray(self.elem_mask_external, dtype=bool)
            if mask_full.ndim != 2:
                raise ValueError("external elem_mask must be 2-D")
            if mask_full.shape != (self.tx_all.size, self.rx_all.size):
                raise ValueError(
                    "external elem_mask must match (n_tx_all, n_rx_all) dimensions"
                )

            tx_idx = self._tx_all_lookup[self.tx_keep]
            rx_idx = self._rx_all_lookup[self.rx_keep]
            if np.any(tx_idx < 0) or np.any(rx_idx < 0):
                raise ValueError("indices provided to select_tx/select_rx are invalid")

            self.elem_mask = mask_full[np.ix_(tx_idx, rx_idx)]
            return

        # default circular-distance-based mask
        n_tx = self.tx_keep.size
        n_rx = self.rx_keep.size
        mask = np.ones((n_tx, n_rx), dtype=bool)

        if self.delta >= 0:
            total = self.n_elem
            for row, tx_elem in enumerate(self.tx_keep):
                for col, rx_elem in enumerate(self.rx_keep):
                    forward = (rx_elem - tx_elem) % total
                    backward = (tx_elem - rx_elem) % total
                    dist = int(min(forward, backward))
                    if dist <= self.delta:
                        mask[row, col] = False

        self.elem_mask = mask

    # ------------------------------------------------------------------
    # selection APIs
    # ------------------------------------------------------------------
    def select_tx(
        self,
        *,
        step: Optional[int] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        if indices is not None:
            arr = np.asarray(indices, dtype=np.int64).ravel()
            if arr.size == 0:
                raise ValueError("indices must contain at least one transmitter")
            if np.any(arr < 0) or np.any(arr >= self.n_elem):
                raise ValueError("transmitter indices out of range")
            if not np.all(np.isin(arr, self.tx_all)):
                raise ValueError("indices must refer to active transmitters")
            self.tx_keep = arr.astype(np.int64, copy=True)
        elif step is not None:
            step = int(step)
            if step <= 0:
                raise ValueError("step must be a positive integer")
            self.tx_keep = self.tx_all[::step]
        else:
            self.tx_keep = self.tx_all.copy()

        if self.tx_keep.size == 0:
            raise ValueError("no transmitters selected after apply step/indices")

        self._rebuild_acceptance_mask()

    def select_rx(
        self,
        *,
        step: Optional[int] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        if indices is not None:
            arr = np.asarray(indices, dtype=np.int64).ravel()
            if arr.size == 0:
                raise ValueError("indices must contain at least one receiver")
            if np.any(arr < 0) or np.any(arr >= self.n_elem):
                raise ValueError("receiver indices out of range")
            if not np.all(np.isin(arr, self.rx_all)):
                raise ValueError("indices must refer to active receivers")
            self.rx_keep = arr.astype(np.int64, copy=True)
        elif step is not None:
            step = int(step)
            if step <= 0:
                raise ValueError("step must be a positive integer")
            self.rx_keep = self.rx_all[::step]
        else:
            self.rx_keep = self.rx_all.copy()

        if self.rx_keep.size == 0:
            raise ValueError("no receivers selected after applying step/indices")

        self._rebuild_acceptance_mask()

    def configure_acceptance(
        self,
        *,
        delta: int = 0,
        elem_mask_external: Optional[np.ndarray] = None,
    ) -> None:
        """
        Configure TX×RX acceptance.

        Parameters
        ----------
        delta : int
            Circular index radius. If >=0, receivers with circular distance
            <= delta from each TX are masked out (False).
        elem_mask_external : array_like, optional
            Full (n_tx_all, n_rx_all) boolean mask in element index space.
            If provided, overrides delta-based construction.
        """
        self.delta = int(delta)
        self.elem_mask_external = elem_mask_external
        self._rebuild_acceptance_mask()

    # ------------------------------------------------------------------
    # getters
    # ------------------------------------------------------------------
    def get_tx_elem_indices(self) -> np.ndarray:
        return self.tx_keep

    def get_rx_elem_indices(self) -> np.ndarray:
        return self.rx_keep

    def get_tx_grid_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.elem_x_idx[self.tx_keep]
        y = self.elem_y_idx[self.tx_keep]
        return x, y

    def get_rx_grid_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        x = self.elem_x_idx[self.rx_keep]
        y = self.elem_y_idx[self.rx_keep]
        return x, y

    def get_rx_lin_idx(self) -> np.ndarray:
        return self.rx_lin_idx

    def get_elem_mask(self) -> np.ndarray:
        return self.elem_mask

    # ------------------------------------------------------------------
    # grid masks (Fortran-order compatible)
    # ------------------------------------------------------------------
    def build_src_mask_full(self) -> np.ndarray:
        """
        Build a 2-D boolean grid mask with all active transmitters.
        """
        mask = np.zeros((self.grid.ny, self.grid.nx), dtype=bool)
        y_idx = self.elem_y_idx[self.tx_keep]
        x_idx = self.elem_x_idx[self.tx_keep]
        mask[y_idx, x_idx] = True
        return mask

    def build_src_mask_for_tx(self, tx_idx: int) -> np.ndarray:
        """
        Build a 2-D boolean grid mask for a single active transmitter.
        """
        if not (0 <= tx_idx < self.tx_keep.size):
            raise IndexError("tx_idx out of range")
        mask = np.zeros((self.grid.ny, self.grid.nx), dtype=bool)
        elem = self.tx_keep[tx_idx]
        mask[self.elem_y_idx[elem], self.elem_x_idx[elem]] = True
        return mask

    def build_rx_mask_full(self) -> np.ndarray:
        """
        Build a 2-D boolean grid mask with all active receivers.
        """
        mask = np.zeros((self.grid.ny, self.grid.nx), dtype=bool)
        y_idx = self.elem_y_idx[self.rx_keep]
        x_idx = self.elem_x_idx[self.rx_keep]
        mask[y_idx, x_idx] = True
        return mask

    def build_rx_mask_for_tx(self, tx_idx: int) -> np.ndarray:
        """
        Build a 2-D boolean grid mask for receivers accepted for a given TX
        according to the current elem_mask.
        """
        if not (0 <= tx_idx < self.tx_keep.size):
            raise IndexError("tx_idx out of range")

        mask = np.zeros((self.grid.ny, self.grid.nx), dtype=bool)
        allow = self.elem_mask[tx_idx]
        rx_subset = self.rx_keep[allow]
        if rx_subset.size:
            y_idx = self.elem_y_idx[rx_subset]
            x_idx = self.elem_x_idx[rx_subset]
            mask[y_idx, x_idx] = True
        return mask

    def get_tx_role_indices(self) -> np.ndarray:
        """Return Tx role indices (0..n_tx_all-1) for the currently kept Tx elements."""
        return self._tx_all_lookup[self.tx_keep]

    def get_rx_role_indices(self) -> np.ndarray:
        """Return Rx role indices (0..n_rx_all-1) for the currently kept Rx elements."""
        return self._rx_all_lookup[self.rx_keep]