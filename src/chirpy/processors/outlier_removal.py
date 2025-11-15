"""
UFWI.processors.outlier_removal
======================================

Zeroes the largest magnitude samples in each (Tx,Rx) slice, restricted to the
*acceptance mask* produced by :class:`~UFWI.processors.acceptance_mask.AcceptanceMask`.
"""

from __future__ import annotations

import numpy as np

from chirpy.processors.base import BaseProcessor
from chirpy.data import AcquisitionData
from chirpy.geometry import GeometryConfigurator


class MagnitudeOutlierFilter(BaseProcessor):
    """
    Zero out the largest-magnitude samples in each (Tx,Rx) slice,
    restricted to the channels allowed by GeometryConfigurator.
    """

    def __init__(self, geom_config: GeometryConfigurator, threshold: float = 0.99) -> None:
        if not (0.0 < threshold <= 1.0):
            raise ValueError("`threshold` must be in (0,1].")
        self._keep = threshold
        self._geom = geom_config

    def __call__(self, data: AcquisitionData) -> AcquisitionData:
        if data.array is None:
            return data

        # Mask is defined in the same (Tx,Rx) space as data.array
        mask = self._geom.get_elem_mask()  # shape (Tx, Rx)

        if mask.shape != data.array.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match data shape {data.array.shape[:2]}."
            )

        n_tx, n_rx, n_f = data.array.shape
        n_total = n_tx * n_rx
        n_drop = int(np.ceil((1.0 - self._keep) * n_total))
        if n_drop == 0:
            return data

        for f in range(n_f):
            slice_abs = np.abs(data.array[:, :, f])  # (Tx, Rx)
            mags = (slice_abs * mask).ravel()

            # skip if all masked entries are zero
            if np.all(mags == 0):
                continue

            # magnitude cutoff for the largest n_drop entries
            thresh = np.partition(mags, -n_drop)[-n_drop]

            drop = (slice_abs >= thresh) & mask
            data.array[:, :, f][drop] = 0.0

        return data
