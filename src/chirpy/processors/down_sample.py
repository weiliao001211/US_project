from __future__ import annotations

import numpy as np

from chirpy.processors.base import BaseProcessor
from chirpy.data import AcquisitionData
from chirpy.geometry import GeometryConfigurator


class DownSample(BaseProcessor):
    """
    Subsample transmitters by keeping every `step`-th TX element.
    This updates both the GeometryConfigurator and AcquisitionData.

    Behaviour
    ---------
    - Calls `geom_config.select_tx(step=step)` to update active TX elements.
    - Uses `geom_config.get_tx_role_indices()` to slice the TX axis of `data.array`.
    - Replaces NaNs with zeros.

    Parameters
    ----------
    geom_config : GeometryConfigurator
        Shared geometry manager.
    step : int
        Subsampling factor (`1` → no change).
    """

    def __init__(self, geom_config: GeometryConfigurator, step: int = 1) -> None:
        if step < 1:
            raise ValueError("`step` must be >= 1.")
        self._geom = geom_config
        self._step = int(step)

    def __call__(self, data: AcquisitionData) -> AcquisitionData:
        """
        Apply TX downsampling in place.

        Returns
        -------
        AcquisitionData
            The same object with reduced TX dimension.
        """
        # 1) Update geometry: select TX elements in element-index space
        self._geom.select_tx(step=self._step)

        # 2) Get TX role indices (0..n_tx_all-1 → data.array axis 0)
        tx_roles = np.asarray(self._geom.get_tx_role_indices(), dtype=np.int64)

        if tx_roles.ndim != 1 or tx_roles.size == 0:
            raise ValueError("GeometryConfigurator returned empty TX role indices.")
        if data.array.shape[0] <= tx_roles.max():
            raise ValueError("TX role indices exceed acquisition array shape.")

        # 3) Slice acquisition tensor
        data.array = data.array[tx_roles, ...]

        # 4) Clean NaNs
        np.nan_to_num(data.array, copy=False)

        return data