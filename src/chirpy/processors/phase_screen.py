from __future__ import annotations

import numpy as np

from chirpy.processors.base import BaseProcessor
from chirpy.data import AcquisitionData
from chirpy.geometry.image_grid_2D import ImageGrid2D
from chirpy.geometry.transducer_array_2D import TransducerArray2D
from chirpy.geometry import GeometryConfigurator


class PhaseScreenCorrection(BaseProcessor):
    """
    Apply phase-screen correction to acquisition data.

    The correction compensates the geometric phase error introduced by 
    snapping transducer element positions to the reconstruction grid.

    For transmitter s and receiver r at frequency f:

        PS_sr(f) = exp( j * sign * 2*pi*f * (TOF_disc_sr - TOF_true_sr) )

    where:
        TOF_true   = true physical time-of-flight using continuous coordinates.
        TOF_disc   = time-of-flight after snapping positions to the nearest grid node.

    Parameters
    ----------
    geom_config : GeometryConfigurator
        Provides snapped grid indices and true element coordinates.
    sign : {+1, -1}
        Sign convention (default -1 → e^{-j ω t}).
    c0 : float, optional
        Background sound speed. Defaults to data.c0 if not given.
    """

    def __init__(
        self,
        geom_config: GeometryConfigurator,
        sign: int = -1,
        c0: float | None = None,
    ) -> None:

        if sign not in (-1, 1):
            raise ValueError("`sign` must be ±1.")
        if not isinstance(geom_config, GeometryConfigurator):
            raise TypeError("`geom_config` must be a GeometryConfigurator instance")

        self.sign = sign
        self.geom = geom_config
        self.c0 = None if c0 is None else float(c0)

    # ------------------------------------------------------------------ #
    def __call__(self, data: AcquisitionData) -> None:
        """
        Apply the phase-screen correction in place to `data.array`.

        Expected array shape: (Tx, Rx, F)
        """

        if data.array is None:
            raise ValueError("AcquisitionData.array must be defined")

        # --- gather frequencies and sound speed ---
        freqs = data.freqs
        if freqs is None:
            raise ValueError("AcquisitionData.freqs must be defined")

        c0 = self.c0 if self.c0 is not None else data.c0

        # --- true element coordinates (continuous) ---
        x_true, y_true = self.geom.tx_array.positions  # (Nelem,)

        # --- snapped coordinates from GeometryConfigurator ---
        ix = self.geom.elem_x_idx  # snapped x indices (Nelem,)
        iy = self.geom.elem_y_idx  # snapped y indices (Nelem,)

        xi = self.geom.grid.xi
        yi = self.geom.grid.yi

        x_disc = xi[ix]
        y_disc = yi[iy]

        # --- compute TOFs ---
        dx_disc = x_disc[:, None] - x_disc[None, :]
        dy_disc = y_disc[:, None] - y_disc[None, :]
        tof_disc = np.sqrt(dx_disc**2 + dy_disc**2) / c0  # (Tx,Rx)

        dx_true = x_true[:, None] - x_true[None, :]
        dy_true = y_true[:, None] - y_true[None, :]
        tof_true = np.sqrt(dx_true**2 + dy_true**2) / c0

        delta_tof = tof_disc - tof_true  # (Tx,Rx)

        # --- assemble phase-screen tensor ---
        phase = np.exp(
            1j
            * self.sign
            * 2
            * np.pi
            * delta_tof[:, :, None]  # broadcast (Tx,Rx,F)
            * freqs[None, None, :]
        ).astype(np.complex128)

        # --- apply in-place ---
        data.array *= phase

        return data