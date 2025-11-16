from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np

# Progress bar (works in terminals & notebooks)
from tqdm.auto import tqdm

from chirpy.geometry import ImageGrid2D, TransducerArray2D, GeometryConfigurator
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

from chirpy.optimization.operator.helmholtz import HelmholtzOperator

from scipy.io import savemat

