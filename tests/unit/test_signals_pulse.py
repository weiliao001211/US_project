import numpy as np
from chirpy.signals import GaussianModulatedPulse


def test_gmpulse_length_and_amplitude():
    p = GaussianModulatedPulse(f0=1e5, frac_bw=0.5, amp=0.7)
    y = p.sample(dt=1e-6, nt=256)
    assert y.shape == (256,)
    assert np.isclose(np.max(np.abs(y)), 0.7, rtol=0.2)  # rough due to truncation
