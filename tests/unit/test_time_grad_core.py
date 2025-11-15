import numpy as np
from chirpy.optimization.gradient.time_grad import AdjointStateGrad


class FakeTDOp:
    def __init__(self, nt=16, ny=6, nx=5, n_tx=3, dt=1e-6):
        self.dt = dt
        self.n_tx = n_tx
        self.nt = nt
        self._cache = None
        self.model_c = np.full((ny, nx), 1500.0, np.float32)
        self.model_a = np.zeros((ny, nx), np.float32)
        self.use_encoding = True
        self.tau_step = 0
        # forward fields: (Tx, nt, ny, nx) filled deterministically
        t = np.linspace(0, 1, nt)[:, None, None]
        base = np.ones((ny, nx))
        self._WF = np.stack(
            [np.sin(2 * np.pi * (k + 1) * t) * base for k in range(n_tx)], axis=0
        )
        self._obs = np.zeros((1, nx * ny, nt))  # k-Wave order placeholder

    def forward(self, m, kind="c"):
        self._cache = type("C", (), {"WF": self._WF})
        # element order (Tx, n_rx, nt) but we just return (Tx,1,nt) for simplicity
        return np.zeros((self.n_tx, 1, self.nt))

    def get_field(self, key):
        if key == "obs_data":
            return np.zeros((1, 1, self.nt))
        if key == "WF":
            return self._WF
        raise KeyError

    def get_forward_fields(self):
        return self._WF

    def adjoint(self, residual):  # residual shape (1,1,nt) or (Tx,1,nt)
        # return (nt, ny, nx): use first Tx field reversed as a dummy adjoint
        return self._WF[0][::-1]

    def renew_encoded_obs(self):
        pass


def test_k_loop_averages():
    op = FakeTDOp()
    ge = AdjointStateGrad(op, K=5, use_first_deriv_product=True)
    m = op.model_c.copy().astype(np.float64)
    g = ge.evaluate(m, None, kind="c")
    # Single-realization result should match the averaged multi-encoding result
    ge2 = AdjointStateGrad(op, K=1, use_first_deriv_product=True)
    g1 = ge2.evaluate(m, None, kind="c")
    # Allow zeros (fake op) but shape must be identical and averaging should match
    assert g.shape == g1.shape
    np.testing.assert_allclose(g, 1 * g1)
