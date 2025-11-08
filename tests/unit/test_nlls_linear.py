import numpy as np
from chirpy.optimization.function import NonlinearLS
from chirpy.optimization.gradient.base import GradientEvaluator
from chirpy.optimization.operator.base import Operator


class LinearOp(Operator):
    def __init__(self, A, d):
        self.A, self._d = A, d

    def forward(self, m, kind=None):
        return self.A @ m.ravel()  # (K,)

    def get_field(self, k):
        if k == "obs_data":
            return self._d
        raise KeyError


class LinearGrad(GradientEvaluator):
    def __init__(self, op):
        super().__init__(None)
        self.op = op

    def evaluate(self, m, q, kind=None):
        # q = w·(A m - d)
        return (self.op.A.T @ q).reshape(m.shape)


def test_linear_nlls_gradient():
    ny, nx = 4, 5
    K = 7
    A = np.random.randn(K, ny * nx)
    m = np.random.randn(ny, nx)
    d = np.random.randn(K)
    op = LinearOp(A, d)
    ge = LinearGrad(op)
    fun = NonlinearLS(op, grad_eval=ge, weight=2.0)  # scalar weight
    r = op.forward(m) - d
    g = fun.gradient(m)
    np.testing.assert_allclose(g, (A.T @ (2.0 * r)).reshape(ny, nx))
