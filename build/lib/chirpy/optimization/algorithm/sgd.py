from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np

from chirpy.utils.visualizer_multi_mode import Visualizer
from chirpy.optimization.function.least_squares import NonlinearLS
from chirpy.data.image_data import ImageData
from chirpy.utils.progress import Progress, ProgressConfig

_VEL_MIN, _VEL_MAX = 800.0, 2500.0
_ALPHA_FLOOR, _ALPHA_CEIL = 1.0e-7, 20.0


class SGD:
    """
    Plain SGD (stochastic gradient descent) with optional momentum and
    progressive iterate averaging (activated after the *first* energy increase).
    """

    def __init__(
        self,
        *,
        lr: float = 1e6,
        schedule_fn: Callable[[int, float], float] | None = None,
        momentum: float = 0.0,  # μ in [0,1). 0 -> plain SGD
        avg_power: float = 3.0,  # w(i) = i**avg_power (paper uses 3)
        viz: Optional[Visualizer] = None,
        progress: Progress | None = None,
    ) -> None:
        """
        Parameters
        ----------
        lr : float
            Base/initial learning rate (constant if no schedule_fn is given).
        schedule_fn : Callable[[int, float], float] | None
            Custom schedule: returns step size for iteration k given lr0.
        momentum : float
            Heavy-ball momentum coefficient μ (0 disables momentum).
        avg_power : float
            Exponent for progressive iterate averaging weights, default 3.0.
        viz : Visualizer | None
            Optional visualizer updated each iteration with current estimates, gradient,
            and search direction.
        """
        self._lr0 = float(lr)
        self._schedule = schedule_fn
        self._mu = float(momentum)
        self._avg_p = float(avg_power)
        self._viz = viz
        self._progress = progress or Progress(ProgressConfig(enabled=False))

        # record
        self._rec: Dict[str, List[np.ndarray]] = {
            "vel": [],
            "atten": [],
            "grad": [],
            "search": [],
            "misfit": [],
        }

    @staticmethod
    def _clip_inplace(arr: np.ndarray, *, kind: str) -> None:
        """Clip model to valid range in place."""
        if kind == "c":
            np.clip(arr, _VEL_MIN, _VEL_MAX, out=arr)
        else:
            np.clip(arr, _ALPHA_FLOOR, _ALPHA_CEIL, out=arr)

    def _step_size(self, k: int) -> float:
        """
        Step size for iteration k.
        - If a schedule is provided, use it.
        - Otherwise, use constant lr0 (plain SGD).
        """
        if self._schedule:
            return float(self._schedule(k, self._lr0))
        return self._lr0

    def _one_update(
        self,
        grad_in: np.ndarray,
        m: ImageData,
        fun: NonlinearLS,
        *,
        k_iter: int,
        verbose: bool,
        prev_energy: float,
        kind: str,
        v_buf: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Perform a single SGD update (with optional momentum).

        Returns
        -------
        g_new : np.ndarray
            New gradient at updated model.
        energy_after : float
            Objective energy after the update.
        step_dir : np.ndarray
            The actual *descent* direction used (i.e., -v_buf after update).
        """
        if np.iscomplexobj(grad_in):
            raise TypeError("SGD expects real-valued gradients, got complex.")

        g = grad_in.astype(np.float64, copy=True)
        eta = float(self._step_size(k_iter))

        # heavy-ball momentum buffer update (v = mu v + g)
        if self._mu > 0.0:
            v_buf *= self._mu
            v_buf += g
            step_dir = -v_buf
        else:
            step_dir = -g

        g_inf = float(np.max(np.abs(g)))
        g_l2 = float(np.linalg.norm(g))
        upd_inf = eta * float(np.max(np.abs(-step_dir)))

        # model update
        m.array += eta * step_dir  # step_dir already negative (descent)
        self._clip_inplace(m.array, kind=kind)
        np.nan_to_num(m.array, copy=False)

        # evaluate new gradient / energy
        g_new = fun.gradient(m.array, kind=kind)
        energy_after = float(fun.last_misfit)

        if verbose:
            msg = (
                f"    iter {k_iter:03d} | "
                f"misfit_before={prev_energy:.3e} -> misfit_after={energy_after:.3e} | "
                f"|g|∞={g_inf:.3e}  η={eta:.3e}  η|step|∞={upd_inf:.3e}  |g|₂={g_l2:.3e}"
            )
            if self._mu > 0.0:
                msg += f"  (momentum μ={self._mu:.2f})"
            print(msg)

        # record
        if kind == "c":
            self._rec["vel"].append(m.array.copy())
            self._rec["atten"].append(np.zeros_like(m.array))
        else:
            self._rec["vel"].append(np.zeros_like(m.array))
            self._rec["atten"].append(m.array.copy())
        self._rec["grad"].append(g.copy())
        self._rec["search"].append(step_dir.copy())
        self._rec["misfit"].append([prev_energy, energy_after])

        return g_new, energy_after, step_dir

    def solve(
        self,
        fun: NonlinearLS,
        m0: ImageData,
        *,
        n_iter: int = 50,
        kind: str,
        verbose: bool = True,
    ):
        """
        Run Plain SGD for a fixed number of iterations.

        Parameters
        ----------
        fun : NonlinearLS
            Objective providing `.value` / `.gradient` and `last_misfit`.
        m0 : ImageData
            Initial model; its `.array` is updated **in place**.
        n_iter : int
            Number of iterations.
        kind : {"c","alpha"}
            Parameter to optimize.
        """
        if kind not in ("c", "alpha"):
            raise ValueError("kind must be 'c' or 'alpha'")

        hdr = (
            f"SGD  kind={kind}, η₀={self._lr0:.1e}, "
            f"μ={self._mu:.2f}, averaging w(i)=i^{self._avg_p:g}, momentum={self._mu:.2f}"
        )
        print("=" * len(hdr))
        print(hdr)

        # init energy & gradient
        energy = float(fun.value(m0.array, kind=kind))
        grad = fun.gradient(m0.array, kind=kind)
        if np.iscomplexobj(grad):
            raise TypeError("SGD expects real-valued gradients, got complex")

        if verbose:
            g0_inf = float(np.max(np.abs(grad)))
            print(
                f"[init] |g0|∞={g0_inf:.3e}  η0={self._lr0:.3e}  misfit0={energy:.3e}"
            )

        # momentum buffer (same shape as model)
        v_buf = np.zeros_like(m0.array, dtype=np.float64)

        # progressive iterate averaging (activated after first energy increase)
        avg_active = False
        sum_w = 0.0
        sum_wu: np.ndarray | None = None
        u_hat = m0.array.copy()

        it = self._progress.iter(range(n_iter), total=n_iter, desc="SGD", unit="iter")
        prev_energy = energy
        for k in it:
            # one SGD update
            grad, energy, step_dir = self._one_update(
                grad_in=grad,
                m=m0,
                fun=fun,
                k_iter=k,
                verbose=verbose,
                prev_energy=prev_energy,
                kind=kind,
                v_buf=v_buf,
            )

            # —— iterate averaging logic ——
            # If not active yet, check the *first* increase to activate averaging.
            if (not avg_active) and (energy > prev_energy):
                avg_active = True

            if avg_active:
                w = float((k + 1) ** self._avg_p)  # k is 0-based -> use (k+1)
                if sum_wu is None:
                    sum_wu = w * m0.array.astype(np.float64, copy=True)
                else:
                    sum_wu += w * m0.array
                sum_w += w
                u_hat = (sum_wu / (sum_w + 1e-12)).astype(m0.array.dtype, copy=True)

            # optional visualization (use the *current* estimate; show averaged as title)
            if self._viz:
                if kind == "c":
                    vel_est = m0.array
                    atten_est = np.zeros_like(m0.array)
                else:
                    vel_est = np.zeros_like(m0.array)
                    atten_est = m0.array
                if avg_active:
                    title = (
                        f"misfit: {self._rec['misfit'][-1][0]:.3e} -> "
                        f"{self._rec['misfit'][-1][1]:.3e}"
                    )
                else:
                    title = (
                        f"misfit: {self._rec['misfit'][-1][0]:.3e} -> "
                        f"{self._rec['misfit'][-1][1]:.3e}"
                    )
                self._viz.update(
                    vel_est=vel_est,
                    atten_est=atten_est,
                    grad=grad,  # show current grad
                    search_dir=step_dir,  # actual descent direction
                    title=title,
                )

            # tqdm postfix
            if hasattr(it, "set_postfix"):
                try:
                    it.set_postfix(
                        {
                            "|g|∞": f"{float(np.max(np.abs(grad))):.2e}",
                            "Φ": f"{float(energy):.3e}",
                            "avg": "on" if avg_active else "off",
                        }
                    )
                except Exception:
                    pass

            prev_energy = energy

        # return the averaged estimate if averaging had been activated
        if avg_active:
            m0.array[:] = u_hat

        return m0

    def get_record(self):
        if self._rec["misfit"]:
            return {k: np.stack(v, axis=-1) for k, v in self._rec.items()}
        else:
            return {k: np.empty((0,)) for k in self._rec}

    def reset(self):
        for lst in self._rec.values():
            lst.clear()

    def save_record(self, filename: str) -> None:
        rec = self.get_record()
        np.savez(filename, **rec)
        print(f"Record saved to {filename}")