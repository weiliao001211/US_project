from __future__ import annotations

from functools import partial
from typing import NamedTuple, Tuple, Callable

import numpy as np
import jax
import jax.numpy as jnp


# ############################################################################
# 1. Stencil Coefficient Calculation (Port of stencilOptParams.py)
# ############################################################################


@partial(jax.jit, static_argnames=("fixB", "n_theta", "r"))
def stencilOptParams_jax(
    vmin: float,
    vmax: float,
    f: float,
    h: float,
    g: float,
    n_theta: int = 100,
    r: int = 10,
    fixB: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JAX port of stencilOptParams to find optimal 9-point stencil coefficients.

    Args:
        vmin: Minimum wave velocity [m/s]
        vmax: Maximum wave velocity [m/s]
        f: Frequency [Hz]
        h: Grid spacing in X [m]
        g: Ratio of (grid spacing in Y) / (grid spacing in X)
        n_theta: Angular resolution for optimization
        r: Wavenumber resolution for optimization
        fixB: If True, fix b=5/6

    Returns:
        Tuple (b, d, e) of optimal stencil parameters.
    """
    # Work in float64 if jax_enable_x64 is set; otherwise default JAX dtype
    vmin = jnp.asarray(vmin)
    vmax = jnp.asarray(vmax)
    f = jnp.asarray(f)
    h = jnp.asarray(h)
    g = jnp.asarray(g)

    Gmin, Gmax = vmin / (f * h), vmax / (f * h)

    m = jnp.arange(1, n_theta + 1)
    n = jnp.arange(1, r + 1)

    theta = (m - 1) * jnp.pi / (4 * (n_theta - 1))
    G = 1.0 / (1.0 / Gmax + ((n - 1) / (r - 1)) * (1.0 / Gmin - 1.0 / Gmax))

    TH, GG = jnp.meshgrid(theta, G, indexing="ij")

    P = jnp.cos(g * 2.0 * jnp.pi * jnp.cos(TH) / GG)
    Q = jnp.cos(2.0 * jnp.pi * jnp.sin(TH) / GG)

    g2 = g**2
    pi2 = jnp.pi**2

    S1 = (1.0 + 1.0 / g2) * (GG**2) * (1.0 - P - Q + P * Q)
    S2 = pi2 * (2.0 - P - Q)
    S3 = (2.0 * pi2) * (1.0 - P * Q)
    S4 = 2.0 * pi2 + (GG**2) * ((1.0 + 1.0 / g2) * P * Q - P - Q / g2)

    if fixB:
        b = jnp.asarray(5.0 / 6.0, dtype=S1.dtype)
        A = jnp.column_stack((S2.ravel(), S3.ravel()))
        y = S4.ravel() - b * S1.ravel()
        params, _, _, _ = jnp.linalg.lstsq(A, y, rcond=None)
        d, e = params[0], params[1]
    else:
        A = jnp.column_stack((S1.ravel(), S2.ravel(), S3.ravel()))
        y = S4.ravel()
        params, _, _, _ = jnp.linalg.lstsq(A, y, rcond=None)
        b, d, e = params[0], params[1], params[2]

    return b, d, e


# ############################################################################
# 2. Block-Tridiagonal Assembly (JAX version of _populate_sparse_matrix)
# ############################################################################


@jax.jit
def assemble_helmholtz_blocks_jax(
    k_sq: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    b: jnp.ndarray,
    d: jnp.ndarray,
    e: jnp.ndarray,
    h: float,
    g: float,
) -> Tuple[jnp.ndarray, ...]:
    """
    Assembles the 9 block-tridiagonal diagonals directly from stencil coefficients.
    This replaces the need to build a full sparse matrix.

    All inputs are (Ny, Nx) arrays.
    Returns 9 arrays, each of shape (Ny, Nx), representing the diagonals:
        (Dd, Dl, Du, Ld, Ll, Lu, Ud, Ul, Uu)
    """
    Ny, Nx = k_sq.shape
    h2 = h**2
    g2 = g**2

    # --- Helper to pad for safe indexing ---
    def pad(arr: jnp.ndarray) -> jnp.ndarray:
        # Pad with one layer of zeros on all sides
        return jnp.pad(arr, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    # Pad all inputs
    k_sq_p = pad(k_sq)
    A_p = pad(A)
    B_p = pad(B)
    C_p = pad(C)

    # We don't actually need the full index grids, just relative offsets.
    iy = jnp.arange(Ny)
    ix = jnp.arange(Nx)
    iy_p = iy[:, None] + 1
    ix_p = ix[None, :] + 1

    # --- Stencil terms (vectorized) ---
    # Indexing: (iy_p, ix_p) refers to the *center* grid point (y, x)
    k_sq_C = k_sq_p[iy_p, ix_p]
    k_sq_L = k_sq_p[iy_p, ix_p - 1]
    k_sq_R = k_sq_p[iy_p, ix_p + 1]
    k_sq_D = k_sq_p[iy_p - 1, ix_p]
    k_sq_U = k_sq_p[iy_p + 1, ix_p]
    k_sq_DL = k_sq_p[iy_p - 1, ix_p - 1]
    k_sq_DR = k_sq_p[iy_p - 1, ix_p + 1]
    k_sq_UL = k_sq_p[iy_p + 1, ix_p - 1]
    k_sq_UR = k_sq_p[iy_p + 1, ix_p + 1]

    C_C = C_p[iy_p, ix_p]
    C_L = C_p[iy_p, ix_p - 1]
    C_R = C_p[iy_p, ix_p + 1]
    C_D = C_p[iy_p - 1, ix_p]
    C_U = C_p[iy_p + 1, ix_p]
    C_DL = C_p[iy_p - 1, ix_p - 1]
    C_DR = C_p[iy_p - 1, ix_p + 1]
    C_UL = C_p[iy_p + 1, ix_p - 1]
    C_UR = C_p[iy_p + 1, ix_p + 1]

    # --- Diagonal (D) blocks ---
    # Dd (center): (x, y) -> (x, y)
    Dd = (1.0 - d - e) * C_C * k_sq_C - (b / h2) * (
        A_p[iy_p, ix_p]
        + A_p[iy_p, ix_p - 1]
        + B_p[iy_p, ix_p] / g2
        + B_p[iy_p - 1, ix_p] / g2
    )

    # Dl (down): (x, y) -> (x, y-1)
    Dl = (
        (b / h2) * (B_p[iy_p - 1, ix_p] / g2)
        - ((1.0 - b) / (2.0 * h2)) * (A_p[iy_p - 1, ix_p] + A_p[iy_p - 1, ix_p - 1])
        + (d / 4.0) * C_D * k_sq_D
    )

    # Du (up): (x, y) -> (x, y+1)
    Du = (
        (b / h2) * (B_p[iy_p, ix_p] / g2)
        - ((1.0 - b) / (2.0 * h2)) * (A_p[iy_p + 1, ix_p] + A_p[iy_p + 1, ix_p - 1])
        + (d / 4.0) * C_U * k_sq_U
    )

    # --- Lower (L) blocks ---
    # Ld (left): (x, y) -> (x-1, y)
    Ld = (
        (b / h2) * A_p[iy_p, ix_p - 1]
        - ((1.0 - b) / (2.0 * h2))
        * (B_p[iy_p, ix_p - 1] / g2 + B_p[iy_p - 1, ix_p - 1] / g2)
        + (d / 4.0) * C_L * k_sq_L
    )

    # Ll (bottom-left): (x, y) -> (x-1, y-1)
    Ll = ((1.0 - b) / (2.0 * h2)) * (
        A_p[iy_p - 1, ix_p - 1] + B_p[iy_p - 1, ix_p - 1] / g2
    ) + (e / 4.0) * C_DL * k_sq_DL

    # Lu (top-left): (x, y) -> (x-1, y+1)
    Lu = ((1.0 - b) / (2.0 * h2)) * (
        A_p[iy_p + 1, ix_p - 1] + B_p[iy_p, ix_p - 1] / g2
    ) + (e / 4.0) * C_UL * k_sq_UL

    # --- Upper (U) blocks ---
    # Ud (right): (x, y) -> (x+1, y)
    Ud = (
        (b / h2) * A_p[iy_p, ix_p]
        - ((1.0 - b) / (2.0 * h2))
        * (B_p[iy_p, ix_p + 1] / g2 + B_p[iy_p - 1, ix_p + 1] / g2)
        + (d / 4.0) * C_R * k_sq_R
    )

    # Ul (bottom-right): (x, y) -> (x+1, y-1)
    Ul = ((1.0 - b) / (2.0 * h2)) * (
        A_p[iy_p - 1, ix_p] + B_p[iy_p - 1, ix_p + 1] / g2
    ) + (e / 4.0) * C_DR * k_sq_DR

    # Uu (top-right): (x, y) -> (x+1, y+1)
    Uu = ((1.0 - b) / (2.0 * h2)) * (A_p[iy_p + 1, ix_p] + B_p[iy_p, ix_p + 1] / g2) + (
        e / 4.0
    ) * C_UR * k_sq_UR

    # --- Handle boundaries ---
    # Set all boundary coefficients to 0, except Dd which is 1
    mask = jnp.ones((Ny, Nx), dtype=bool)
    mask = mask.at[0, :].set(False)
    mask = mask.at[-1, :].set(False)
    mask = mask.at[:, 0].set(False)
    mask = mask.at[:, -1].set(False)

    Dd = jnp.where(mask, Dd, 1.0)
    Dl = jnp.where(mask, Dl, 0.0)
    Du = jnp.where(mask, Du, 0.0)
    Ld = jnp.where(mask, Ld, 0.0)
    Ll = jnp.where(mask, Ll, 0.0)
    Lu = jnp.where(mask, Lu, 0.0)
    Ud = jnp.where(mask, Ud, 0.0)
    Ul = jnp.where(mask, Ul, 0.0)
    Uu = jnp.where(mask, Uu, 0.0)

    # The original code's L, U blocks are (Ny, Nx-1). We return (Ny, Nx) for all.
    return Dd, Dl, Du, Ld, Ll, Lu, Ud, Ul, Uu


# ############################################################################
# 3. Block LU Decomposition (JAX version of decompBlockLU.py)
# ############################################################################


@partial(jax.jit, static_argnames=("Ny",))
def _blocks_to_dense_jax(
    Ny: int,
    d: jnp.ndarray,
    l: jnp.ndarray,  # noqa
    u: jnp.ndarray,
) -> jnp.ndarray:
    """Helper to build a dense (Ny, Ny) tridiagonal matrix from diagonals."""
    return jnp.diag(d) + jnp.diag(l[1:], k=-1) + jnp.diag(u[:-1], k=1)


class Factors(NamedTuple):
    """Container for the computed LU factors."""

    invT: jnp.ndarray  # Shape (Nx, Ny, Ny)
    L_blocks: jnp.ndarray  # Shape (Nx-1, Ny, Ny)
    U_blocks: jnp.ndarray  # Shape (Nx-1, Ny, Ny)


def _decomp_step(
    T_prev: jnp.ndarray,
    blocks: Tuple[jnp.ndarray, ...],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX scan function for one step of the block LU decomposition.
    T_j = D_j - L_j @ inv(T_j-1) @ U_j-1
    """
    (
        D_curr_d,
        D_curr_l,
        D_curr_u,
        L_curr_d,
        L_curr_l,
        L_curr_u,
        U_prev_d,
        U_prev_l,
        U_prev_u,
    ) = blocks
    Ny = D_curr_d.shape[0]

    # 1. Build dense (Ny, Ny) blocks for this step
    D_curr = _blocks_to_dense_jax(Ny, D_curr_d, D_curr_l, D_curr_u)
    L_curr = _blocks_to_dense_jax(Ny, L_curr_d, L_curr_l, L_curr_u)
    U_prev = _blocks_to_dense_jax(Ny, U_prev_d, U_prev_l, U_prev_u)

    # 2. Compute T_j
    invT_prev = jnp.linalg.inv(T_prev)
    T_curr = D_curr - L_curr @ invT_prev @ U_prev

    # 3. Return new carry (T_curr) and output (invT_prev)
    return T_curr, invT_prev


@partial(jax.jit, static_argnames=("Ny", "Nx"))
def decompBlockLU_jax(
    Ny: int,
    Nx: int,
    Dd: jnp.ndarray,
    Dl: jnp.ndarray,
    Du: jnp.ndarray,
    Ld: jnp.ndarray,
    Ll: jnp.ndarray,
    Lu: jnp.ndarray,
    Ud: jnp.ndarray,
    Ul: jnp.ndarray,
    Uu: jnp.ndarray,
) -> Factors:
    """
    Performs the block LU decomposition using jax.lax.scan.

    Args:
        Ny, Nx: Grid dimensions
        Dd, Dl, ...: 9 diagonal arrays, shape (Ny, Nx)

    Returns:
        Factors: A NamedTuple containing invT, L_blocks, and U_blocks.
    """
    # Transpose all to (Nx, Ny) for easier slicing along x
    Dd_x, Dl_x, Du_x = Dd.T, Dl.T, Du.T
    Ld_x, Ll_x, Lu_x = Ld.T, Ll.T, Lu.T
    Ud_x, Ul_x, Uu_x = Ud.T, Ul.T, Uu.T

    # Inputs for the scan (D_curr, L_curr, U_prev)
    D_curr_diags = (Dd_x[1:], Dl_x[1:], Du_x[1:])
    L_curr_diags = (Ld_x[1:], Ll_x[1:], Lu_x[1:])
    U_prev_diags = (Ud_x[:-1], Ul_x[:-1], Uu_x[:-1])

    scan_inputs = (*D_curr_diags, *L_curr_diags, *U_prev_diags)

    # Initial state (T_0)
    T_0 = _blocks_to_dense_jax(Ny, Dd_x[0], Dl_x[0], Du_x[0])

    # Run the scan
    # T_last is T_{Nx-1}; invT_partial contains [invT_0, ..., invT_{Nx-2}]
    T_last, invT_partial = jax.lax.scan(_decomp_step, T_0, scan_inputs)

    # Compute final invT block (invT_{Nx-1})
    invT_last = jnp.linalg.inv(T_last)

    # Assemble final invT: (Nx, Ny, Ny)
    invT = jnp.concatenate(
        (invT_partial, invT_last[jnp.newaxis, :, :]),
        axis=0,
    )

    # vmap _blocks_to_dense over the x-dimension
    vmap_blocks = jax.vmap(
        partial(_blocks_to_dense_jax, Ny),
        in_axes=(0, 0, 0),
        out_axes=0,
    )

    # L_blocks[j] = L_{j+1}, U_blocks[j] = U_j
    L_blocks = vmap_blocks(Ld_x[1:], Ll_x[1:], Lu_x[1:])
    U_blocks = vmap_blocks(Ud_x[:-1], Ul_x[:-1], Uu_x[:-1])

    return Factors(invT=invT, L_blocks=L_blocks, U_blocks=U_blocks)


# ############################################################################
# 4. Block LU Solver (JAX version of applyBlockLU.py)
# ############################################################################


def _forward_step_jax(
    y_prev: jnp.ndarray,
    inputs: Tuple[jnp.ndarray, ...],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward substitution step: y_j = invT_j @ (s_j - L_{j-1} @ y_{j-1})
    """
    invT_curr, s_curr, L_prev = inputs
    tmp = s_curr - L_prev @ y_prev
    y_curr = invT_curr @ tmp
    return y_curr, y_curr


def _backward_step_jax(
    u_next: jnp.ndarray,
    inputs: Tuple[jnp.ndarray, ...],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Backward substitution step: u_j = y_j - (invT_j @ U_j @ u_{j+1})
    """
    y_curr, invT_curr, U_curr = inputs
    tmp = invT_curr @ (U_curr @ u_next)
    u_curr = y_curr - tmp
    return u_curr, u_curr


def _adj_forward_step_jax(
    y_prev: jnp.ndarray,
    inputs: Tuple[jnp.ndarray, ...],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Adjoint forward step: y_j = s_j - (invT_{j-1} @ U_{j-1})^H @ y_{j-1}
    """
    invT_prev, s_curr, U_prev = inputs
    tmp = (invT_prev.conj().T @ U_prev.conj().T) @ y_prev
    y_curr = s_curr - tmp
    return y_curr, y_curr


def _adj_backward_step_jax(
    u_next: jnp.ndarray,
    inputs: Tuple[jnp.ndarray, ...],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Adjoint backward step: u_j = invT_j^H @ (y_j - L_j^H @ u_{j+1})
    """
    y_curr, invT_curr, L_curr = inputs
    tmp = y_curr - L_curr.conj().T @ u_next
    u_curr = invT_curr.conj().T @ tmp
    return u_curr, u_curr


@partial(jax.jit, static_argnames=("adjoint",))
def _applyBlockLU_jax_single(
    s: jnp.ndarray,
    factors: Factors,
    adjoint: bool,
) -> jnp.ndarray:
    """
    Solves H * u = s (or H^H * u = s) for a single source, given factors.

    Args:
        s: shape (Nx, Ny)
        factors: LU factors from decompBlockLU_jax
        adjoint: If True, solve the adjoint system H^H u = s

    Returns:
        u: solution, shape (Nx, Ny)
    """
    invT, L_blocks, U_blocks = factors.invT, factors.L_blocks, factors.U_blocks

    if adjoint:
        # --- Adjoint Solve: U^H * L^H * u = s ---

        # 1. Adjoint Forward (solve U^H * y = s)
        y_0 = s[0]
        adj_fwd_inputs = (invT[:-1], s[1:], U_blocks)
        _, y_partial = jax.lax.scan(_adj_forward_step_jax, y_0, adj_fwd_inputs)
        y = jnp.concatenate((y_0[jnp.newaxis, :], y_partial), axis=0)

        # 2. Adjoint Backward (solve L^H * u = y)
        u_last = invT[-1].conj().T @ y[-1]
        adj_bwd_inputs = (y[:-1], invT[:-1], L_blocks)
        _, u_partial = jax.lax.scan(
            _adj_backward_step_jax, u_last, adj_bwd_inputs, reverse=True
        )
        u = jnp.concatenate((u_partial, u_last[jnp.newaxis, :]), axis=0)

    else:
        # --- Forward Solve: L * U * u = s ---

        # 1. Forward substitution (solve L * y = s)
        y_0 = invT[0] @ s[0]
        fwd_inputs = (invT[1:], s[1:], L_blocks)
        _, y_partial = jax.lax.scan(_forward_step_jax, y_0, fwd_inputs)
        y = jnp.concatenate((y_0[jnp.newaxis, :], y_partial), axis=0)

        # 2. Backward substitution (solve U * u = y)
        u_last = y[-1]
        bwd_inputs = (y[:-1], invT[:-1], U_blocks)
        _, u_partial = jax.lax.scan(
            _backward_step_jax, u_last, bwd_inputs, reverse=True
        )
        u = jnp.concatenate((u_partial, u_last[jnp.newaxis, :]), axis=0)

    return u


# ############################################################################
# 5. Jitted setup function with dynamic frequency
# ############################################################################


@partial(jax.jit, static_argnames=("Nx", "Ny"))
def setup_helmholtz_jax(
    vel_arr: jnp.ndarray,
    atten_arr: jnp.ndarray,
    f: float,
    sign: float,
    a0: float,
    L_PML: float,
    Nx: int,
    Ny: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    h: float,
    g: float,
) -> Tuple[Factors, jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled setup function that builds PML, stencil, and block-LU factors
    for a given velocity, attenuation, and frequency on a fixed grid.

    Nx, Ny, geometry and grid spacing are static args (compile-time),
    while vel_arr, atten_arr, f, sign, a0, L_PML are dynamic (run-time).
    """
    # --- Complex velocity and wavenumber ---
    SI = atten_arr / (2.0 * jnp.pi)
    V = 1.0 / (1.0 / vel_arr + 1j * SI * jnp.sign(sign))
    k_sq = ((2.0 * jnp.pi * f) / V) ** 2

    # --- PML ---
    xe = jnp.linspace(xmin, xmax, 2 * (Nx - 1) + 1)
    ye = jnp.linspace(ymin, ymax, 2 * (Ny - 1) + 1)
    Xe, Ye = jnp.meshgrid(xe, ye, indexing="xy")

    xctr = 0.5 * (xmin + xmax)
    xspan = 0.5 * (xmax - xmin)
    yctr = 0.5 * (ymin + ymax)
    yspan = 0.5 * (ymax - ymin)

    sx = (
        2.0
        * jnp.pi
        * a0
        * f
        * (jnp.maximum(jnp.abs(Xe - xctr) - xspan + L_PML, 0.0) / L_PML) ** 2
    )
    sy = (
        2.0
        * jnp.pi
        * a0
        * f
        * (jnp.maximum(jnp.abs(Ye - yctr) - yspan + L_PML, 0.0) / L_PML) ** 2
    )

    ex = 1.0 + 1j * sx * jnp.sign(sign) / (2.0 * jnp.pi * f)
    ey = 1.0 + 1j * sy * jnp.sign(sign) / (2.0 * jnp.pi * f)
    bigA, bigB, bigC = ey / ex, ex / ey, ex * ey

    A = bigA[0::2, 1::2]
    B = bigB[1::2, 0::2]
    C = bigC[0::2, 0::2]

    # --- Stencil parameters ---
    b, d, e = stencilOptParams_jax(jnp.min(vel_arr), jnp.max(vel_arr), f, h, g)

    # --- Assemble 9 block diagonals ---
    all_diags = assemble_helmholtz_blocks_jax(k_sq, A, B, C, b, d, e, h, g)

    # --- Compute LU factors ---
    factors = decompBlockLU_jax(Ny, Nx, *all_diags)

    return factors, C, V


# ############################################################################
# 6. JAX Solver Class
# ############################################################################


class HelmholtzSolverJAX:
    """
    JAX-based Helmholtz solver using 9-point stencil and block-LU factorization.

    This class replicates the algorithm from the original HelmholtzSolver,
    but uses JAX for all computations.

    The heavy setup path is jitted with frequency as a dynamic argument, so
    code is compiled once per grid size and reused across frequencies.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        vel: np.ndarray,
        atten: np.ndarray,
        f: float,
        signConvention: int,
        a0: float,
        L_PML: float,
        **kwargs,  # For compatibility, progress_cb is ignored
    ):
        # --- Grid and Parameters ---
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.f = float(f)

        vel = np.asarray(vel)
        atten = np.asarray(atten)
        self.Ny, self.Nx = vel.shape

        # Use float64-friendly computations if enabled globally
        float_dtype = (
            jnp.float64 if getattr(jax.config, "x64_enabled", False) else jnp.float32
        )

        self.h = float(jnp.mean(jnp.diff(self.x.ravel()).astype(float_dtype)))
        self.gh = float(jnp.mean(jnp.diff(self.y.ravel()).astype(float_dtype)))
        self.g = self.gh / self.h

        self.xmin, self.xmax = float(jnp.min(self.x)), float(jnp.max(self.x))
        self.ymin, self.ymax = float(jnp.min(self.y)), float(jnp.max(self.y))

        self._sign = float(signConvention)
        self._a0 = float(a0)
        self._L_PML = float(L_PML)

        print(f"[HelmholtzSolverJAX] Initializing {self.Ny}x{self.Nx} solver...")

        # Coerce medium fields to JAX arrays
        self._vel_j = jnp.asarray(vel, dtype=float_dtype)
        self._atten_j = jnp.asarray(atten, dtype=float_dtype)

        # --- JIT-compiled setup (frequency is dynamic) ---
        self.factors, self.PML, self.V = setup_helmholtz_jax(
            self._vel_j,
            self._atten_j,
            float(self.f),
            self._sign,
            self._a0,
            self._L_PML,
            self.Nx,
            self.Ny,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.h,
            self.g,
        )

        print("[HelmholtzSolverJAX] Factors computed (JIT compiled).")

        # --- Compile vmapped solvers ---
        self._solve_forward, self._solve_adjoint = self._compile_solvers()
        print("[HelmholtzSolverJAX] Solvers JIT-compiled.")

    def refactor_for_frequency(
        self,
        f: float,
        vel: np.ndarray | None = None,
        atten: np.ndarray | None = None,
    ) -> None:
        """
        Optional helper: recompute factors for a new frequency and/or medium,
        reusing the compiled setup_helmholtz_jax for this grid.

        Args:
            f: new frequency [Hz]
            vel: optional new velocity model (Ny, Nx)
            atten: optional new attenuation model (Ny, Nx)
        """
        if vel is not None:
            vel = np.asarray(vel)
            assert vel.shape == (self.Ny, self.Nx)
            float_dtype = self._vel_j.dtype
            self._vel_j = jnp.asarray(vel, dtype=float_dtype)

        if atten is not None:
            atten = np.asarray(atten)
            assert atten.shape == (self.Ny, self.Nx)
            float_dtype = self._atten_j.dtype
            self._atten_j = jnp.asarray(atten, dtype=float_dtype)

        self.f = float(f)

        self.factors, self.PML, self.V = setup_helmholtz_jax(
            self._vel_j,
            self._atten_j,
            float(self.f),
            self._sign,
            self._a0,
            self._L_PML,
            self.Nx,
            self.Ny,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.h,
            self.g,
        )

    def _compile_solvers(self) -> Tuple[Callable, Callable]:
        """
        JIT-compiles and vmaps the forward and adjoint solvers over source index.
        """

        # Fix adjoint flag first so the JIT sees a static boolean
        fwd_func = partial(_applyBlockLU_jax_single, adjoint=False)
        adj_func = partial(_applyBlockLU_jax_single, adjoint=True)

        # vmap over source dimension (K); factors is shared
        solve_fwd_vmap = jax.vmap(fwd_func, in_axes=(0, None), out_axes=0)
        solve_adj_vmap = jax.vmap(adj_func, in_axes=(0, None), out_axes=0)

        return solve_fwd_vmap, solve_adj_vmap

    def solve(
        self,
        src: np.ndarray,
        adjoint: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the Helmholtz equation H*u = src or H^H*u = src.

        Args:
            src: Source array, shape (Ny, Nx, K)
            adjoint: If True, solve the adjoint problem.

        Returns:
            Tuple (wv, virt):
                wv: Wavefield solution, shape (Ny, Nx, K)
                virt: Virtual source, shape (Ny, Nx, K)
        """
        # Ensure src is a JAX array with a consistent complex dtype
        dtype = (
            jnp.complex128
            if getattr(jax.config, "x64_enabled", False)
            else jnp.complex64
        )
        src_jax = jnp.asarray(src, dtype=dtype)

        # Transpose from (Ny, Nx, K) to (K, Nx, Ny) for vmapped solver
        src_vmap = jnp.transpose(src_jax, (2, 1, 0))

        # Select and run the appropriate JIT-compiled solver
        if adjoint:
            u_vmap = self._solve_adjoint(src_vmap, self.factors)
        else:
            u_vmap = self._solve_forward(src_vmap, self.factors)

        # Transpose result from (K, Nx, Ny) back to (Ny, Nx, K)
        wv = jnp.transpose(u_vmap, (2, 1, 0))

        # Compute virtual source
        sf = 8.0 * (jnp.pi**2) * (self.f**2)
        mat = sf * (self.PML / self.V)
        virt = mat[..., jnp.newaxis] * wv

        # Block until computation is complete before converting to numpy
        wv.block_until_ready()

        # Return as numpy arrays (as original class did)
        return np.array(wv), np.array(virt)
