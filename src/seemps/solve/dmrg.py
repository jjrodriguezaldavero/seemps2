from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Callable
import numpy as np
import scipy.sparse.linalg
from ..tools import make_logger
from ..typing import Tensor4
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, Strategy, scprod
from ..state.simplification import AntilinearForm
from ..cython import _contract_last_and_first
from ..operators import MPO
from ..operators.quadratic import QuadraticForm


SolverFn = Callable[..., tuple[np.ndarray, int]]

_SOLVERS: dict[str, SolverFn] = {
    "cg": scipy.sparse.linalg.cg,
    "bicg": scipy.sparse.linalg.bicg,
    "gmres": scipy.sparse.linalg.gmres,
    "lgmres": scipy.sparse.linalg.lgmres,
    "bicgstab": scipy.sparse.linalg.bicgstab,
}


@dataclass
class _LocalSolveStats:
    solves: int = 0
    iterations: int = 0
    matvecs: int = 0


def _relative_change(a: MPS, b: MPS) -> float:
    a_norm_sq = a.norm_squared()
    b_norm_sq = b.norm_squared()
    if a_norm_sq == 0:
        if b_norm_sq == 0:
            return 0.0
        else:
            return np.inf

    d_sq = a_norm_sq - 2.0 * scprod(a, b).real + b_norm_sq
    return sqrt(max(d_sq, 0.0) / a_norm_sq)


def _counted_operator(
    Op: scipy.sparse.linalg.LinearOperator, stats: _LocalSolveStats
) -> scipy.sparse.linalg.LinearOperator:
    def _matvec(v: np.ndarray) -> np.ndarray:
        stats.matvecs += 1
        return Op.matvec(v)

    def _rmatvec(v: np.ndarray) -> np.ndarray:
        stats.matvecs += 1
        return Op.rmatvec(v)

    return scipy.sparse.linalg.LinearOperator(
        shape=Op.shape,
        dtype=Op.dtype,
        matvec=_matvec,
        rmatvec=_rmatvec,
    )


def _solve_local(
    QF: QuadraticForm,
    i: int,
    b_tensor: Tensor4,
    atol: float,
    rtol: float,
    maxiter: int | None,
    solver: SolverFn,
    check_residual: bool = False,
    stats: _LocalSolveStats | None = None,
) -> Tensor4:
    """Solve the local two-site linear system at bond `i`."""
    Op = QF.two_site_operator(i)
    if stats is not None:
        stats.solves += 1
        Op = _counted_operator(Op, stats)
    v = _contract_last_and_first(QF.state[i], QF.state[i + 1])
    b_flat = b_tensor.reshape(-1)
    x0 = v.reshape(-1)
    if check_residual:
        residual_norm = float(np.linalg.norm(Op @ x0 - b_flat))
        tol = max(atol, rtol * float(np.linalg.norm(b_flat)))
        if residual_norm <= tol:
            return v

    def _callback(_xk: np.ndarray) -> None:
        if stats is not None:
            stats.iterations += 1

    callback = _callback if stats is not None else None
    x, _ = solver(
        Op,
        b_flat,
        x0,
        atol=atol,
        rtol=rtol,
        maxiter=maxiter,
        callback=callback,
    )
    return x.reshape(v.shape)


def _sweep(
    QF: QuadraticForm,
    LF: AntilinearForm,
    direction: int,
    atol: float,
    rtol: float,
    maxiter: int | None,
    solver: SolverFn,
    strategy: Strategy,
    check_residual: bool = False,
    stats: _LocalSolveStats | None = None,
) -> None:
    """One full two-site sweep updating `QF` and `LF` in place."""
    size = QF.state.size
    sites = range(size - 1) if direction > 0 else range(size - 2, -1, -1)
    for i in sites:
        AB = _solve_local(
            QF,
            i,
            LF.tensor2site(direction),
            atol,
            rtol,
            maxiter,
            solver,
            check_residual,
            stats,
        )
        if direction > 0:
            QF.update_2site_right(AB, i, strategy)
            LF.update_right()
        else:
            QF.update_2site_left(AB, i, strategy)
            LF.update_left()


def dmrg_solve(
    A: MPO,
    b: MPS,
    guess: MPS | None = None,
    maxiter: int = 20,
    atol: float = 0,
    rtol: float = 1e-5,
    strategy: Strategy = DEFAULT_STRATEGY,
    method: str = "bicgstab",
    compute_residuals: bool = True,
    local_atol: float | None = None,
    local_rtol: float | None = None,
    local_maxiter: int | None = None,
    check_local_residual: bool = False,
) -> tuple[MPS, float | None]:
    r"""Solve :math:`A x = b` for an MPO `A` and an MPS `b` using two-site DMRG.

    Parameters
    ----------
    A : MPO
        Linear operator on the left-hand side.
    b : MPS
        Right-hand side.
    guess : MPS | None, default = None
        Initial guess (defaults to `b`).
    maxiter : int, default = 20
        Maximum number of sweeps.
    atol, rtol : float
        Convergence tolerance. With ``compute_residuals=True``, convergence
        requires ``norm(A@x - b) <= max(rtol * norm(b), atol)``. With
        ``compute_residuals=False``, the relative change of the solution
        between sweeps is used instead. Defaults are `rtol=1e-5` and `atol=0`.
        These are outer sweep tolerances; local solver tolerances can be
        controlled separately with ``local_atol`` and ``local_rtol``.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for the bond dimension.
    method : str, deafult = 'bicgstab'
        Iterative solver: ``'cg'``, ``'bicg'``, ``'bicgstab'``, ``'gmres'``,
        or ``'lgmres'``.
    compute_residuals : bool, default = True
        If True, check the full residual at each sweep for convergence (robust).
        If False, use a cheaper relative-change criterion (faster).
    local_atol, local_rtol : float | None, default = None
        Absolute and relative tolerances passed to the local two-site iterative
        solver. When omitted, they default to ``atol`` and ``rtol`` respectively.
    local_maxiter : int | None, default = None
        Maximum number of iterations for each local two-site solve. ``None`` uses
        SciPy's default for the selected solver.
    check_local_residual : bool, default = False
        If True, compute the initial local residual before calling SciPy and skip
        the local solve when it is already below the local tolerance. The default
        avoids an extra effective-operator matvec because SciPy computes this
        residual internally.

    Returns
    -------
    MPS
        The solution `x`.
    float | None
        Residual :math:``\Vert A x - b\Vert_2``, or None when ``compute_residuals=False``.
    """
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if local_maxiter is not None and local_maxiter < 1:
        raise ValueError("local_maxiter must be positive or None")
    if local_atol is None:
        local_atol = atol
    if local_rtol is None:
        local_rtol = rtol
    if guess is None:
        guess = b.copy()
    solver = _SOLVERS.get(method)
    if solver is None:
        raise ValueError(f'Unknown solver "{method}"')

    b_norm = b.norm()
    tol = max(atol, rtol * b_norm)
    strat = strategy.replace(normalize=False)
    logger = make_logger()
    logger_message = (
        f"DMRG solver initiated with maxiter={maxiter}, tolerance={tol}, "
        + f"local_atol={local_atol}, local_rtol={local_rtol}, "
        + f"local_maxiter={local_maxiter}"
    )
    logger(logger_message)
    local_stats = _LocalSolveStats() if logger else None

    if not isinstance(guess, CanonicalMPS) or guess.center not in (0, A.size - 1):
        guess = CanonicalMPS(guess, center=0, strategy=strat)
    if guess.center == 0:
        direction = +1
        QF = QuadraticForm(A, guess, start=0)
        LF = AntilinearForm(guess, b, center=0)
    else:
        direction = -1
        QF = QuadraticForm(A, guess, start=A.size - 2)
        LF = AntilinearForm(guess, b, center=A.size - 1)

    residual: float | None = np.inf
    change_tol = max(rtol, atol / b_norm) if b_norm > 0 else rtol
    if compute_residuals:
        residual = (A @ QF.state - b).norm()
        logger(f"initial residual={residual}")
        if residual <= tol:
            logger(f"Converged below tolerance {tol}")
            logger.close()
            return QF.state, residual

    psi_old: MPS | None = None
    for sweep in range(1, maxiter + 1):
        if not compute_residuals:
            psi_old = QF.state.copy()
        sweep_start_solves = local_stats.solves if local_stats else 0
        sweep_start_iterations = local_stats.iterations if local_stats else 0
        sweep_start_matvecs = local_stats.matvecs if local_stats else 0
        _sweep(
            QF,
            LF,
            direction,
            local_atol,
            local_rtol,
            local_maxiter,
            solver,
            strat,
            check_local_residual,
            local_stats,
        )
        direction = -direction

        converged = False
        if compute_residuals:
            residual = (A @ QF.state - b).norm()
            logger(f"sweep={sweep}, residual={residual}")
            if residual <= tol:
                logger(f"Converged below tolerance {tol}")
                converged = True
        else:
            assert psi_old is not None  # To make pyright happy
            change = _relative_change(QF.state, psi_old)
            logger(f"sweep={sweep}, relative_change={change}")
            if change <= change_tol:
                logger(f"Converged below tolerance {change_tol}")
                converged = True
        if local_stats is not None:
            logger_message = (
                f"sweep={sweep}, local_solves="
                + f"{local_stats.solves - sweep_start_solves}, "
                + "local_iterations="
                + f"{local_stats.iterations - sweep_start_iterations}, "
                + f"local_matvecs={local_stats.matvecs - sweep_start_matvecs}"
            )
            logger(logger_message)
        if converged:
            break

    logger.close()
    if not compute_residuals:
        residual = None

    return QF.state, residual
