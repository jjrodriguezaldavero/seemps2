import numpy as np
import scipy.linalg
from dataclasses import dataclass
from collections import defaultdict
from time import perf_counter
from typing import Optional

from ...typing import Matrix
from ...tools import make_logger

from .black_box import BlackBox
from .cross import (
    CrossStrategy,
    CrossInterpolant,
    CrossResults,
    CrossCost,
    check_convergence,
    combine_indices,
    maxvol_square,
)


@dataclass
class CrossStrategyMaxvol(CrossStrategy):
    rank_kick: tuple = (0, 1)
    max_iter_maxvol: int = 10
    tol_maxvol_square: float = 1.05
    tol_maxvol_rect: float = 1.05
    """
    Extends the `CrossStrategy` dataclass with parameters specific to the "maxvol" TCI variant.
    
    Parameters
    ----------
    rank_kick : tuple, default=(0, 1)
        Minimum and maximum rank increments allowed for each half-sweep.
    max_iter_maxvol : int, default=10
        Maximum iterations allowed for the square maxvol decomposition.
    tol_maxvol_square : float, default=1.05
        Tolerance required for the square maxvol decomposition.
    tol_maxvol_rect : float, default=1.05
        Maximum iterations allowed for the rectangular maxvol decomposition.
    """


def cross_maxvol(
    black_box: BlackBox,
    cross_strategy: CrossStrategyMaxvol = CrossStrategyMaxvol(),
    initial_points: Optional[Matrix] = None,
) -> CrossResults:
    """
    Computes the MPS representation of the given black box using TCI based on
    one-site optimizations following the rectangular "maxvol" decomposition.

    Parameters
    ----------
    black_box : BlackBox
        Black box representation of the function to be interpolated.
    cross_strategy : CrossStrategyMaxvol = CrossStrategyMaxvol()
        Dataclass containing the parameters of the algorithm.
    initial_points : Optional[Matrix], default=None
        Coordinates of initial discretization points used to initialize the algorithm.
        Defaults to zero coordinates.

    Returns
    -------
    CrossResults
        Dataclass containing the results of the interpolation.
    """
    interpolant = CrossInterpolant(black_box, initial_points)
    cost_calculator = CrossCost(cross_strategy)

    converged = False
    trajectories = defaultdict(list)
    for i in range(cross_strategy.max_half_sweeps // 2):
        tick = perf_counter()

        # Left-to-right half sweep
        for k in range(interpolant.sites):
            _update_interpolant(interpolant, k, True, cross_strategy)

        # Right-to-left half sweep
        for k in reversed(range(interpolant.sites)):
            _update_interpolant(interpolant, k, False, cross_strategy)

        sweep_time = perf_counter() - tick

        # Update trajectories
        trajectories["costs"].append(cost_calculator.get_cost(interpolant))
        trajectories["bonds"].append(interpolant.mps.bond_dimensions())
        trajectories["times"].append(sweep_time)
        trajectories["evals"].append(interpolant.black_box.evals)

        # Evaluate convergence
        if converged := check_convergence(2 * (i + 1), trajectories, cross_strategy):
            break

    if not converged:
        with make_logger(2) as logger:
            logger("Maximum number of iterations reached")

    return CrossResults(
        mps=interpolant.mps,
        costs=np.array(trajectories["costs"]),
        bonds=np.array(trajectories["bonds"]),
        times=np.cumsum(trajectories["times"]),
        evals=np.array(trajectories["evals"]),
    )


def _update_interpolant(
    interpolant: CrossInterpolant,
    k: int,
    left_to_right: bool,
    cross_strategy: CrossStrategyMaxvol,
) -> None:
    """
    Updates the `CrossInterpolant` object performing a one-site "maxvol" optimization
    at site `k`.
    """
    fiber = interpolant.sample_fiber(k)
    r_l, s, r_g = fiber.shape

    if left_to_right:
        C = fiber.reshape(r_l * s, r_g, order="F")
        Q, _ = scipy.linalg.qr(C, mode="economic", overwrite_a=True)  # type: ignore
        I, _ = _choose_maxvol(
            Q,  # type: ignore
            cross_strategy.rank_kick,
            cross_strategy.max_iter_maxvol,
            cross_strategy.tol_maxvol_square,
            cross_strategy.tol_maxvol_rect,
        )
        if k < interpolant.sites - 1:
            interpolant.I_l[k + 1] = combine_indices(
                interpolant.I_l[k], interpolant.I_s[k], fortran=True
            )[I]

    else:
        if k > 0:
            R = fiber.reshape(r_l, s * r_g, order="F")
            Q, _ = scipy.linalg.qr(R.T, mode="economic", overwrite_a=True)  # type: ignore
            I, G = _choose_maxvol(
                Q,  # type: ignore
                cross_strategy.rank_kick,
                cross_strategy.max_iter_maxvol,
                cross_strategy.tol_maxvol_square,
                cross_strategy.tol_maxvol_rect,
            )
            interpolant.mps[k] = (G.T).reshape(-1, s, r_g, order="F")
            interpolant.I_g[k - 1] = combine_indices(
                interpolant.I_s[k], interpolant.I_g[k], fortran=True
            )[I]
        else:
            interpolant.mps[0] = fiber


def _choose_maxvol(
    A: Matrix,
    rank_kick: tuple,
    max_iter: int,
    tol: float,
    tol_rect: float,
) -> tuple[Matrix, Matrix]:
    """Chooses whether to compute the square or rectangular submatrix with maximal volume in `A`."""
    n, r = A.shape
    min_kick, max_kick = rank_kick
    max_kick = min(max_kick, n - r)
    min_kick = min(min_kick, max_kick)
    if n <= r:
        I, B = np.arange(n, dtype=int), np.eye(n)
    elif rank_kick == 0:
        I, B = maxvol_square(A, max_iter, tol)
    else:
        I, B = _maxvol_rectangular(A, (min_kick, max_kick), max_iter, tol, tol_rect)
    return I, B


def _maxvol_rectangular(
    A: Matrix,
    rank_kick: tuple = (0, 1),
    max_iter: int = 10,
    tol: float = 1.05,
    tol_rect: float = 1.05,
) -> tuple[Matrix, Matrix]:
    """
    Computes the rectangular submatrix with maximal volume in `A` by extending the
    square "maxvol" decomposition.
    """
    n, r = A.shape
    min_rank = r + rank_kick[0]
    max_rank = min(r + rank_kick[1], n)
    if min_rank < r or min_rank > max_rank or max_rank > n:
        raise ValueError("Invalid rank_kick")

    I0, B = maxvol_square(A, max_iter, tol)
    I = np.hstack([I0, np.zeros(max_rank - r, dtype=I0.dtype)])
    S = np.ones(n, dtype=int)
    S[I0] = 0
    F = S * np.linalg.norm(B) ** 2

    for k in range(r, max_rank):
        i = np.argmax(F)
        if k >= min_rank and F[i] <= tol_rect**2:
            break
        I[k] = i
        S[i] = 0
        v = B.dot(B[i])
        l = 1.0 / (1 + v[i])
        B = np.hstack([B - l * np.outer(v, B[i]), l * v.reshape(-1, 1)])
        F = S * (F - l * v * v)
    I = I[: B.shape[1]]
    B[I] = np.eye(B.shape[1], dtype=B.dtype)

    return I, B
