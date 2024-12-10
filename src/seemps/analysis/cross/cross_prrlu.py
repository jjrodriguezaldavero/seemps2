"""
Implementation of tensor cross-interpolation based on partial rank-revealing LU decompositions (prrLU).
"""

import numpy as np
from time import perf_counter
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict

from ...typing import Vector
from ...tools import make_logger

from .black_box import BlackBox
from .cross import CrossStrategy, CrossInterpolant, CrossResults, check_convergence
from .tools import Matrix


@dataclass
class CrossStrategyLU(CrossStrategy):
    prrlu_method: str = "block_rook"
    pivot_update_method: str = "reset"
    error_method: str = "environment"
    pivot_tolerance: Optional[float] = 1e-8
    rook_iterations: int = 3


def cross_prrlu(
    black_box: BlackBox,
    cross_strategy: CrossStrategyLU,
    initial_points: Vector,
) -> CrossResults:

    interpolant = CrossInterpolant(black_box, initial_points)

    converged = False
    trajectories = defaultdict(list)
    for i in range(cross_strategy.max_half_sweeps // 2):

        # Left-to-right half sweep
        tick = perf_counter()
        for k in range(interpolant.sites - 1):
            _update_interpolant(interpolant, k, True, cross_strategy)
        time_ltr = perf_counter() - tick

        # Update trajectories
        trajectories["costs"].append(cross_strategy.cost_function.cost(interpolant))
        trajectories["bonds"].append(interpolant.mps.bond_dimensions())
        trajectories["times"].append(time_ltr)
        trajectories["evals"].append(interpolant.black_box.evals)

        # Evaluate convergence
        if converged := check_convergence(2 * i + 1, trajectories, cross_strategy):
            break

        # Right-to-left half sweep
        tick = perf_counter()
        for k in reversed(range(interpolant.sites - 1)):
            _update_interpolant(interpolant, k, False, cross_strategy)
        time_rtl = perf_counter() - tick

        # Update trajectories
        trajectories["costs"].append(cross_strategy.cost_function.cost(interpolant))
        trajectories["bonds"].append(interpolant.mps.bond_dimensions())
        trajectories["times"].append(time_rtl)
        trajectories["evals"].append(interpolant.black_box.evals)

        # Evaluate convergence
        if converged := check_convergence(2 * i + 2, trajectories, cross_strategy):
            break

    if not converged:
        with make_logger(2) as logger:
            logger("Maximum number of iterations reached")

    return CrossResults(
        mps=interpolant.mps,
        costs=np.array(trajectories["costs"]),
        bonds=np.array(trajectories["bonds"]),
        times=np.array(trajectories["times"]),
        evals=np.array(trajectories["evals"]),
    )


def _update_interpolant(
    interp: CrossInterpolant,
    k: int,
    left_to_right: bool,
    cross_strategy: CrossStrategyLU,
) -> None:
    superblock = interp.sample_superblock(k)
    r_l, s1, s2, r_g = superblock.shape
    A = superblock.reshape(r_l * s1, s2 * r_g)
    L, D, U = choose_prrlu(A, cross_strategy.prrlu_method)
    r = D.shape[0]

    if left_to_right:
        if k < interp.sites - 2:
            pass
        else:
            pass
    else:
        if k > 0:
            pass
        else:
            pass


def choose_prrlu(A: Matrix, method: str) -> tuple[Matrix, Matrix, Matrix]:
    if method == "full":
        return prrlu_full(A)
    elif method == "rook":
        return prrlu_rook(A)
    elif method == "block_rook":
        return prrlu_block_rook(A)
    else:
        raise ValueError("Invalid method")


def prrlu_full(A: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    pass


def prrlu_rook(A: Matrix, iterations: int) -> tuple[Matrix, Matrix, Matrix]:
    pass


def prrlu_block_rook(A: Matrix, iterations: int) -> tuple[Matrix, Matrix, Matrix]:
    pass
