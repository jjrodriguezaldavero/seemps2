import numpy as np
import scipy.linalg
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict
from time import perf_counter

from ...state import Strategy
from ...state.core import destructively_truncate_vector
from ...truncate import SIMPLIFICATION_STRATEGY
from ...typing import Matrix
from ...tools import make_logger

from .black_box import BlackBox
from .cross import (
    CrossStrategy,
    CrossInterpolant,
    CrossCost,
    CrossResults,
    check_convergence,
    combine_indices,
    maxvol_square,
)


DEFAULT_CROSS_DMRG_STRATEGY = SIMPLIFICATION_STRATEGY.replace(normalize=False)


@dataclass
class CrossStrategyDMRG(CrossStrategy):
    strategy: Strategy = DEFAULT_CROSS_DMRG_STRATEGY
    max_iter_maxvol: int = 10
    tol_maxvol_square: float = 1.05
    """
    Extends the `CrossStrategy` dataclass with parameters specific to the "DMRG" TCI variant.
    
    Parameters
    ----------
    strategy : Strategy, default=DEFAULT_CROSS_DMRG_STRATEGY
        The `strategy` object used for the truncation at each SVD split.
    max_iter_maxvol : int, default=10
        Maximum iterations allowed for the square maxvol decomposition.
        tol_maxvol_square : float, default=1.05
        Tolerance required for the square maxvol decomposition.
    """


def cross_dmrg(
    black_box: BlackBox,
    cross_strategy: CrossStrategyDMRG = CrossStrategyDMRG(),
    initial_points: Optional[Matrix] = None,
) -> CrossResults:
    """
    Computes the MPS representation of the black box object using TCI based on
    two-site optimizations in a DMRG-like fashion.

    Parameters
    ----------
    black_box : BlackBox
        Black box representation of the function to be interpolated.
    cross_strategy : CrossStrategyDMRG, default=CrossStrategyDMRG()
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

        # Left-to-right half sweep
        tick = perf_counter()
        for k in range(interpolant.sites - 1):
            _update_interpolant(interpolant, k, True, cross_strategy)
        time_ltr = perf_counter() - tick

        # Update trajectories
        trajectories["costs"].append(cost_calculator.get_cost(interpolant))
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
        trajectories["costs"].append(cost_calculator.get_cost(interpolant))
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
    cross_strategy: CrossStrategyDMRG,
) -> None:
    """
    Updates the `CrossInterpolant` object performing a two-site DMRG optimization
    at site `k`.
    """
    superblock = interp.sample_superblock(k)
    r_l, s1, s2, r_g = superblock.shape
    A = superblock.reshape(r_l * s1, s2 * r_g)
    ## Non-destructive SVD
    U, S, V = scipy.linalg.svd(A, check_finite=False)
    destructively_truncate_vector(S, cross_strategy.strategy)
    r = S.size
    U, S, V = U[:, :r], np.diag(S), V[:r, :]
    ##
    r = S.shape[0]

    if left_to_right:
        if k < interp.sites - 2:
            C = U.reshape(r_l * s1, r)
            Q, _ = scipy.linalg.qr(C, mode="economic", overwrite_a=True)  # type: ignore
            I, G = maxvol_square(
                Q, cross_strategy.max_iter_maxvol, cross_strategy.tol_maxvol_square  # type: ignore
            )
            interp.I_l[k + 1] = combine_indices(interp.I_l[k], interp.I_s[k])[I]
            interp.mps[k] = G.reshape(r_l, s1, r)
        else:
            interp.mps[k] = U.reshape(r_l, s1, r)
            interp.mps[k + 1] = (S @ V).reshape(r, s2, r_g)

    else:
        if k > 0:
            R = V.reshape(r, s2 * r_g)
            Q, _ = scipy.linalg.qr(R.T, mode="economic", overwrite_a=True)  # type: ignore
            I, G = maxvol_square(
                Q, cross_strategy.max_iter_maxvol, cross_strategy.tol_maxvol_square  # type: ignore
            )
            interp.I_g[k] = combine_indices(interp.I_s[k + 1], interp.I_g[k + 1])[I]
            interp.mps[k + 1] = (G.T).reshape(r, s2, r_g)
        else:
            interp.mps[k] = (U @ S).reshape(r_l, s1, r)
            interp.mps[k + 1] = V.reshape(r, s2, r_g)
