import numpy as np
import scipy.linalg
from dataclasses import dataclass
from collections import defaultdict
from time import perf_counter

from ...typing import Vector
from ...tools import make_logger

from .black_box import BlackBox
from .cross import CrossStrategy, CrossInterpolant, CrossResults, check_convergence
from .tools import choose_maxvol, combine_indices


@dataclass
class CrossStrategyMaxvol(CrossStrategy):
    rank_kick: tuple = (0, 1)
    max_iter_maxvol: int = 10
    tol_maxvol_square: float = 1.05
    tol_maxvol_rect: float = 1.05


def cross_maxvol(
    black_box: BlackBox,
    cross_strategy: CrossStrategyMaxvol,
    initial_points: Vector,
) -> CrossResults:

    interpolant = CrossInterpolant(black_box, initial_points)

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
        trajectories["costs"].append(cross_strategy.cost_function.cost(interpolant))
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
    fiber = interpolant.sample_fiber(k)
    r_l, s, r_g = fiber.shape

    if left_to_right:
        C = fiber.reshape(r_l * s, r_g, order="F")
        Q, _ = scipy.linalg.qr(C, mode="economic", overwrite_a=True)
        I, _ = choose_maxvol(
            Q,
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
            Q, _ = scipy.linalg.qr(R.T, mode="economic", overwrite_a=True)
            I, G = choose_maxvol(
                Q,
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
