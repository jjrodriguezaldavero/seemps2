from __future__ import annotations
import numpy as np
import dataclasses

# Typing
from typing import Optional
from ...typing import Vector, Tensor3, Tensor4
from .tools import Matrix

from ...state import MPS
from ...tools import make_logger
from .black_box import BlackBox
from .cost_function import CostFunction, CostNormP
from .tools import combine_indices, points_to_indices


@dataclasses.dataclass
class CrossStrategy:
    cost_function: CostFunction = CostNormP()
    min_cost: float = 1e-8
    max_half_sweeps: int = 200
    max_bond: int = 1000
    max_time: Optional[float] = None
    max_evals: Optional[int] = None


class CrossInterpolant:
    def __init__(self, black_box: BlackBox, initial_points: Matrix):
        self.black_box = black_box
        self.sites = len(black_box.physical_dimensions)
        self.I_l, self.I_g = points_to_indices(initial_points)
        self.I_s = [np.arange(s).reshape(-1, 1) for s in black_box.physical_dimensions]
        self.mps = MPS([np.ones((1, s, 1)) for s in black_box.physical_dimensions])

    def sample_fiber(self, k: int) -> Tensor3:
        i_l, i_s, i_g = self.I_l[k], self.I_s[k], self.I_g[k]
        mps_indices = combine_indices(i_l, i_s, i_g)
        fiber_shape = (len(i_l), len(i_s), len(i_g))
        return self.black_box[mps_indices].reshape(fiber_shape)

    def sample_superblock(self, k: int) -> Tensor4:
        i_l, i_g = self.I_l[k], self.I_g[k + 1]
        i_s1, i_s2 = self.I_s[k], self.I_s[k + 1]
        mps_indices = combine_indices(i_l, i_s1, i_s2, i_g)
        superblock_shape = (len(i_l), len(i_s1), len(i_s2), len(i_g))
        return self.black_box[mps_indices].reshape(superblock_shape)


@dataclasses.dataclass
class CrossResults:
    mps: MPS
    costs: Vector
    bonds: Matrix
    times: Vector
    evals: Vector


def cross_interpolation(
    black_box: BlackBox,
    cross_strategy: CrossStrategy,
    initial_points: Optional[Matrix] = None,
) -> CrossResults:

    # Avoid circular dependency with local imports
    # TODO: Redesign the module structure to avoid this.
    from .cross_maxvol import cross_maxvol, CrossStrategyMaxvol
    from .cross_dmrg import cross_dmrg, CrossStrategyDMRG
    from .cross_greedy import cross_greedy, CrossStrategyGreedy

    sites = len(black_box.physical_dimensions)
    if initial_points is None:
        initial_points = np.zeros(sites, dtype=int)

    # TODO: Redesign the algorithm structure to avoid having cross_strategy with a state that needs to be reset.
    cross_strategy.cost_function.reset()
    if isinstance(cross_strategy, CrossStrategyMaxvol):
        cross_results = cross_maxvol(black_box, cross_strategy, initial_points)
    elif isinstance(cross_strategy, CrossStrategyDMRG):
        cross_results = cross_dmrg(black_box, cross_strategy, initial_points)
    elif isinstance(cross_strategy, CrossStrategyGreedy):
        cross_results = cross_greedy(black_box, cross_strategy, initial_points)
    else:
        raise ValueError("Invalid cross_strategy")

    return cross_results


def check_convergence(
    half_sweep: int, trajectories: dict, cross_strategy: CrossStrategy
) -> bool:
    maxbond = np.max(trajectories["bonds"][-1])
    maxbond_prev = np.max(trajectories["bonds"][-2]) if half_sweep > 2 else 0
    time = np.sum(trajectories["times"])
    evals = trajectories["evals"][-1]
    with make_logger(2) as logger:
        logger(
            f"Cross half-sweep: {half_sweep:3}/{cross_strategy.max_half_sweeps}, "
            f"cost: {trajectories['costs'][-1]:1.15e}/{cross_strategy.min_cost:.2e}, "
            f"maxbond: {maxbond:3}/{cross_strategy.max_bond}, "
            f"time: {time:8.6f}/{cross_strategy.max_time}, "
            f"evals: {evals:8}/{cross_strategy.max_evals}."
        )

    if trajectories["costs"][-1] <= cross_strategy.min_cost:
        logger(f"State converged within tolerance {cross_strategy.min_cost}")
        return True
    elif maxbond >= cross_strategy.max_bond:
        logger(f"Max. bond reached above the threshold {cross_strategy.max_bond}")
        return True
    elif cross_strategy.max_time is not None and time >= cross_strategy.max_time:
        logger(f"Max. time reached above the threshold {cross_strategy.max_time}")
        return True
    elif cross_strategy.max_evals is not None and evals >= cross_strategy.max_evals:
        logger(f"Max. evals reached above the threshold {cross_strategy.max_evals}")
        return True
    elif maxbond - maxbond_prev <= 0:
        logger(f"Max. bond dimension converged with value {maxbond}")
        return True

    return False
