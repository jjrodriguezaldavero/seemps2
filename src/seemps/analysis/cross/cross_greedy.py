import numpy as np
import dataclasses
from typing import TypeVar, Union
from collections import defaultdict
from time import perf_counter

from ...state import MPS
from ...state._contractions import _contract_last_and_first
from ...typing import Vector
from ...tools import make_logger

from .black_box import BlackBox
from .cross import CrossStrategy, CrossInterpolant, CrossResults, check_convergence
from .tools import Matrix, combine_indices, fiber_to_Q3R, Q3_to_core


@dataclasses.dataclass
class CrossStrategyGreedy(CrossStrategy):
    pivot_tolerance: float = 1e-10
    max_iterations: int = 5
    initial_random_points: int = 10
    rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng()
    )


class CrossInterpolantGreedy(CrossInterpolant):
    def __init__(self, black_box: BlackBox, initial_points: Matrix):
        super().__init__(black_box, initial_points)

        self.list_fibers = [self.sample_fiber(k) for k in range(self.sites)]
        self.list_Q3 = []
        self.list_R = []

        for fiber in self.list_fibers[:-1]:
            Q3, R = fiber_to_Q3R(fiber)
            self.list_Q3.append(Q3)
            self.list_R.append(R)

        ### TODO: This should be refactored.
        # Translate initial multi-indices I_l and I_g to integer indices J_l and J_g.
        def get_row_indices(rows, all_rows):
            large_set = {tuple(row): idx for idx, row in enumerate(all_rows)}
            return np.array([large_set[tuple(row)] for row in rows])

        J_l = []
        J_g = []
        for k in range(self.sites - 1):
            i_l = combine_indices(self.I_l[k], self.I_s[k])
            i_g = combine_indices(self.I_l[k], self.I_s[k])
            J_l.append(get_row_indices(self.I_l[k + 1], i_l))
            J_g.append(get_row_indices(self.I_l[k + 1], i_g))

        self.J_l = [np.array([])] + J_l  # add empty indices to respect convention
        self.J_g = J_g[::-1] + [np.array([])]
        ###

        mps_cores = [Q3_to_core(Q3, j_l) for Q3, j_l in zip(self.list_Q3, self.J_l[1:])]
        self.mps = MPS(mps_cores + [self.list_fibers[-1]])

    _Index = TypeVar("_Index", bound=Union[Vector, slice])

    def sample_superblock_slice(
        self, k: int, j_l: _Index = slice(None), j_g: _Index = slice(None)
    ) -> Matrix:
        i_ls = combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_ls = i_ls.reshape(1, -1) if i_ls.ndim == 1 else i_ls  # Prevent collapse to 1D
        i_sg = combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        i_sg = i_sg.reshape(1, -1) if i_sg.ndim == 1 else i_sg
        mps_indices = combine_indices(i_ls, i_sg)
        return self.black_box[mps_indices].reshape((len(i_ls), len(i_sg)))

    def sample_skeleton_slice(
        self, k: int, j_l: _Index = slice(None), j_g: _Index = slice(None)
    ) -> Matrix:
        r_l, r_s1, chi = self.mps[k].shape
        chi, r_s2, r_g = self.list_fibers[k + 1].shape
        G = self.mps[k].reshape(r_l * r_s1, chi)[j_l]
        R = self.list_fibers[k + 1].reshape(chi, r_s2 * r_g)[:, j_g]
        return _contract_last_and_first(G, R)

    def update_indices(self, k: int, j_l: Vector, j_g: Vector) -> None:
        i_l = combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_g = combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        self.I_l[k + 1] = np.vstack((self.I_l[k + 1], i_l))
        self.J_l[k + 1] = np.append(self.J_l[k + 1], j_l)
        self.I_g[k] = np.vstack((self.I_g[k], i_g))
        self.J_g[k] = np.append(self.J_g[k], j_g)

    def update_tensors(self, k: int, row: Vector, column: Vector) -> None:
        # Update left fiber using the column vector
        fiber_1 = self.list_fibers[k]
        r_l, r_s1, chi = fiber_1.shape
        C = fiber_1.reshape(r_l * r_s1, chi)
        fiber_1_new = np.hstack((C, column.reshape(-1, 1))).reshape(r_l, r_s1, chi + 1)
        self.list_fibers[k] = fiber_1_new

        # Update left Q3, R and MPS core
        self.list_Q3[k], self.list_R[k] = fiber_to_Q3R(fiber_1_new)
        self.mps[k] = Q3_to_core(self.list_Q3[k], self.J_l[k + 1])

        # Update right fiber using the row vector
        fiber_2 = self.list_fibers[k + 1]
        chi, r_s2, r_g = fiber_2.shape
        R = fiber_2.reshape(chi, r_s2 * r_g)
        fiber_2_new = np.vstack((R, row)).reshape(chi + 1, r_s2, r_g)
        self.list_fibers[k + 1] = fiber_2_new

        # Update right Q3, R and MPS core
        if k < self.sites - 2:
            self.list_Q3[k + 1], self.list_R[k + 1] = fiber_to_Q3R(fiber_2_new)
            self.mps[k + 1] = Q3_to_core(self.list_Q3[k + 1], self.J_l[k + 2])
        else:
            self.mps[k + 1] = self.list_fibers[k + 1]


def cross_greedy(
    black_box: BlackBox, cross_strategy: CrossStrategyGreedy, initial_points: Vector
) -> CrossResults:

    interpolant = CrossInterpolantGreedy(black_box, initial_points)

    converged = False
    trajectories = defaultdict(list)
    for i in range(cross_strategy.max_half_sweeps // 2):
        # Left-to-right half sweep
        tick = perf_counter()
        for k in range(interpolant.sites - 1):
            _update_interpolant(interpolant, k, cross_strategy)
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
            _update_interpolant(interpolant, k, cross_strategy)
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
    interp: CrossInterpolantGreedy,
    k: int,
    cross_strategy: CrossStrategyGreedy,
) -> None:
    max_pivots = interp.black_box.physical_dimensions[k] ** (
        1 + min(k, interp.sites - (k + 2))
    )
    if len(interp.I_g[k]) >= max_pivots or len(interp.I_l[k + 1]) >= max_pivots:
        return

    # Choose random indices (lottery)
    j_l_random = cross_strategy.rng.integers(
        low=0,
        high=len(interp.I_l[k]) * len(interp.I_s[k]),
        size=cross_strategy.initial_random_points,
    )
    j_g_random = cross_strategy.rng.integers(
        low=0,
        high=len(interp.I_s[k + 1]) * len(interp.I_g[k + 1]),
        size=cross_strategy.initial_random_points,
    )

    # Choose initial point from the lottery
    A_random = interp.sample_superblock_slice(k, j_l=j_l_random, j_g=j_g_random)
    B_random = interp.sample_skeleton_slice(k, j_l=j_l_random, j_g=j_g_random)

    error_function = lambda A, B: np.abs(A - B)  # TODO: Consider other functions
    diff = error_function(A_random, B_random)
    i, j = np.unravel_index(np.argmax(diff), A_random.shape)
    j_l, j_g = j_l_random[i], j_g_random[j]

    # Start row-column alternating search for maximum error pivots
    for i in range(cross_strategy.max_iterations):

        # Traverse column residual
        c_A = interp.sample_superblock_slice(k, j_g=j_g).reshape(-1)
        c_B = interp.sample_skeleton_slice(k, j_g=j_g)
        new_j_l = np.argmax(error_function(c_A, c_B))
        if new_j_l == j_l and i > 0:
            break
        j_l = new_j_l

        # Traverse row residual
        r_A = interp.sample_superblock_slice(k, j_l=j_l).reshape(-1)
        r_B = interp.sample_skeleton_slice(k, j_l=j_l)
        new_j_g = np.argmax(error_function(r_A, r_B))
        if new_j_g == j_g:
            break
        j_g = new_j_g

    # Add pivot if its error is large enough
    pivot_error = error_function(c_A[j_l], c_B[j_l])
    if pivot_error >= cross_strategy.pivot_tolerance:
        interp.update_indices(k, j_l=j_l, j_g=j_g)
        interp.update_tensors(k, row=r_A, column=c_A)
