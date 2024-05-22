import numpy as np
from typing import TypeVar, Union
from dataclasses import dataclass

from .cross import (
    CrossInterpolation,
    CrossResults,
    CrossStrategy,
    BlackBox,
    _check_convergence,
)
from ...state import MPS
from ...state._contractions import _contract_last_and_first
from ...tools import make_logger


@dataclass
class CrossStrategyGreedy(CrossStrategy):
    greedy_method: str = "full_search"
    greedy_tol: float = 1e-12
    partial_maxiter: int = 5
    partial_points: int = 10
    """
    Dataclass containing the parameters for the maxvol-based TCI.
    The common parameters are documented in the base `CrossStrategy` class.
    Parameters
    ----------
    greedy_method : str, default = "full_search"
        Method used to perform the greedy pivot updates. Options:
        - "full_search": finds the pivot of maximum error in the superblock doing a full search.
        - "partial_search": looks for a pivot that locally maximizes the error by a partial search.
        The partial search uses much less function evaluations (O(chi) instead of O(chi^2)) but
        can have a worse convergence than the full search.
    greedy_tol : float, default = 1e-12
        Tolerance in Frobenius norm between the superblock and the skeleton decomposition
        after which the pivots are no longer added.
    partial_maxiter : int, default = 5
        How many row-column iterations to perform in the pivot partial search.
    partial_points : int, default = 10
        Number of initial random points for each pivot partial search.
    """


class CrossInterpolationGreedy(CrossInterpolation):
    def __init__(self, black_box: BlackBox, initial_point: np.ndarray):
        super().__init__(black_box, initial_point)
        self.fibers = [self.sample_fiber(k) for k in range(self.sites)]
        pivots = [self.sample_pivot(k) for k in range(self.sites - 1)]
        tensors = [
            _contract_last_and_first(fiber, pivot)
            for fiber, pivot in zip(self.fibers, pivots)
        ]
        tensors.append(self.fibers[-1])
        self.mps = MPS(tensors)
        self.J_l, self.J_g = self.points_to_integers(initial_point)

    def sample_fiber(self, k: int) -> np.ndarray:
        i_l, i_s, i_g = self.I_l[k], self.I_s[k], self.I_g[k]
        mps_indices = self.combine_indices(i_l, i_s, i_g)
        return self.black_box[mps_indices].reshape((len(i_l), len(i_s), len(i_g)))

    def sample_pivot(self, k: int) -> np.ndarray:
        i_l, i_g = self.I_l[k + 1], self.I_g[k]
        mps_indices = self.combine_indices(i_l, i_g)
        P = self.black_box[mps_indices].reshape(len(i_l), len(i_g))
        return np.linalg.inv(P)

    _Index = TypeVar("_Index", bound=Union[int, np.intp, np.ndarray, slice])

    def sample_superblock(
        self, k: int, j_l: _Index = slice(None), j_g: _Index = slice(None)
    ) -> np.ndarray:
        i_ls = self.combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_ls = i_ls.reshape(1, -1) if i_ls.ndim == 1 else i_ls  # Prevent collapse to 1D
        i_sg = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        i_sg = i_sg.reshape(1, -1) if i_sg.ndim == 1 else i_sg
        mps_indices = self.combine_indices(i_ls, i_sg)
        return self.black_box[mps_indices].reshape((len(i_ls), len(i_sg)))

    def sample_skeleton(
        self,
        k: int,
        j_l: _Index = slice(None),
        j_g: _Index = slice(None),
    ) -> np.ndarray:
        i_l, i_s1, r = self.mps[k].shape
        r, i_s2, i_g = self.fibers[k + 1].shape
        tensor_L = self.mps[k].reshape(i_l * i_s1, r)[j_l]
        tensor_R = self.fibers[k + 1].reshape(r, i_s2 * i_g)[:, j_g]
        return _contract_last_and_first(tensor_L, tensor_R)

    def update_indices(self, k: int, j_l: _Index, j_g: _Index) -> None:
        i_l = self.combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_g = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        self.I_l[k + 1] = np.vstack((self.I_l[k + 1], i_l))
        self.J_l[k + 1] = np.append(self.J_l[k + 1], j_l)  # type: ignore
        self.I_g[k] = np.vstack((self.I_g[k], i_g))
        self.J_g[k] = np.append(self.J_g[k], j_g)  # type: ignore

    def update_tensors(
        self,
        k: int,
        r: np.ndarray,
        c: np.ndarray,
    ) -> None:
        # TODO: El algoritmo funciona pero sigue sin ser estable.
        # El truco de la QR es CRUCIAL aún haciendo updates a G en lugar de a P.

        # Get left tensor core and fiber
        r_l, r_s1, chi = self.mps[k].shape
        G_L = self.mps[k].reshape(r_l * r_s1, chi)
        R_L = self.fibers[k].reshape(r_l * r_s1, chi)

        # Get right tensor core and fiber
        chi, r_s2, r_g = self.mps[k + 1].shape
        R_R = self.fibers[k + 1]
        G_R = self.mps[k + 1]

        # Get integer indices
        j_l = self.J_l[k + 1][:-1]
        j = self.J_l[k + 1][-1]

        # Update left tensor core and fiber
        S = c[j] - np.dot(G_L[j], c[j_l])  # Schur complement
        Gu = _contract_last_and_first(G_L, c[j_l])
        G_L1 = (np.outer(Gu, G_L[j]) - np.outer(c, G_L[j])) / S
        G_L2 = (c - Gu) / S
        G_L = np.hstack((G_L + G_L1, G_L2.reshape(-1, 1)))
        R_L = np.hstack((R_L, c.reshape(-1, 1)))

        # Update right fiber
        R_R_updated = np.vstack((R_R.reshape(chi, r_s2 * r_g), r))

        # Update right tensor core from right fiber
        rect_inverse = lambda A: np.linalg.inv(A.T @ A) @ A.T
        R_R_old = R_R.reshape(chi * r_s2, r_g)
        R_R_new = R_R_updated.reshape((chi + 1) * r_s2, r_g)
        G_R_old = G_R.reshape(chi * r_s2, r_g)
        G_R_new = R_R_new @ rect_inverse(R_R_old) @ G_R_old

        # Apply the updates to self
        self.mps[k] = G_L.reshape(r_l, r_s1, chi + 1)
        self.fibers[k] = R_L.reshape(r_l, r_s1, chi + 1)
        self.mps[k + 1] = G_R_new.reshape(chi + 1, r_s2, r_g)
        self.fibers[k + 1] = R_R_new.reshape(chi + 1, r_s2, r_g)

    def points_to_integers(self, initial_point: np.ndarray):
        # TODO: Refactor
        def find_row_indices(small_array: np.ndarray, large_array: np.ndarray):
            large_set = {tuple(row): idx for idx, row in enumerate(large_array)}
            return np.array([large_set[tuple(row)] for row in small_array])

        J_l = []
        for k in range(len(initial_point) - 1):
            i_small = self.I_l[k + 1]
            i_large = self.combine_indices(self.I_l[k], self.I_s[k])
            J_l.append(find_row_indices(i_small, i_large))
        J_l.insert(0, None)  # Insert padding on the left to respect convention

        J_g = []
        for k in reversed(range(len(initial_point) - 1)):
            i_small = self.I_g[k]
            i_large = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])
            J_g.append(find_row_indices(i_small, i_large))
        J_g.append(None)  # Insert padding on the right to respect convention

        return J_l, J_g


def cross_greedy(
    black_box: BlackBox,
    cross_strategy: CrossStrategyGreedy = CrossStrategyGreedy(),
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on two-site optimizations following greedy updates of the pivot matrices.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    cross_strategy : CrossStrategy, default=CrossStrategy()
        A dataclass containing the parameters of the algorithm.

    Returns
    -------
    mps : MPS
        The MPS representation of the black-box function.
    """
    initial_point = cross_strategy.rng.integers(
        low=0, high=black_box.base, size=black_box.sites
    )
    cross = CrossInterpolationGreedy(black_box, initial_point)

    if cross_strategy.greedy_method == "full":
        update_method = _update_full_search
    elif cross_strategy.greedy_method == "partial":
        update_method = _update_partial_search

    converged = False
    with make_logger(2) as logger:
        for i in range(cross_strategy.maxiter):
            # Forward sweep
            for k in range(cross.sites - 1):
                update_method(cross, k, cross_strategy)
            if converged := _check_convergence(cross, i, cross_strategy, logger):
                break
            # Backward sweep
            for k in reversed(range(cross.sites - 1)):
                update_method(cross, k, cross_strategy)
            if converged := _check_convergence(cross, i, cross_strategy, logger):
                break
    if not converged:
        logger("Maximum number of TT-Cross iterations reached")
    return CrossResults(mps=cross.mps, evals=black_box.evals)


def _update_full_search(
    cross: CrossInterpolationGreedy,
    k: int,
    cross_strategy: CrossStrategyGreedy,
) -> None:
    A = cross.sample_superblock(k)
    B = cross.sample_skeleton(k)

    diff = np.abs(A - B)
    j_l, j_g = np.unravel_index(np.argmax(diff), A.shape)
    if diff[j_l, j_g] < cross_strategy.greedy_tol:
        return

    cross.update_indices(k, j_l=j_l, j_g=j_g)
    cross.update_tensors(k, r=A[j_l, :], c=A[:, j_g])


def _update_partial_search(
    cross: CrossInterpolationGreedy,
    k: int,
    cross_strategy: CrossStrategyGreedy,
) -> None:
    j_l_random = cross_strategy.rng.integers(
        low=0,
        high=len(cross.I_l[k]) * len(cross.I_s[k]),
        size=cross_strategy.partial_points,
    )
    j_g_random = cross_strategy.rng.integers(
        low=0,
        high=len(cross.I_s[k + 1]) * len(cross.I_g[k + 1]),
        size=cross_strategy.partial_points,
    )
    A_random = cross.sample_superblock(k, j_l=j_l_random, j_g=j_g_random)
    B_random = cross.sample_skeleton(k, j_l=j_l_random, j_g=j_g_random)

    cost_function = lambda A, B: np.abs(A - B)
    diff = cost_function(A_random, B_random)
    i, j = np.unravel_index(np.argmax(diff), A_random.shape)
    j_l, j_g = j_l_random[i], j_g_random[j]

    for iter in range(cross_strategy.partial_maxiter):
        # Traverse column residual
        c_A = cross.sample_superblock(k, j_g=j_g).reshape(-1)
        c_B = cross.sample_skeleton(k, j_g=j_g)
        new_j_l = np.argmax(cost_function(c_A, c_B))
        if new_j_l == j_l and iter > 0:
            break
        j_l = new_j_l

        # Traverse row residual
        r_A = cross.sample_superblock(k, j_l=j_l).reshape(-1)
        r_B = cross.sample_skeleton(k, j_l=j_l)
        new_j_g = np.argmax(cost_function(r_A, r_B))
        if new_j_g == j_g:
            break
        j_g = new_j_g

    cross.update_indices(k, j_l=j_l, j_g=j_g)
    cross.update_tensors(k, r=r_A, c=c_A)
