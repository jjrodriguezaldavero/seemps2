import numpy as np
import scipy.linalg
from typing import TypeVar, Union
from dataclasses import dataclass

from .cross import (
    CrossInterpolation,
    CrossResults,
    CrossStrategy,
    BlackBox,
    _check_convergence,
)
from ..sampling import random_mps_indices
from ...state import MPS
from ...state._contractions import _contract_last_and_first
from ...tools import make_logger


@dataclass
class CrossStrategyGreedy(CrossStrategy):
    greedy_method: str = "full"
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
        - "full"    : finds the pivot of maximum error in the superblock doing a full search.
        - "partial" : looks for a pivot that locally maximizes the error by a partial search.
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
        self.Q_factors = []
        self.R_matrices = []
        for fiber in self.fibers[:-1]:
            Q, R = self.fiber_to_QR(fiber)
            self.Q_factors.append(Q)
            self.R_matrices.append(R)
        self.J_l, self.J_g = self.points_to_J(initial_point)
        data = [self.Q_to_G(Q, j_l) for Q, j_l in zip(self.Q_factors, self.J_l[1:])]
        self.mps = MPS(data + [self.fibers[-1]])

    def sample_fiber(self, k: int) -> np.ndarray:
        i_l, i_s, i_g = self.I_l[k], self.I_s[k], self.I_g[k]
        mps_indices = self.combine_indices(i_l, i_s, i_g)
        return self.black_box[mps_indices].reshape((len(i_l), len(i_s), len(i_g)))

    _Index = TypeVar("_Index", bound=Union[np.intp, np.ndarray, slice])

    def sample_superblock(
        self,
        k: int,
        j_l: _Index = slice(None),
        j_g: _Index = slice(None),
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
        r_l, r_s1, chi = self.mps[k].shape
        chi, r_s2, r_g = self.fibers[k + 1].shape
        G = self.mps[k].reshape(r_l * r_s1, chi)[j_l]
        R = self.fibers[k + 1].reshape(chi, r_s2 * r_g)[:, j_g]
        return _contract_last_and_first(G, R)

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
        # Update fibers
        r_l, r_s1, chi = self.fibers[k].shape
        C = self.fibers[k].reshape(r_l * r_s1, chi)
        self.fibers[k] = np.hstack((C, c.reshape(-1, 1))).reshape(r_l, r_s1, chi + 1)

        chi, r_s2, r_g = self.fibers[k + 1].shape
        R = self.fibers[k + 1].reshape(chi, r_s2 * r_g)
        self.fibers[k + 1] = np.vstack((R, r)).reshape(chi + 1, r_s2, r_g)

        # Update Q-factors and MPS sites
        # self.Q_factors[k], self.R_matrices[k] = self.fiber_to_QR(self.fibers[k])
        Q = self.Q_factors[k].reshape(r_l * r_s1, chi)
        Q, self.R_matrices[k] = scipy.linalg.qr_insert(
            Q,
            self.R_matrices[k],
            u=c,
            k=Q.shape[1],
            which="col",
            check_finite=False,
        )
        self.Q_factors[k] = Q.reshape(r_l, r_s1, chi + 1)
        self.mps[k] = self.Q_to_G(self.Q_factors[k], self.J_l[k + 1])

        if k < self.sites - 2:
            # self.Q_factors[k + 1], self.R_matrices[k + 1] = self.fiber_to_QR(self.fibers[k + 1])
            Q = self.Q_factors[k + 1].reshape(chi * r_s2, r_g)
            Q, self.R_matrices[k + 1] = scipy.linalg.qr_insert(
                Q,
                self.R_matrices[k + 1],
                u=r.reshape(-1, Q.shape[1]),
                k=Q.shape[0],
                which="row",
                check_finite=False,
            )
            self.Q_factors[k + 1] = Q.reshape(chi + 1, r_s2, r_g)
            self.mps[k + 1] = self.Q_to_G(self.Q_factors[k + 1], self.J_l[k + 2])
        else:
            self.mps[k + 1] = self.fibers[k + 1]

    def points_to_J(self, initial_point: np.ndarray):
        # TODO: Refactor
        def find_row_indices(small_array: np.ndarray, large_array: np.ndarray):
            large_set = {tuple(row): idx for idx, row in enumerate(large_array)}
            return np.array([large_set[tuple(row)] for row in small_array])

        J_l = []
        for k in range(len(initial_point) - 1):
            i_small = self.I_l[k + 1]
            i_large = self.combine_indices(self.I_l[k], self.I_s[k])
            J_l.append(find_row_indices(i_small, i_large))
        J_l.insert(0, None)  # to respect convention

        J_g = []
        for k in reversed(range(len(initial_point) - 1)):
            i_small = self.I_g[k]
            i_large = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])
            J_g.append(find_row_indices(i_small, i_large))
        J_g.append(None)  # to respect convention

        return J_l, J_g

    @staticmethod
    def fiber_to_QR(fiber: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r_l, r_s, r_g = fiber.shape
        Q, R = scipy.linalg.qr(  # type: ignore
            fiber.reshape(r_l * r_s, r_g), mode="economic", check_finite=False
        )
        Q_factor = Q.reshape(r_l, r_s, r_g)  # type: ignore
        return Q_factor, R

    @staticmethod
    def Q_to_G(Q_factor: np.ndarray, j_l: np.ndarray) -> np.ndarray:
        """Transforms a Q-factor into a MPS tensor core G."""
        r_l, r_s, r_g = Q_factor.shape
        Q = Q_factor.reshape(r_l * r_s, r_g)
        P = scipy.linalg.inv(Q[j_l], check_finite=False)
        G = _contract_last_and_first(Q, P)
        return G.reshape(r_l, r_s, r_g)


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
    initial_point = random_mps_indices(
        black_box.physical_dimensions,
        num_indices=1,
        allowed_indices=getattr(black_box, "allowed_indices", None),
        rng=cross_strategy.rng,
    )[0]
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

    error_function = lambda A, B: np.abs(A - B)
    diff = error_function(A, B)
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

    error_function = lambda A, B: np.abs(A - B)
    diff = error_function(A_random, B_random)
    i, j = np.unravel_index(np.argmax(diff), A_random.shape)
    if diff[i, j] < cross_strategy.greedy_tol:
        return

    j_l, j_g = j_l_random[i], j_g_random[j]
    for iter in range(cross_strategy.partial_maxiter):
        # Traverse column residual
        c_A = cross.sample_superblock(k, j_g=j_g).reshape(-1)
        c_B = cross.sample_skeleton(k, j_g=j_g)
        new_j_l = np.argmax(error_function(c_A, c_B))
        if new_j_l == j_l and iter > 0:
            break
        j_l = new_j_l

        # Traverse row residual
        r_A = cross.sample_superblock(k, j_l=j_l).reshape(-1)
        r_B = cross.sample_skeleton(k, j_l=j_l)
        new_j_g = np.argmax(error_function(r_A, r_B))
        if new_j_g == j_g:
            break
        j_g = new_j_g

    cross.update_indices(k, j_l=j_l, j_g=j_g)
    cross.update_tensors(k, r=r_A, c=c_A)
