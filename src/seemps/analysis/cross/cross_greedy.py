import numpy as np
import scipy
from dataclasses import dataclass

import scipy.linalg

from .cross import (
    CrossInterpolation,
    CrossResults,
    CrossStrategy,
    BlackBox,
    _check_convergence,
)
from ...state import MPS
from ...state._contractions import _contract_last_and_first
from ...tools import log


@dataclass
class CrossStrategyGreedy(CrossStrategy):
    greedy_method: str = "full_search"
    greedy_tol: float = 1e-12
    partial_maxiter: int = 5
    partial_points: int = 10
    parallelize: bool = False
    num_cores: int = 1
    """
    Dataclass containing the parameters for the maxvol-based TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    greedy_method : str, default = "full_search"
        Method used to perform the greedy pivot updates. Options:
        - "full_search": finds the pivot of maximum error in each superblock.
        - "partial_search": looks for a pivot of maximum error using a partial search.
    greedy_tol : float, default = 1e-14
        Tolerance in Frobenius norm between the superblock and the skeleton decomposition
        after which the pivots are no longer added.
    partial_search_maxiter : int, default = 5
        How many row-column iterations to perform in the pivot partial search.
    partial_search_points : int, default = 10
        Number of initial random points for each pivot partial search.
    parallelize : bool, default = False
        Whether to parallelize the greedy updates along each site for num_cores.
    num_cores : int, default = 1
        The number of cores to parallelize the greedy updates.
    """


class CrossInterpolationGreedy(CrossInterpolation):
    def __init__(self, black_box: BlackBox, initial_point: np.ndarray):
        super().__init__(black_box, initial_point)
        self.fibers = [self.sample_fiber(k) for k in range(self.sites)]
        self.pivots = [self.sample_pivot(k) for k in range(self.sites - 1)]

    def sample_fiber(self, k: int) -> np.ndarray:
        i_l, i_s, i_g = self.I_l[k], self.I_s[k], self.I_g[k]
        mps_indices = self.combine_indices(i_l, i_s, i_g)
        return self.black_box[mps_indices].reshape((len(i_l), len(i_s), len(i_g)))

    def sample_pivot(self, k: int) -> np.ndarray:
        i_l, i_g = self.I_l[k + 1], self.I_g[k]
        mps_indices = self.combine_indices(i_l, i_g)
        P = self.black_box[mps_indices].reshape(len(i_l), len(i_g))
        return scipy.linalg.inv(P)

    def sample_superblock(self, k: int) -> np.ndarray:
        i_l, i_g = self.I_l[k], self.I_g[k + 1]
        i_s1, i_s2 = self.I_s[k], self.I_s[k + 1]
        mps_indices = self.combine_indices(i_l, i_s1, i_s2, i_g)
        return self.black_box[mps_indices].reshape(
            (len(i_l), len(i_s1), len(i_s2), len(i_g))
        )

    def skeleton(self, k: int) -> np.ndarray:
        return _contract_last_and_first(
            self.fibers[k], _contract_last_and_first(self.pivots[k], self.fibers[k + 1])
        )

    def translate_indices(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        I_small = self.I_l[k + 1]
        I_large = self.combine_indices(self.I_l[k], self.I_s[k])
        J_small = self.I_g[k]
        J_large = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])
        # Function to find indices of rows in a larger matrix

        def find_indices(small, large):
            # This will store the indices of each row of `small` in `large`
            indices = np.array(
                [
                    np.nonzero((large == single_row).all(axis=1))[0][0]
                    for single_row in small
                ]
            )
            return indices

        # Find indices of I_small in I_large
        I = find_indices(I_small, I_large)

        # Find indices of J_small in J_large
        J = find_indices(J_small, J_large)
        return I, J

    def to_mps(self) -> MPS:
        return MPS(
            [fiber @ pivot for fiber, pivot in zip(self.fibers[:-1], self.pivots)]
            + [self.fibers[-1]]
        )


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

    for i in range(cross_strategy.maxiter):
        # Forward sweep
        for k in range(cross.sites - 1):
            _update_full_search(cross, k, True, cross_strategy)
        cross.mps = cross.to_mps()  # Contract fibers and pivots to evaluate the error
        converged, message = _check_convergence(cross, i, cross_strategy)
        # Backward sweep
        for k in reversed(range(cross.sites - 1)):
            _update_full_search(cross, k, False, cross_strategy)
        cross.mps = cross.to_mps()  # Contract fibers and pivots to evaluate the error
        converged, message = _check_convergence(cross, i, cross_strategy)
        if converged:
            break
    log(message)
    return CrossResults(mps=cross.mps, evals=black_box.evals)


def _update_full_search(
    cross: CrossInterpolationGreedy,
    k: int,
    forward: bool,
    cross_strategy: CrossStrategyGreedy,
) -> None:
    # Continue only if the MPS sites allow for more pivots
    max_pivots = cross.black_box.base ** (1 + min(k, cross.sites - (k + 2)))
    if len(cross.I_g[k]) >= max_pivots or len(cross.I_l[k + 1]) >= max_pivots:
        return

    # Compute the skeleton decomposition at site k
    skeleton = cross.skeleton(k)
    r_l, s1, s2, r_g = skeleton.shape
    A = skeleton.reshape(r_l * s1, s2 * r_g)

    # Sample the superblock at site k
    superblock = cross.sample_superblock(k)
    B = superblock.reshape(r_l * s1, s2 * r_g)

    # Find the pivots that have a maximum error on the whole superblock and update the indices
    diff = np.abs(A - B)
    i, j = np.unravel_index(np.argmax(diff), A.shape)  # type: ignore
    if diff[i, j] < cross_strategy.greedy_tol:
        return
    i_l = cross.combine_indices(cross.I_l[k], cross.I_s[k])[i]
    i_g = cross.combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])[j]
    cross.I_g[k] = np.vstack((cross.I_g[k], i_g))
    cross.I_l[k + 1] = np.vstack((cross.I_l[k + 1], i_l))

    # # Translate the binary indices to integer indices
    # # Maybe I can progressively keep track of it in cross
    # I, J = cross.translate_indices(k)
    # C = B[:, J]
    # R = B[I, :]

    # # Update the tensors
    # if forward:
    #     Q, T = np.linalg.qr(C)
    #     P = Q[I]
    #     P_inv = np.linalg.inv(P)
    #     cross.fibers[k] = Q.reshape(r_l, s1, -1)
    #     cross.pivots[k] = P_inv
    #     cross.fibers[k + 1] = R.reshape(-1, s2, r_g)
    #     if k == cross.sites - 2:
    #         cross.fibers[k + 1] = _contract_last_and_first(
    #             Q[I] @ T, cross.fibers[k + 1]
    #         )
    # else:
    #     Q, T = np.linalg.qr(R.T)
    #     P = Q[J].T
    #     P_inv = np.linalg.inv(P)
    #     cross.fibers[k] = C.reshape(r_l, s1, -1)
    #     cross.pivots[k] = P_inv
    #     cross.fibers[k + 1] = Q.reshape(-1, s2, r_g)
    #     if k == 0:
    #         cross.pivots[k] = (Q[I] @ T).T @ cross.pivots[k]

    # i_l = cross.combine_indices(cross.I_l[k], cross.I_s[k])[i]
    # i_g = cross.combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])[j]

    # # Update the indices and the tensors
    # # TODO: The updates are not stable. Maybe I have to use the QR-trick

    #     cross.I_g[k] = np.vstack((cross.I_g[k], i_g))
    #     cross.I_l[k + 1] = np.vstack((cross.I_l[k + 1], i_l))
    #     cross.update_tensors(k, i_l, i_g)

    # # cross.update_fiber(k, i_g=i_g)
    # # cross.update_fiber(k + 1, i_l=i_l)
    # # cross.update_pivot(k, i_l, i_g)

    # def update_pivot(
    #     self, k: int, i_l: Optional[np.ndarray] = None, i_g: Optional[np.ndarray] = None
    # ):
    #     self.pivots[k] = self.sample_pivot(k)

    # def update_fiber(
    #     self, k: int, i_l: Optional[np.ndarray] = None, i_g: Optional[np.ndarray] = None
    # ) -> None:
    #     self.fibers[k] = self.sample_fiber(k)

    # def update_tensors(self, k: int, i_l: np.ndarray, i_g: np.ndarray) -> None:
    #     C = _contract_last_and_first(self.fibers[k], self.pivots[k])
    #     Q, T = np.linalg.qr(C)
    #     P = np.linalg.inv(Q[self.I_l[k + 1]])


#### Partial search code

# if cross_strategy.greedy_method == "full_search":
#     update_method = _update_full_search
# elif cross_strategy.greedy_method == "partial_search":
#     update_method = _update_partial_search

# def sample_submatrix(
#     self,
#     k: int,
#     row_idx: Optional[Union[int, np.ndarray]] = None,
#     col_idx: Optional[Union[int, np.ndarray]] = None,
# ) -> np.ndarray:
#     """
#     TODO: This implementation evaluates the whole superblock, reshapes it to matrix form,
#     and slices it using row_idx and col_idx. It would be more efficient to just sample the
#     required elements instead of the whole superblock.
#     """
#     i_l, i_g = self.I_l[k], self.I_g[k + 1]
#     i_s1, i_s2 = self.I_s[k], self.I_s[k + 1]
#     r_l, s1, s2, r_g = len(i_l), len(i_s1), len(i_s2), len(i_g)
#     row_idx = np.arange(r_l * s1) if row_idx is None else np.asarray(row_idx)
#     col_idx = np.arange(s2 * r_g) if col_idx is None else np.asarray(col_idx)
#     mps_indices = self.combine_indices(i_l, i_s1, i_s2, i_g)
#     return self.black_box[mps_indices].reshape(r_l * s1, s2 * r_g)[row_idx, col_idx]


# def _update_partial_search(
#     cross: CrossInterpolationGreedy,
#     k: int,
#     cross_strategy: CrossStrategyGreedy,
# ) -> None:
#     # Compute the skeleton decomposition at site k
#     skeleton = cross.skeleton(k)
#     r_l, s1, s2, r_g = skeleton.shape
#     A = skeleton.reshape(r_l * s1, s2 * r_g)

#     # Choose an initial point that has maximum error from a random set
#     rng = cross_strategy.rng
#     I_random = rng.integers(low=0, high=r_l * s1, size=cross_strategy.partial_points)
#     J_random = rng.integers(low=0, high=s2 * r_g, size=cross_strategy.partial_points)
#     random_set = cross.sample_submatrix(k, I_random, J_random)
#     idx = np.argmax(np.abs(A[I_random, J_random] - random_set))
#     i, j = I_random[idx], J_random[idx]

#     # Find the pivots that have a maximum error doing an alternate row-column search
#     # Note: the residuals `col` and `row` may be used as `u` and `v` to update the pivot matrix
#     cost_function = lambda A, B: np.abs(A - B)
#     for _ in range(cross_strategy.partial_maxiter):
#         col = cross.sample_submatrix(k, col_idx=j)  # type: ignore
#         i_k = np.argmax(cost_function(A[:, j], col))
#         row = cross.sample_submatrix(k, row_idx=i_k)  # type: ignore
#         diff = cost_function(A[i_k, :], row)
#         j_k = np.argmax(diff)
#         if (i_k, j_k) == (i, j):
#             break
#         (i, j) = (i_k, j_k)
#     i_l = cross.combine_indices(cross.I_l[k], cross.I_s[k])[i]
#     i_g = cross.combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])[j]

#     # Update the pivots and the cross-interpolation
#     if (
#         i_l.tolist() not in cross.I_l[k + 1].tolist()
#         and i_g.tolist() not in cross.I_g[k].tolist()
#     ):
#         cross.I_g[k] = np.vstack((cross.I_g[k], i_g))
#         cross.I_l[k + 1] = np.vstack((cross.I_l[k + 1], i_l))
#         cross.update_fiber(k, i_g=i_g)
#         cross.update_fiber(k + 1, i_l=i_l)
#         cross.update_pivot(k, i_l, i_g)
