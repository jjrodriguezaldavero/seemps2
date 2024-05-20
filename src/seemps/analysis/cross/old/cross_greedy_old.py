import numpy as np
from dataclasses import dataclass

from ..cross import (
    CrossInterpolation,
    CrossResults,
    CrossStrategy,
    BlackBox,
    _check_convergence,
)
from ....state import MPS
from ....state._contractions import _contract_last_and_first
from ....tools import log


@dataclass
class CrossStrategyGreedy(CrossStrategy):

    greedy_tol: float = 1e-10
    """
    Dataclass containing the parameters for the maxvol-based TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    greedy_tol : float, default = 1e-10
        Minimum error in Frobenius norm committed by the chosen pivots between the 
        superblock and the skeleton decomposition.
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
        return np.linalg.inv(P)

    def sample_superblock(self, k: int) -> np.ndarray:
        i_l, i_g = self.I_l[k], self.I_g[k + 1]
        i_s1, i_s2 = self.I_s[k], self.I_s[k + 1]
        mps_indices = self.combine_indices(i_l, i_s1, i_s2, i_g)
        return self.black_box[mps_indices].reshape(
            (len(i_l), len(i_s1), len(i_s2), len(i_g))
        )

    def contract_skeleton(self, k: int) -> np.ndarray:
        contraction = _contract_last_and_first(self.fibers[k], self.pivots[k])
        contraction = _contract_last_and_first(contraction, self.fibers[k + 1])
        return contraction

    def to_mps(self) -> MPS:
        contractions = [
            _contract_last_and_first(fiber, pivot)
            for fiber, pivot in zip(self.fibers[:-1], self.pivots)
        ]
        contractions.append(self.fibers[-1])
        return MPS(contractions)

    def translate_indices(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        I_small = self.I_l[k + 1]
        J_small = self.I_g[k]
        I_large = self.combine_indices(self.I_l[k], self.I_s[k])
        J_large = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])

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
        J = find_indices(J_small, J_large)
        return I, J


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
        cross.mps = cross.to_mps()
        converged, message = _check_convergence(cross, i, cross_strategy)
        # Backward sweep
        for k in reversed(range(cross.sites - 1)):
            _update_full_search(cross, k, False, cross_strategy)
        cross.mps = cross.to_mps()
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
    # I am pretty sure it is due to the matrices being pretty close to rank-deficiency due to the QTT format
    # The pivot matrices grow up to 1e+8, and there is catastrophic precision loss when computing the skeleton decomposition
    # reaching a maximum precision of 1e-8.
    # Then, this induces a wrong pivot selection that produces a singular matrix.
    # This problem does not appear for TT because the pivot matrix rows are quite different from each other.
    # It should have been explained in the QTT paper but they didn'nt mention it.
    # The QR trick is crucial
    # max_pivots = cross.black_box.base ** (1 + min(k, cross.sites - (k + 2)))
    # if len(cross.I_g[k]) >= max_pivots or len(cross.I_l[k + 1]) >= max_pivots:
    #     return
    # Efectivamente es esto. Ahora tan solo tengo que rediseñar el algoritmo de cero teniendo esto en cuenta.

    skeleton = cross.contract_skeleton(k)
    r_l, s1, s2, r_g = skeleton.shape
    A = skeleton.reshape(r_l * s1, s2 * r_g)

    superblock = cross.sample_superblock(k)
    B = superblock.reshape(r_l * s1, s2 * r_g)

    i_l_global = cross.I_l[k]
    i_g_global = cross.I_g[k + 1]

    i_l_global_s = cross.combine_indices(i_l_global, cross.I_s[k])
    i_g_global_s = cross.combine_indices(cross.I_s[k + 1], i_g_global)

    i_l_local = cross.I_l[k + 1]
    i_g_local = cross.I_g[k]

    diff = np.abs(A - B)
    j_row, j_col = np.unravel_index(np.argmax(diff), A.shape)  # type: ignore
    if diff[j_row, j_col] < cross_strategy.greedy_tol:
        if forward:
            cross.fibers[k + 1] = cross.sample_fiber(k + 1)
        else:
            cross.fibers[k] = cross.sample_fiber(k)
        return

    i_l_local = np.vstack((i_l_local, i_l_global_s[j_row]))
    i_g_local = np.vstack((i_g_local, i_g_global_s[j_col]))

    cross.I_l[k + 1] = i_l_local
    cross.I_g[k] = i_g_local

    J_rows = find_row_indices(i_l_local, i_l_global_s)
    J_cols = find_row_indices(i_g_local, i_g_global_s)

    C = B[:, J_cols]
    R = B[J_rows, :]

    if forward:
        Q, _ = np.linalg.qr(C)
        cross.fibers[k] = Q.reshape(r_l, s1, -1)
        cross.pivots[k] = np.linalg.inv(Q[J_rows])
        cross.fibers[k + 1] = R.reshape(-1, s2, r_g)
        # Test if the QR trick is working:
        core_1 = _contract_last_and_first(cross.fibers[k], cross.pivots[k])
        core_2 = _contract_last_and_first(cross.sample_fiber(k), cross.sample_pivot(k))
        pass  # print(core_1 - core_2)

    else:
        Q, _ = np.linalg.qr(R.T)
        cross.fibers[k + 1] = (Q.T).reshape(-1, s2, r_g)
        cross.pivots[k] = np.linalg.inv(Q[J_cols]).T
        cross.fibers[k] = C.reshape(r_l, s1, -1)
        # Test if the QR trick is working:
        core_1 = _contract_last_and_first(cross.pivots[k], cross.fibers[k + 1])
        core_2 = _contract_last_and_first(
            cross.sample_pivot(k), cross.sample_fiber(k + 1)
        )
        pass  # print(core_1 - core_2)


def find_row_indices(A, B):
    indices = []
    for row in A:
        index = np.where((B == row).all(axis=1))[0]
        if index.size > 0:
            indices.append(index[0])
    return np.array(indices)
