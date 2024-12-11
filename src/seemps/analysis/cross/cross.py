from __future__ import annotations
import numpy as np
import scipy
import dataclasses
import functools
from typing import Optional

from ...state import MPS
from ...tools import make_logger
from ...typing import Vector, Matrix, Tensor3, Tensor4
from ..evaluation import random_mps_indices, evaluate_mps
from .black_box import BlackBox

# TODO: Create a master function `cross_interpolation` that enables calling every
# TCI variant by passing the correct CrossStrategy


@dataclasses.dataclass
class CrossStrategy:
    cost_tol: float = 1e-8
    cost_norm: float = np.inf
    cost_evals: int = 2**10
    cost_relative: bool = False
    max_half_sweeps: int = 200
    max_bond: int = 1000
    max_time: Optional[float] = None
    max_evals: Optional[int] = None
    rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng()
    )
    """
    Dataclass encapsulating the base TCI hyperparameters.

    Parameters
    ----------
    cost_tol : float, default=1e-8
        Tolerance for the randomly evaluated error between the MPS and the function.
    cost_norm : float, default=np.inf
        Norm L^p used for calculating the cost.
    cost_evals : int, default=1024
        Number of evaluations to measure the cost.
    cost_relative : bool, default=False
        Flag indicating whether to compute the absolute or relative cost.
    max_half_sweeps : int, default=200
        Maximum number of half-sweeps allowed.
    max_bond : int, default=1000
        Maximum allowed MPS bond dimension.
    max_time : Optional[float], default=None
        Maximum time allowed for the computation.
    max_evals : Optional[int], default=None
        Maximum number of evaluations allowed.
    rng : np.random.Generator, default=np.random.default_rng()
        Random number generator, used for the random cost evaluation.
    """


class CrossInterpolant:
    """
    Helper class that stores the information of the interpolation.
    It keeps track of the current MPS approximation and of the nested indices in
    the lists `I_l` (at sites lower than k), `I_s` (at site k) and `I_g`
    (at sites greater than k) for every site k. Also, it provides methods to evaluate
    the "fiber" and "superblock" of the current interpolation at site k.
    """

    def __init__(self, black_box: BlackBox, initial_points: Optional[Matrix] = None):
        self.black_box = black_box
        self.sites = len(black_box.physical_dimensions)
        if initial_points is None:
            initial_points = np.zeros(self.sites, dtype=int)
        self.I_l, self.I_g = _points_to_indices(initial_points)
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
    """
    Dataclass store the TCI results.

    Parameters
    ----------
    mps : MPS
        The resulting MPS interpolation of the black-box function.
    costs : Vector
        Vector of cost values for each iteration (half-sweep).
    bonds : Matrix
        Matrix storing the bond dimensions of the MPS for each iteration.
    times : Vector
        Vector of cumulative computation times for each iteration.
    evals : Vector
        Vector of cumulative function evaluations for each iteration.
    """


class CrossCost:
    """
    Helper class that facilitates the computation of TCI cost, caching intermediate
    results for efficiency.
    """

    def __init__(self, cross_strategy: CrossStrategy):
        self.cost_norm = cross_strategy.cost_norm
        self.cost_evals = cross_strategy.cost_evals
        self.cost_relative = cross_strategy.cost_relative
        self.rng = cross_strategy.rng
        # Cache
        self.mps_indices = None
        self.black_box_evals = None
        self.norm = 1.0

    def lp_distance(self, x: Vector) -> float:
        p = self.cost_norm
        if np.isfinite(p):
            dist = ((1 / len(x)) * np.sum(np.abs(x) ** p)) ** (1 / p)
        else:
            dist = np.max(np.abs(x))
        return float(dist)

    def get_cost(self, interpolant: CrossInterpolant) -> float:
        if self.mps_indices is None:
            # Consider the allowed indices to impose restrictions (e.g. diagonal MPO)
            allowed_indices = getattr(interpolant.black_box, "allowed_indices", None)
            self.mps_indices = random_mps_indices(
                interpolant.black_box.physical_dimensions,
                self.cost_evals,
                allowed_indices,
                self.rng,
            )
            self.black_box_evals = interpolant.black_box[self.mps_indices].reshape(-1)
            self.norm = self.lp_distance(self.black_box_evals)
        mps_evals = evaluate_mps(interpolant.mps, self.mps_indices)
        cost = self.lp_distance(mps_evals - self.black_box_evals)
        return cost / self.norm if self.cost_relative else cost


def check_convergence(
    half_sweep: int, trajectories: dict, cross_strategy: CrossStrategy
) -> bool:
    """Checks the convergence of the algorithm and logs the results for each iteration."""
    maxbond = np.max(trajectories["bonds"][-1])
    maxbond_prev = np.max(trajectories["bonds"][-2]) if half_sweep > 2 else 0
    time = np.sum(trajectories["times"])
    evals = trajectories["evals"][-1]
    with make_logger(2) as logger:
        logger(
            f"Cross half-sweep: {half_sweep:3}/{cross_strategy.max_half_sweeps}, "
            f"cost: {trajectories['costs'][-1]:1.15e}/{cross_strategy.cost_tol:.2e}, "
            f"maxbond: {maxbond:3}/{cross_strategy.max_bond}, "
            f"time: {time:8.6f}/{cross_strategy.max_time}, "
            f"evals: {evals:8}/{cross_strategy.max_evals}."
        )

    if trajectories["costs"][-1] <= cross_strategy.cost_tol:
        logger(f"State converged within tolerance {cross_strategy.cost_tol}")
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


def combine_indices(*indices: Matrix, fortran: bool = False) -> Matrix:
    """
    Combines matrices of indices by taking their Cartesian product, either
    following C or Fortran convention.
    """

    def cartesian(A: Matrix, B: Matrix) -> Matrix:
        A_repeated = np.repeat(A, repeats=B.shape[0], axis=0)
        B_tiled = np.tile(B, (A.shape[0], 1))
        return np.hstack((A_repeated, B_tiled))

    def cartesian_fortran(A: Matrix, B: Matrix) -> Matrix:
        A_tiled = np.tile(A, (B.shape[0], 1))
        B_repeated = np.repeat(B, repeats=A.shape[0], axis=0)
        return np.hstack((A_tiled, B_repeated))

    if fortran:
        return functools.reduce(cartesian_fortran, indices)
    else:
        return functools.reduce(cartesian, indices)


def _points_to_indices(points: Matrix) -> tuple[list[Matrix], list[Matrix]]:
    """Extract the left and right multi-indices of a matrix of "points" on the mesh."""
    if points.ndim == 1:
        points = points.reshape(1, -1)
    sites = points.shape[1]
    I_l = [points[:, :k] for k in range(sites)]
    I_g = [points[:, (k + 1) : sites] for k in range(sites)]
    return I_l, I_g


def maxvol_square(
    A: Matrix,
    max_iter: int = 10,
    tol: float = 1.05,
) -> tuple[Matrix, Matrix]:
    """
    Performs the "maxvol" decomposition to compute a submatrix `B` of maximal volume in `A`.

    Parameters
    ----------
    A : Matrix
        A "tall" matrix with more rows than columns from which to extract the submatrix.
    max_iter : int, default=10
        The maximum number of iterations allowed.
    tol : float, default=1.05
        The tolerance with respect to the volume improvement for `B` required for convergence.

    Returns
    -------
    tuple[Matrix, Matrix]
        A tuple containing:
        `I`: An array of indices representing the rows in `A` that form the submatrix with maximal volume.
        `B`: The square submatrix of approximately maximal volume in `A`.
    """
    n, r = A.shape

    if n <= r:
        I, B = np.arange(n, dtype=int), np.eye(n)
        return I, B

    P, L, U = scipy.linalg.lu(A)
    I = P[:, :r].argmax(axis=0)
    Q = scipy.linalg.solve_triangular(U, A.T, trans=1)
    B = scipy.linalg.solve_triangular(
        L[:r, :], Q, trans=1, unit_diagonal=True, lower=True
    ).T

    for _ in range(max_iter):
        i, j = np.divmod(abs(B).argmax(), r)
        if abs(B[i, j]) <= tol:
            break
        I[j] = i
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])

    return I, B
