from __future__ import annotations
import numpy as np
import scipy.spatial
import scipy.stats
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from ...state.sampling import sample_mps
from ...typing import Vector
from ..sampling import evaluate_mps, sobol_mps_indices, random_mps_indices
from .tools import Matrix

if TYPE_CHECKING:  # Avoid circular dependency
    from .cross import CrossInterpolant


class CostFunction(ABC):
    @abstractmethod
    def cost(self, interpolant: CrossInterpolant) -> float: ...

    @abstractmethod
    def reset(self) -> None: ...


class CostNormP(CostFunction):
    def __init__(
        self,
        p: float = np.inf,
        relative: bool = False,
        num_evals: int = 2**10,
        sobol: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        self.p = p
        self.relative = relative
        self.num_evals = num_evals
        self.sobol = sobol
        self.rng = np.random.default_rng() if rng is None else rng
        # Cache
        self.mps_indices = None
        self.y = None
        self.norm_y = None

    def norm_p(self, x: Vector) -> float:
        if np.isfinite(self.p):
            return ((1 / len(x)) * np.sum(np.abs(x) ** self.p)) ** (1 / self.p)
        else:
            return np.max(x)

    def cost(self, interpolant: CrossInterpolant) -> float:
        if self.mps_indices is None:
            s = interpolant.black_box.physical_dimensions
            self.mps_indices = (
                sobol_mps_indices(s, self.num_evals, self.rng)
                if self.sobol
                else random_mps_indices(s, self.num_evals, rng=self.rng)
            )
            self.y = interpolant.black_box[self.mps_indices].reshape(-1)
            self.norm_y = self.norm_p(self.y)
        x = evaluate_mps(interpolant.mps, self.mps_indices)
        dist = self.norm_p(x - self.y)
        return dist / self.norm_y if self.relative else dist

    def reset(self):
        self.mps_indices = None
        self.y = None
        self.norm_y = None


class CostNormPFull(CostFunction):
    def __init__(self, p: float = np.inf):
        self.p = p

    def norm_p(self, x: Vector) -> float:
        if np.isfinite(self.p):
            return ((1 / len(x)) * np.sum(np.abs(x) ** self.p)) ** (1 / self.p)
        else:
            return np.max(x)

    def cost(self, interpolant: CrossInterpolant) -> float:
        import itertools

        s = interpolant.black_box.physical_dimensions
        n = len(s)
        all_mps_indices = np.array(list(itertools.product(list(range(s[0])), repeat=n)))
        Z_exact = interpolant.black_box[all_mps_indices]
        Z_test = interpolant.mps.to_vector()
        return self.norm_p(np.abs(Z_exact - Z_test))

    def reset(self):
        pass


class CostKL(CostFunction):
    def __init__(
        self,
        num_evals: int = 2**10,
        sobol: bool = True,
        regularize: bool = True,
        symmetrize: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_evals = num_evals
        self.sobol = sobol
        self.regularize = regularize
        self.symmetrize = symmetrize
        self.rng = np.random.default_rng() if rng is None else rng
        # Cache
        self.mps_indices = None
        self.p = None

    def cost(self, interpolant: CrossInterpolant) -> float:
        if self.mps_indices is None:
            s = interpolant.black_box.physical_dimensions
            self.mps_indices = (
                sobol_mps_indices(s, self.num_evals, self.rng)
                if self.sobol
                else random_mps_indices(s, self.num_evals, rng=self.rng)
            )
            self.p = interpolant.black_box[self.mps_indices].reshape(-1)
            if self.regularize:
                p2 = self.p**2
                self.p = p2 / np.sum(p2)

        q = evaluate_mps(interpolant.mps, self.mps_indices)
        if self.regularize:
            q2 = q**2
            q = q2 / np.sum(q2)

        if self.symmetrize:
            m = 0.5 * (self.p + q)
            kl_pm = scipy.stats.entropy(self.p, m)
            kl_qm = scipy.stats.entropy(q, m)
            return 0.5 * (kl_pm + kl_qm)
        else:
            return scipy.stats.entropy(self.p, q)

    def reset(self):
        self.mps_indices = None
        self.p = None


class CostMMD(CostFunction):
    def __init__(
        self,
        func_samples: Matrix,
        gamma: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.func_samples = func_samples  # Convention: (k, m) for k samples and m dims.
        self.num_samples, self.dim = func_samples.shape
        self.gamma = gamma
        self.rng = np.random.default_rng() if rng is None else rng

    def cost(self, interpolant: CrossInterpolant) -> float:
        mps_indices = sample_mps(interpolant.mps, size=self.num_samples, rng=self.rng)
        mps_samples = interpolant.black_box._indices_to_points(mps_indices)
        return self.mmd(self.func_samples, mps_samples.T)

    def mmd(self, X: Matrix, Y: Matrix) -> float:
        XX = self.gaussian_kernel(X, X)
        YY = self.gaussian_kernel(Y, Y)
        XY = self.gaussian_kernel(X, Y)
        mean_XX = np.mean(XX)
        mean_YY = np.mean(YY)
        mean_XY = np.mean(XY)
        return np.sqrt(mean_XX + mean_YY - 2 * mean_XY)

    def gaussian_kernel(self, X: Matrix, Y: Matrix) -> Vector:
        sq_dists = scipy.spatial.distance.cdist(X, Y, "sqeuclidean")
        return np.exp(-0.5 * sq_dists / self.gamma**2)

    def reset(self):
        pass
