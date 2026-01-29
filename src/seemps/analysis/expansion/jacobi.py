from __future__ import annotations
import numpy as np
from scipy.special import roots_jacobi, eval_jacobi, gammaln

from ...state import MPS
from ...operators import MPO
from ...typing import Vector
from ..mesh import array_affine
from ..factories import mps_affine
from ..operators import mpo_affine
from .expansion import PolynomialExpansion, ScalarFunction

# TODO: Add tests


class JacobiExpansion(PolynomialExpansion):
    r"""
    Expansion in the Jacobi polynomial basis.

    The Jacobi polynomials :math:`P_k^{(\alpha,\beta)}(x)` are orthogonal on
    the interval :math:`[-1,1]` with respect to the weight :math:`(1-x)^\alpha (1+x)^\beta`.
    They are useful for function approximation since they can match functions with algebraic
    singularities on the boundaries, enhancing convergence.

    The Chebyshev and Legendre polynomials are special cases of this basis.

    See https://en.wikipedia.org/wiki/Jacobi_polynomials for more information.
    """

    orthogonality_domain = (-1.0, 1.0)

    def __init__(
        self,
        coefficients: Vector,
        approximation_domain: tuple[float, float],
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.approximation_domain = approximation_domain
        self.alpha = alpha
        self.beta = beta
        self.affine_fix = (0.5 * (alpha + beta + 2.0), 0.5 * (alpha - beta))
        super().__init__(coefficients)

    def recurrence_coefficients(self, k: int) -> tuple[float, float, float]:
        """
        Returns the three-term coefficients of the Jacobi recursion.

        See https://en.wikipedia.org/wiki/Jacobi_polynomials.
        """
        if k == 0:
            μ, σ = self.affine_fix
            return (μ, σ, 0.0)

        a = self.alpha
        b = self.beta

        kab = k + a + b
        c = kab + k  # 2k + a + b

        α_k = (c + 1) * (c + 2) / (2 * (k + 1) * (k + a + b + 1))
        β_k = (c + 1) * (a**2 - b**2) / (2 * (k + 1) * (k + a + b + 1) * c)
        γ_k = (k + a) * (k + b) * (c + 2.0) / ((k + 1) * (k + a + b + 1) * c)
        return (α_k, β_k, γ_k)

    def rescale_mps(self, mps: MPS) -> MPS:
        orig = self.approximation_domain
        dest: tuple[float, float] = self.orthogonality_domain  # pyright: ignore
        return mps_affine(mps, orig, dest)

    def rescale_mpo(self, mpo: MPO) -> MPO:
        orig = self.approximation_domain
        dest: tuple[float, float] = self.orthogonality_domain  # pyright: ignore
        return mpo_affine(mpo, orig, dest)

    @classmethod
    def project(
        cls,
        func: ScalarFunction,
        order: int,
        alpha: float,
        beta: float,
        approximation_domain: tuple[float, float] = (-1.0, 1.0),
    ) -> JacobiExpansion:
        x, w = roots_jacobi(order, alpha, beta)
        x_affine = array_affine(
            x,
            orig=cls.orthogonality_domain,  # pyright: ignore
            dest=approximation_domain,
        )
        P = np.vstack([eval_jacobi(k, alpha, beta, x) for k in range(order)])
        k = np.arange(order)
        norm = 2.0 ** (alpha + beta + 1.0) * np.exp(
            gammaln(k + alpha + 1.0)
            + gammaln(k + beta + 1.0)
            - gammaln(k + 1.0)
            - gammaln(k + alpha + beta + 1.0)
            - np.log(2 * k + alpha + beta + 1.0)
        )
        coefficients = (P * func(x_affine)).dot(w) / norm
        return cls(
            coefficients,
            approximation_domain=approximation_domain,
            alpha=alpha,
            beta=beta,
        )
