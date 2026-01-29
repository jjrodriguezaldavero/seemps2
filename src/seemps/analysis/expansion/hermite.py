from __future__ import annotations
import numpy as np
from scipy.special import roots_hermite, eval_hermite, factorial

from ...typing import Vector
from .expansion import PolynomialExpansion, ScalarFunction

# TODO: Add tests


class HermiteExpansion(PolynomialExpansion):
    r"""
    Expansion in the physicist's Hermite basis.

    The Hermite polynomials :math:`H_k(x)` are orthogonal on the interval
    :math:`(-\infty,\infty)` with respect to the Gaussian weight :math:`e^{-x^2}`.

    See https://en.wikipedia.org/wiki/Hermite_polynomials for more information.
    """

    orthogonality_domain = (-np.inf, np.inf)
    affine_fix = (2.0, 0.0)

    def __init__(self, coefficients: Vector, scale: float):
        self.scale = float(scale)
        super().__init__(coefficients)

    def recurrence_coefficients(self, k: int) -> tuple[float, float, float]:
        r"""
        Returns the three-term coefficients of the Hermite recursion:

        .. math::
           H_{k+1}(x) = 2x H_k(x) - 2k H_{k-1}(x)
        """
        return (2.0, 0.0, 2.0 * k)

    @classmethod
    def project(
        cls,
        func: ScalarFunction,
        order: int,
        scale: float = 1.0,
    ) -> HermiteExpansion:
        r"""
        Project a scalar function onto the scaled Hermite basis :math:`H_k(\sqrt{s} x)`.

        Unlike compactly supported bases (e.g. Chebyshev), Hermite polynomials are
        orthogonal on :math:`(-\infty, \infty)` and therefore do not require an explicit
        approximation domain. Instead, the `scale` parameter controls the effective "region
        of significance" of the basis by dilating the gaussian weight, concentrating resolution
        where the function has most mass.
        """
        y, w = roots_hermite(order)
        x = y / np.sqrt(scale)
        H = np.vstack([eval_hermite(k, y) for k in range(order)])
        k = np.arange(order)
        norm = np.sqrt(np.pi) * (2.0**k) * factorial(k) / np.sqrt(scale)
        coefficients = (H * func(x)).dot(w) / norm
        return cls(coefficients, scale=scale)
