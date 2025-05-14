from __future__ import annotations
import numpy as np
from scipy.fft import dct  # type: ignore
from typing import Optional, Literal

from ...typing import Vector
from ..mesh import array_affine, ChebyshevInterval
from .expansion import PolynomialExpansion, ScalarFunction


class ChebyshevExpansion(PolynomialExpansion):
    basis_domain = (-1, 1)

    def __init__(self, coeffs: Vector, domain: tuple[float, float]):
        super().__init__(coeffs, domain)

    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        """Chebyshev recurrence: T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)"""
        _ = k  # Ignore k
        α_k = 2.0
        β_k = 0.0
        γ_k = 1.0
        return (α_k, β_k, γ_k)

    @classmethod
    def project(
        cls,
        func: ScalarFunction,
        start: float = -1.0,
        stop: float = 1.0,
        order: Optional[int] = None,
    ) -> ChebyshevExpansion:
        if order is None:
            order = cls.estimate_order(func, start, stop)
        nodes = np.cos(np.pi * np.arange(1, 2 * order, 2) / (2.0 * order))
        nodes_affine = array_affine(nodes, orig=(-1, 1), dest=(start, stop))
        weights = np.ones(order) * (np.pi / order)
        T_matrix = np.cos(np.outer(np.arange(order), np.arccos(nodes)))
        coeffs = (2 / np.pi) * (T_matrix * func(nodes_affine)) @ weights
        coeffs[0] /= 2
        return cls(coeffs, domain=(start, stop))

    @classmethod
    def interpolate(
        cls,
        func: ScalarFunction,
        start: float,
        stop: float,
        order: Optional[int] = None,
        nodes: Literal["zeros", "extrema"] = "zeros",
    ) -> ChebyshevExpansion:
        if order is None:
            order = cls.estimate_order(func, start, stop)
        if nodes == "zeros":
            nodes = ChebyshevInterval(start, stop, order).to_vector()
            coeffs = (1 / order) * dct(np.flip(func(nodes)), type=2)
        elif nodes == "extrema":
            nodes = ChebyshevInterval(start, stop, order, endpoints=True).to_vector()
            coeffs = 2 * dct(np.flip(func(nodes)), type=1, norm="forward")
        coeffs[0] /= 2
        return cls(coeffs, domain=(start, stop))

    def deriv(self, m: int = 1) -> ChebyshevExpansion:
        """Return the m-th derivative as a new ChebyshevExpansion."""
        poly = np.polynomial.Chebyshev(self.coeffs, domain=self.domain)
        poly_deriv: np.polynomial.Chebyshev = poly.deriv(m)
        return ChebyshevExpansion(poly_deriv.coef, domain=tuple(poly_deriv.domain))

    def integ(self, m: int = 1, lbnd: float = 0.0) -> ChebyshevExpansion:
        """Return the m-th integral as a new ChebyshevExpansion."""
        poly = np.polynomial.Chebyshev(self.coeffs, domain=self.domain)
        poly_integ: np.polynomial.Chebyshev = poly.integ(m=m, lbnd=lbnd)
        return ChebyshevExpansion(poly_integ.coef, domain=tuple(poly_integ.domain))
