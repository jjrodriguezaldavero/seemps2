from __future__ import annotations
import numpy as np
from typing import Optional

from ...typing import Vector
from ..mesh import array_affine
from .expansion import PolynomialExpansion, ScalarFunction


class LegendreExpansion(PolynomialExpansion):
    basis_domain = (-1, 1)

    def __init__(self, coeffs: Vector, domain: tuple[float, float]):
        super().__init__(coeffs, domain)

    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        """Legendre recurrence: (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)"""
        α_k = (2 * k + 1) / (k + 1)
        β_k = 0.0
        γ_k = k / (k + 1)
        return (α_k, β_k, γ_k)

    @classmethod
    def project(
        cls,
        func: ScalarFunction,
        start: float = -1.0,
        stop: float = 1.0,
        order: Optional[int] = None,
    ) -> LegendreExpansion:
        if order is None:
            order = cls.estimate_order(func, start, stop)
        nodes, weights = np.polynomial.legendre.leggauss(order)
        nodes_affine = array_affine(nodes, orig=(-1, 1), dest=(start, stop))
        P_matrix = np.vstack(
            [np.polynomial.legendre.legval(nodes, [0] * k + [1]) for k in range(order)]
        )
        coeffs = (P_matrix * func(nodes_affine)).dot(weights)
        coeffs *= (2 * np.arange(order) + 1) / 2
        return cls(coeffs, domain=(start, stop))
