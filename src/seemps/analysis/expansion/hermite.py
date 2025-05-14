from __future__ import annotations
import numpy as np

from ...typing import Vector
from .expansion import PolynomialExpansion, ScalarFunction


class HermiteExpansion(PolynomialExpansion):
    basis_domain = (-np.inf, np.inf)

    def __init__(self, coeffs: Vector, domain: tuple[float, float]):
        super().__init__(coeffs, domain)

    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        """Hermite recurrence: H_{k+1}(x) = 2x H_k(x) - 2k H_{k-1}(x)"""
        α_k = 2.0
        β_k = 0.0
        γ_k = 2.0 * k
        return (α_k, β_k, γ_k)

    @classmethod
    def project(
        cls, order: int, func: ScalarFunction, start: float, stop: float
    ) -> HermiteExpansion:
        nodes, weights = np.polynomial.hermite.hermgauss(order)
