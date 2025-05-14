from __future__ import annotations
import numpy as np

from ...typing import Vector
from ..mesh import array_affine
from .expansion import PolynomialExpansion, ScalarFunction


class GegenbauerExpansion(PolynomialExpansion):
    basis_domain = (-1, 1)

    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        """Gegenbauer recurrence: (n+1) C_{n+1}^{(λ)}(x) = 2(n+λ)x C_n^{(λ)}(x) - (n+2λ-1) C_{n-1}^{(λ)}(x)"""
        γ_k = 2.0 * (k + self.λ) / (k + 1)
        α_k = 0.0
        β_k = (k + 2.0 * self.λ - 1.0) / (k + 1)
        return (γ_k, α_k, β_k)
