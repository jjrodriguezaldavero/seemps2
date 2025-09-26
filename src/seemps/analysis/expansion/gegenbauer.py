from __future__ import annotations
import numpy as np
from scipy.special import roots_gegenbauer, eval_gegenbauer, gamma

from ...typing import Vector
from ..mesh import array_affine
from .expansion import OrthogonalExpansion, ScalarFunction


class GegenbauerExpansion(OrthogonalExpansion):
    canonical_domain = (-1, 1)

    def __init__(self, coeffs: Vector, domain: tuple[float, float], α: float):
        self.α = α
        super().__init__(coeffs, domain)

    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        """Gegenbauer recurrence: (k+1) C_{k+1}^{(α)}(x) = 2(k+α)x C_k^{(α)}(x) - (k+2α-1) C_{k-1}^{(α)}(x)"""
        α_k = 2.0 * (k + self.α) / (k + 1)
        β_k = 0.0
        γ_k = (k + 2.0 * self.α - 1) / (k + 1)
        return α_k, β_k, γ_k

    @property
    def p1_factor(self) -> float:
        return 2 * self.α

    @classmethod
    def project(
        cls,
        func: ScalarFunction,
        start: float = -1.0,
        stop: float = 1.0,
        order: int | None = None,
        α: float | None = None,
    ) -> GegenbauerExpansion:
        if α is None:
            raise ValueError("Missing required parameter α for GegenbauerExpansion.")
        elif α <= -0.5 or α == 0.0:
            raise ValueError("Gegenbauer parameter α must satisfy α > -1/2 and α != 0.")

        if order is None:
            order = cls.estimate_order(func, start, stop, α=α)

        nodes, weights = roots_gegenbauer(order, α)
        nodes_affine = array_affine(
            nodes, orig=GegenbauerExpansion.canonical_domain, dest=(start, stop)
        )

        # TODO: Refactor this
        C_matrix = np.vstack([eval_gegenbauer(k, α, nodes) for k in range(order)])
        integrals = (C_matrix * func(nodes_affine)).dot(weights)
        ks = np.arange(order, dtype=float)
        norm_factors = (
            np.pi
            * 2.0 ** (1.0 - 2.0 * α)
            * gamma(ks + 2.0 * α)
            / (gamma(α) ** 2 * (ks + α) * gamma(ks + 1.0))
        )
        coeffs = integrals / norm_factors

        return cls(coeffs, domain=(start, stop), α=α)
