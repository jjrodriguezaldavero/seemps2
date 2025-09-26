from __future__ import annotations
import numpy as np
from scipy.special import roots_hermite, hermite
from math import factorial

from ...typing import Vector
from ..mesh import array_affine
from .expansion import OrthogonalExpansion, ScalarFunction


class HermiteExpansion(OrthogonalExpansion):
    canonical_domain = (-np.inf, np.inf)

    def __init__(self, coeffs: Vector, domain: tuple[float, float]):
        super().__init__(coeffs, domain)

    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        """Hermite recurrence: H_{k+1}(x) = 2x H_k(x) - 2k H_{k-1}(x)"""
        α_k = 2.0
        β_k = 0.0
        γ_k = 2.0 * k
        return (α_k, β_k, γ_k)

    @property
    def p1_factor(self) -> float:
        return 2.0

    @classmethod
    def project(
        cls,
        func: ScalarFunction,
        start: float = -1.0,
        stop: float = 1.0,
        order: int | None = None,
    ) -> HermiteExpansion:
        if order is None:
            order = cls.estimate_order(func, start, stop)

        nodes, weights = roots_hermite(order)
        L = np.max(nodes)
        eff_canonical_domain = (-L, L)

        nodes_affine = array_affine(
            nodes, orig=eff_canonical_domain, dest=(start, stop)
        )
        f_vals = func(nodes_affine)

        coeffs = np.zeros(order)
        for k in range(order):
            Hk = hermite(k)(nodes)  # H_k at quadrature nodes
            integral = np.sum(weights * f_vals * Hk)  # ⟨f, H_k⟩_w
            norm = 2.0**k * factorial(k) * np.sqrt(np.pi)  # ||H_k||²
            coeffs[k] = integral / norm  # a_k

        # Override the unbounded canonical domain by the effective canonical domain (-L, L)
        expansion = cls(coeffs, domain=(start, stop))
        expansion.canonical_domain = eff_canonical_domain
        return expansion
