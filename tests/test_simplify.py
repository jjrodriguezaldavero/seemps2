import numpy as np
from seemps.expectation import scprod
from seemps.state import (DEFAULT_STRATEGY, MPS, NO_TRUNCATION, Simplification,
                          random_uniform_mps)
from seemps.truncate import simplify

from .tools import *


class TestSimplify(TestCase):

    def test_no_truncation(self):
        d = 2
        for n in range(3,9):
            ψ = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ = ψ * (1/ψ.norm())
            φ = simplify(ψ, truncation=NO_TRUNCATION)
            self.assertSimilar(ψ.to_vector(), φ.to_vector())

    def test_tolerance(self):
        d = 2
        tolerance = 1e-10
        strategy = DEFAULT_STRATEGY.replace(simplification_tolerance=tolerance)
        for n in range(3,15):
            ψ = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ = ψ * (1/ψ.norm())
            φ = simplify(ψ, truncation=strategy)
            err = 2 * abs(
            1.0 - scprod(ψ, φ).real / (ψ.norm() * φ.norm()))
            self.assertTrue(err < tolerance)

    def test_max_bond_dimensions(self):
        d = 2
        n = 14
        for D in range(2,15):
            strategy = DEFAULT_STRATEGY.replace(max_bond_dimension=D)
            ψ = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ = ψ * (1/ψ.norm())
            φ = simplify(ψ, truncation=strategy)
            max_D_φ = max([max(t.shape) for t in φ])
            self.assertTrue(max_D_φ <= D)
    
    def test_simplification_method(self):
        d = 2
        strategy_0 = DEFAULT_STRATEGY.replace(simplification_method=Simplification.CANONICAL_FORM)
        strategy_1 = DEFAULT_STRATEGY.replace(simplification_method=Simplification.VARIATIONAL)
        for n in range(3,9):
            ψ = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ = ψ * (1/ψ.norm())
            φ0 = simplify(ψ, truncation=strategy_0)
            φ1 = simplify(ψ, truncation=strategy_1)
            self.assertSimilar(ψ.to_vector(), φ0.to_vector())
            self.assertSimilar(ψ.to_vector(), φ1.to_vector())
            self.assertSimilar(φ0.to_vector(), φ1.to_vector())