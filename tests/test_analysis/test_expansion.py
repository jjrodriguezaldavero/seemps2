import numpy as np
from scipy.special import erf

from seemps.state import MPS, NO_TRUNCATION, DEFAULT_STRATEGY
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.factories import mps_tensor_sum, mps_interval, mpo_x
from seemps.analysis.expansion import (
    ChebyshevExpansion,
    mps_polynomial_expansion,
    mpo_polynomial_expansion,
)

from ..tools import TestCase


class TestChebyshevExpansion(TestCase):
    def test_interpolation_coefficients(self):
        f = lambda x: np.exp(x)
        expansion = ChebyshevExpansion.interpolate(order=15, func=f, start=-1, stop=1)
        correct_coeffs = [
            1.266065877752008,
            1.130318207984970,
            0.271495339534077,
            0.044336849848664,
            0.005474240442094,
            0.000542926311914,
            0.000044977322954,
            0.000003198436462,
            0.000000199212481,
            0.000000011036772,
            0.000000000550590,
            0.000000000024980,
            0.000000000001039,
            0.000000000000040,
            0.000000000000001,
        ]
        self.assertTrue(np.allclose(expansion.coeffs, correct_coeffs, atol=1e-15))

    def test_estimate_order(self):
        tol = 1e-12
        func = lambda x: np.exp(-(x**2))
        start, stop = -1, 1
        order = ChebyshevExpansion.estimate_order(func, start, stop, tol=tol)
        projection = ChebyshevExpansion.project(func, start, stop, order=order)
        initial = RegularInterval(-1, 1, 2**6)
        mps = mps_polynomial_expansion(projection, initial, strategy=NO_TRUNCATION)
        y_vec = func(initial.to_vector())
        y_mps = mps.to_vector()
        self.assertTrue(abs(projection.coeffs[-1]) <= tol)
        self.assertSimilar(y_mps, y_vec, atol=tol)

    def test_gaussian_1d(self):
        func = lambda x: np.exp(-(x**2))
        interval = RegularInterval(-1, 2, 2**5)
        y_vec = func(interval.to_vector())
        expansion = ChebyshevExpansion.interpolate(
            func, interval.start, interval.stop, order=30
        )
        mps_cheb_clen = mps_polynomial_expansion(
            expansion, initial=interval, clenshaw=True
        )
        mps_cheb_poly = mps_polynomial_expansion(
            expansion,
            initial=interval,
            clenshaw=False,
        )
        self.assertSimilar(y_vec, mps_cheb_clen.to_vector())
        self.assertSimilar(y_vec, mps_cheb_poly.to_vector())

    def test_gaussian_deriv_1d(self):
        func = lambda x: np.exp(-(x**2))
        func_diff = lambda x: -2 * x * np.exp(-(x**2))
        interval = RegularInterval(-1, 2, 2**5)
        y_vec = func_diff(interval.to_vector())
        expansion = ChebyshevExpansion.interpolate(
            func, interval.start, interval.stop, order=30
        )
        expansion_deriv = expansion.deriv(1)
        mps_cheb_clen = mps_polynomial_expansion(
            expansion_deriv,
            initial=interval,
            clenshaw=True,
        )
        mps_cheb_poly = mps_polynomial_expansion(
            expansion_deriv,
            initial=interval,
            clenshaw=False,
        )
        self.assertSimilar(y_vec, mps_cheb_clen.to_vector())
        self.assertSimilar(y_vec, mps_cheb_poly.to_vector())

    def test_gaussian_integral_1d(self):
        func = lambda x: np.exp(-(x**2))
        func_intg = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(-1))
        interval = RegularInterval(-1, 2, 2**5)
        y_vec = func_intg(interval.to_vector())
        expansion = ChebyshevExpansion.interpolate(
            func, interval.start, interval.stop, order=30
        )
        expansion_integ = expansion.integ(m=1, lbnd=interval.start)
        mps_cheb_clen = mps_polynomial_expansion(
            expansion_integ,
            initial=interval,
            clenshaw=True,
        )
        mps_cheb_poly = mps_polynomial_expansion(
            expansion_integ,
            initial=interval,
            clenshaw=False,
        )
        self.assertSimilar(y_vec, mps_cheb_clen.to_vector())
        self.assertSimilar(y_vec, mps_cheb_poly.to_vector())

    def test_gaussian_2d(self):
        func = lambda z: np.exp(-(z**2))
        sites = 6
        interval_x = RegularInterval(-0.5, 2, 2**sites)
        interval_y = RegularInterval(-0.5, 3, 2**sites)
        expansion = ChebyshevExpansion.interpolate(func, start=-1, stop=5, order=30)
        mps_x_plus_y = mps_tensor_sum(
            [mps_interval(interval_y), mps_interval(interval_x)]  # type: ignore
        )
        tol = 1e-10
        strategy = DEFAULT_STRATEGY.replace(
            tolerance=tol**2,
            simplification_tolerance=tol**2,
        )
        mps_cheb_clen = mps_polynomial_expansion(
            expansion,
            initial=mps_x_plus_y,
            strategy=strategy,
            clenshaw=True,
        )
        mps_cheb_poly = mps_polynomial_expansion(
            expansion,
            initial=mps_x_plus_y,
            strategy=strategy,
            clenshaw=False,
        )
        X, Y = np.meshgrid(interval_x.to_vector(), interval_y.to_vector())
        Z_vector = func(X + Y)
        Z_mps_clen = mps_cheb_clen.to_vector().reshape([2**sites, 2**sites])
        Z_mps_poly = mps_cheb_poly.to_vector().reshape([2**sites, 2**sites])
        self.assertSimilar(Z_vector, Z_mps_clen)
        self.assertSimilar(Z_vector, Z_mps_poly)

    def test_gaussian_mpo(self):
        a, b, n = -1, 1, 5
        dx = (b - a) / 2**n
        x = np.linspace(a, b, 2**n, endpoint=False)
        func = lambda x: np.sin(-(x**2))
        y_vec = func(x)

        expansion = ChebyshevExpansion.interpolate(func, a, b, order=30)
        I = MPS([np.ones((1, 2, 1))] * n)
        x_op = mpo_x(n, a, dx)
        mpo_gaussian_clen = mpo_polynomial_expansion(
            expansion,
            initial=x_op,
            clenshaw=True,
        )
        mpo_gaussian_poly = mpo_polynomial_expansion(
            expansion,
            initial=x_op,
            clenshaw=False,
        )
        self.assertSimilar(y_vec, mpo_gaussian_clen.apply(I).to_vector())
        self.assertSimilar(y_vec, mpo_gaussian_poly.apply(I).to_vector())


class TestLegendreExpansion(TestCase):
    pass
