import numpy as np

from seemps.state import MPS, scprod
from seemps.analysis.factories import mps_exponential, mps_tensor_product
from seemps.analysis.mesh import (
    Mesh,
    RegularInterval,
    ChebyshevInterval,
    mps_to_mesh_matrix,
)
from seemps.analysis.integration import (
    mesh_to_quadrature_mesh,
    quadrature_mesh_to_mps,
    vector_trapezoidal,
    vector_simpson38,
    vector_fifth_order,
    vector_fejer,
    mps_trapezoidal,
    mps_simpson38,
    mps_fifth_order,
    mps_fejer,
)

from ..tools import TestCase


class TestQuadBinaryMPS(TestCase):
    def test_trapezoidal(self):
        a, b, n = -1, 1, 4
        quad_vector = vector_trapezoidal(a, b, 2**n)
        quad_mps = mps_trapezoidal(a, b, n)
        self.assertSimilar(quad_vector, quad_mps.to_vector())

    def test_simpson(self):
        a, b, n = -1, 1, 4
        quad_vector = vector_simpson38(a, b, 2**n)
        quad_mps = mps_simpson38(a, b, n)
        self.assertSimilar(quad_vector, quad_mps.to_vector())

    def test_fifth_order(self):
        a, b, n = -1, 1, 4
        quad_vector = vector_fifth_order(a, b, 2**n)
        quad_mps = mps_fifth_order(a, b, n)
        self.assertSimilar(quad_vector, quad_mps.to_vector())

    def test_fejer(self):
        a, b, n = -1, 1, 4
        quad_vector = vector_fejer(a, b, 2**n)
        quad_mps = mps_fejer(a, b, n)
        self.assertSimilar(quad_vector, quad_mps.to_vector())


def setup_integral(n: int, m: int):
    a, b = -2, 2
    h = (b - a) / (2**n - 1)
    func = lambda x: np.exp(x)
    integral = (func(b) - func(a)) ** m
    return a, b, h, func, integral


class TestIntegrals(TestCase):
    def test_trapezoidal(self):
        n, m = 12, 1
        a, b, h, _, intg_exact = setup_integral(n, m)
        mps_func = mps_exponential(a, b + h, n)
        mps_quad = mps_trapezoidal(a, b, n)
        intg_test = scprod(mps_func, mps_quad)
        self.assertAlmostEqual(intg_exact, intg_test, places=5)

    def test_simpson(self):
        n, m = 10, 1
        a, b, h, _, intg_exact = setup_integral(n, m)
        mps_func = mps_exponential(a, b + h, n)
        mps_quad = mps_simpson38(a, b, n)
        intg_test = scprod(mps_func, mps_quad)
        self.assertAlmostEqual(intg_exact, intg_test, places=5)

    def test_fifth_order(self):
        n, m = 8, 1
        a, b, h, _, intg_exact = setup_integral(n, m)
        mps_func = mps_exponential(a, b + h, n)
        mps_quad = mps_fifth_order(a, b, n)
        intg_test = scprod(mps_func, mps_quad)
        self.assertAlmostEqual(intg_exact, intg_test, places=5)

    def test_fejer(self):
        n, m = 5, 1
        a, b, _, func, intg_exact = setup_integral(n, m)
        interval = ChebyshevInterval(a, b, 2**n)
        x = interval.to_vector()
        mps_func = MPS.from_vector(func(x), [2] * n, normalize=False)
        mps_quad = mps_fejer(a, b, n)
        intg_test = scprod(mps_func, mps_quad)
        self.assertAlmostEqual(intg_exact, intg_test)


class TestMultivariateIntegrals(TestCase):
    def test_newton_cotes_5d(self):
        n, m = 4, 5
        a, b, h, _, intg_exact = setup_integral(n, m)
        interval = RegularInterval(a, b, 2**n)
        mesh = Mesh([interval] * m)
        quad_mesh = mesh_to_quadrature_mesh(mesh)

        map_matrix = mps_to_mesh_matrix([n] * m)
        physical_dimensions = [2] * (n * m)
        mps_quad = quadrature_mesh_to_mps(quad_mesh, map_matrix, physical_dimensions)

        mps_func_1d = mps_exponential(a, b + h, n)
        mps_func = mps_tensor_product([mps_func_1d] * m)
        intg_test = scprod(mps_quad, mps_func)

        relative_error = abs(intg_exact - intg_test) / abs(intg_exact)
        self.assertLess(relative_error, 1e-5)
