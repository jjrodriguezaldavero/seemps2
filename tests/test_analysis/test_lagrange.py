import numpy as np

from seemps.analysis.mesh import RegularInterval, Mesh
from seemps.analysis.lagrange import (
    lagrange_basic,
    lagrange_rank_revealing,
    lagrange_local_rank_revealing,
)

from .tools_analysis import reorder_tensor
from ..tools import TestCase


def gaussian_setup(dim: int):
    start, stop = -2, 2
    sites = 6
    interval = RegularInterval(start, stop, 2**sites)
    if dim == 1:
        func = lambda x: np.exp(-(x**2))
        domain = interval
    elif dim > 1:
        func = lambda tensor: np.exp(-np.sum(tensor**2, axis=0))
        domain = Mesh([interval] * dim)
    return func, domain


class TestLagrangeBasic(TestCase):
    def test_gaussian_1d(self):
        func, interval = gaussian_setup(1)
        mps = lagrange_basic(func, interval, 20)
        Z_exact = func(interval.to_vector())
        Z_test = mps.to_vector()
        self.assertSimilar(Z_exact, Z_test)

    def test_gaussian_2d_serial(self):
        func, mesh = gaussian_setup(2)
        mps = lagrange_basic(func, mesh, 20, interleave=False)
        Z_exact = func(mesh.to_tensor(True))
        Z_test = mps.to_vector().reshape(mesh.dimensions)
        self.assertSimilar(Z_exact, Z_test)

    def test_gaussian_2d_interleaved(self):
        func, mesh = gaussian_setup(2)
        mps = lagrange_basic(func, mesh, 20, interleave=True)
        Z_exact = func(mesh.to_tensor(True))
        Z_test = mps.to_vector().reshape(mesh.dimensions)
        n = int(np.log2(mesh.dimensions[0]))
        Z_test = reorder_tensor(Z_test, [n, n])
        self.assertSimilar(Z_exact, Z_test)


class TestLagrangeRankRevealing(TestCase):
    def test_gaussian_1d(self):
        func, interval = gaussian_setup(1)
        mps = lagrange_rank_revealing(func, interval, 20)
        Z_exact = func(interval.to_vector())
        Z_test = mps.to_vector()
        self.assertSimilar(Z_exact, Z_test)

    def test_gaussian_2d_serial(self):
        func, mesh = gaussian_setup(2)
        mps = lagrange_rank_revealing(func, mesh, 20, interleave=False)
        Z_exact = func(mesh.to_tensor(True))
        Z_test = mps.to_vector().reshape(mesh.dimensions)
        self.assertSimilar(Z_exact, Z_test)

    def test_gaussian_2d_interleaved(self):
        func, mesh = gaussian_setup(2)
        mps = lagrange_rank_revealing(func, mesh, 20, interleave=True)
        Z_exact = func(mesh.to_tensor(True))
        Z_test = mps.to_vector().reshape(mesh.dimensions)
        n = int(np.log2(mesh.dimensions[0]))
        Z_test = reorder_tensor(Z_test, [n, n])
        self.assertSimilar(Z_exact, Z_test)


class TestLagrangeLocalRankRevealing(TestCase):
    def test_gaussian_1d(self):
        func, interval = gaussian_setup(1)
        mps = lagrange_local_rank_revealing(func, interval, 20, 10)
        Z_exact = func(interval.to_vector())
        Z_test = mps.to_vector()
        self.assertSimilar(Z_exact, Z_test)

    def test_gaussian_2d_serial(self):
        func, mesh = gaussian_setup(2)
        mps = lagrange_local_rank_revealing(func, mesh, 20, 10, interleave=False)
        Z_exact = func(mesh.to_tensor(True))
        Z_test = mps.to_vector().reshape(mesh.dimensions)
        self.assertSimilar(Z_exact, Z_test)

    def test_gaussian_2d_interleaved(self):
        func, mesh = gaussian_setup(2)
        mps = lagrange_local_rank_revealing(func, mesh, 30, 10, interleave=True)
        Z_exact = func(mesh.to_tensor(True))
        Z_test = mps.to_vector().reshape(mesh.dimensions)
        n = int(np.log2(mesh.dimensions[0]))
        Z_test = reorder_tensor(Z_test, [n, n])
        self.assertSimilar(Z_exact, Z_test)
