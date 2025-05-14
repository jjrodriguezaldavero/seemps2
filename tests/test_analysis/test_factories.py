import numpy as np
from seemps.state import NO_TRUNCATION
from seemps.analysis.factories import (
    mps_equispaced,
    mps_exponential,
    mps_sin,
    mps_cos,
    mps_interval,
    mps_tensor_sum,
    mps_tensor_product,
)
from seemps.analysis.mesh import RegularInterval, ChebyshevInterval
from seemps.analysis.factories.mpo_factories import *

from ..tools import TestCase, random_uniform_mps
from .tools_analysis import reorder_tensor


class TestMPSFactories(TestCase):
    def test_mps_equispaced(self):
        self.assertSimilar(
            mps_equispaced(-1, 1, 5).to_vector(),
            np.linspace(-1, 1, 2**5, endpoint=False),
        )

    def test_mps_exponential(self):
        self.assertSimilar(
            mps_exponential(-1, 1, 5, c=1).to_vector(),
            np.exp(np.linspace(-1, 1, 2**5, endpoint=False)),
        )
        self.assertSimilar(
            mps_exponential(-1, 1, 5, c=-1).to_vector(),
            np.exp(-np.linspace(-1, 1, 2**5, endpoint=False)),
        )

    def test_mps_sin(self):
        self.assertSimilar(
            mps_sin(-1, 1, 5).to_vector(),
            np.sin(np.linspace(-1, 1, 2**5, endpoint=False)),
        )

    def test_mps_cos(self):
        self.assertSimilar(
            mps_cos(-1, 1, 5).to_vector(),
            np.cos(np.linspace(-1, 1, 2**5, endpoint=False)),
        )

    def test_mps_interval(self):
        start, stop, sites = -1, 1, 5
        N = 2**sites
        mps_half_open = mps_interval(RegularInterval(start, stop, N))
        mps_closed = mps_interval(RegularInterval(start, stop, N, endpoint_right=True))

        mps_zeros = mps_interval(ChebyshevInterval(start, stop, N))
        mps_extrema = mps_interval(ChebyshevInterval(start, stop, N, endpoints=True))
        zeros = np.flip(
            [np.cos(np.pi * (2 * k + 1) / (2 * N)) for k in np.arange(0, N)]
        )
        extrema = np.flip([np.cos(np.pi * k / (N - 1)) for k in np.arange(0, N)])

        self.assertSimilar(mps_half_open, np.linspace(start, stop, N, endpoint=False))
        self.assertSimilar(mps_closed, np.linspace(start, stop, N, endpoint=True))
        self.assertSimilar(mps_zeros, zeros)
        self.assertSimilar(mps_extrema, extrema)


class TestMPOFactories(TestCase):
    n = 6
    N = 2**n
    L = 10
    a = -L / 2
    dx = L / N
    x = a + dx * np.arange(N)
    k = 2 * np.pi * np.arange(N) / L
    p = k - (np.arange(N) >= (N / 2)) * 2 * np.pi / dx
    f = random_uniform_mps(2, n)

    def test_mpo_identity(self):
        self.assertSimilar(self.f, mpo_identity(self.n) @ self.f)

    def test_mpo_x(self):
        self.assertSimilar(
            self.x * self.f.to_vector(), mpo_x(self.n, self.a, self.dx) @ self.f
        )

    def test_mpo_x_n(self):
        n = 3
        self.assertSimilar(
            self.x**n * self.f.to_vector(),
            (mpo_x(self.n, self.a, self.dx) ** n) @ self.f,
        )

    def test_mpo_p(self):
        self.assertSimilar(self.p * self.f.to_vector(), mpo_p(self.n, self.dx) @ self.f)

    def test_mpo_p_n(self):
        n = 3
        self.assertSimilar(
            self.p**n * self.f.to_vector(),
            (mpo_p(self.n, self.dx) ** n) @ self.f,
        )

    def test_mpo_exponential(self):
        c = -2j
        self.assertSimilar(
            np.exp(c * self.x) * self.f.to_vector(),
            mpo_exponential(self.n, self.a, self.dx, c) @ self.f,
        )

    def test_mpo_cos(self):
        self.assertSimilar(
            np.cos(self.x) * self.f.to_vector(),
            mpo_cos(self.n, self.a, self.dx) @ self.f,
        )

    def test_mpo_sin(self):
        self.assertSimilar(
            np.sin(self.x) * self.f.to_vector(),
            mpo_sin(self.n, self.a, self.dx) @ self.f,
        )


class TestMPSOperations(TestCase):
    def test_tensor_product(self):
        sites = 5
        interval = RegularInterval(-1, 2, 2**sites)
        mps_x = mps_interval(interval)
        X, Y = np.meshgrid(interval.to_vector(), interval.to_vector())
        # Order A
        mps_x_times_y_A = mps_tensor_product([mps_x, mps_x], mps_order="A")
        Z_mps_A = mps_x_times_y_A.to_vector().reshape((2**sites, 2**sites))
        self.assertSimilar(Z_mps_A, X * Y)
        # Order B
        mps_x_times_y_B = mps_tensor_product([mps_x, mps_x], mps_order="B")
        Z_mps_B = mps_x_times_y_B.to_vector().reshape((2**sites, 2**sites))
        Z_mps_B = reorder_tensor(Z_mps_B, [sites, sites])
        self.assertSimilar(Z_mps_B, X * Y)

    def test_tensor_sum(self):
        sites = 5
        interval = RegularInterval(-1, 2, 2**sites)
        mps_x = mps_interval(interval)
        X, Y = np.meshgrid(interval.to_vector(), interval.to_vector())
        # Order A
        mps_x_plus_y_A = mps_tensor_sum([mps_x, mps_x], mps_order="A")
        Z_mps_A = mps_x_plus_y_A.to_vector().reshape((2**sites, 2**sites))
        self.assertSimilar(Z_mps_A, X + Y)
        # Order B
        mps_x_plus_y_B = mps_tensor_sum([mps_x, mps_x], mps_order="B")
        Z_mps_B = mps_x_plus_y_B.to_vector().reshape((2**sites, 2**sites))
        Z_mps_B = reorder_tensor(Z_mps_B, [sites, sites])
        self.assertSimilar(Z_mps_B, X + Y)

    def test_tensor_sum_one_site(self):
        A = self.random_mps([2, 3, 4])
        self.assertSimilar(
            mps_tensor_sum([A], mps_order="A").to_vector(), A.to_vector()
        )

    def test_tensor_sum_small_size_A_order(self):
        A = self.random_mps([2, 3, 4])
        B = self.random_mps([5, 3, 2])
        AB = mps_tensor_sum([A, B], mps_order="A", strategy=NO_TRUNCATION)
        A_v = A.to_vector()
        B_v = B.to_vector()
        self.assertSimilar(
            AB.to_vector(),
            np.kron(A.to_vector(), np.ones(B_v.shape))
            + np.kron(np.ones(A_v.shape), B.to_vector()),
        )

    def test_tensor_sum_small_size_B_order(self):
        A = self.random_mps([2, 3, 4])
        B = self.random_mps([2, 3, 4])
        AB = mps_tensor_sum([A, B], mps_order="B", strategy=NO_TRUNCATION)
        AB_t = AB.to_vector().reshape([2, 2, 3, 3, 4, 4])
        AB_v = AB_t.transpose([0, 2, 4, 1, 3, 5]).reshape(-1)
        A_v = A.to_vector()
        B_v = B.to_vector()
        self.assertSimilar(
            AB_v,
            np.kron(A.to_vector(), np.ones(B_v.shape))
            + np.kron(np.ones(A_v.shape), B.to_vector()),
        )

    def test_tensor_sum_small_size_B_order_different_sizes(self):
        A = self.random_mps([2, 3, 4])
        B = self.random_mps([6, 3, 4])
        AB = mps_tensor_sum([A, B], mps_order="B", strategy=NO_TRUNCATION)
        AB_t = AB.to_vector().reshape([2, 6, 3, 3, 4, 4])
        AB_v = AB_t.transpose([0, 2, 4, 1, 3, 5]).reshape(-1)
        A_v = A.to_vector()
        B_v = B.to_vector()
        self.assertSimilar(
            AB_v,
            np.kron(A.to_vector(), np.ones(B_v.shape))
            + np.kron(np.ones(A_v.shape), B.to_vector()),
        )
