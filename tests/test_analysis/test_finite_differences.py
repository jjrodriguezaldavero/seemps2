import numpy as np
from scipy.sparse import spdiags, csr_matrix, eye
from seemps.analysis.finite_differences import (
    mpo_finite_differences,
    mpo_smooth_finite_differences,
)
from seemps.analysis.space import Space
from seemps.state import MPS, NO_TRUNCATION

from .. import tools


def gaussian(x):
    f = np.exp(-(x**2))
    return f / np.linalg.norm(f)


def S_plus_v(n, closed=True):
    S = np.diag(np.ones(2**n - 1), +1)
    if closed:
        S[-1, 0] = 1
    return S


def S_minus_v(n, closed=True):
    S = np.diag(np.ones(2**n - 1), -1)
    if closed:
        S[0, -1] = 1
    return S


def finite_differences_v(n, Δx, closed=True):
    return (1 / Δx**2) * (
        S_plus_v(n, closed=closed) + S_minus_v(n, closed=closed) - 2 * np.eye(2**n)
    )


class TestFiniteDifferences(tools.TestCase):
    def test_finite_differences(self):
        for n in range(2, 10):
            qubits_per_dimension = [n]
            L = 10
            space = Space(qubits_per_dimension, L=[[-L / 2, L / 2]])
            x = space.x[0]
            Δx = space.dx[0]
            v = gaussian(x)
            fd_sol = finite_differences_v(n, Δx, closed=True) @ v
            ψ = MPS.from_vector(v, [2] * n, normalize=False, strategy=NO_TRUNCATION)
            fd_mps_sol = (
                mpo_finite_differences(n, Δx, closed=True, strategy=NO_TRUNCATION) @ ψ
            )
            self.assertSimilar(fd_sol, fd_mps_sol)


class TestFiniteDifferences(tools.TestCase):
    def Down(self, nqubits: int, periodic: bool = False) -> csr_matrix:
        """Moves f[i] to f[i-1]"""
        L = 2**nqubits
        if periodic:
            M = spdiags([np.ones(L), np.ones(L)], [1, -L + 1], format="csr")
        else:
            M = spdiags([np.ones(L)], [1], format="csr")
        return M

    def Up(self, nqubits: int, periodic: bool = False) -> csr_matrix:
        """Moves f[i] to f[i+1]"""
        return self.Down(nqubits, periodic).T

    def test_first_derivative_two_qubits_perodic(self):
        dx = 0.1
        D2 = mpo_smooth_finite_differences(
            2, order=1, filter=3, periodic=True, dx=dx
        ).to_matrix()
        self.assertSimilar(
            D2,
            [
                [0, 0.5 / dx, 0, -0.5 / dx],
                [-0.5 / dx, 0, 0.5 / dx, 0],
                [0, -0.5 / dx, 0, 0.5 / dx],
                [0.5 / dx, 0, -0.5 / dx, 0],
            ],
        )
        D = self.Down(2, periodic=True)
        self.assertSimilar(D2, (D - D.T) / (2.0 * dx))

    def test_second_derivative_two_qubits_perodic(self):
        dx = 0.1
        D2 = mpo_smooth_finite_differences(
            2, order=2, filter=3, periodic=True, dx=dx
        ).to_matrix()
        dx2 = dx * dx
        self.assertSimilar(
            D2,
            [
                [-2 / dx2, 1 / dx2, 0, 1 / dx2],
                [1 / dx2, -2 / dx2, 1 / dx2, 0],
                [0, 1 / dx2, -2 / dx2, 1 / dx2],
                [1 / dx2, 0, 1 / dx2, -2 / dx2],
            ],
        )
        D = self.Down(2, periodic=True)
        I = eye(4)
        self.assertSimilar(D2, (D - 2 * I + D.T) / dx2)

    def test_first_derivative_two_qubits_non_perodic(self):
        dx = 0.1
        D2 = mpo_smooth_finite_differences(
            2, order=1, filter=3, periodic=False, dx=dx
        ).to_matrix()
        self.assertSimilar(
            D2,
            [
                [0, 0.5 / dx, 0, 0],
                [-0.5 / dx, 0, 0.5 / dx, 0],
                [0, -0.5 / dx, 0, 0.5 / dx],
                [0, 0, -0.5 / dx, 0],
            ],
        )
        D = self.Down(2, periodic=False)
        self.assertSimilar(D2, (D - D.T) / (2.0 * dx))

    def test_second_derivative_two_qubits_perodic(self):
        dx = 0.1
        D2 = mpo_smooth_finite_differences(
            2, order=2, filter=3, periodic=False, dx=dx
        ).to_matrix()
        dx2 = dx * dx
        self.assertSimilar(
            D2,
            [
                [-2 / dx2, 1 / dx2, 0, 0],
                [1 / dx2, -2 / dx2, 1 / dx2, 0],
                [0, 1 / dx2, -2 / dx2, 1 / dx2],
                [0, 0, 1 / dx2, -2 / dx2],
            ],
        )
        D = self.Down(2, periodic=False)
        I = eye(4)
        self.assertSimilar(D2, (D - 2 * I + D.T) / dx2)

    def test_second_derivative_two_qubits_perodic(self):
        dx = 0.1
        D2 = mpo_smooth_finite_differences(
            2, order=2, filter=3, periodic=False, dx=dx
        ).to_matrix()
        dx2 = dx * dx
        self.assertSimilar(
            D2,
            [
                [-2 / dx2, 1 / dx2, 0, 0],
                [1 / dx2, -2 / dx2, 1 / dx2, 0],
                [0, 1 / dx2, -2 / dx2, 1 / dx2],
                [0, 0, 1 / dx2, -2 / dx2],
            ],
        )
        D = self.Down(2, periodic=False)
        I = eye(4)
        self.assertSimilar(D2, (D - 2 * I + D.T) / dx2)

    def test_second_derivative_two_qubits_smooth_and_ordinary(self):
        dx = 1 / 4.0
        for nqubits in range(2, 10):
            for periodic in [False, True]:
                D2a = mpo_smooth_finite_differences(
                    nqubits, order=2, filter=3, periodic=periodic, dx=dx
                )
                D2b = mpo_finite_differences(nqubits, dx, closed=periodic)
                self.assertSimilar(D2a.to_matrix(), D2b.to_matrix())
