import numpy as np
import scipy.linalg
import functools

# Typing
from numpy.typing import NDArray
from typing import TypeAlias
from ...typing import Vector, Tensor3

from ...state import Strategy
from ...state._contractions import _contract_last_and_first
from ...state.schmidt import svd
from ...state.core import destructively_truncate_vector

Matrix: TypeAlias = NDArray


def combine_indices(*indices: Matrix, fortran: bool = False) -> Matrix:

    def cartesian(A: Matrix, B: Matrix) -> Matrix:
        A_repeated = np.repeat(A, repeats=B.shape[0], axis=0)
        B_tiled = np.tile(B, (A.shape[0], 1))
        return np.hstack((A_repeated, B_tiled))

    def cartesian_fortran(A: Matrix, B: Matrix) -> Matrix:
        A_tiled = np.tile(A, (B.shape[0], 1))
        B_repeated = np.repeat(B, repeats=A.shape[0], axis=0)
        return np.hstack((A_tiled, B_repeated))

    if fortran:
        return functools.reduce(cartesian_fortran, indices)
    else:
        return functools.reduce(cartesian, indices)


def points_to_indices(points: Matrix) -> tuple[Matrix, Matrix]:
    if points.ndim == 1:
        points = points.reshape(1, -1)
    sites = points.shape[1]
    I_l = [points[:, :k] for k in range(sites)]
    I_g = [points[:, (k + 1) : sites] for k in range(sites)]
    return I_l, I_g


def fiber_to_Q3R(fiber: Tensor3) -> tuple[Tensor3, Matrix]:
    r_l, r_s, r_g = fiber.shape
    Q, R = scipy.linalg.qr(fiber.reshape(r_l * r_s, r_g), mode="economic")
    Q3 = Q.reshape(r_l, r_s, r_g)
    return Q3, R


def Q3_to_core(Q3: Tensor3, row_indices: Vector) -> Tensor3:
    r_l, r_s, r_g = Q3.shape
    Q = Q3.reshape(r_l * r_s, r_g)
    P = scipy.linalg.inv(Q[row_indices])
    G = _contract_last_and_first(Q, P)
    return G.reshape(r_l, r_s, r_g)


def non_destructive_svd(A: Matrix, strategy: Strategy) -> tuple[Matrix, Matrix, Matrix]:
    U, S, V = svd(A)
    destructively_truncate_vector(S, strategy)
    r = S.size
    return U[:, :r], np.diag(S), V[:r, :]


def choose_maxvol(
    A: Matrix,
    rank_kick: tuple,
    max_iter: int,
    tol: float,
    tol_rect: float,
) -> tuple[Matrix, Matrix]:
    n, r = A.shape
    min_kick, max_kick = rank_kick
    max_kick = min(max_kick, n - r)
    min_kick = min(min_kick, max_kick)
    if n <= r:
        I, B = np.arange(n, dtype=int), np.eye(n)
    elif rank_kick == 0:
        I, B = maxvol_square(A, max_iter, tol)
    else:
        I, B = maxvol_rectangular(A, (min_kick, max_kick), max_iter, tol, tol_rect)
    return I, B


def maxvol_square(
    A: Matrix,
    max_iter: int,
    tol: float,
) -> tuple[Matrix, Matrix]:
    n, r = A.shape

    if n <= r:
        I, B = np.arange(n, dtype=int), np.eye(n)
        return I, B

    P, L, U = scipy.linalg.lu(A)
    I = P[:, :r].argmax(axis=0)
    Q = scipy.linalg.solve_triangular(U, A.T, trans=1)
    B = scipy.linalg.solve_triangular(
        L[:r, :], Q, trans=1, unit_diagonal=True, lower=True
    ).T

    for _ in range(max_iter):
        i, j = np.divmod(abs(B).argmax(), r)
        if abs(B[i, j]) <= tol:
            break
        I[j] = i
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])

    return I, B


def maxvol_rectangular(
    A: Matrix,
    rank_kick: tuple,
    max_iter: int,
    tol: float,
    tol_rect: float,
) -> tuple[Matrix, Matrix]:
    n, r = A.shape
    min_rank = r + rank_kick[0]
    max_rank = min(r + rank_kick[1], n)
    if min_rank < r or min_rank > max_rank or max_rank > n:
        raise ValueError("Invalid rank_kick")

    I0, B = maxvol_square(A, max_iter, tol)
    I = np.hstack([I0, np.zeros(max_rank - r, dtype=I0.dtype)])
    S = np.ones(n, dtype=int)
    S[I0] = 0
    F = S * np.linalg.norm(B) ** 2

    for k in range(r, max_rank):
        i = np.argmax(F)
        if k >= min_rank and F[i] <= tol_rect**2:
            break
        I[k] = i
        S[i] = 0
        v = B.dot(B[i])
        l = 1.0 / (1 + v[i])
        B = np.hstack([B - l * np.outer(v, B[i]), l * v.reshape(-1, 1)])
        F = S * (F - l * v * v)
    I = I[: B.shape[1]]
    B[I] = np.eye(B.shape[1], dtype=B.dtype)

    return I, B
