from __future__ import annotations

import numpy as np
from typing import Union

from ...state import Strategy, DEFAULT_STRATEGY
from ...operator import MPO, MPOList, MPOSum


def mpo_identity(n: int, strategy=DEFAULT_STRATEGY) -> MPO:
    """Return the identity MPO with `n` qubits."""
    B = np.zeros((1, 2, 2, 1))
    B[0, 0, 0, 0] = 1
    B[0, 1, 1, 0] = 1
    return MPO([B for _ in range(n)], strategy=strategy)


def mpo_x(n: int, a: float, dx: float, strategy=DEFAULT_STRATEGY) -> MPO:
    """
    Returns the MPO for the x operator.

    Parameters:
    ----------
    n: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        x operator MPO.
    """
    MPO_x = []

    if n == 1:
        B = np.zeros((1, 2, 2, 1))
        B[0, 0, 0, 0] = a
        B[0, 1, 1, 0] = a + dx
        MPO_x.append(B)
        return MPO(MPO_x, strategy=strategy)
    else:
        for i in range(n):
            if i == 0:
                Bi = np.zeros((1, 2, 2, 2))
                Bi[0, 0, 0, 0] = 1
                Bi[0, 1, 1, 0] = 1
                Bi[0, 0, 0, 1] = a
                Bi[0, 1, 1, 1] = a + dx * 2 ** (n - 1)
                MPO_x.append(Bi)
            elif i == n - 1:
                Bf = np.zeros((2, 2, 2, 1))
                Bf[1, 0, 0, 0] = 1
                Bf[1, 1, 1, 0] = 1
                Bf[0, 1, 1, 0] = dx
                MPO_x.append(Bf)
            else:
                B = np.zeros((2, 2, 2, 2))
                B[0, 0, 0, 0] = 1
                B[0, 1, 1, 0] = 1
                B[1, 0, 0, 1] = 1
                B[1, 1, 1, 1] = 1
                B[0, 1, 1, 1] = dx * 2 ** (n - 1 - i)
                MPO_x.append(B)

        return MPO(MPO_x, strategy=strategy)


def mpo_x_to_n(
    n: int, a: float, dx: float, m: int, strategy: Strategy = DEFAULT_STRATEGY
) -> MPO:
    """
    Returns the MPO for the x^m operator.

    Parameters:
    ----------
    n: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    m: int
        Order of the x polynomial.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        x^m operator MPO.
    """
    return MPOList([mpo_x(n, a, dx) for _ in range(m)]).join(strategy=strategy)


def mpo_p(n: int, dx: float, strategy: Strategy = DEFAULT_STRATEGY) -> MPO:
    """
    Returns the MPO for the p operator.

    Parameters:
    ----------
    n: int
        Number of qubits.
    dx: float
        Spacing of the position interval.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        p operator MPO.
    """
    dk = 2 * np.pi / (dx * 2**n)
    MPO_p = []

    if n == 1:
        B = np.zeros((1, 2, 2, 1))
        B[0, 1, 1, 0] = dk * (1 - 2**n)
        MPO_p.append(B)
        return MPO(MPO_p, strategy=strategy)

    for i in range(n):
        if i == 0:
            Bi = np.zeros((1, 2, 2, 2))
            Bi[0, 0, 0, 0] = 1
            Bi[0, 1, 1, 0] = 1
            Bi[0, 1, 1, 1] = dk * 2 ** (n - 1) - dk * 2**n
            MPO_p.append(Bi)
        elif i == n - 1:
            Bf = np.zeros((2, 2, 2, 1))
            Bf[1, 0, 0, 0] = 1
            Bf[1, 1, 1, 0] = 1
            Bf[0, 1, 1, 0] = dk
            MPO_p.append(Bf)
        else:
            B = np.zeros((2, 2, 2, 2))
            B[0, 0, 0, 0] = 1
            B[0, 1, 1, 0] = 1
            B[1, 0, 0, 1] = 1
            B[1, 1, 1, 1] = 1
            B[0, 1, 1, 1] = dk * 2 ** (n - 1 - i)
            MPO_p.append(B)

    return MPO(MPO_p, strategy=strategy)


def mpo_p_to_n(n: int, dx: float, m: int, strategy: Strategy = DEFAULT_STRATEGY) -> MPO:
    """
    Returns the MPO for the p^m operator.

    Parameters:
    ----------
    n_qubits: int
        Number of qubits.
    dx: float
        Spacing of the position interval.
    n: int
        Order of the x polynomial.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        p^m operator MPO.
    """
    return MPOList([mpo_p(n, dx) for _ in range(m)]).join(strategy=strategy)


def mpo_exponential(
    n: int,
    a: float,
    dx: float,
    c: Union[float, complex] = 1,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """
    Returns the MPO for the exp(cx) operator.

    Parameters:
    ----------
    n: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    c: float | complex, default = 1
        Constant preceeding the x coordinate.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        exp(x) operator MPO.
    """
    MPO_exp = []
    if n == 1:
        B = np.zeros((1, 2, 2, 1), complex)
        B[0, 0, 0, 0] = np.exp(c * a)
        B[0, 1, 1, 0] = np.exp(c * (a + dx))
        MPO_exp.append(B)
        return MPO(MPO_exp, strategy=strategy)
    else:
        for i in range(n):
            if i == 0:
                Bi = np.zeros((1, 2, 2, 1), complex)
                Bi[0, 0, 0, 0] = np.exp(c * (a))
                Bi[0, 1, 1, 0] = np.exp(c * (a + dx * 2 ** (n - 1)))
                MPO_exp.append(Bi)
            elif i == n - 1:
                Bf = np.zeros((1, 2, 2, 1), complex)
                Bf[0, 0, 0, 0] = 1
                Bf[0, 1, 1, 0] = np.exp(c * dx)
                MPO_exp.append(Bf)
            else:
                B = np.zeros((1, 2, 2, 1), complex)
                B[0, 0, 0, 0] = 1
                B[0, 1, 1, 0] = np.exp(c * dx * 2 ** (n - 1 - i))
                MPO_exp.append(B)

        return MPO(MPO_exp, strategy=strategy)


def mpo_cos(n: int, a: float, dx: float, strategy=DEFAULT_STRATEGY) -> MPO:
    """
    Returns the MPO for the cos(x) operator.

    Parameters:
    ----------
    n: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        cos(x) operator MPO.
    """
    exp1 = mpo_exponential(n, a, dx, c=+1j, strategy=strategy)
    exp2 = mpo_exponential(n, a, dx, c=-1j, strategy=strategy)
    cos_mpo = 0.5 * (exp1 + exp2)
    return cos_mpo.join(strategy=strategy)


def mpo_sin(n: int, a: float, dx: float, strategy=DEFAULT_STRATEGY) -> MPO:
    """
    Returns the MPO for the sin(x) operator.

    Parameters:
    ----------
    n: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        sin(x) operator MPO.
    """
    exp1 = mpo_exponential(n, a, dx, c=+1j, strategy=strategy)
    exp2 = mpo_exponential(n, a, dx, c=-1j, strategy=strategy)
    sin_mpo = (-1j) * 0.5 * (exp1 - exp2)
    return sin_mpo.join(strategy=strategy)


def mpo_affine(
    mpo: MPO,
    orig: tuple,
    dest: tuple,
) -> MPO:
    """Performs an affine transformation of the given MPO from `orig` to `dest`."""
    x0, x1 = orig
    u0, u1 = dest
    a = (u1 - u0) / (x1 - x0)
    b = 0.5 * ((u1 + u0) - a * (x0 + x1))
    mpo_affine = a * mpo
    if abs(b) > np.finfo(np.float64).eps:
        I = MPO([np.ones((1, 2, 2, 1))] * len(mpo_affine))
        mpo_affine = MPOSum(mpos=[mpo_affine, I], weights=[1, b]).join()
    return mpo_affine
