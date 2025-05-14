from __future__ import annotations

import numpy as np

from ..state import MPS, MPSSum, CanonicalMPS, Strategy, DEFAULT_STRATEGY
from ..operator import MPO
from ..truncate import simplify


def mps_fourier_interpolation(
    initial_mps: MPS,
    sites_per_dimension: tuple[int, ...],
    target_sites_per_dimension: tuple[int, ...],
    mps_order: str = "A",
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    _check_sites(initial_mps, sites_per_dimension, target_sites_per_dimension)
    mps = (
        CanonicalMPS(mps, strategy=strategy)
        if not isinstance(initial_mps, CanonicalMPS)
        else initial_mps
    )

    # 1. Fourier transform the initial MPS
    qft_mpo = None
    two_mpo = None
    fourier_mps = two_mpo @ (qft_mpo @ initial_mps)

    # 2. Extend the state with zero-qubits
    fourier_mps_ext = fourier_mps.extend()

    # 3. Invert the Fourier transform
    iqft_mpo = None
    itwo_mpo = None
    final_mps = iqft_mpo @ (itwo_mpo @ fourier_mps_ext)
    if strategy.get_normalize_flag():
        final_mps = final_mps.normalize_inplace()

    return final_mps


def mps_fd_interpolation(
    initial_mps: MPS,
    sites_per_dimension: tuple[int, ...],
    target_sites_per_dimension: tuple[int, ...],
    mps_order: str = "A",
    fd_order: int = 1,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    _check_sites(initial_mps, sites_per_dimension, target_sites_per_dimension)
    mps = (
        CanonicalMPS(mps, strategy=strategy)
        if not isinstance(initial_mps, CanonicalMPS)
        else initial_mps
    )

    # 1. Construct the shift operator
    shift_mpo = None

    # 2. Construct the interpolated MPS using FD formulas
    if fd_order == 1:
        weights = np.array([1, 1]) / 2
        states = [initial_mps, shift_mpo @ initial_mps]

    elif fd_order == 2:
        weights = np.array([-1, 9, 9, -1]) / 16
        f1 = initial_mps
        f2 = shift_mpo @ f1
        f3 = shift_mpo @ f2
        f0 = shift_mpo.T @ f1
        states = [f0, f1, f2, f3]

    elif fd_order == 3:
        weights = np.array([-3, 21, -70, 210, 105, -7]) / 256
        shift_mpo_T = shift_mpo.T
        f2 = initial_mps
        f1 = shift_mpo_T @ f2
        f0 = shift_mpo_T @ f1
        f3 = shift_mpo @ f2
        f4 = shift_mpo @ f3
        f5 = shift_mpo @ f4
        states = [f0, f1, f2, f3, f4, f5]

    else:
        raise ValueError("Invalid interpolation order.")

    interp_mps = simplify(MPSSum(weights, states), strategy=strategy)

    # 3. Combine both MPS on odd and even points, respectively.
    odd_mps = None
    even_mps = None
    return simplify(odd_mps + even_mps, strategy=strategy)


def _check_sites(mps: MPS, sites: tuple[int, ...], target_sites: tuple[int, ...]):
    if np.sum(sites) != len(mps):
        raise ValueError("The number of sites must match the MPS length.")
    if len(sites) != len(target_sites):
        raise ValueError("Sites and target sites must have the same dimension.")
    for s, t in zip(sites, target_sites):
        if not t >= s:
            raise ValueError("Target sites must be increasing.")
        if not (isinstance(s, int) and isinstance(t, int)):
            raise ValueError("Sites must be integers.")


def _twoscomplement(L: int, **kwdargs) -> MPO:
    """Two's complement operation."""
    A0 = np.zeros((1, 2, 2, 2))
    A0[0, 0, 0, 0] = 1.0
    A0[0, 1, 1, 1] = 1.0

    A = np.zeros((2, 2, 2, 2))
    A[0, 0, 0, 0] = 1.0
    A[0, 1, 1, 0] = 1.0
    A[1, 1, 0, 1] = 1.0
    A[1, 0, 1, 1] = 1.0

    Aend = A[:, :, :, [0]] + A[:, :, :, [1]]
    return MPO([A0] + [A] * (L - 2) + [Aend], **kwdargs)


# No tiene sentido plantear la interpolación de Fourier en 1D
# para una funcion multidimensional pudiendo hacer directamente
# la interpolación en varias dimensiones.

# def mps_fourier_interpolation_1d(
#     initial_mps: MPS,
#     sites_per_dimension: list[int],
#     dim: int,
#     new_sites: tuple[int, int],
#     strategy: Strategy = DEFAULT_STRATEGY,
# ) -> MPS:

#     # 1. Fourier transform the initial MPS
#     qft_mpo = None
#     two_mpo = None
#     fourier_mps = two_mpo @ (qft_mpo @ initial_mps)

#     # 2. Extend the state with zero-qubits
#     fourier_mps_ext = fourier_mps.extend()

#     # 3. Invert the Fourier transform
#     iqft_mpo = None
#     itwo_mpo = None
#     final_mps = np.sqrt(2**Δn) * iqft_mpo @ (itwo_mpo @ fourier_mps_ext)
#     if strategy.get_normalize_flag():
#         final_mps = final_mps.normalize_inplace()

#     return final_mps
