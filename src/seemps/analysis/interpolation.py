from __future__ import annotations

import numpy as np
import copy
from math import sqrt

from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, MPSSum, Strategy
from ..operator import MPO
from ..qft import qft_mpo, mpo_qft_flip
from ..truncate import simplify
from .finite_differences import mpo_combined
from .space import Space


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


def mps_fourier_interpolation_1D(
    ψ0_mps: MPS,
    space: Space,
    n0: int,
    n1: int,
    dim: int,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> tuple[MPS, Space]:
    """
    Obtain the Fourier interpolated MPS over the chosen dimension with a new
    number of sites `n1`.

    Parameters
    ----------
    ψ0_mps: MPS
        Discretized multidimensional function MPS.
    space: Space
        Space object of the defined ψ0_mps.
    n0: int
        Initial number of sites.
    n1: int
        Final number of sites.
    dim: int
        Dimension to perform the interpolation.
    strategy : Strategy, optional
        Truncation strategy, defaults to DEFAULT_STRATEGY

    Returns
    -------
    ψ1_mps: MPS
        Interpolated MPS.
    new_space: Space
        New space of the interpolated MPS.
    """
    # Perform Fourier transform on ψ0_mps
    old_sites = space.sites
    u2c_op = space.extend(mpo_qft_flip(_twoscomplement(n0)), dim)
    qft_op = space.extend(qft_mpo(len(old_sites[dim]), sign=+1, strategy=strategy), dim)
    fourier_ψ0_mps = u2c_op @ (qft_op @ ψ0_mps)

    # Extend the state with zero qubits
    new_qubits_per_dimension = space.qubits_per_dimension.copy()
    Δn = n1 - n0
    new_qubits_per_dimension[dim] += Δn
    new_space = Space(new_qubits_per_dimension, space.L, space.closed)
    new_sites = new_space.sites
    idx_old_sites = new_sites.copy()
    idx_old_sites[dim] = list(
        np.append(idx_old_sites[dim][: (-Δn - 1)], idx_old_sites[dim][-1])
    )
    new_size = fourier_ψ0_mps.size + Δn
    fourier_ψ1_mps = fourier_ψ0_mps.extend(L=new_size, sites=sum(idx_old_sites, []))

    # Undo Fourier transform on fourier_ψ1_mps
    iqft_op = new_space.extend(
        mpo_qft_flip(qft_mpo(len(new_sites[dim]), sign=-1, strategy=strategy)), dim
    )
    u2c_op = new_space.extend(mpo_qft_flip(_twoscomplement(n1, strategy=strategy)), dim)
    ψ1_mps = iqft_op @ (u2c_op @ fourier_ψ1_mps)
    ψ1_mps = sqrt(2**Δn) * ψ1_mps
    if strategy.get_normalize_flag():
        ψ1_mps = ψ1_mps.normalize_inplace()
    return ψ1_mps, new_space


def mps_fourier_interpolation(
    ψ_mps: MPS,
    space: Space,
    old_sites: list,
    new_sites: list,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    """
    Fourier interpolation on an MPS.

    Parameters
    ----------
    ψ_mps : MPS
        Discretized multidimensional function MPS.
    space: Space
        Space object of the defined ψ_mps.
    old_sites : list[int]
        List of integers with the original number of sites for each dimension.
    new_sites : list[int]
        List of integers with the new number of sites for each dimension.
    strategy : Strategy, optional
            Truncation strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPS
        Interpolated multidimensional function MPS.
    """
    space = copy.copy(space)
    if not isinstance(ψ_mps, CanonicalMPS):
        ψ_mps = CanonicalMPS(ψ_mps, strategy=strategy)

    for i, sites in enumerate(new_sites):
        ψ_mps, space = mps_fourier_interpolation_1D(
            ψ_mps, space, old_sites[i], sites, dim=i, strategy=strategy
        )
    return ψ_mps


def mps_finite_differences_interpolation_1D(
    ψ0_mps: MPS,
    space: Space,
    dim: int = 0,
    strategy: Strategy = DEFAULT_STRATEGY,
    closed: bool = True,
    order: int = 1,
) -> tuple[CanonicalMPS, Space]:
    """
    Finite differences interpolation of dimension `dim` of an MPS representing
    a multidimensional function.

    Parameters
    ----------
    ψ0_mps : MPS
        MPS representing a multidimensional function.
    space : Space
        Space on which the function is defined.
    dim : int
        Dimension to perform the interpolation.
    strategy : Strategy, optional
        Truncation strategy, defaults to DEFAULT_STRATEGY.
    closed : bool, default=True
        Whether the space of definition of the function is open or closed.
    order : int, default=1
        The order for the finite difference interpolation.

    Returns
    -------
    MPS
        Interpolated MPS with one more site for the given dimension.
    """

    # Shift operator for finite difference formulas
    shift_op = space.extend(
        mpo_combined(len(space.sites[dim]), 0, 0, 1, closed=closed, strategy=strategy),
        dim,
    )

    # First order finite differences
    if order == 1:
        interpolated_points = simplify(
            MPSSum([0.5, 0.5], [ψ0_mps, shift_op @ ψ0_mps]), strategy=strategy
        )

    # Second order finite differences
    elif order == 2:
        f1 = ψ0_mps
        f2 = shift_op @ f1
        f3 = shift_op @ f2
        f0 = shift_op.T @ f1
        interpolated_points = simplify(
            MPSSum([-1 / 16, 9 / 16, 9 / 16, -1 / 16], [f0, f1, f2, f3]),
            strategy=strategy,
        )

    # Third order finite differences
    elif order == 3:
        shift_op_T = shift_op.T
        f2 = ψ0_mps
        f3 = shift_op @ f2
        f4 = shift_op @ f3
        f5 = shift_op @ f4
        f1 = shift_op_T @ f2
        f0 = shift_op_T @ f1
        interpolated_points = simplify(
            MPSSum(
                [-3 / 256, 21 / 256, -35 / 128, 105 / 128, 105 / 256, -7 / 256],
                [f0, f1, f2, f3, f4, f5],
            ),
            strategy=strategy,
        )

    else:
        raise ValueError("Invalid interpolation order")
    #
    # The new space representation with one more qubit
    new_space = space.enlarge_dimension(dim, 1)
    new_positions = new_space.new_positions_from_old_space(space)
    #
    # We create an MPS by extending the old one to the even sites
    # and placing the interpolating polynomials in an MPS that
    # is only nonzero in the odd sites. We then add. There are better
    # ways for sure.
    odd_mps = ψ0_mps.extend(
        L=new_space.n_sites,
        sites=new_positions,
        dimensions=2,
        state=np.asarray([1.0, 0.0]),
    )
    even_mps = interpolated_points.extend(
        L=new_space.n_sites,
        sites=new_positions,
        dimensions=2,
        state=np.asarray([0.0, 1.0]),
    )
    return simplify(odd_mps + even_mps, strategy=strategy), new_space


def mps_finite_differences_interpolation(
    ψ_mps: MPS, space: Space, strategy: Strategy = DEFAULT_STRATEGY
) -> CanonicalMPS:
    """
    Finite differences interpolation of an MPS representing
    a multidimensional function.

    Parameters
    ----------
    ψ0mps : MPS
        MPS representing a multidimensional function.
    space : Space
        Space on which the function is defined.
    strategy : Strategy, optional
            Truncation strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPS
        Interpolated MPS with one more site for each dimension.
    """
    space = copy.deepcopy(space)

    if not isinstance(ψ_mps, CanonicalMPS):
        ψ_mps = CanonicalMPS(ψ_mps, strategy=strategy)

    for i, _ in enumerate(space.qubits_per_dimension):
        ψ_mps, space = mps_finite_differences_interpolation_1D(
            ψ_mps, space, dim=i, strategy=strategy
        )
    return ψ_mps
