from __future__ import annotations

import numpy as np
from typing import TypeVar, Union, Optional

from ...state import Strategy, MPS, MPSSum, CanonicalMPS
from ...truncate import simplify
from ...typing import Tensor3, Matrix
from ..mesh import Interval, Mesh, RegularInterval, ChebyshevInterval


def mps_identity(sites: int, base: int = 2) -> MPS:
    I = np.ones((1, base, 1))
    return MPS([I] * sites)


def mps_identity_like(mps: MPS) -> MPS:
    return MPS([np.ones((1, s, 1)) for s in mps.physical_dimensions()])


def mps_equispaced(start: float, stop: float, sites: int) -> MPS:
    """
    Returns an MPS representing a discretized interval with equispaced points.

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.

    Returns
    -------
    MPS
        An MPS representing an equispaced discretization within [start, stop].
    """
    step = (stop - start) / 2**sites
    tensor_1 = np.zeros((1, 2, 2))
    tensor_1[0, :, :] = np.array([[[1, start], [1, start + step * 2 ** (sites - 1)]]])
    tensor_2 = np.zeros((2, 2, 1))
    tensor_2[:, :, 0] = np.array([[0, step], [1, 1]])
    tensors_bulk = [np.zeros((2, 2, 2)) for _ in range(sites - 2)]
    for idx, tensor in enumerate(tensors_bulk):
        tensor[0, :, 0] = np.ones(2)
        tensor[1, :, 1] = np.ones(2)
        tensor[0, 1, 1] = step * 2 ** (sites - (idx + 2))
    tensors = [tensor_1] + tensors_bulk + [tensor_2]
    return MPS(tensors)


def mps_exponential(start: float, stop: float, sites: int, c: complex = 1) -> MPS:
    """
    Returns an MPS representing an exponential function discretized over a
    half-open interval [start, stop).

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    c : complex, default=1
        The coefficient in the exponent of the exponential function.

    Returns
    -------
    MPS
        An MPS representing the discretized exponential function over the interval.
    """
    step = (stop - start) / 2**sites
    tensor_1 = np.zeros((1, 2, 1), dtype=complex)
    tensor_1[0, 0, 0] = np.exp(c * start)
    tensor_1[0, 1, 0] = np.exp(c * start + c * step * 2 ** (sites - 1))
    tensor_2 = np.zeros((1, 2, 1), dtype=complex)
    tensor_2[0, 0, 0] = 1
    tensor_2[0, 1, 0] = np.exp(c * step)
    tensors_bulk = [np.zeros((1, 2, 1), dtype=complex) for _ in range(sites - 2)]
    for idx, tensor in enumerate(tensors_bulk):
        tensor[0, 0, 0] = 1
        tensor[0, 1, 0] = np.exp(c * step * 2 ** (sites - (idx + 2)))
    tensors = [tensor_1] + tensors_bulk + [tensor_2]
    return MPS(tensors)


def mps_sin(start: float, stop: float, sites: int) -> MPS:
    """
    Returns an MPS representing a sine function discretized over a
    half-open interval [start, stop).

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy to apply.

    Returns
    -------
    MPS
        An MPS representing the discretized sine function over the interval.
    """
    mps_1 = mps_exponential(start, stop, sites, c=1j)
    mps_2 = mps_exponential(start, stop, sites, c=-1j)
    return -0.5j * (mps_1 - mps_2).join()


def mps_cos(start: float, stop: float, sites: int) -> MPS:
    """
    Returns an MPS representing a cosine function discretized over a
    half-open interval [start, stop).

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy to apply.

    Returns
    -------
    MPS
        An MPS representing the discretized cosine function over the interval.
    """
    mps_1 = mps_exponential(start, stop, sites, c=1j)
    mps_2 = mps_exponential(start, stop, sites, c=-1j)
    return 0.5 * (mps_1 + mps_2).join()


_State = TypeVar("_State", bound=Union[MPS, MPSSum])


def mps_step(
    start: float,
    stop: float,
    sites: int,
    c_x: float = 0.0,
    c_y: float = 0.5,
) -> MPS:
    """
    Returns an MPS representing the univariate Heaviside step function.

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    c_x : float, default=0.0
        The position of the discontinuity.
    c_y : float, default=0.5
        The value of the function at the discontinuity.
    """

    if not (c_x >= start and c_x <= stop):
        raise ValueError("c_x must be within [start, stop]")
    if not (c_y >= 0.0 and c_y <= 1.0):
        raise ValueError("c_y must be within [0, 1]")

    idx = int((2**sites - 1) * (c_x - start) / (stop - start))
    s = [(idx >> i) & 1 for i in range(sites)][::-1]

    tensor_L = np.zeros((1, 2, 2))
    tensor_L[0, s[0], 0] = 1
    tensor_L[0, (1 + s[0]) :, 1] = 1

    tensors_bulk = []
    for s_k in s[1:-1]:
        tensor = np.zeros((2, 2, 2))
        tensor[0, s_k, 0] = 1
        tensor[0, (1 + s_k) :, 1] = 1
        tensor[1, :, 1] = 1
        tensors_bulk.append(tensor)

    tensor_R = np.zeros((2, 2, 1))
    tensor_R[0, s[-1], 0] = c_y
    tensor_R[0, (1 + s[-1]) :, 0] = 1
    tensor_R[1, :, 0] = 1

    return MPS([tensor_L] + tensors_bulk + [tensor_R])


def mps_affine(mps: _State, orig: tuple, dest: tuple) -> _State:
    """
    Applies an affine transformation to an MPS, mapping it from one interval [x0, x1] to another [u0, u1].
    This is a transformation u = a * x + b, with u0 = a * x0 + b and and  u1 = a * x1 + b.
    Hence, a = (u1 - u0) / (x1 - x0) and b = ((u1 + u0) - a * (x0 + x1)) / 2.

    Parameters
    ----------
    mps : Union[MPS, MPSSum]
        The MPS to be transformed.
    orig : tuple
        A tuple (x0, x1) representing the original interval.
    dest : tuple
        A tuple (u0, u1) representing the destination interval.

    Returns
    -------
    mps_affine : Union[MPS, MPSSum]
        The MPS affinely transformed from (x0, x1) to (u0, u1).
    """
    x0, x1 = orig
    u0, u1 = dest
    a = (u1 - u0) / (x1 - x0)
    b = 0.5 * ((u1 + u0) - a * (x0 + x1))
    mps_affine = a * mps
    if abs(b) > np.finfo(np.float64).eps:
        physical_dimensions = (
            mps.states[0].physical_dimensions()
            if isinstance(mps, MPSSum)
            else mps.physical_dimensions()
        )
        I = MPS([np.ones((1, s, 1)) for s in physical_dimensions])
        mps_affine = mps_affine + b * I
        # Preserve the input type
        if isinstance(mps, MPS):
            return mps_affine.join()
    return mps_affine


def mps_interval(interval: Interval):
    """
    Returns an MPS corresponding to a specific type of interval.

    Parameters
    ----------
    interval : Interval
        The interval object containing start and stop points and the interval type.
        Currently supports `RegularInterval` and `ChebyshevInterval`.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy to apply.

    Returns
    -------
    MPS
        An MPS representing the interval according to its type.
    """
    start = interval.start
    stop = interval.stop
    sites = int(np.log2(interval.size))

    if isinstance(interval, RegularInterval):
        start_reg = start + interval.step if not interval.endpoint_left else start
        stop_reg = stop + interval.step if interval.endpoint_right else stop
        return mps_equispaced(start_reg, stop_reg, sites)

    elif isinstance(interval, ChebyshevInterval):
        if interval.endpoints is True:  # Extrema
            start_cheb = 0
            stop_cheb = np.pi + np.pi / (2**sites - 1)
        else:  # Zeros
            start_cheb = np.pi / (2 ** (sites + 1))
            stop_cheb = np.pi + start_cheb
        return mps_affine(
            mps_cos(start_cheb, stop_cheb, sites),
            (1, -1),  # Reverse order
            (start, stop),
        )
    else:
        raise ValueError(f"Unsupported interval type {type(interval)}")


def tt_quadratic(Q: Matrix, mesh: Mesh) -> MPS:
    """
    Returns an MPS corresponding to the quadratic form of matrix Q on `mesh`.
    This represents the sum :math:`\\sum_{ij} Q_{ij} x_i x_j` evaluated over every
    possible vector `x_i`.

    Parameters
    ----------
    Q : Matrix
        A symmetric matrix representing the quadratic form.
        The shape of Q must be (m, m), where 'm' is the dimension of the mesh (number of intervals).
    mesh : Mesh
        A `Mesh` object that defines the domain over which the quadratic form is evaluated.

    Returns
    -------
    MPS
        An MPS with physical dimensions (N_1, N_2, ..., N_m), where 'N_k' is the size of the k-th interval.
    """
    m = mesh.dimension
    if not (m == Q.shape[0] == Q.shape[1]):
        raise ValueError("Dimensions don't match.")

    # Left tensor
    x_L = mesh.intervals[0].to_vector()
    A_L = np.zeros((1, len(x_L), 3))
    A_L[0, :, 0] = Q[0, 0] * x_L**2  # "Channel 0": accumulates the sum
    A_L[0, :, 1] = x_L  # "Channel k": carries the value of the k-th core
    A_L[0, :, 2] = 1

    # Bulk tensors
    bulk_tensors = []
    for k, interval in enumerate(mesh.intervals[1:-1], start=1):
        x_k = interval.to_vector()
        A_k = np.zeros((k + 2, len(x_k), k + 3))
        A_k[0, :, 0] = 1
        for i in range(1, k + 1):
            A_k[i, :, 0] = (Q[i - 1, k] + Q[k, i - 1]) * x_k
            A_k[i, :, i] = 1
        A_k[k + 1, :, 0] = Q[k, k] * x_k**2
        A_k[k + 1, :, k + 1] = x_k
        A_k[k + 1, :, k + 2] = 1
        bulk_tensors.append(A_k)

    # Right tensor
    x_R = mesh.intervals[-1].to_vector()
    A_R = np.zeros((m + 1, len(x_R), 1))
    A_R[0, :, 0] = 1
    for i in range(1, m):
        A_R[i, :, 0] = (Q[i - 1, m - 1] + Q[m - 1, i - 1]) * x_R
    A_R[m, :, 0] = Q[m - 1, m - 1] * x_R**2

    return MPS([A_L] + bulk_tensors + [A_R])


def qtt_quadratic(Q: Matrix, mesh: Mesh) -> MPS:
    """
    Returns an MPS analogous to `tt_quadratic` but with quantized physical dimensions.
    The quantization procedure requires every interval in the mesh to be regular and
    have a size that is a power of 2.

    Parameters
    ----------
    Q : np.ndarray
        A symmetric matrix representing the quadratic form.
        The shape of Q must be (m, m), where 'm' is the dimension of the mesh (number of intervals).
    mesh : Mesh
        A `Mesh` object that defines the domain over which the quadratic form is evaluated.
        Each interval's size in the mesh must be a power of 2.

    Returns
    -------
    MPS
        An MPS with physical dimensions (2, 2, ..., 2).
    """
    # TODO: Implement in other qubit orders
    m = mesh.dimension
    if not (m == Q.shape[0] == Q.shape[1]):
        raise ValueError("Dimensions don't match.")

    for interval in mesh.intervals:
        if interval.size & (interval.size - 1) != 0:
            raise ValueError("All intervals must have a size that is a power of 2.")
        if not isinstance(interval, RegularInterval):
            raise ValueError("All intervals must be regular.")

    # Expand Q into a block matrix with the appropriate size
    num_qubits = [int(np.log2(interval.size)) for interval in mesh.intervals]
    blocks = []
    for i in range(m):
        row_blocks = []
        for j in range(m):
            row_blocks.append(Q[i, j] * np.ones((num_qubits[i], num_qubits[j])))
        blocks.append(np.block(row_blocks))
    Q_quantized = np.block(blocks)

    # Quantize each interval of a regular mesh into binary intervals
    quantized_intervals = []
    for interval in mesh.intervals:
        n = int(np.log2(interval.size))
        a, b = interval.start, interval.stop
        for k in range(1, n + 1):
            x_min = a / n
            x_max = x_min + (b - a) * (2 ** (-k))
            quantized_interval = interval.update_size(2).map_to(x_min, x_max)
            quantized_intervals.append(quantized_interval)
    mesh_quantized = Mesh(quantized_intervals)

    # Compute the quadratic form MPS
    return tt_quadratic(Q_quantized, mesh_quantized)


def _map_mps_locations(
    mps_list: list[MPS], mps_order: str
) -> list[tuple[int, Tensor3]]:
    """Create a vector that lists which MPS and which tensor is
    associated to which position in the joint Hilbert space.
    """
    tensors = [(0, mps_list[0][0])] * sum(len(mps) for mps in mps_list)
    if mps_order == "A":
        k = 0
        for mps_id, mps in enumerate(mps_list):
            for i, Ai in enumerate(mps):
                tensors[k] = (mps_id, Ai)  # type: ignore
                k += 1
    elif mps_order == "B":
        k = 0
        i = 0
        while k < len(tensors):
            for mps_id, mps in enumerate(mps_list):
                if i < mps.size:
                    tensors[k] = (mps_id, mps[i])  # type: ignore
                    k += 1
            i += 1
    else:
        raise ValueError(f"Invalid mps order {mps_order}")
    return tensors  # type: ignore


def _mps_tensor_terms(mps_list: list[MPS], mps_order: str) -> list[MPS]:
    """
    Extends each MPS of a given input list by appending identity tensors to it according
    to the specified MPS order ('A' or 'B'). The resulting list of MPS can be given as terms
    to a tensorized operation between MPS, such as a tensor product or tensor sum.

    Parameters
    ----------
    mps_list : list[MPS]
        The MPS input list.
    mps_order : str
        The order in which to arrange the qubits for each resulting MPS term ('A' or 'B').

    Returns
    -------
    list[MPS]
        The resulting list of MPS terms.
    """

    def extend_mps(mps_id: int, mps_map: list[tuple[int, Tensor3]]) -> MPS:
        D = 1
        output = [mps_map[0][1]] * len(mps_map)
        for k, (site_mps, site_tensor) in enumerate(mps_map):
            if mps_id == site_mps:
                output[k] = site_tensor
                D = site_tensor.shape[-1]
            else:
                site_dimension = site_tensor.shape[1]
                output[k] = np.eye(D).reshape(D, 1, D) * np.ones((1, site_dimension, 1))
        return MPS(output)

    mps_map = _map_mps_locations(mps_list, mps_order)
    return [extend_mps(mps_id, mps_map) for mps_id, _ in enumerate(mps_list)]


def mps_tensor_product(
    mps_list: list[MPS],
    mps_order: str = "A",
    strategy: Optional[Strategy] = None,
    simplify_steps: bool = False,
) -> Union[MPS, CanonicalMPS]:
    """
    Returns the tensor product of a list of MPS, with the sites arranged
    according to the specified MPS order.

    Parameters
    ----------
    mps_list : list[MPS]
        The list of MPS objects to multiply.
    mps_order : str
        The order in which to arrange the resulting MPS ('A' or 'B').
    strategy : Strategy, optional
        The strategy to use when multiplying the MPS. If None, the tensor product is not simplified.
    simplify_steps : bool, default=False
        Whether to simplify the intermediate steps with `strategy` (if provided)
        or simplify at the end.

    Returns
    -------
    result : MPS | CanonicalMPS
        The resulting MPS from the tensor product of the input list.
    """
    if mps_order == "A":
        nested_sites = [mps._data for mps in mps_list]
        flattened_sites = [site for sites in nested_sites for site in sites]
        result = MPS(flattened_sites)
    elif mps_order == "B":
        terms = _mps_tensor_terms(mps_list, mps_order)
        result = terms[0]
        for _, mps in enumerate(terms[1:]):
            result = (
                simplify(result * mps, strategy=strategy)
                if (strategy and simplify_steps)
                else result * mps
            )
    else:
        raise ValueError(f"Invalid mps order {mps_order}")
    if strategy and not simplify_steps:
        result = simplify(result, strategy=strategy)
    return result


def mps_tensor_sum(
    mps_list: list[MPS],
    mps_order: str = "A",
    strategy: Optional[Strategy] = None,
    simplify_steps: bool = False,
) -> Union[MPS, CanonicalMPS]:
    """
    Returns the tensor sum of a list of MPS, with the sites arranged
    according to the specified MPS order.

    Parameters
    ----------
    mps_list : list[MPS]
        The list of MPS objects to sum.
    mps_order : str, default='A'
        The order in which to arrange the resulting MPS ('A' or 'B').
    strategy : Strategy, optional
        The strategy to use when summing the MPS. If None, the tensor sum is not simplified.
    simplify_steps : bool, default=False
        Whether to simplify the intermediate steps with `strategy` (if provided)
        or simplify at the end.

    Returns
    -------
    result : MPS | CanonicalMPS
        The resulting MPS from the tensor sum of the input list.
    """
    if mps_order == "A":
        result = _mps_tensor_sum_serial_order(mps_list)
    elif mps_order == "B":
        terms = _mps_tensor_terms(mps_list, mps_order)
        result = terms[0]
        for _, mps in enumerate(terms[1:]):
            result = (
                simplify(result + mps, strategy=strategy)
                if (strategy and simplify_steps)
                else (result + mps).join()
            )
    else:
        raise ValueError(f"Invalid mps order {mps_order}")
    if strategy and not simplify_steps:
        result = simplify(result, strategy=strategy)
    return result


def _mps_tensor_sum_serial_order(mps_list: list[MPS]) -> MPS:
    """
    Computes the MPS tensor sum in serial order in an optimized manner.
    """

    def extend_tensor(A: Tensor3, first: bool, last: bool) -> Tensor3:
        a, d, b = A.shape
        output = np.zeros((a + 2, d, b + 2), dtype=A.dtype)
        output[0, :, 0] = np.ones(d)  # No MPS applied
        output[1, :, 1] = np.ones(d)  # One MPS applied
        if first:
            if last:
                output[[0], :, [1]] = A
            else:
                output[[0], :, 2:] = A
        elif last:
            output[2:, :, [1]] = A
        else:
            output[2:, :, 2:] = A
        return output

    output = [
        extend_tensor(Ai, i == 0, i == len(A) - 1)
        for _, A in enumerate(mps_list)
        for i, Ai in enumerate(A)
    ]
    output[0] = output[0][[0], :, :]
    output[-1] = output[-1][:, :, [1]]
    return MPS(output)
