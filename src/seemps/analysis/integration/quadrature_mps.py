import numpy as np
from typing import Optional

from ...state import MPS, Strategy, DEFAULT_STRATEGY
from ...qft import iqft, qft_flip
from ...typing import Matrix
from ..cross import (
    CrossStrategy,
    BlackBoxMesh,
    CrossStrategyMaxvol,
    cross_interpolation,
)
from ..factories import mps_affine
from ..mesh import (
    IntegerInterval,
    RegularInterval,
    ChebyshevInterval,
    Mesh,
    mps_to_mesh_matrix,
)
from .quadrature_vector import (
    vector_best_newton_cotes,
    vector_clenshaw_curtis,
    vector_fejer,
)


def quadrature_mesh_to_mps(
    quadrature_mesh: Mesh,
    map_matrix: Optional[Matrix] = None,
    physical_dimensions: Optional[list] = None,
    cross_strategy: CrossStrategy = CrossStrategyMaxvol(),
    **kwargs
) -> MPS:
    """
    Constructs a multivariate quadrature MPS from a `Mesh` object encapsulating
    quadrature vectors using tensor cross-interpolation.
    """
    black_box = BlackBoxMesh(
        lambda q: np.prod(q, axis=0),
        quadrature_mesh,
        map_matrix,
        physical_dimensions,
    )
    return cross_interpolation(black_box, cross_strategy, **kwargs).mps


def mesh_to_quadrature_mesh(mesh: Mesh) -> Mesh:
    """
    Generates a quadrature mesh by transforming each interval of a `Mesh` object
    into its correspondent quadrature vector.
    """
    quad_vectors = []
    for interval in mesh.intervals:
        start, stop, size = interval.start, interval.stop, interval.size

        if isinstance(interval, RegularInterval):
            quad_vectors.append(vector_best_newton_cotes(start, stop, size))
        elif isinstance(interval, ChebyshevInterval):
            if interval.endpoints:
                quad_vectors.append(vector_clenshaw_curtis(start, stop, size))
            else:
                quad_vectors.append(vector_fejer(start, stop, size))
        else:
            raise ValueError("Invalid Interval")

    return Mesh(quad_vectors)


def mps_trapezoidal(start: float, stop: float, sites: int) -> MPS:
    step = (stop - start) / (2**sites - 1)

    tensor_L = np.zeros((1, 2, 3))
    tensor_L[0, 0, 0] = 1
    tensor_L[0, 1, 1] = 1
    tensor_L[0, 0, 2] = 1
    tensor_L[0, 1, 2] = 1

    tensor_C = np.zeros((3, 2, 3))
    tensor_C[0, 0, 0] = 1
    tensor_C[1, 1, 1] = 1
    tensor_C[2, 0, 2] = 1
    tensor_C[2, 1, 2] = 1

    tensor_R = np.zeros((3, 2, 1))
    tensor_R[0, 0, 0] = -0.5
    tensor_R[1, 1, 0] = -0.5
    tensor_R[2, 0, 0] = 1
    tensor_R[2, 1, 0] = 1

    tensors = [tensor_L] + [tensor_C for _ in range(sites - 2)] + [tensor_R]
    return step * MPS(tensors)


def mps_simpson38(start: float, stop: float, sites: int) -> MPS:
    if sites % 2 != 0:
        raise ValueError("The number of sites must be even.")

    step = (stop - start) / (2**sites - 1)

    tensor_L1 = np.zeros((1, 2, 4))
    tensor_L1[0, 0, 0] = 1
    tensor_L1[0, 1, 1] = 1
    tensor_L1[0, 0, 2] = 1
    tensor_L1[0, 1, 3] = 1

    if sites == 2:
        tensor_R = np.zeros((4, 2, 1))
        tensor_R[0, 0, 0] = -1
        tensor_R[1, 1, 0] = -1
        tensor_R[2, 0, 0] = 2
        tensor_R[2, 1, 0] = 3
        tensor_R[3, 0, 0] = 3
        tensor_R[3, 1, 0] = 2
        tensors = [tensor_L1, tensor_R]
    else:
        tensor_L2 = np.zeros((4, 2, 5))
        tensor_L2[0, 0, 0] = 1
        tensor_L2[1, 1, 1] = 1
        tensor_L2[2, 0, 2] = 1
        tensor_L2[2, 1, 3] = 1
        tensor_L2[3, 0, 4] = 1
        tensor_L2[3, 1, 2] = 1

        tensor_C = np.zeros((5, 2, 5))
        tensor_C[0, 0, 0] = 1
        tensor_C[1, 1, 1] = 1
        tensor_C[2, 0, 2] = 1
        tensor_C[2, 1, 3] = 1
        tensor_C[3, 0, 4] = 1
        tensor_C[3, 1, 2] = 1
        tensor_C[4, 0, 3] = 1
        tensor_C[4, 1, 4] = 1

        tensor_R = np.zeros((5, 2, 1))
        tensor_R[0, 0, 0] = -1
        tensor_R[1, 1, 0] = -1
        tensor_R[2, 0, 0] = 2
        tensor_R[2, 1, 0] = 3
        tensor_R[3, 0, 0] = 3
        tensor_R[3, 1, 0] = 2
        tensor_R[4, 0, 0] = 3
        tensor_R[4, 1, 0] = 3

        tensors = (
            [tensor_L1, tensor_L2] + [tensor_C for _ in range(sites - 3)] + [tensor_R]
        )

    return (3 * step / 8) * MPS(tensors)


def mps_fifth_order(start: float, stop: float, sites: int) -> MPS:
    if sites % 4 != 0:
        raise ValueError("The number of sites must be divisible by 4.")

    step = (stop - start) / (2**sites - 1)

    tensor_L1 = np.zeros((1, 2, 4))
    tensor_L1[0, 0, 0] = 1
    tensor_L1[0, 1, 1] = 1
    tensor_L1[0, 0, 2] = 1
    tensor_L1[0, 1, 3] = 1

    tensor_L2 = np.zeros((4, 2, 6))
    tensor_L2[0, 0, 0] = 1
    tensor_L2[1, 1, 1] = 1
    tensor_L2[2, 0, 2] = 1
    tensor_L2[2, 1, 3] = 1
    tensor_L2[3, 0, 4] = 1
    tensor_L2[3, 1, 5] = 1

    tensor_L3 = np.zeros((6, 2, 7))
    tensor_L3[0, 0, 0] = 1
    tensor_L3[1, 1, 1] = 1
    tensor_L3[2, 0, 2] = 1
    tensor_L3[2, 1, 3] = 1
    tensor_L3[3, 0, 4] = 1
    tensor_L3[3, 1, 5] = 1
    tensor_L3[4, 0, 6] = 1
    tensor_L3[4, 1, 2] = 1
    tensor_L3[5, 0, 3] = 1
    tensor_L3[5, 1, 4] = 1

    tensor_C = np.zeros((7, 2, 7))
    tensor_C[0, 0, 0] = 1
    tensor_C[1, 1, 1] = 1
    tensor_C[2, 0, 2] = 1
    tensor_C[2, 1, 3] = 1
    tensor_C[3, 0, 4] = 1
    tensor_C[3, 1, 5] = 1
    tensor_C[4, 0, 6] = 1
    tensor_C[4, 1, 2] = 1
    tensor_C[5, 0, 3] = 1
    tensor_C[5, 1, 4] = 1
    tensor_C[6, 0, 5] = 1
    tensor_C[6, 1, 6] = 1

    tensor_R = np.zeros((7, 2, 1))
    tensor_R[0, 0, 0] = -19
    tensor_R[1, 1, 0] = -19
    tensor_R[2, 0, 0] = 38
    tensor_R[2, 1, 0] = 75
    tensor_R[3, 0, 0] = 50
    tensor_R[3, 1, 0] = 50
    tensor_R[4, 0, 0] = 75
    tensor_R[4, 1, 0] = 38
    tensor_R[5, 0, 0] = 75
    tensor_R[5, 1, 0] = 50
    tensor_R[6, 0, 0] = 50
    tensor_R[6, 1, 0] = 75

    tensors = (
        [tensor_L1, tensor_L2, tensor_L3]
        + [tensor_C for _ in range(sites - 4)]
        + [tensor_R]
    )
    return (5 * step / 288) * MPS(tensors)


def mps_best_newton_cotes(start: float, stop: float, sites: int) -> MPS:
    if sites % 4 == 0:
        return mps_fifth_order(start, stop, sites)
    elif sites % 2 == 0:
        return mps_simpson38(start, stop, sites)
    else:
        return mps_trapezoidal(start, stop, sites)


def mps_fejer(
    start: float,
    stop: float,
    sites: int,
    qft_strategy: Strategy = DEFAULT_STRATEGY,
    cross_strategy: CrossStrategy = CrossStrategyMaxvol(),
) -> MPS:
    N = int(2**sites)

    # Encode 1/(1 - 4*k**2) term with TCI
    func = lambda k: np.where(k < N / 2, 2 / (1 - 4 * k**2), 2 / (1 - 4 * (N - k) ** 2))
    black_box = BlackBoxMesh(
        func,
        mesh=Mesh([IntegerInterval(0, N)]),
        map_matrix=mps_to_mesh_matrix([sites]),
        physical_dimensions=[2] * sites,
    )
    mps_k2 = cross_interpolation(black_box, cross_strategy).mps

    # Encode phase term analytically
    pref = 1j * np.pi / N
    expn = pref * N / 2

    tensor_L = np.zeros((1, 2, 5), dtype=complex)
    tensor_L[0, 0, 0] = 1
    tensor_L[0, 1, 1] = np.exp(-expn)
    tensor_L[0, 1, 2] = np.exp(expn)
    tensor_L[0, 1, 3] = -np.exp(-expn)
    tensor_L[0, 1, 4] = -np.exp(expn)

    tensor_R = np.zeros((5, 2, 1), dtype=complex)
    tensor_R[0, 0, 0] = 1
    tensor_R[0, 1, 0] = np.exp(pref)
    tensor_R[1, 0, 0] = 1
    tensor_R[1, 1, 0] = np.exp(pref)
    tensor_R[2, 0, 0] = 1
    tensor_R[3, 0, 0] = 1
    tensor_R[4, 0, 0] = 1

    tensors_C = [np.zeros((5, 2, 5), dtype=complex) for _ in range(sites - 2)]
    for idx, tensor_C in enumerate(tensors_C):
        expn = pref * 2 ** (sites - (idx + 2))
        tensor_C[0, 0, 0] = 1
        tensor_C[0, 1, 0] = np.exp(expn)
        tensor_C[1, 0, 1] = 1
        tensor_C[1, 1, 1] = np.exp(expn)
        tensor_C[2, 0, 2] = 1
        tensor_C[3, 0, 3] = 1
        tensor_C[4, 0, 4] = 1

    tensors = [tensor_L] + tensors_C + [tensor_R]
    mps_phase = MPS(tensors)

    # Encode Fejér quadrature with iQFT
    mps = (1 / np.sqrt(2) ** sites) * qft_flip(
        iqft(mps_k2 * mps_phase, strategy=qft_strategy)
    )

    return mps_affine(mps, (-1, 1), (start, stop))
