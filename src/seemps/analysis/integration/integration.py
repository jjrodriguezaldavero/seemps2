from __future__ import annotations
import numpy as np

from ...state import MPS, scprod
from ...typing import Matrix
from ..cross import cross_interpolation, CrossStrategyMaxvol, BlackBoxLoadMPS
from ..factories import mps_tensor_product
from ..mesh import (
    Interval,
    RegularInterval,
    ChebyshevInterval,
    Mesh
)
from .mps_quadratures import mps_trapezoidal, mps_simpson38, mps_fifth_order, mps_clenshaw_curtis, mps_fejer
from .vector_quadratures import (
    vector_best_newton_cotes,
    vector_clenshaw_curtis,
    vector_fejer,
)


def integrate_mps(mps: MPS, domain: Interval | Mesh, mps_order: str = "A") -> complex:
    """
    Returns the integral of a multivariate function represented as a MPS.
    Uses the 'best possible' quadrature rule according to the intervals that compose the mesh.
    Intervals of type `RegularInterval` employ high-order Newton-Côtes rules, while
    those of type `ChebyshevInterval` employ Clenshaw-Curtis or Fejer rules.

    Parameters
    ----------
    mps : MPS
        The MPS representation of the multivariate function to be integrated.
    domain : Interval | Mesh
        An object defining the discretization domain of the function.
        Can be either an `Interval` or a `Mesh` given by a collection of intervals.
        The quadrature rules are selected based on the properties of these intervals.
    mps_order : str, default='A'
        Specifies the ordering of the qubits in the quadrature. Possible values:.
        - 'A': Qubits are serially ordered (by variable).
        - 'B': Qubits are interleaved (by significance).

    Returns
    -------
    complex
        The integral of the MPS representation of the function discretized in the given Mesh.

    Notes
    -----
    - This algorithm assumes that all function variables are in the standard MPS form, i.e.
    quantized in base 2, are discretized either on a `RegularInterval or `ChebyshevInterval`,
    and are quantized either in serial or interleaved order.

    - For more general structures, the quadrature MPS can be constructed using the univariate 
    quadrature rules and the `mps_tensor_product` routine, which can be subsequently contracted 
    using the `scprod` routine. Equivalently, tensor cross-interpolation can be used to construct 
    the quadrature rule using the `quadrature_mesh_to_mps` routine.

    Examples
    --------
    .. code-block:: python

        # Integrate a given bivariate function using the Clenshaw-Curtis quadrature.
        # Assuming that the MPS is already loaded (for example, using TT-Cross or Chebyshev).
        mps_function_2d = ...

        # Define a domain that matches the MPS to integrate.
        start, stop = -1, 1
        n_qubits = 10
        interval = ChebyshevInterval(-1, 1, 2**n_qubits, endpoints=True)
        mesh = Mesh([interval, interval])

        # Integrate the MPS on the given discretization domain.
        integral = integrate_mps(mps_function_2d, mesh)
    """
    mesh = domain if isinstance(domain, Mesh) else Mesh([domain])
    quads = []
    for interval in mesh.intervals:
        a, b, N = interval.start, interval.stop, interval.size
        n = int(np.log2(N))
        if isinstance(interval, RegularInterval):
            if n % 4 == 0:
                quads.append(mps_fifth_order(a, b, n))
            elif n % 2 == 0:
                quads.append(mps_simpson38(a, b, n))
            else:
                quads.append(mps_trapezoidal(a, b, n))
        elif isinstance(interval, ChebyshevInterval):
            if interval.endpoints:
                quads.append(mps_clenshaw_curtis(a, b, n))
            else:
                quads.append(mps_fejer(a, b, n))
        else:
            raise ValueError("Invalid interval in mesh")
    mps_quad = quads[0] if len(quads) == 1 else mps_tensor_product(quads, mps_order)
    return scprod(mps, mps_quad)


def mesh_to_quadrature_mesh(mesh: Mesh) -> Mesh:
    """
    Returns a `Mesh` composed of quadrature vectors corresponding to the best
    quadrature rules of each of the `Interval` objects of the given mesh.
    Can be used to construct the multidimensional quadrature space in any quantization 
    or qubit ordering using tensor cross-interpolation with the `quadrature_mesh_to_mps` routine.

    Note: any additional arbitrary vector or quadrature rule can be appended to the mesh 
    afterward if required.
    """
    quads = []
    for interval in mesh.intervals:
        start, stop, size = interval.start, interval.stop, interval.size

        if isinstance(interval, RegularInterval):
            quads.append(vector_best_newton_cotes(start, stop, size))
        elif isinstance(interval, ChebyshevInterval):
            if interval.endpoints:
                quads.append(vector_clenshaw_curtis(start, stop, size))
            else:
                quads.append(vector_fejer(start, stop, size))
        else:
            raise ValueError("Invalid Interval")

    return Mesh(quads)


def quadrature_mesh_to_mps(
    quadrature_mesh: Mesh,
    map_matrix: Matrix | None = None,
    physical_dimensions: list | None = None,
    cross_strategy: CrossStrategyMaxvol = CrossStrategyMaxvol(),
    **kwargs,
) -> MPS:
    """
    Constructs the MPS representation of the given multidimensional quadrature mesh
    composed of quadrature rules or arbitrary vectors in any qubit arrangement or
    quantization. Employs tensor cross-interpolation with the given strategy. 
    """
    black_box = BlackBoxLoadMPS(
        lambda q: np.prod(q, axis=0),
        quadrature_mesh,
        map_matrix,
        physical_dimensions,
    )
    return cross_interpolation(black_box, cross_strategy, **kwargs).mps