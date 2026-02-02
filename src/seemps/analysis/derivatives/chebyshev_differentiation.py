from ...state import MPS, MPSSum, Strategy, DEFAULT_STRATEGY
from ...operators import MPOList
from ..mesh import ChebyshevInterval

# TODO: Implement


def chebyshev_derivative_mpo(
    order: int,
    interval: ChebyshevInterval,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPOList:
    """Nth-order derivative operator via Chebyshev spectral differentiation.

    Construct the `order`-th spatial derivative operator acting on (non-periodic) functions
    represented on a Chebyshev grid (either Chebyshev-Gauss or Chebyshev-Lobatto), using the
    Chebyshev polynomial recurrence relations. This allows Fourier-like spectral differentiation
    without requiring periodicity.

    The operator accounts for the affine rescaling between the domain of definition of the function
    and the domain of orthogonality :math:`[-1, 1]` of the Chebyshev basis.

    Parameters
    ----------
    order : int
        Order of the derivative.
    interval : ChebyshevInterval
        Physical interval over the Chebyshev nodes in which the function is defined.
    strategy : Strategy, optional
        Truncation strategy for the MPO.

    Returns
    -------
    MPOList
        Operator implementing the spectral Chebyshev derivative.
    """
    raise NotImplementedError()


def chebyshev_derivative(
    psi: MPS | MPSSum,
    order: int,
    interval: ChebyshevInterval,
) -> MPS | MPSSum:
    """Nth-order derivative via Quantum Chebyshev Transform.

    Compute the `order`-th spatial derivative of a quantum state `psi`
    represented on a Chebyshev grid, using spectral Chebyshev differentiation.

    Parameters
    ----------
    psi : MPS or MPSSum
        Quantum state in position representation.
    order : int
        Order of the derivative.
    interval : ChebyshevInterval
        Physical interval over the Chebyshev nodes in which the function is defined.

    Returns
    -------
    d_psi : MPS or MPSSum
        Quantum state encoding the derivative.
    """
    raise NotImplementedError()
