from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from abc import ABC, abstractmethod

from ...state import MPS, MPSSum, CanonicalMPS, Strategy, DEFAULT_STRATEGY
from ...operator import MPO, MPOList, MPOSum
from ...truncate import simplify, simplify_mpo
from ...typing import Vector
from ...tools import make_logger
from ..mesh import Interval
from ..factories import mps_interval, mps_affine, mpo_affine


ScalarFunction = Callable[[float], float]


class PolynomialExpansion(ABC):
    basis_domain: tuple[float, float]

    def __init__(self, coeffs: Vector, domain: tuple[float, float]):
        self.coeffs = coeffs
        self.domain = domain

    @abstractmethod
    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        """
        Return the three-term recurrence coefficients (α_k, β_k, γ_k) for
        P_{k+1}(x) = (α_k x + β_k) P_k(x) - γ_k P_{k-1}(x).
        """
        ...

    @classmethod
    @abstractmethod
    def project(
        cls,
        func: ScalarFunction,
        start: float,
        stop: float,
        order: Optional[int] = None,
    ) -> PolynomialExpansion:
        """Project `func` defined on (`start`, `stop`) onto the orthogonal polynomial basis up to the given `order`."""
        ...

    @classmethod
    def estimate_order(
        cls,
        func: ScalarFunction,
        start: float,
        stop: float,
        tol: float = 100 * float(np.finfo(np.float64).eps),
        initial_order: int = 2,
        max_order: int = 2**12,  # 4096
    ) -> float:
        """
        Generic order estimator: doubling until |c_N| < tolerance.
        Relies on cls.project(...) to build the coefficients.
        """
        order = initial_order
        while order <= max_order:
            # Build the expansion at this trial order
            expansion = cls.project(func, start, stop, order)
            c = expansion.coeffs
            pairs = np.maximum(np.abs(c[0::2]), np.abs(c[1::2]))
            idx = np.where(pairs < tol)[0]
            if idx.size > 0 and idx[0] != 0:
                return 2 * idx[0] + 1
            order *= 2
        raise ValueError("Order exceeds max_order without achieving tolerance.")


def mps_polynomial_expansion(
    expansion: PolynomialExpansion,
    initial: Interval | MPS,
    clenshaw: bool = True,
    strategy: Strategy = DEFAULT_STRATEGY,
    rescale: bool = True,
) -> MPS:
    """
    Compose a function on an initial MPS by expanding it on any PolynomialExpansion basis.

    Parameters
    ----------
    expansion : PolynomialExpansion
        The polynomial expansion object (Chebyshev, Legendre, Hermite, Gegenbauer, etc.)
    initial : Interval | MPS
        The initial Interval or MPS representing the input function.
    clenshaw : bool, default=True
        Whether to use Clenshaw algorithm (recommended).
    strategy : Strategy, default=DEFAULT_STRATEGY
        Strategy to simplify intermediate MPS operations.
    rescale : bool, default=True
        Whether to rescale `initial` to the intrinsic domain of the polynomial family.

    Returns
    -------
    MPS
        The MPS representing the polynomial expansion applied to the input MPS.
    """
    if isinstance(initial, Interval):
        initial_mps = mps_interval(initial)
    elif isinstance(initial, MPS):
        initial_mps = initial
    else:
        raise ValueError("Either an Interval or an initial MPS must be provided.")

    if rescale:
        orig = expansion.domain
        dest = expansion.basis_domain
        initial_mps = mps_affine(initial_mps, orig, dest)

    I = MPS([np.ones((1, s, 1)) for s in initial_mps.physical_dimensions()])
    I_norm = I.norm()
    normalized_I = CanonicalMPS(I, center=0, normalize=True, strategy=strategy)

    x_norm = initial_mps.norm()
    normalized_x = CanonicalMPS(
        initial_mps, center=0, normalize=True, strategy=strategy
    )

    c = expansion.coeffs
    steps = len(c)
    logger = make_logger(2)
    recurrences = [expansion.get_recurrence(l) for l in range(steps + 1)]

    if clenshaw:
        # Recurrence rules:
        # y_k = c_k + (α_k x + β_k I) y_{k+1} - γ_{k+1} y_{k+2}
        # f = y_0 + [(1 - α_0) x + β_0)] y_1
        logger("MPS Clenshaw evaluation started")
        y_k = y_k_plus_1 = normalized_I.zero_state()

        # Main loop
        for k, c_k in enumerate(reversed(c)):
            y_k_plus_1, y_k_plus_2 = y_k, y_k_plus_1

            # Since we reversed(c), loop index k=0 corresponds to c_{d-1} (highest degree).
            # Thus, l = (d - 1) - k gives the true degree index.
            l = (steps - 1) - k
            α_k, β_k, _ = recurrences[l]
            _, _, γ_k_plus_1 = recurrences[l + 1]

            # Avoid the zero branch when β_k == 0
            weights = [c_k * I_norm, α_k * x_norm, -γ_k_plus_1]
            states = [normalized_I, normalized_x * y_k_plus_1, y_k_plus_2]
            if β_k != 0:
                weights.append(β_k * I_norm)
                states.append(normalized_I * y_k_plus_1)
            y_k = simplify(MPSSum(weights, states, check_args=False), strategy=strategy)
            logger(
                f"MPS Clenshaw step {k+1}/{steps}, maxbond={y_k.max_bond_dimension()}, error={y_k.error():6e}"
            )

        α_0, β_0, _ = recurrences[0]
        weights = [1.0, (1 - α_0) * x_norm]
        states = [y_k, normalized_x * y_k_plus_1]
        if β_0 != 0:
            weights.append(-β_0 * I_norm)
            states.append(normalized_I * y_k_plus_1)
        f_mps = simplify(MPSSum(weights, states, check_args=False), strategy=strategy)

    else:
        # Recurrence rules:
        # f_2 = c_0 + c_1 x
        # f_{k+1} = f_k + c_{k+1} T_{k+1}
        # T_{k+1} = (α_{k} x + β_{k}) T_k - γ_k T_{k-1}
        logger("MPS expansion (direct) started")
        f_mps = simplify(
            MPSSum(
                weights=[c[0] * I_norm, c[1] * x_norm],
                states=[normalized_I, normalized_x],
                check_args=False,
            ),
            strategy=strategy,
        )
        T_k_minus_1, T_k = I_norm * normalized_I, x_norm * normalized_x
        for k, c_k in enumerate(c[2:], start=2):
            α_k, β_k, γ_k = recurrences[k - 1]
            weights = [α_k * x_norm, -γ_k]
            states = [normalized_x * T_k, T_k_minus_1]
            if β_k != 0:
                weights.append(β_k * I_norm)
                states.append(normalized_I * T_k)

            T_k_plus_1 = simplify(
                MPSSum(weights, states, check_args=False), strategy=strategy
            )
            f_mps = simplify(
                MPSSum(
                    weights=[1.0, c_k], states=[f_mps, T_k_plus_1], check_args=False
                ),
                strategy=strategy,
            )
            logger(
                f"MPS expansion step {k+1}/{steps}, maxbond={f_mps.max_bond_dimension()}, error={f_mps.error():6e}"
            )
            T_k_minus_1, T_k = T_k, T_k_plus_1

    logger.close()
    return f_mps


def mpo_polynomial_expansion(
    expansion: PolynomialExpansion,
    initial: MPO,
    clenshaw: bool = True,
    strategy: Strategy = DEFAULT_STRATEGY,
    rescale: bool = True,
) -> MPO:
    """
    Compose a function on an initial MPO by expanding it on a generic PolynomialExpansion basis.

    Parameters
    ----------
    expansion : PolynomialExpansion
        The polynomial expansion object (Chebyshev, Legendre, Hermite, etc).
    initial : MPO
        The initial MPO representing the input operator.
    clenshaw : bool, default=True
        Whether to use the Clenshaw algorithm (recommended).
    strategy : Strategy, default=DEFAULT_STRATEGY
        The simplification strategy for intermediate MPO operations.
    rescale : bool, default=True
        Whether to rescale the initial MPO to the intrinsic domain of the expansion basis.

    Returns
    -------
    MPO
        The MPO representing the function applied to the input MPO.
    """
    if rescale:
        orig = expansion.domain
        dest = expansion.basis_domain
        initial_mpo = mpo_affine(initial, orig, dest)
    else:
        initial_mpo = initial

    c = expansion.coeffs
    steps = len(c)
    I = MPO([np.eye(2).reshape(1, 2, 2, 1)] * len(initial_mpo))
    logger = make_logger(1)
    recurrences = [expansion.get_recurrence(l) for l in range(steps + 1)]

    if clenshaw:
        logger("MPO Clenshaw evaluation started")
        y_k = y_k_plus_1 = MPO([np.zeros((1, 2, 2, 1))] * len(initial_mpo))

        # Main loop
        for k, c_k in enumerate(reversed(c)):
            y_k_plus_1, y_k_plus_2 = y_k, y_k_plus_1

            l = (steps - 1) - k
            α_k, β_k, _ = recurrences[l]
            _, _, γ_k_plus_1 = recurrences[l + 1]

            weights = [c_k, α_k, -γ_k_plus_1]
            mpos = [I, MPOList([initial_mpo, y_k_plus_1]), y_k_plus_2]
            if β_k != 0:
                weights.append(β_k)
                mpos.append(MPOList([I, y_k_plus_1]))
            y_k = simplify_mpo(MPOSum(mpos, weights), strategy=strategy)
            logger(
                f"MPO Clenshaw step {k+1}/{steps}, maxbond={y_k.max_bond_dimension()}"
            )

        α_0, β_0, _ = recurrences[0]
        weights = [1.0, (1 - α_0)]
        mpos = [y_k, MPOList([initial_mpo, y_k_plus_1])]
        if β_0 != 0:
            weights.append(-β_0)
            mpos.append(MPOList([I, y_k_plus_1]))
        f_mpo = simplify_mpo(MPOSum(mpos, weights), strategy=strategy)

    else:
        logger("MPO expansion (direct) started")
        f_mpo = simplify_mpo(
            MPOSum(mpos=[I, initial_mpo], weights=[c[0], c[1]]),
            strategy=strategy,
        )
        T_k_minus_1, T_k = I, initial_mpo
        for k, c_k in enumerate(c[2:], start=2):
            α_k, β_k, γ_k = recurrences[k - 1]
            weights = [α_k, -γ_k]
            mpos = [MPOList([initial_mpo, T_k]), T_k_minus_1]
            if β_k != 0:
                weights.append(β_k)
                mpos.append(MPOList([I, T_k]))

            T_k_plus_1 = simplify_mpo(MPOSum(mpos, weights), strategy=strategy)
            f_mpo = simplify_mpo(
                MPOSum(mpos=[f_mpo, T_k_plus_1], weights=[1.0, c_k]), strategy=strategy
            )
            logger(
                f"MPO expansion step {k+1}/{steps}, maxbond={f_mpo.max_bond_dimension()}"
            )
            T_k_minus_1, T_k = T_k, T_k_plus_1

    logger.close()
    return f_mpo
