from __future__ import annotations

import numpy as np
from typing import overload, Union, Optional, Sequence

from ..state import DEFAULT_STRATEGY, MPS, MPSSum, Strategy, scprod
from ..tools import InvalidOperation
from ..typing import Tensor4, Operator, Weight


class MPOList(object):
    """Sequence of matrix-product operators.

    This implements a list of MPOs that are applied sequentially. It can impose
    its own truncation or simplification strategy on top of the one provided by
    the individual operators.

    Parameters
    ----------
    mpos : list[MPO]
        Operators in this sequence, to be applied from mpos[0] to mpos[-1]. Must
        contain at least one operator.
    strategy : Strategy, optional
        Truncation and simplification strategy, defaults to DEFAULT_STRATEGY

    Attributes
    ----------
    mpos : list[MPO]
        Operators in this sequence, to be applied from mpos[0] to mpos[-1]. Must
        contain at least one operator.
    strategy : Strategy
        Truncation and simplification strategy.
    size : int
        Number of quantum subsystems in each MPO. Computed from the supplied
        MPOs. Not checked for consistency.
    """

    __array_priority__ = 10000

    mpos: list[MPO]
    strategy: Strategy
    size: int

    def __init__(self, mpos: Sequence[MPO], strategy: Strategy = DEFAULT_STRATEGY):
        assert len(mpos) > 1
        self.mpos = mpos = list(mpos)
        self.size = mpos[0].size
        self.strategy = strategy

    def copy(self) -> MPOList:
        """Shallow copy of the MPOList, without copying the MPOs themselves."""
        return MPOList(self.mpos.copy(), self.strategy)

    def __add__(self, A: Union[MPO, MPOList, MPOSum]) -> MPOSum:
        """Represent `self + A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, 1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + A.weights, A.strategy)
        raise TypeError(f"Cannod add MPO and {type(A)}")

    def __sub__(self, A: Union[MPO, MPOList, MPOSum]) -> MPOSum:
        """Represent `self - A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, -1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + [-w for w in A.weights], A.strategy)
        raise TypeError(f"Cannod subtract MPO and {type(A)}")

    def __mul__(self, n: Weight) -> MPOList:
        """Multiply an MPO by a scalar `n` as in `n * self`."""
        if isinstance(n, (int, float, complex)):
            return MPOList([n * self.mpos[0]] + self.mpos[1:], self.strategy)
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPOList:
        """Multiply an MPO by a scalar `n` as in `self * n`."""
        if isinstance(n, (int, float, complex)):
            return MPOList([n * self.mpos[0]] + self.mpos[1:], self.strategy)
        raise InvalidOperation("*", n, self)

    @property
    def T(self) -> MPOList:
        return MPOList([A.T for A in reversed(self.mpos)], self.strategy)

    def physical_dimensions(self) -> list[int]:
        """Return the physical dimensions of the MPOList."""
        return self.mpos[0].physical_dimensions()

    def to_matrix(self) -> Operator:
        """Convert this MPO to a dense or sparse matrix."""
        A = self.mpos[0].to_matrix()
        for mpo in self.mpos[1:]:
            A = mpo.to_matrix() @ A
        return A

    def set_strategy(self, strategy, strategy_components=None) -> MPOList:
        """Return MPOList with the given strategy."""
        if strategy_components is not None:
            mpos = [mpo.set_strategy(strategy_components) for mpo in self.mpos]
        else:
            mpos = self.mpos
        return MPOList(mpos=mpos, strategy=strategy)

    @overload
    def apply(
        self,
        state: MPS,
        strategy: Optional[Strategy] = None,
        simplify: Optional[bool] = None,
    ) -> MPS: ...

    @overload
    def apply(
        self,
        state: MPSSum,
        strategy: Optional[Strategy] = None,
        simplify: Optional[bool] = None,
    ) -> MPS | MPSSum: ...

    # TODO: Describe how `strategy` and simplify act as compared to
    # the values provided by individual operators.
    def apply(
        self,
        state: Union[MPS, MPSSum],
        strategy: Optional[Strategy] = None,
        simplify: Optional[bool] = None,
    ) -> Union[MPS, MPSSum]:
        """Implement multiplication `A @ state` between a matrix-product operator
        `A` and a matrix-product state `state`.

        Parameters
        ----------
        state : MPS | MPSSum
            Transformed state.
        strategy : Strategy, optional
            Truncation strategy, defaults to DEFAULT_STRATEGY
        simplify : bool, optional
            Whether to simplify the state after the contraction.
            Defaults to `strategy.get_simplify_flag()`

        Returns
        -------
        CanonicalMPS
            The result of the contraction.
        """
        if strategy is None:
            strategy = self.strategy
        if simplify is None:
            simplify = strategy.get_simplify_flag()

        for mpo in self.mpos:
            # log(f'Total error before applying MPOList {b.error()}')
            state = mpo.apply(state)
        if simplify:
            state = truncate.simplify(state, strategy=strategy)
        return state

    @overload
    def __matmul__(self, b: MPS) -> MPS: ...

    @overload
    def __matmul__(self, b: MPSSum) -> MPS | MPSSum: ...

    def __matmul__(self, b: Union[MPS, MPSSum]) -> Union[MPS, MPSSum]:
        """Implement multiplication `self @ b`."""
        return self.apply(b)

    def extend(
        self, L: int, sites: Optional[list[int]] = None, dimensions: int = 2
    ) -> MPOList:
        """Enlarge an MPOList so that it acts on a larger Hilbert space with 'L' sites.

        See also
        --------
        :py:meth:`MPO.extend`
        """
        return MPOList(
            [mpo.extend(L, sites=sites, dimensions=dimensions) for mpo in self.mpos],
            strategy=self.strategy,
        )

    def _joined_tensors(self, i: int, L: int) -> Tensor4:
        """Join the tensors from all MPOs into bigger tensors."""

        def join(A, *args):
            if not args:
                return A
            B = join(*args)
            a, d, d, b = A.shape
            c, d, d, e = B.shape
            # A, B, args[1],... are the tensors of the MPO to
            # join. They are applied to the MPS in this order, hence the
            # particular position of elements in opt_einsum
            # return opt_einsum.contract("aijb,cjkd->acikbd", B, A).reshape(
            #    a * c, d, d, b * e
            # )
            # aijbc,cjkd->aibckd->acikbd
            aux = np.tensordot(B, A, ((2,), (1,)))
            return np.ascontiguousarray(
                aux.transpose(0, 3, 1, 4, 2, 5).reshape(a * c, d, d, b * e)
            )

        return join(*[mpo[i] for mpo in self.mpos])

    def join(self, strategy: Optional[Strategy] = None) -> MPO:
        """Create an `MPO` by combining all tensors from all MPOs.

        Returns
        -------
        MPO
            Quantum operator implementing the product of tensors.
        """
        L = self.mpos[0].size
        return MPO(
            [self._joined_tensors(i, L) for i in range(L)],
            strategy=self.strategy if strategy is None else strategy,
        )

    def expectation(self, bra: MPS, ket: Optional[MPS] = None) -> Weight:
        """Expectation value of MPOList on one or two MPS states.

        If one state is given, this state is interpreted as :math:`\\psi`
        and this function computes :math:`\\langle{\\psi|O\\psi}\\rangle`
        If two states are given, the first one is the bra :math:`\\psi`,
        the second one is the ket :math:`\\phi`, and this computes
        :math:`\\langle\\psi|O|\\phi\\rangle`.

        Parameters
        ----------
        bra : MPS
            The state :math:`\\psi` on which the expectation value
            is computed.
        ket : Optional[MPS]
            The ket component of the expectation value. Defaults to `bra`.

        Returns
        -------
        float | complex
            :math:`\\langle\\psi\\vert{O}\\vert\\phi\\rangle` where `O`
            is the matrix-product operator.
        """
        if ket is None:
            ket = bra
        return scprod(bra, self.apply(ket))  # type: ignore


from .mpo import MPO  # noqa: E402
from .mposum import MPOSum  # noqa: E402
from .. import truncate  # noqa: E402
