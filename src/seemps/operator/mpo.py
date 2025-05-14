from __future__ import annotations

import numpy as np
from math import isqrt
from typing import overload, Union, Optional, Sequence

from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, MPSSum, Strategy, TensorArray
from ..state.environments import (
    begin_mpo_environment,
    update_left_mpo_environment,
    update_right_mpo_environment,
    join_mpo_environments,
)
from ..truncate import simplify as do_simplify
from ..tools import InvalidOperation
from ..typing import Tensor3, Tensor4, Operator, Weight


def _mpo_multiply_tensor(A: Tensor4, B: Tensor3) -> Tensor3:
    """
    Implements np.einsum("cjd,aijb->caidb", B, A)

    Matmul takes two arguments
        B(c, 1, 1, d, j)
        A(1, a, i, j, b)
    It broadcasts, repeating the indices that are of size 1
        B(c, a, i, d, j)
        A(c, a, i, j, b)
    And then multiplies the matrices that are formed by the last two
    indices, (d,j) * (j,b) -> (b,d) so that the outcome has size
        C(c, a, i, d, b)
    """
    a, i, j, b = A.shape
    c, j, d = B.shape
    # np.matmul(...) -> C(a,i,b,c,d)
    return np.matmul(
        B.transpose(0, 2, 1).reshape(c, 1, 1, d, j), A.reshape(1, a, i, j, b)
    ).reshape(c * a, i, d * b)


class MPO(TensorArray):
    """Matrix Product Operator class.

    This implements a bare-bones Matrix Product Operator object with open
    boundary conditions. The tensors have four indices, A[α,i,j,β], where
    'α,β' are the internal labels and 'i,j' the physical indices ar the given
    site.

    Parameters
    ----------
    data: Sequence[Tensor4]
        Sequence of four-legged tensors forming the structure.
    strategy: Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for algorithms.
    """

    strategy: Strategy

    __array_priority__ = 10000

    def __init__(self, data: Sequence[Tensor4], strategy: Strategy = DEFAULT_STRATEGY):
        super().__init__(data)
        self.strategy = strategy

    def copy(self) -> MPO:
        """Return a shallow copy of the MPO, without duplicating the tensors."""
        # We use the fact that TensorArray duplicates the list
        return MPO(self, self.strategy)

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

    # TODO: The deep copy also copies the tensors. This should be improved.
    def __mul__(self, n: Weight) -> MPO:
        """Multiply an MPO by a scalar `n * self`"""
        if isinstance(n, (int, float, complex)):
            absn = abs(n)
            if absn:
                phase = n / absn
                factor = np.exp(np.log(absn) / self.size)
            else:
                phase = 0.0
                factor = 0.0
            return MPO(
                [
                    (factor if i > 0 else (factor * phase)) * A
                    for i, A in enumerate(self)
                ],
                self.strategy,
            )
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPO:
        """Multiply an MPO by a scalar `self * self`"""
        if isinstance(n, (int, float, complex)):
            absn = abs(n)
            if absn:
                phase = n / absn
                factor = np.exp(np.log(absn) / self.size)
            else:
                phase = 0.0
                factor = 0.0
            return MPO(
                [
                    (factor if i > 0 else (factor * phase)) * A
                    for i, A in enumerate(self)
                ],
                self.strategy,
            )
        raise InvalidOperation("*", n, self)

    def __pow__(self, n: int) -> MPOList:
        """Exponentiate a MPO to n."""
        if isinstance(n, int):
            return MPOList([self.copy() for _ in range(n)])
        raise InvalidOperation("**", n, self)

    def physical_dimensions(self) -> list[int]:
        """Return the physical dimensions of the MPO."""
        return [A.shape[2] for A in self]

    def bond_dimensions(self) -> list[int]:
        """Return the bond dimensions of the MPO."""
        return [A.shape[-1] for A in self][:-1]

    def max_bond_dimension(self) -> int:
        """Return the largest bond dimension."""
        return max(A.shape[0] for A in self)

    @property
    def T(self) -> MPO:
        return MPO([A.transpose(0, 2, 1, 3) for A in self], self.strategy)

    @classmethod
    def from_mps(cls, mps: MPS, strategy: Strategy = DEFAULT_STRATEGY) -> MPO:
        _, S, _ = mps[0].shape
        s = isqrt(S)
        if s**2 != S:
            raise ValueError("The MPS dimensions must be perfect squares.")
        return cls(
            [t.reshape(t.shape[0], s, s, t.shape[-1]) for t in mps._data],
            strategy=strategy,
        )

    def to_mps(self) -> MPS:
        """Recast the MPO as MPS by combining the physical dimensions."""
        _, i, j, _ = self._data[0].shape
        return MPS([t.reshape(t.shape[0], i * j, t.shape[-1]) for t in self._data])

    def to_matrix(self) -> Operator:
        """Convert this MPO to a dense or sparse matrix."""
        Di = 1  # Total physical dimension so far
        Dj = 1
        out = np.array([[[1.0]]])
        for A in self:
            _, i, j, b = A.shape
            out = np.einsum("lma,aijb->limjb", out, A)
            Di *= i
            Dj *= j
            out = out.reshape(Di, Dj, b)
        return out[:, :, 0]

    def set_strategy(self, strategy) -> MPO:
        """Return MPO with the given strategy."""
        return MPO(self, strategy)

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
    ) -> MPS: ...

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
        # TODO: Remove implicit conversion of MPSSum to MPS
        if strategy is None:
            strategy = self.strategy
        if simplify is None:
            simplify = strategy.get_simplify_flag()
        if isinstance(state, MPSSum):
            assert self.size == state.size
            for i, (w, mps) in enumerate(zip(state.weights, state.states)):
                Ostate = w * MPS(
                    [_mpo_multiply_tensor(A, B) for A, B in zip(self, mps)],
                    error=mps.error(),
                )
                state = Ostate if i == 0 else state + Ostate
        elif isinstance(state, MPS):
            assert self.size == state.size
            state = MPS(
                [_mpo_multiply_tensor(A, B) for A, B in zip(self, state)],
                error=state.error(),
            )
        else:
            raise TypeError(f"Cannot multiply MPO with {state}")

        if simplify:
            state = do_simplify(state, strategy=strategy)
        return state

    @overload
    def __matmul__(self, b: MPS) -> MPS: ...

    @overload
    def __matmul__(self, b: MPSSum) -> MPS | MPSSum: ...

    def __matmul__(self, b: Union[MPS, MPSSum]) -> Union[MPS, MPSSum]:
        """Implement multiplication `self @ b`."""
        return self.apply(b)

    # TODO: We have to change the signature and working of this function, so that
    # 'sites' only contains the locations of the _new_ sites, and 'L' is no longer
    # needed. In this case, 'dimensions' will only list the dimensions of the added
    # sites, not all of them.
    def extend(
        self,
        L: int,
        sites: Optional[Sequence[int]] = None,
        dimensions: Union[int, list[int]] = 2,
    ) -> MPO:
        """Enlarge an MPO so that it acts on a larger Hilbert space with 'L' sites.

        Parameters
        ----------
        L : int
            The new size of the MPS. Must be strictly larger than `self.size`.
        sites : Iterable[int], optional
            Sequence of integers describing the sites that occupied by the
            tensors in this state.
        dimensions : Union[int, list[int]], default = 2
            Dimension of the added sites. It can be the same integer or a list
            of integers with the same length as `sites`.

        Returns
        -------
        MPO
            Extended MPO.
        """
        if isinstance(dimensions, int):
            final_dimensions = [dimensions] * max(L - self.size, 0)
        else:
            final_dimensions = dimensions.copy()
            assert len(dimensions) == L - self.size
        if sites is None:
            sites = range(self.size)
        assert L >= self.size
        assert len(sites) == self.size

        data: list[np.ndarray] = [np.ndarray(())] * L
        for ndx, A in zip(sites, self):
            data[ndx] = A
        D = 1
        k = 0
        for i, A in enumerate(data):
            if A.ndim == 0:
                d = final_dimensions[k]
                A = np.eye(D).reshape(D, 1, 1, D) * np.eye(d).reshape(1, d, d, 1)
                data[i] = A
                k = k + 1
            else:
                D = A.shape[-1]
        return MPO(data, strategy=self.strategy)

    def expectation(self, bra: MPS, ket: Optional[MPS] = None) -> Weight:
        """Expectation value of MPO on one or two MPS states.

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
        if isinstance(bra, CanonicalMPS):
            center = bra.center
        elif isinstance(bra, MPS):
            center = self.size - 1
        else:
            raise Exception("MPS required")
        if ket is None:
            ket = bra
        elif not isinstance(ket, MPS):
            raise Exception("MPS required")
        left = right = begin_mpo_environment()
        for i in range(0, center):
            left = update_left_mpo_environment(left, bra[i].conj(), self[i], ket[i])
        for i in range(self.size - 1, center - 1, -1):
            right = update_right_mpo_environment(right, bra[i].conj(), self[i], ket[i])
        return join_mpo_environments(left, right)


from .mpolist import MPOList  # noqa: E402
from .mposum import MPOSum  # noqa: E402
