from __future__ import annotations

import numpy as np
import scipy.sparse
import functools
from scipy.sparse import dok_array, csr_array
from typing import Callable, Optional

from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..state.schmidt import _destructive_svd
from ..state._contractions import _contract_last_and_first
from ..state.core import destructively_truncate_vector
from ..truncate import simplify
from ..typing import Tensor3
from .mesh import Interval, Mesh, array_affine


def _validate_mesh(mesh: Mesh):
    num_qubits = [int(np.log2(N)) for N in mesh.dimensions]
    if not all(2**n == N for n, N in zip(num_qubits, mesh.dimensions)):
        raise ValueError("The mesh must be quantizable in qubits.")
    if len(set(num_qubits)) != 1:
        raise ValueError("The qubits per dimension must be constant.")


def mps_lagrange_chebyshev_basic(
    func: Callable,
    domain: Interval | Mesh,
    order: int,
    strategy: Strategy = DEFAULT_STRATEGY,
    interleave: bool = False,
    use_logs: bool = True,
) -> MPS:
    """
    Encodes a multivariate function in an MPS using a basic Lagrange-Chebyshev
    interpolation method.

    Parameters
    ----------
    func : Callable
        The function to encode.
    domain : Interval | Mesh
        The Interval or multivariate mesh where the function is defined.
        However, the interpolation is always constructed on a regular discretization.
    order : int
        The order of the Lagrange-Chebyshev interpolation.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy.
    interleave : bool, default=False
        Whether to construct the interleaved MPS representation.
    use_logs : bool, default=True
        Whether to compute the Chebyshev cardinal function using
        logarithms to avoid overflow. Defaults to True, but much faster if False.

    Returns
    -------
    mps : MPS
        The MPS corresponding to the basic Lagrange-Chebyshev interpolation.
    """
    mesh = Mesh([domain]) if isinstance(domain, Interval) else domain
    _validate_mesh(mesh)

    builder = _CoreBuilder(order)
    left_core = builder.get_left_core(func, mesh)
    center_core = builder.get_center_core(use_logs)
    right_core = builder.get_right_core(use_logs)
    cores = _arrange_dense_cores(center_core, right_core, mesh, interleave)
    cores[0] = left_core

    mps = MPS(cores)
    return simplify(mps, strategy=strategy)


def mps_lagrange_chebyshev_rr(
    func: Callable,
    domain: Interval | Mesh,
    order: int,
    strategy: Strategy = DEFAULT_STRATEGY,
    interleave: bool = False,
    use_logs: bool = True,
) -> MPS:
    """
    Encodes a multivariate function in an MPS using a rank-revealing Lagrange-Chebyshev
    interpolation method.

    Parameters
    ----------
    func : Callable
        The function to encode.
    domain : Interval | Mesh
        The Interval or multivariate mesh where the function is defined.
        However, the interpolation is always constructed on a regular discretization.
    order : int
        The order of the Lagrange-Chebyshev interpolation.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy.
    interleave : bool, default=False
        Whether to construct the interleaved MPS representation.
    use_logs : bool, default=True
        Whether to compute the Chebyshev cardinal function using
        logarithms to avoid overflow. Defaults to True, but much faster if False.

    Returns
    -------
    mps : MPS
        The MPS corresponding to the rank-revealing Lagrange-Chebyshev interpolation.
    """
    mesh = Mesh([domain]) if isinstance(domain, Interval) else domain
    _validate_mesh(mesh)

    builder = _CoreBuilder(order)
    left_core = builder.get_left_core(func, mesh)
    center_core = builder.get_center_core(use_logs)
    right_core = builder.get_right_core(use_logs)

    # Construct truncated cores with rank-revealing construction
    cores = _arrange_dense_cores(center_core, right_core, mesh, interleave)

    U_L, R = np.linalg.qr(left_core.reshape((2, -1)))
    truncated_tensors = [U_L.reshape(1, 2, 2)]
    for core in cores[1:-1]:
        B = _contract_last_and_first(R, core)
        r1, _, r2 = B.shape
        ## SVD
        U, S, V = _destructive_svd(B.reshape(r1 * 2, r2))
        destructively_truncate_vector(S, strategy)
        D = S.size
        U = U[:, :D]
        R = S.reshape(D, 1) * V[:D, :]
        ##
        truncated_tensors.append(U.reshape(r1, 2, -1))
    truncated_tensors.append(_contract_last_and_first(R, cores[-1]))
    return MPS(truncated_tensors)


def mps_lagrange_chebyshev_lrr(
    func: Callable,
    domain: Interval | Mesh,
    order: int,
    local_order: int,
    strategy: Strategy = DEFAULT_STRATEGY,
    interleave: bool = False,
) -> MPS:
    """
    Encodes a multivariate function in an MPS using a local rank-revealing Lagrange-Chebyshev
    interpolation method.

    Parameters
    ----------
    func : Callable
        The function to encode.
    domain : Interval | Mesh
        The Interval or multivariate mesh where the function is defined.
        However, the interpolation is always constructed on a regular discretization.
    order : int
        The order of the Lagrange-Chebyshev interpolation.
    local_order : int
        The local order of the Lagrange-Chebyshev interpolation.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy.
    interleave : bool, default=False
        Whether to construct the interleaved MPS representation.
    use_logs : bool, default=True
        Whether to compute the Chebyshev cardinal function using
        logarithms to avoid overflow. Defaults to True, but much faster if False.

    Returns
    -------
    mps : MPS
        The MPS corresponding to the local rank-revealing Lagrange-Chebyshev interpolation.
    """
    # TODO: Perform sparse matrix multiplications
    mesh = Mesh([domain]) if isinstance(domain, Interval) else domain
    _validate_mesh(mesh)

    builder = _CoreBuilder(order, local_order)
    left_core = builder.get_left_core(func, mesh)
    center_core = builder.get_center_sparse_core()
    right_core = builder.get_right_sparse_core()

    # Construct truncated cores with rank-revealing construction (sparse)
    cores = _arrange_sparse_cores(center_core, right_core, mesh, interleave)

    U_L, R = np.linalg.qr(left_core.reshape((2, -1)))
    truncated_cores = [U_L.reshape(1, 2, 2)]
    for core in cores[1:-1]:
        B = R @ core
        r1 = B.shape[0]
        ## SVD
        U, S, V = _destructive_svd(B.reshape(r1 * 2, -1))
        destructively_truncate_vector(S, strategy)
        D = S.size
        U = U[:, :D]
        R = S.reshape(D, 1) * V[:D, :]
        ##
        truncated_cores.append(U.reshape(r1, 2, -1))
    U_R = R @ cores[-1]
    truncated_cores.append(U_R.reshape(-1, 2, 1))
    return MPS(truncated_cores)


class _CoreBuilder:
    """
    Auxiliar class used to build the tensor cores required for Lagrange-Chebyshev interpolation.
    Source: "Multiscale interpolative construction of quantized tensor trains" (2311.12554)
    """

    def __init__(
        self,
        order: int,
        local_order: Optional[int] = None,
    ):
        self.d = order
        self.m = local_order if local_order else order
        self.D = order + 1
        self.c = np.array(
            [0.5 * (np.cos(np.pi * i / self.d) + 1) for i in range(self.d + 1)]
        )
        self.angular_grid = np.array([i * np.pi / self.d for i in range(self.d + 1)])
        if local_order is not None:
            self.extended_grid = np.array(
                [(i * np.pi) / self.d for i in range(-self.d, 2 * self.d + 1)]
            )
        # Precompute cardinal terms
        self.den = self.c[:, np.newaxis] - self.c
        np.fill_diagonal(self.den, 1)
        self.log_den = np.log(abs(self.den))
        self.sign_den = np.sign(self.den)

    @functools.lru_cache(maxsize=None)  # Unbound cache
    def angular_index(self, theta: float) -> int:
        return int(np.argmin(abs(theta - self.angular_grid)))

    def chebyshev_cardinal(self, x: np.ndarray, j: int, use_logs: bool) -> float:
        num = np.delete(x[:, np.newaxis] - self.c, j, axis=1)
        if use_logs:  # Prevents overflow
            with np.errstate(divide="ignore"):  # Ignore warning of log(0)
                log_num = np.log(abs(num))
            log_den = np.delete(self.log_den[j], j)
            log_div = np.sum(log_num - log_den, axis=1)
            sign_num = np.sign(num)
            sign_den = np.delete(self.sign_den[j], j)
            sign_div = np.prod(sign_num * sign_den, axis=1)
            return sign_div * np.exp(log_div)
        else:
            den = np.delete(self.den[j], j)
            return np.prod(num / den, axis=1)

    def local_chebyshev_cardinal(self, x: float, j: int) -> float:
        θ = np.arccos(2 * x - 1)
        idx = self.angular_index(θ)

        P = 0.0
        for γ in range(idx - self.m, idx + self.m + 1):
            γ_rep = -γ if γ < 0 else self.d - (γ - self.d) if γ > self.d else γ
            if j == γ_rep:
                P += self.local_angular_cardinal(θ, γ)
        return P

    def local_angular_cardinal(self, θ: float, γ: int) -> float:
        idx = self.angular_index(θ)
        L = 1
        for β in range(idx - self.m, idx + self.m + 1):
            if β != γ:
                L *= (θ - self.extended_grid[self.d + β]) / (
                    self.extended_grid[self.d + γ] - self.extended_grid[self.d + β]
                )
        return L

    def get_left_core(
        self, func: Callable, mesh: Mesh, channels_first: bool = True
    ) -> Tensor3:
        m = mesh.dimension
        A = np.zeros((1, 2, self.D**m))
        for σ in [0, 1]:
            intervals = []
            for i in range(m):
                a, b = mesh.intervals[i].start, mesh.intervals[i].stop
                c = (σ + self.c) / 2 if i == 0 else self.c
                interval = array_affine(c, (0, 1), (a, b))
                intervals.append(interval)
            c_mesh = Mesh(intervals)
            tensor = c_mesh.to_tensor(channels_first)
            A[0, σ, :] = func(tensor).reshape(-1)
        return A

    def get_center_core(self, use_logs: bool) -> Tensor3:
        A = np.zeros((self.D, 2, self.D))
        for σ in range(2):
            for i in range(self.D):
                A[i, σ, :] = self.chebyshev_cardinal(0.5 * (σ + self.c), i, use_logs)
        return A

    def get_right_core(self, use_logs: bool) -> Tensor3:
        A = np.zeros((self.D, 2, 1))
        for σ in range(2):
            for i in range(self.D):
                A[i, σ, 0] = self.chebyshev_cardinal(np.array([0.5 * σ]), i, use_logs)
        return A

    def get_center_sparse_core(self) -> csr_array:
        A = dok_array((self.D, 2 * self.D), dtype=np.float64)
        for σ in range(2):
            for i in range(self.D):
                for j, c_j in enumerate(self.c):
                    A[i, σ * self.D + j] = self.local_chebyshev_cardinal(
                        0.5 * (σ + c_j), i
                    )
        return A.tocsr()

    def get_right_sparse_core(self) -> csr_array:
        A = dok_array((self.D, 2), dtype=np.float64)
        for σ in range(2):
            for i in range(self.D):
                A[i, σ] = self.local_chebyshev_cardinal(0.5 * σ, i)
        return A.tocsr()


def _arrange_dense_cores(
    A_C: Tensor3, A_R: Tensor3, mesh: Mesh, interleave: bool
) -> list[Tensor3]:
    m = mesh.dimension
    n = int(np.log2(mesh.dimensions[0]))

    A_R_kron = [_kron_dense(A_R, m - i, 0) for i in range(m)]
    if interleave:
        A_C_kron = [_kron_dense(A_C, m, i) for i in range(m)]
        cores = A_C_kron * (n - 1) + A_R_kron
    else:
        A_C_kron = [_kron_dense(A_C, m - i, 0) for i in range(m)]
        cores = []
        for A_C, A_R in zip(A_C_kron, A_R_kron):
            cores.extend([A_C] * (n - 1) + [A_R])
    return cores


def _kron_dense(A: Tensor3, m: int, i: int) -> Tensor3:
    """
    Take the Kronecker product of the tensor A with identity matrices along m dimensions.
    The function reshapes A from (i, s, j) to (s, i, j) and back to (i*i, s, j*i) after the operation.
    """
    I = np.eye(A.shape[0])
    tensors = [np.swapaxes(A, 0, 1) if j == i else I for j in range(m)]
    B = tensors[0]
    for tensor in tensors[1:]:
        B = np.kron(B, tensor)
    return np.swapaxes(B, 0, 1)


def _arrange_sparse_cores(
    A_C: csr_array, A_R: csr_array, mesh: Mesh, interleave: bool
) -> list[csr_array]:
    m = mesh.dimension
    n = int(np.log2(mesh.dimensions[0]))

    A_R_kron = [_kron_sparse(A_R, m - i, 0) for i in range(m)]
    if interleave:
        A_C_kron = [_kron_sparse(A_C, m, i) for i in range(m)]
        cores = A_C_kron * (n - 1) + A_R_kron
    else:
        A_C_kron = [_kron_sparse(A_C, m - i, 0) for i in range(m)]
        cores = []
        for A_C, A_R in zip(A_C_kron, A_R_kron):
            cores.extend([A_C] * (n - 1) + [A_R])
    return cores


def _kron_sparse(A: csr_array, m: int, i: int) -> csr_array:
    """
    Take the Kronecker product of the sparse tensor A with identity matrices along m dimensions.
    This operation is implemented converting the CSR matrix to a dense format as a temporary workaround.
    """
    # TODO: Fix without transforming to dense matrices.
    A_dense = A.toarray().reshape(A.shape[0], 2, A.shape[1] // 2)
    B = _kron_dense(A_dense, m, i)
    return scipy.sparse.csr_array(B.reshape(B.shape[0], 2 * B.shape[2]))
