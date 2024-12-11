import numpy as np
from typing import Callable, Optional
from abc import ABC, abstractmethod

from ...state import MPS
from ...analysis.mesh import Mesh, mps_to_mesh_matrix
from ...typing import Vector
from ..evaluation import evaluate_mps
from ...typing import Matrix


class BlackBox(ABC):
    """
    Abstract base class representing generic black-box functions.
    These are implicit representation of functions that can be indexed with indices
    similarly as a multidimensional array.
    """

    def __init__(self, func: Callable, physical_dimensions: list):
        self.func = func
        self.physical_dimensions = physical_dimensions
        self.evals = 0

    @abstractmethod
    def __getitem__(self, mps_indices: Matrix) -> Vector: ...


# TODO: Think of a better name for these black box classes.
# The names should not describe what the objects *do* but what the objects *are*.


class BlackBoxLoadMPS(BlackBox):
    """
    Black-box representation of a multivariate scalar function, discretized on a `Mesh`
    object. The structure of the discretization is represented by the `map_matrix` and
    `physical_dimensions` objects, which respectively encode the quantization and
    arrangement of the degrees of freedom and the physical dimensions of the representation.

    This class is helpful for encoding functions in MPS using tensor cross-interpolation,
    hence the name.

    Parameters
    ----------
    func : Callable
        A multivariate scalar function acting on a tensor. By convention, this assumes that
        the tensor index that represents the degrees of freedom is placed first instead of last.
    mesh : Mesh
        The multivariate domain where the function is discretized.
    map_matrix : Optional[Matrix], default None
        A matrix that encodes the quantization and arrangement of the MPS tensors.
        If None, a non-quantized "tensor-train" structure is assumed, where each degree
        of freedom of the tensor is assigned a unique MPS tensor.
    physical_dimensions: Optional[Vector], default None
        A vector that stores the physical sizes of the resulting MPS tensors. This is
        only required when `map_matrix` is not None, as it defaults to the sizes of the `mesh`.

    Example
    -------
        .. code-block:: python

        # Load a bivariate Gaussian function in interleaved qubit order.

        # Define the tensorized function.
        # This follows the "channels-first" convention (the dimension index goes first).
        func = lambda tensor: np.exp(-(tensor[0]**2 + tensor[1]**2))

        # Define the bivariate domain implictly using a `Mesh` object:
        start, stop = -1, 1
        n = 10 # Number of qubits
        interval = RegularInterval(start, stop, 2**n)
        mesh = Mesh([interval, interval])

        # Define the map matrix. For standard binary quantizations in serial and interleaved
        orders, this is implemented using the `mps_to_mesh_matrix` function:
        map_matrix = mps_to_mesh_matrix([n, n], mps_order='B')
        physical_dimensions = [2] * (2 * n)

        # Define the black box. This defaults to a non-quantized "tensor-train" structure:
        black_box = BlackBoxLoadMPS(func, mesh, map_matrix, physical_dimensions)

        # Load the function using some TCI variant, either `cross_maxvol`, `cross_dmrg`
        # or `cross_greedy`. This returns a `CrossResults` object with all the information.
        cross_results = cross_maxvol(black_box)
        mps = cross_results.mps
    """

    def __init__(
        self,
        func: Callable,
        mesh: Mesh,
        map_matrix: Optional[Matrix] = None,
        physical_dimensions: Optional[list] = None,
    ):
        # Defaults to tensor-train (not quantized).
        if map_matrix is None:
            map_matrix = np.eye(mesh.dimension, dtype=int)
        if physical_dimensions is None:
            physical_dimensions = list(mesh.dimensions)

        super().__init__(func, physical_dimensions)
        self.mesh = mesh
        self.map_matrix = map_matrix

    def __getitem__(self, mps_indices: Matrix) -> Vector:
        self.evals += len(mps_indices)
        mesh_indices = mps_indices @ self.map_matrix
        coordinates = self.mesh[mesh_indices]
        return self.func(coordinates.T)  # type: ignore


class BlackBoxComposeMPS(BlackBox):
    """
    Black-box representation of a multivariate scalar function, applied on a collection
    of `MPS` objects by considering each MPS as one degree of freedom of the function.

    Parameters
    ----------
    func : Callable
        A multivariate scalar function acting on a tensor. By convention, this assumes that
        the tensor index that represents the degrees of freedom is placed first instead of last.
    mps_list : list[MPS]
        A list of MPS. They must be of the same physical dimension, which must also be constant.
        Also, the length of this list must match the dimension of the multivariate function.

    Example
    -------
    .. code-block:: python

        # Use TCI to apply a three-dimensional function on three MPS.

        # Assume the three initial MPS are given and are of the same structure:
        mps_0, mps_1, mps_2 = ...

        # Define the three dimensional function by its action on the MPS:
        func = lambda x: x[0]**2 + np.sin(x[0]*x[1]) + np.cos(x[0]*x[2])

        # Define the black-box:
        black_box = BlackBoxComposeMPS(func, [mps_0, mps_1, mps_2])

        # Load the function using some TCI variant, either `cross_maxvol`, `cross_dmrg`
        # or `cross_greedy`. This returns a `CrossResults` object with all the information.
        cross_results = cross_dmrg(black_box)
        mps = cross_results.mps
    """

    def __init__(self, func: Callable, mps_list: list[MPS]):
        physical_dimensions = mps_list[0].physical_dimensions()
        for mps in mps_list:
            if mps.physical_dimensions() != physical_dimensions:
                raise ValueError("The MPS physical dimensions do not match.")

        super().__init__(func, physical_dimensions)
        self.mps_list = mps_list

    def __getitem__(self, mps_indices: Matrix) -> Vector:
        self.evals += len(mps_indices)
        samples = np.array([evaluate_mps(mps, mps_indices) for mps in self.mps_list])
        return self.func(samples)


class BlackBoxLoadMPO(BlackBox):
    """
    Black-box representation of a bivariate scalar function, discretized on a `Mesh`
    object. This can be understood as a specialization of the `BlackBoxLoadMPS` object,
    for the case of bivariate functions, and can be used to encode MPO of physical dimension
    `base_mpo` as MPS with physical dimension `base_mpo ** 2`. The corresponding MPS can be
    unfolded back to MPO using the function `mps_as_mpo`.

    Parameters
    ----------
    func : Callable
        The bivariate scalar function to be represented as MPO.
    mesh : Mesh
        The two-dimensional discretization where the function is discretized.
    base_mpo : int, default=2
        The required physical dimension of each index of the MPO.
    is_diagonal : bool, default=True
        Flag that helps in the convergence of TCI for diagonal operators. It restricts
        the function evaluations to the main diagonal.

    Example
    -------
        .. code-block:: python

        # Load a 2D Gaussian function in a MPO.

        # Define the bivariate function:
        func = lambda tensor: np.exp(-(tensor[0]**2 + tensor[1]**2))

        # Define the bivariate domain as a `Mesh` object:
        start, stop = -1, 1
        n = 10 # Number of qubits
        interval = RegularInterval(start, stop, 2**n)
        mesh = Mesh([interval, interval])

        # Define the black box:
        black_box = BlackBoxLoadMPO(func, mesh)

        # Load the function using some TCI variant, either `cross_maxvol`, `cross_dmrg`
        # or `cross_greedy`. This returns a `CrossResults` object with all the information.
        cross_results = cross_greedy(black_box)
        mps = cross_results.mps

        # Unfold the result MPS as a MPO using the routine `mps_as_mpo`.
        mpo = mps_as_mpo(mps)
    """

    # TODO: Generalize for an arbitrary `map_matrix` as for MPS.
    # TODO: Generalize for multivariate MPOs.
    # TODO: Think of a more robust way of handling convergence than the `is_diagonal` flag.
    def __init__(
        self,
        func: Callable,
        mesh: Mesh,
        base_mpo: int = 2,
        is_diagonal: bool = False,
    ):
        self.mesh = mesh
        self.base_mpo = base_mpo
        self.is_diagonal = is_diagonal

        if not (mesh.dimension == 2 and mesh.dimensions[0] == mesh.dimensions[1]):
            raise ValueError("The mesh must be bivariate.")
        self.sites = int(np.emath.logn(self.base_mpo, mesh.dimensions[0]))
        if not self.base_mpo**self.sites == mesh.dimensions[0]:
            raise ValueError(f"The mesh cannot be quantized with base {self.base_mpo}")

        physical_dimensions = [base_mpo**2] * self.sites
        super().__init__(func, physical_dimensions)

        sites_per_dimension = [self.sites]
        self.map_matrix = mps_to_mesh_matrix(sites_per_dimension, base=self.base_mpo)

        # If the MPO is diagonal, restrict the allowed indices for random sampling to the main diagonal.
        self.allowed_indices = (
            [s * base_mpo + s for s in range(base_mpo)] if self.is_diagonal else None
        )

    def __getitem__(self, mps_indices: np.ndarray) -> np.ndarray:
        self.evals += len(mps_indices)
        row_indices = (mps_indices // self.base_mpo) @ self.map_matrix
        col_indices = (mps_indices % self.base_mpo) @ self.map_matrix
        mesh_indices = np.hstack((row_indices, col_indices))
        return self.func(*self.mesh[mesh_indices].T)  # type: ignore
