import numpy as np
from typing import Callable
from abc import ABC, abstractmethod

from ...state import MPS
from ...analysis.mesh import Mesh, mps_to_mesh_matrix
from ...typing import Vector
from ..evaluation import evaluate_mps
from ...typing import Matrix

# ✅ Code review de Juanjo hecha

class BlackBox(ABC):
	"""
    Abstract base class representing generic black-box functions.
    A black-box function represents an implicit representation of a function
    that can be indexed with indices similarly as a multidimensional array.
    These objects are fundamental for the efficient implementation of TCI algorithms.
    """

	def __init__(self, func: Callable, physical_dimensions: list):
		self.func = func
		self.physical_dimensions = physical_dimensions
		self.evals = 0

	@abstractmethod
	def __getitem__(self, mps_indices: Matrix) -> Vector: ...

	def record_evaluations(self, n: int = 1) -> None:
		self.evals += n


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

		# Define the bivariate domain implictly using `Interval` and `Mesh`.
		start, stop = -1, 1
		n_qubits = 10
		interval = RegularInterval(start, stop, 2**n_qubits)
		mesh = Mesh([interval, interval])

		# Define the quantization map using the `mps_to_mesh_matrix` function.
		permutation = interleaving_permutation([n_qubits, n_qubits])
		map_matrix = mps_to_mesh_matrix([n_qubits, n_qubits], permutation=permutation)
		physical_dimensions = [2] * (2 * n_qubits)

		# Define the black box.
		black_box = BlackBoxLoadMPS(func, mesh, map_matrix, physical_dimensions)

		# Load the function in the given domain using some TCI variant (e.g. DMRG, Maxvol or Greedy).
		cross_results = cross_X(black_box)
		mps = cross_results.mps
	"""

	mesh: Mesh
	map_matrix: Matrix | None
	physical_dimensions: list | None

	def __init__(
		self,
		func: Callable,
		mesh: Mesh,
		map_matrix: Matrix | None = None,
		physical_dimensions: list | None = None,
	):
		if physical_dimensions is None:
			physical_dimensions = list(mesh.dimensions)

		super().__init__(func, physical_dimensions)
		self.mesh = mesh
		self.map_matrix = map_matrix

	def __getitem__(self, mps_indices: Matrix) -> Vector:
		self.record_evaluations(len(mps_indices))
		mesh_indices = (
			mps_indices if self.map_matrix is None else mps_indices @ self.map_matrix
		)
		coordinates = self.mesh[mesh_indices]
		# Transpose because of opposite conventions for mesh (dimension index last)
        # and cross (dimension index first).
		return self.func(coordinates.T)  # type: ignore


class BlackBoxComposeMPS(BlackBox):
	"""
	Black-box representing the composition of a multivariate scalar function with
	a collection of MPS objects.

	Parameters
	----------
	func : Callable
		The function to compose with the collection of MPS objects. Must be
		scalar, and each of its degrees of freedom must refer to each of the MPS
		in the `mps_list` collection. For example, `f(x, y) = sin(x + y**2)`
		acts on two MPS representing respectively `x` and `y`.
	mps_list : list[MPS]
		A list of MPS of the same physical dimension, to be composed with `func`.
		Their physical dimensions are assumed similar and constant. The number of MPS
		must match the dimension of `func`

	Example
	-------
	.. code-block:: python

		# Use TCI to apply a three-dimensional function on three MPS.

		# Assume the three initial MPS are given and are of the same structure.
		mps_0, mps_1, mps_2 = ...

		# Define the three dimensional function by its action on the MPS.
		func = lambda x: x[0]**2 + np.sin(x[0]*x[1]) + np.cos(x[0]*x[2])

		# Define the black-box.
		black_box = BlackBoxComposeMPS(func, [mps_0, mps_1, mps_2])

		# Compose the three MPS with the function `func`.
		cross_results = cross_X(black_box)
		mps = cross_results.mps
	"""

	mps_list: list[MPS]

	def __init__(self, func: Callable, mps_list: list[MPS]):
		physical_dimensions = mps_list[0].physical_dimensions()
		for mps in mps_list:
			if mps.physical_dimensions() != physical_dimensions:
				raise ValueError("All MPS must have the same physical dimensions.")

		super().__init__(func, physical_dimensions)
		self.mps_list = mps_list

	def __getitem__(self, mps_indices: Matrix) -> Vector:
		self.record_evaluations(len(mps_indices))
		mps_values = []
		for mps in self.mps_list:
			mps_values.append(evaluate_mps(mps, mps_indices))
		return self.func(mps_values)


class BlackBoxLoadMPO(BlackBox):
	"""
	Black-box representing a 2-dimensional function discretized on a 2D `Mesh`
	and quantized in a MPO with physical dimensions given by `base_mpo`. Can be
	used to load operators in MPO using tensor cross-interpolation. In practice,
	this object is equivalently represented as a MPS with physical dimensions
	of size `base_mpo**2`, whose indices can be subsequently split to form
	the required MPO.

	Parameters
	----------
	func : Callable
		The bivariate scalar function to be represented as MPO.
	mesh : Mesh
		The two-dimensional discretization where the function is discretized.
	base_mpo : int, default=2
		The required physical dimension of each index of the MPO.
	is_diagonal : bool, default=True
		Flag that helps in the convergence of TCI for diagonal operators by restricting
		the convergence evaluation to the main diagonal.

	Example
	-------
		.. code-block:: python

		# Load a 2D Gaussian function in a non-diagonal MPO using some TCI variant.

		# Define the tensorized function following the convention of having the dimension index first.
		func = lambda tensor: np.exp(-(tensor[0]**2 + tensor[1]**2))

		# Define the bivariate domain implictly using `Interval` and `Mesh`.
		start, stop = -1, 1
		num_qubits = 10
		interval = RegularInterval(start, stop, 2**n)
		mesh = Mesh([interval, interval])

		# Define the black box.
		black_box = BlackBoxLoadMPO(func, mesh)

		# Load the function in the given domain using some tci variant (e.g. DMRG, Maxvol or Greedy).
		cross_results = cross_X(black_box)
		mpo = mps_as_mpo(cross_results.mps) # Unfold into a MPO.
	"""

	mesh: Mesh
	base_mpo: int
	is_diagonal: bool
	map_matrix: np.ndarray

	# TODO: Generalize for multivariate MPOs.
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
		self.sites = int(np.lib.scimath.logn(base_mpo, mesh.dimensions[0]))

		if not (mesh.dimension == 2 and mesh.dimensions[0] == mesh.dimensions[1]):
			raise ValueError("The mesh must be bivariate for a 1d MPO.")
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
		self.record_evaluations(len(mps_indices))
		row_indices = (mps_indices // self.base_mpo) @ self.map_matrix
		col_indices = (mps_indices % self.base_mpo) @ self.map_matrix
		mesh_indices = np.hstack((row_indices, col_indices))
		return self.func(*self.mesh[mesh_indices].T)  # type: ignore
