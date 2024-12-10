import numpy as np
from typing import Callable, Optional
from abc import ABC, abstractmethod

from ...state import MPS
from ...analysis.mesh import Mesh
from ...typing import Vector
from ..sampling import evaluate_mps
from .tools import Matrix


class BlackBox(ABC):
    def __init__(self, func: Callable, physical_dimensions: list):
        self.func = func
        self.physical_dimensions = physical_dimensions
        self.evals = 0

    @abstractmethod
    def _indices_to_points(self, mps_indices: Matrix) -> Matrix: ...

    @abstractmethod
    def __getitem__(self, mps_indices: Matrix) -> Vector: ...


class BlackBoxMesh(BlackBox):
    """Implicit black-box evaluation of a function on a mesh with the given structure."""

    # TODO: Figure out how to use this to load MPO.
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

    def _indices_to_points(self, mps_indices: Matrix) -> Matrix:
        mesh_indices = mps_indices @ self.map_matrix
        return self.mesh[mesh_indices].T

    def __getitem__(self, mps_indices: Matrix) -> Vector:
        self.evals += len(mps_indices)
        points = self._indices_to_points(mps_indices)
        return self.func(points)  # Convention: channel first


class BlackBoxMPS(BlackBox):
    """Implicit black-box evaluation of a function on a list of MPS."""

    def __init__(self, func: Callable, mps_list: list[MPS]):
        physical_dimensions = mps_list[0].physical_dimensions()
        for mps in mps_list:
            if mps.physical_dimensions() != physical_dimensions:
                raise ValueError("The MPS physical dimensions do not match.")

        super().__init__(func, physical_dimensions)
        self.mps_list = mps_list

    def _indices_to_points(self, mps_indices: Matrix) -> Matrix:
        return np.array([evaluate_mps(mps, mps_indices)] for mps in self.mps_list)

    def __getitem__(self, mps_indices: Matrix) -> Vector:
        self.evals += len(mps_indices)
        samples = self._indices_to_points(mps_indices)
        return self.func(samples)
