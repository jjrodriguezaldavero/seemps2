from __future__ import annotations

import numpy as np
from scipy.sparse import lil_array
from typing import Callable, Sequence

from ...tools import Logger, make_logger
from ...typing import Vector
from .sparse_mps import SparseCore


class BranchNode:
    def __init__(
        self,
        func: Callable,
        grid: Sequence,
        binning_tol: float | None = None,
        max_rank: int | None = None,
    ):
        """
        Internal node of a computational tree, used in both `BinaryTree` and `UnaryTree`.

        A branch node represents an intermediate functional dependency in the
        computational tree of a multivariate function. It encodes a mapping
        between the values of its child nodes and a discretization grid, which
        acts as the "physical index" of the MPS representation. Together with the 
        root node and leaves, `BranchNode` objects provide a complete description 
        of the algebraic structure of the function to be approximated as an MPS.

        Parameters
        ----------
        func : Callable
            The functional dependence associated with the node. This function
            determines how values from the child nodes are combined with the
            discretization grid.
        grid : Sequence
            A finite set of grid points that discretize the functional variable
            at this node. The grid index serves as a local degree of freedom
            (a physical dimension in the resulting MPS).
        binning_tol : float, optional
            A tolerance parameter controlling the grouping (binning) of similar
            function values. By merging values that fall within this tolerance,
            the resulting MPS can achieve a more compact representation at the
            cost of some approximation error. Smaller tolerances increase accuracy
            but may also increase the MPS bond dimensions.
        max_rank : int, optional
            An upper bound on the bond dimension of the MPS at this node. This
            allows explicit control over resource usage, potentially at the cost
            of approximation fidelity.

        Branch nodes form the "interior" of the functional tree: in a `BinaryTree`,
        they sit between the `BinaryRootNode` and the leaves representing the
        physical input variables; in a `UnaryTree`, they occur along the chain of
        functional dependencies leading up to the root.
        """
        self.func = func
        self.grid = grid
        self.binning_tol = binning_tol
        self.max_rank = max_rank
        self.N = len(grid)

    def evaluate(self, x_in: float | None, s: int) -> float | None:
        if x_in is None:
            return None
        x_s = self.grid[s]
        return self.func(x_in, x_s)

    def compute_image(
        self, values: Vector, default_tol: float = 1e-4, tol_multiplier: float = 1.25
    ) -> Vector:
        logger = make_logger(3)

        # Compute the image. TODO: Vectorize
        image_matrix = np.zeros((len(values), self.N))
        for j, x_in in enumerate(values):
            for s in range(self.N):
                value = self.evaluate(x_in, s)
                image_matrix[j, s] = np.nan if value is None else value

        # Format the image
        image = image_matrix.reshape(-1)
        image = image[~np.isnan(image)]
        image = np.unique(image)
        logger(f"\tIncoming image of size {len(image)}.")

        # Compress the image
        if self.binning_tol is not None or self.max_rank is not None:
            binning_tol = default_tol if self.binning_tol is None else self.binning_tol
            image = self._bin_image(image, binning_tol)
            logger(
                f"\tImage compressed to {len(image)} bins with tolerance {binning_tol:.3e}."
            )

            if self.max_rank is not None:
                while len(image) > self.max_rank:
                    binning_tol *= tol_multiplier
                    image = self._bin_image(image, binning_tol)
                    logger(
                        f"\tImage compressed to {len(image)} bins with tolerance {binning_tol:.3e}."
                    )

        logger.close()
        return image

    @staticmethod
    def _bin_image(image: Vector, binning_tol: float) -> Vector:
        binned_image = []
        bin = [image[0]]
        for x in image[1:]:
            error = abs((x - bin[0]) / bin[0])
            if error <= binning_tol:
                bin.append(x)
            else:
                binned_image.append(np.mean(bin))
                bin = [x]
        binned_image.append(np.mean(bin))
        return np.array(binned_image)


def _compute_images(nodes: list[BranchNode], logger: Logger = Logger()) -> list[Vector]:
    l = len(nodes)
    images = [np.array([0.0])]
    for i, node in enumerate(nodes):
        image = node.compute_image(images[-1])
        logger(f"Node {(i + 1)}/{l} | Image of size {len(image)}.")
        images.append(image)
    return images


def _compute_transitions(
    nodes: list[BranchNode], images: list[Vector], logger: Logger = Logger()
) -> list[dict]:
    l = len(nodes)
    transitions = []
    for i, node in enumerate(nodes):
        R_in = images[i]
        R_out = images[i + 1]

        # Create lookup tables for fast O(1) search
        R_out_lookup = {value: idx for idx, value in enumerate(R_out)}

        transition = {}
        for s in range(node.N):
            for k_in, x_in in enumerate(R_in):
                x_out = node.evaluate(x_in, s)
                if x_out is not None:
                    k_out = R_out_lookup.get(x_out, None)
                    # If not found, find closest index in R_out with np.searchsorted
                    if k_out is None:
                        k_out = np.searchsorted(R_out, x_out, side="left")
                        k_out = min(k_out, len(R_out) - 1)
                    transition[(k_in, s)] = k_out
        logger(f"Node {(i + 1)}/{l} | Transition of size {len(transition)}.")
        transitions.append(transition)

    return transitions


def _compute_cores(
    transitions: list[dict], logger: Logger = Logger()
) -> list[SparseCore]:
    cores = []
    l = len(transitions)
    for i, transition in enumerate(transitions):
        coords = np.array([(k_in, s, k_out) for (k_in, s), k_out in transition.items()])
        χ_L = 1 + np.max(coords[:, 0])
        N = 1 + np.max(coords[:, 1])
        χ_R = 1 + np.max(coords[:, 2])

        data = [lil_array((χ_L, χ_R)) for _ in range(N)]
        for k_in, s, k_out in coords:
            data[s][k_in, k_out] += 1

        core = SparseCore([matrix.tocsr() for matrix in data])
        logger(f"Node {(i + 1)}/{l} | Core of shape {core.shape}.")
        cores.append(core)

    return cores
