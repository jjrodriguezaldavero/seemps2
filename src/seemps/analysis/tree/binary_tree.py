from __future__ import annotations

import numpy as np
from typing import Callable, Sequence
from dataclasses import dataclass

from ...tools import make_logger
from .branch import (
    BranchNode,
    _compute_images,
    _compute_transitions,
    _compute_cores,
)
from .sparse_mps import SparseMPS


class BinaryRootNode:
    """
    Root node of a `BinaryTree`. Stores the root function and its discretization grid. 
    The node evaluates the functional dependency `f(x_L, x_s, x_R)` where:
      - `x_L` and `x_R` are values from the left and right subtrees
      - `x_s` is selected from the grid by index `s`
    """
    def __init__(self, func: Callable, grid: Sequence):
        self.func = func
        self.grid = grid
        self.N = len(grid)

    def evaluate(self, x_L: float | None, s: int, x_R: float | None) -> float:
        if x_L is None or x_R is None:
            return 0
        x_s = self.grid[s]
        return self.func(x_L, x_s, x_R)


@dataclass
class BinaryTree:
    """
    A binary-tree representation of a multivariate function.

    This structure encodes a functional algebraic dependency as a tree composed
    of a `BinaryRootNode` (the final function applied) and a collection of
    `BranchNode` objects (the inner functional dependencies). The physical leaves
    of the tree correspond to the input dimensions of the function.

    Each node may define its own binning tolerance, which controls approximation
    accuracy by grouping similar values. The resulting binary-tree representation
    can be efficiently converted into a Matrix Product State (MPS) using
    :func:`mps_binary_tree`.

    Examples
    --------
    A function of the form

        f(g1(g11(...), g12(...)), g2(g21(...), g22(...)))

    is naturally encoded as a `BinaryTree` with:
      - a `BinaryRootNode` representing ``f``
      - `BranchNode`s representing the ``g*`` dependencies
      - leaves representing the physical variables.

    See Also
    --------
    UnaryTree
        The corresponding unary-tree representation for chain-like (non-branching)
        functional dependencies.
    mps_binary_tree
        Construct the MPS approximation from a binary tree.
    """
    left_nodes: list[BranchNode]
    root_node: BinaryRootNode
    right_nodes: list[BranchNode]

    def __post_init__(self):
        self.center = len(self.left_nodes)
        left_dimensions = [node.N for node in self.left_nodes]
        right_dimensions = [node.N for node in self.right_nodes]
        self.physical_dimensions = (
            left_dimensions + [self.root_node.N] + right_dimensions
        )
        self.length = len(self.physical_dimensions)


def mps_binary_tree(binary_tree: BinaryTree) -> SparseMPS:
    """
    Compute the MPS representation of a function encoded as a `BinaryTree`.

    These are functions with algebraic structure that can be represented as a 
    binary computational tree, where each node encodes a functional dependency
    and the root node is the final function applied. For example,

        f(g1(g11(...), g12(...)), g2(g21(...), g22(...)))

    corresponds to a binary tree with `f` as the root node, `g1` and `g2`
    as branch nodes, and the physical leaves representing the MPS input
    dimensions. Each node specifies its own binning tolerance, which
    determines the approximation accuracy by grouping similar values.

    The result is an MPS with sparse cores, structured as collections of
    sparse matrices, that efficiently approximates the multivariate
    functional dependency defined by the binary tree.

    Parameters
    ----------
    binary_tree : BinaryTree
        The binary tree representation of the function to approximate.

    Returns
    -------
    MPS
        The MPS approximation of the functional dependency described
        by the binary tree.
    """

    with make_logger(2) as logger:
        logger("Computing branch images:")
        left_images = _compute_images(binary_tree.left_nodes, logger)
        right_images = _compute_images(binary_tree.right_nodes, logger)

        logger("Computing transitions:")
        left_transitions = _compute_transitions(
            binary_tree.left_nodes, left_images, logger
        )
        right_transitions = _compute_transitions(
            binary_tree.right_nodes, right_images, logger
        )
        root_transition = {
            (k_L, s, k_R): binary_tree.root_node.evaluate(x_L, s, x_R)
            for k_L, x_L in enumerate(left_images[-1])
            for s in range(binary_tree.root_node.N)
            for k_R, x_R in enumerate(right_images[-1])
        }

        logger("Computing MPS cores:")
        left_cores = _compute_cores(left_transitions, logger)
        right_cores = _compute_cores(right_transitions, logger)

        # Compute root core
        coords = np.array(list(root_transition.keys()))
        values = np.array(list(root_transition.values()))
        χ_L = 1 + np.max(coords[:, 0])
        N = 1 + np.max(coords[:, 1])
        χ_R = 1 + np.max(coords[:, 2])
        shape = (χ_L, N, χ_R)
        root_core = np.zeros(shape)
        root_core[tuple(coords.T)] = values
        logger(f"Center core of shape {shape}.")

    cores = left_cores + [root_core] + right_cores[::-1]
    return SparseMPS(cores)
