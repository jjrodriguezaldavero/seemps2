import numpy as np
from typing import Optional, Callable, Sequence
from dataclasses import dataclass

from seemps.tools import make_logger

from .branch import (
    BranchNode,
    _compute_images,
    _compute_transitions,
    _compute_cores,
)
from .sparse_mps import SparseMPS


class BinaryRootNode:
    def __init__(self, func: Callable, grid: Sequence):
        self.func = func
        self.grid = grid
        self.N = len(grid)

    def evaluate(self, x_L: Optional[float], s: int, x_R: Optional[float]) -> float:
        if x_L is None or x_R is None:
            return 0
        x_s = self.grid[s]
        return self.func(x_L, x_s, x_R)


@dataclass
class BinaryTree:
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
