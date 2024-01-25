from __future__ import annotations
import numpy as np
from abc import abstractmethod
from seemps.state import MPS, product_state
from seemps.operators import MPO
from seemps.analysis.evolution import EvolutionResults
from ..tools import TestCase


class TestItimeCase(TestCase):
    Sz = np.diag([0.5, -0.5])

    def make_problem_and_solution(self, size: int) -> tuple[MPO, MPS]:
        A = np.zeros((2, 2, 2, 2))
        A[0, :, :, 0] = np.eye(2)
        A[1, :, :, 1] = np.eye(2)
        A[0, :, :, 1] = self.Sz
        tensors = [A] * size
        tensors[0] = tensors[0][[0], :, :, :]
        tensors[-1] = tensors[-1][:, :, :, [1]]
        return MPO(tensors), product_state([0, 1], size)

    def make_callback(self):
        norms = []

        def callback_func(state: MPS, results: EvolutionResults):
            self.assertIsInstance(results, EvolutionResults)
            self.assertIsInstance(state, MPS)
            norms.append(np.sqrt(state.norm_squared()))
            return None

        return callback_func, norms

    @abstractmethod
    def solve(self, H: MPO, state: MPS, **kwdargs) -> EvolutionResults:
        raise Exception("solve() not implemented")

    def test_itime_solver_with_local_field(self):
        if type(self) != TestItimeCase:
            N = 4
            H, exact = self.make_problem_and_solution(N)
            guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
            result = self.solve(H, guess)
            self.assertAlmostEqual(result.energy, H.expectation(exact))
            self.assertSimilar(result.state, exact, atol=1e-4)

    def test_itime_solver_with_callback(self):
        if type(self) != TestItimeCase:
            N = 4
            H, exact = self.make_problem_and_solution(N)
            guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
            callback_func, norms = self.make_callback()
            result = self.solve(H, guess, maxiter=10, callback=callback_func)
            self.assertSimilar(norms, np.ones(len(norms)))
