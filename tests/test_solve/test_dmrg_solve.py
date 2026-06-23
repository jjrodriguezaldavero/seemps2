import importlib
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .problems import TestSolveProblems, make_complex_problem, make_identity_problem
from seemps.solve.dmrg import _solve_local, dmrg_solve
from seemps.state import CanonicalMPS


dmrg_module = importlib.import_module("seemps.solve.dmrg")


class _FakeQuadraticForm:
    def __init__(self, op):
        self.state = [
            np.ones((1, 2, 1)),
            np.ones((1, 3, 1)),
        ]
        self.op = op

    def two_site_operator(self, _i):
        return self.op


class TestDMRGSolve(TestSolveProblems):
    def test_basic_problems(self):
        for p in self.DMRG_PROBLEMS:
            with self.subTest(msg=p.name):
                x, r = dmrg_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    atol=p.tolerance,
                    rtol=0.0,
                )
                self.assertTrue(r < p.tolerance)
                exact_x = np.linalg.solve(
                    p.invertible_mpo.to_matrix(), p.get_rhs().to_vector()
                )
                self.assertTrue(np.linalg.norm(x.to_vector() - exact_x) < p.tolerance)

    def test_atol_is_respected_with_residuals(self):
        for p in self.DMRG_PROBLEMS:
            with self.subTest(msg=p.name):
                atol = p.tolerance
                x, r = dmrg_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    atol=atol,
                    rtol=0.0,
                    compute_residuals=True,
                )
                self.assertIsNotNone(r)
                self.assertLess(r, atol)

    def test_atol_is_respected_without_residuals(self):
        for p in self.DMRG_PROBLEMS:
            with self.subTest(msg=p.name):
                atol = p.tolerance
                x, r = dmrg_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    atol=atol,
                    rtol=0.0,
                    compute_residuals=False,
                )
                self.assertIsNone(r)
                residual = float(
                    (p.invertible_mpo @ x - p.get_rhs()).norm()
                )
                self.assertLess(residual, 10 * atol)

    def test_atol_greater_than_rtol(self):
        for p in self.DMRG_PROBLEMS:
            with self.subTest(msg=p.name):
                b_norm = p.get_rhs().norm()
                atol = 10.0 * b_norm
                rtol = 1e-5
                x, r = dmrg_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    atol=atol,
                    rtol=rtol,
                    compute_residuals=True,
                )
                self.assertLess(r, atol)

    def test_accepts_canonical_guess_with_interior_center(self):
        p = make_identity_problem(5, self.rng)
        b = p.get_rhs()
        guess = CanonicalMPS(b, center=2)

        x, r = dmrg_solve(
            p.invertible_mpo,
            b,
            guess=guess,
            atol=1e-5,
            rtol=0.0,
        )

        self.assertLessEqual(r, 1e-5)
        self.assertLess(np.linalg.norm(x.to_vector() - b.to_vector()), 1e-5)

    def test_maxiter_one_performs_one_sweep(self):
        p = make_complex_problem(3, self.rng)
        b = p.get_rhs()

        x, r = dmrg_solve(
            p.invertible_mpo,
            b,
            guess=b,
            maxiter=1,
            atol=1e-5,
            rtol=0.0,
        )

        exact_x = np.linalg.solve(p.invertible_mpo.to_matrix(), b.to_vector())
        self.assertLess(r, p.tolerance)
        self.assertLess(np.linalg.norm(x.to_vector() - exact_x), p.tolerance)

    def test_local_solve_does_not_check_residual_by_default(self):
        matvecs = 0

        def matvec(v):
            nonlocal matvecs
            matvecs += 1
            return v

        op = LinearOperator((6, 6), matvec=matvec, dtype=float)
        qf = _FakeQuadraticForm(op)
        b_tensor = np.ones((1, 2, 3, 1))

        def solver(Op, b_flat, x0, atol, rtol, maxiter, callback):
            self.assertEqual(matvecs, 0)
            self.assertEqual(atol, 1e-4)
            self.assertEqual(rtol, 1e-3)
            self.assertEqual(maxiter, 7)
            self.assertIsNone(callback)
            return x0.copy(), 0

        x = _solve_local(qf, 0, b_tensor, 1e-4, 1e-3, 7, solver)

        self.assertEqual(matvecs, 0)
        self.assertEqual(x.shape, b_tensor.shape)

    def test_local_solve_can_skip_when_residual_check_is_enabled(self):
        matvecs = 0

        def matvec(v):
            nonlocal matvecs
            matvecs += 1
            return v

        op = LinearOperator((6, 6), matvec=matvec, dtype=float)
        qf = _FakeQuadraticForm(op)
        b_tensor = np.ones((1, 2, 3, 1))

        def solver(*_args, **_kwargs):
            raise AssertionError("solver should not be called")

        x = _solve_local(qf, 0, b_tensor, 1e-4, 1e-3, 7, solver, True)

        self.assertEqual(matvecs, 1)
        self.assertEqual(x.shape, b_tensor.shape)

    def test_local_solver_controls_are_forwarded(self):
        p = make_identity_problem(3, self.rng)
        calls = []

        def recording_solver(Op, b_flat, _x0, atol, rtol, maxiter, callback):
            calls.append((atol, rtol, maxiter, callback))
            return b_flat.copy(), 0

        dmrg_module._SOLVERS["recording"] = recording_solver
        try:
            dmrg_solve(
                p.invertible_mpo,
                p.get_rhs(),
                guess=p.get_rhs(),
                maxiter=1,
                atol=1e-8,
                rtol=1e-8,
                compute_residuals=False,
                method="recording",
                local_atol=1e-4,
                local_rtol=1e-3,
                local_maxiter=5,
            )
        finally:
            del dmrg_module._SOLVERS["recording"]

        self.assertGreater(len(calls), 0)
        self.assertTrue(all(call[:3] == (1e-4, 1e-3, 5) for call in calls))
