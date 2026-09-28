# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Jacobi diagonal preconditioner kernel construction and FK call sites."""

import inspect
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.linalg.blas import block_sparse_ATA_inv_diagonal_2d
from newton._src.solvers.kamino._src.linalg.sparse_matrix import BlockDType, BlockSparseMatrices
from newton._src.solvers.kamino._src.solvers.fk import solver as fk_solver
from newton._src.solvers.kamino._src.solvers.fk.solver import ForwardKinematicsSolver
from newton.tests.utils.testing import build_unary_revolute_joint_test

_ORACLE = np.array([[0.2, 0.1]], dtype=np.float32)
_FK_CALLERS = ("_run_newton_iteration", "_solve_for_body_velocities")


def _inv_diagonal(device: str) -> np.ndarray:
    matrix = BlockSparseMatrices(nzb_dtype=BlockDType(shape=(1,), dtype=wp.float32), device=device)
    matrix.finalize(max_dims=[(2, 2)], capacities=[2])
    matrix.dims.assign([[2, 2]])
    matrix.num_nzb.assign([2])
    matrix.nzb_coords.assign([[0, 0], [1, 1]])
    matrix.nzb_values.view(dtype=wp.float32).assign([2.0, 3.0])
    mask = wp.array([True], dtype=wp.bool, device=device)
    result = wp.empty((1, 2), dtype=wp.float32, device=device)
    block_sparse_ATA_inv_diagonal_2d(matrix, result, mask, diag_offset=1.0)
    wp.synchronize_device(device)
    return result.numpy()


def _exercise_fk_call_sites(device: str) -> set[str]:
    builder = build_unary_revolute_joint_test(ground=False, limits=False)
    model = ModelKamino.from_newton(builder.finalize(device=device))
    solver = ForwardKinematicsSolver(
        model,
        ForwardKinematicsSolver.Config(use_sparsity=True, preconditioner="jacobi_diagonal"),
    )
    seen: set[str] = set()
    real = fk_solver.block_sparse_ATA_inv_diagonal_2d

    def _wrapped(*args, **kwargs):
        for frame in inspect.stack():
            if frame.function in _FK_CALLERS:
                seen.add(frame.function)
        return real(*args, **kwargs)

    fk_solver.block_sparse_ATA_inv_diagonal_2d = _wrapped
    try:
        with wp.ScopedDevice(device):
            body_q = wp.zeros(model.size.sum_of_num_bodies, dtype=wp.transformf)
            body_u = wp.zeros(model.size.sum_of_num_bodies, dtype=wp.spatial_vectorf)
            actuator_q = wp.zeros((int(solver.data.dimensions.num_actuated_coords),), dtype=wp.float32)
            actuator_u = wp.zeros((int(solver.data.dimensions.num_actuated_dofs),), dtype=wp.float32)
        solver.solve_fk(
            actuator_q,
            body_q,
            actuator_u=actuator_u,
            body_u=body_u,
            use_graph=False,
        )
        # A feasible initial pose can skip the Newton loop; call it once so both sites run.
        solver._run_newton_iteration(body_q)
        wp.synchronize_device(device)
    finally:
        fk_solver.block_sparse_ATA_inv_diagonal_2d = real
    return seen


class TestJacobiDiagonalDtype(unittest.TestCase):
    def test_ata_inverse_diagonal_matches_cpu_and_cuda(self):
        cpu = _inv_diagonal("cpu")
        cuda = _inv_diagonal("cuda:0")
        np.testing.assert_allclose(cpu, _ORACLE, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(cuda, _ORACLE, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(cpu, cuda, atol=1e-6, rtol=0.0)

    def test_fk_jacobi_diagonal_call_sites(self):
        for device in ("cpu", "cuda:0"):
            with self.subTest(device=device):
                seen = _exercise_fk_call_sites(device)
                self.assertEqual(seen, set(_FK_CALLERS))


if __name__ == "__main__":
    unittest.main()
