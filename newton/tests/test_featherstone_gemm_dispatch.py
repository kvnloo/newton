# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.featherstone.kernels import (
    eval_dense_gemm_batched,
    eval_dense_gemm_batched_elementwise,
)
from newton._src.solvers.featherstone.solver_featherstone import _prefer_elementwise_dense_gemm
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestFeatherstoneGemmDispatch(unittest.TestCase):
    def test_dispatch_policy_preserves_tiny_products(self):
        """Verify the measured dispatch boundary keeps tiny products on the serial kernel."""
        self.assertFalse(_prefer_elementwise_dense_gemm(False, 1, 1024))
        self.assertFalse(_prefer_elementwise_dense_gemm(True, 256, 1))
        self.assertFalse(_prefer_elementwise_dense_gemm(True, 256, 4))
        self.assertFalse(_prefer_elementwise_dense_gemm(True, 1, 6))
        self.assertTrue(_prefer_elementwise_dense_gemm(True, 8, 6))
        self.assertTrue(_prefer_elementwise_dense_gemm(True, 1, 7))


def _make_batch(cases, rng):
    m = np.asarray([case[0] for case in cases], dtype=np.int32)
    n = np.asarray([case[1] for case in cases], dtype=np.int32)
    p = np.asarray([case[2] for case in cases], dtype=np.int32)

    a_counts = m * p
    b_counts = p * n
    c_counts = m * n
    a_start = np.concatenate(([0], np.cumsum(a_counts[:-1]))).astype(np.int32)
    b_start = np.concatenate(([0], np.cumsum(b_counts[:-1]))).astype(np.int32)
    c_start = np.concatenate(([0], np.cumsum(c_counts[:-1]))).astype(np.int32)

    return {
        "m": m,
        "n": n,
        "p": p,
        "a_start": a_start,
        "b_start": b_start,
        "c_start": c_start,
        "A": rng.standard_normal(int(np.sum(a_counts))).astype(np.float32),
        "B": rng.standard_normal(int(np.sum(b_counts))).astype(np.float32),
        "C_size": int(np.sum(c_counts)),
        "max_output_elements": int(np.max(c_counts)),
    }


def _launch_pair(device, batch, transpose_A, transpose_B, requires_grad=False):
    m = wp.array(batch["m"], dtype=wp.int32, device=device)
    n = wp.array(batch["n"], dtype=wp.int32, device=device)
    p = wp.array(batch["p"], dtype=wp.int32, device=device)
    a_start = wp.array(batch["a_start"], dtype=wp.int32, device=device)
    b_start = wp.array(batch["b_start"], dtype=wp.int32, device=device)
    c_start = wp.array(batch["c_start"], dtype=wp.int32, device=device)

    def run(kernel, dim):
        A = wp.array(batch["A"], dtype=wp.float32, device=device, requires_grad=requires_grad)
        B = wp.array(batch["B"], dtype=wp.float32, device=device, requires_grad=requires_grad)
        C = wp.zeros(batch["C_size"], dtype=wp.float32, device=device, requires_grad=requires_grad)
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel,
                dim=dim,
                inputs=[
                    m,
                    n,
                    p,
                    transpose_A,
                    transpose_B,
                    a_start,
                    b_start,
                    c_start,
                    A,
                    B,
                ],
                outputs=[C],
                device=device,
            )

        if requires_grad:
            tape.backward(grads={C: wp.ones_like(C)})
            result = (C.numpy(), A.grad.numpy(), B.grad.numpy())
            tape.zero()
            return result
        return (C.numpy(),)

    serial = run(eval_dense_gemm_batched, len(batch["m"]))
    element = run(
        eval_dense_gemm_batched_elementwise,
        (len(batch["m"]), batch["max_output_elements"]),
    )
    return serial, element


def test_elementwise_gemm_matches_serial(test: TestFeatherstoneGemmDispatch, device):
    """Verify element-parallel GEMM matches the serial kernel in forward and backward passes."""
    rng = np.random.default_rng(7)
    cases = [
        (1, 1, 6),
        (2, 4, 12),
        (6, 3, 6),
        (8, 5, 8),
    ]
    batch = _make_batch(cases, rng)

    for transpose_A, transpose_B in ((False, False), (True, False), (False, True)):
        with test.subTest(transpose_A=transpose_A, transpose_B=transpose_B):
            serial, element = _launch_pair(
                device,
                batch,
                transpose_A,
                transpose_B,
                requires_grad=True,
            )
            for serial_array, element_array in zip(serial, element, strict=True):
                np.testing.assert_allclose(serial_array, element_array, atol=2.0e-5, rtol=2.0e-5)


devices = get_test_devices()
add_function_test(
    TestFeatherstoneGemmDispatch,
    "test_elementwise_gemm_matches_serial",
    test_elementwise_gemm_matches_serial,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
