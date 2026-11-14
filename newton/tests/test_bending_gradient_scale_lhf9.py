# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for the dihedral-angle gradient shared by two solvers."""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.semi_implicit.kernels_particle import eval_bending
from newton._src.solvers.xpbd.kernels import bending_constraint


class TestBendingGradientScale(unittest.TestCase):
    @staticmethod
    def _hinge(length: float) -> np.ndarray:
        fold = math.pi / 6.0
        return np.array(
            [
                [0.0, length * math.cos(fold), length * math.sin(fold)],
                [0.0, -length, 0.0],
                [0.0, 0.0, 0.0],
                [length, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _inputs(length: float) -> tuple:
        device = "cpu"
        return (
            wp.array(TestBendingGradientScale._hinge(length), dtype=wp.vec3, device=device),
            wp.zeros(4, dtype=wp.vec3, device=device),
            wp.array([[0, 1, 2, 3]], dtype=int, device=device),
            wp.zeros(1, dtype=float, device=device),
        )

    def test_semi_implicit_bending_force_is_scale_invariant(self):
        """Preserve the physical bending force from 1 m to submillimeter hinges."""
        for length in (1.0, 0.1, 0.01, 0.0005):
            with self.subTest(length=length):
                x, v, indices, rest = self._inputs(length)
                force = wp.zeros(4, dtype=wp.vec3, device="cpu")
                wp.launch(
                    eval_bending,
                    dim=1,
                    inputs=[
                        x,
                        v,
                        indices,
                        rest,
                        wp.array([[1.0, 0.0]], dtype=float, device="cpu"),
                    ],
                    outputs=[force],
                    device="cpu",
                )
                # |e|=L, |d(theta)/dx0|=1/L, delta(theta)=pi/6.
                expected_force = math.pi / 6.0
                self.assertAlmostEqual(
                    float(np.linalg.norm(force.numpy()[0])),
                    expected_force,
                    delta=1.0e-4,
                )

    def test_xpbd_bending_correction_scales_with_length(self):
        """Scale position corrections with hinge size, including tiny triangles."""
        for length in (1.0, 0.1, 0.01, 0.0005):
            with self.subTest(length=length):
                x, v, indices, rest = self._inputs(length)
                delta = wp.zeros(4, dtype=wp.vec3, device="cpu")
                wp.launch(
                    bending_constraint,
                    dim=1,
                    inputs=[
                        x,
                        v,
                        wp.ones(4, dtype=float, device="cpu"),
                        indices,
                        rest,
                        wp.array([[1.0e20, 0.0]], dtype=float, device="cpu"),
                        0.001,
                        wp.zeros(1, dtype=float, device="cpu"),
                    ],
                    outputs=[wp.zeros(1, dtype=float, device="cpu"), delta],
                    device="cpu",
                )
                # Sum of four squared gradient norms is (4+sqrt(3))/L^2.
                expected_delta = (math.pi / 6.0) * length / (4.0 + math.sqrt(3.0))
                self.assertAlmostEqual(
                    float(np.linalg.norm(delta.numpy()[0])),
                    expected_delta,
                    delta=max(1.0e-8, length * 5.0e-5),
                )


if __name__ == "__main__":
    unittest.main()
