# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU-only affine pressure-tangent witness for upstream Newton PR #4169."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction_hydroelastic import _linearize_contact
from newton._src.geometry.sdf_hydroelastic import _mc_pressure_gradient, vec8f
from newton.geometry import HydroelasticSDF


@wp.kernel
def _probe_affine_springs(
    moduli: wp.array[wp.vec2],
    gradients: wp.array[float],
    springs: wp.array[wp.vec2],
):
    i = wp.tid()
    # Unit-voxel affine SDFs: sdf_a = z and sdf_b = -z.
    a = vec8f(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    b = vec8f(0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0)
    gradient = _mc_pressure_gradient(
        a, b,
        wp.vec3(0.25, 0.5, 0.75),
        wp.vec3(1.0, 1.0, 1.0),
        wp.vec3(0.0, 0.0, 1.0),
        moduli[i][0], moduli[i][1],
    )
    # Geometric secant: area=1 m^2, pressure=100 Pa, depth=-0.02 m.
    stiffness, distance = _linearize_contact(5000.0, -0.02, 100.0, gradient)
    gradients[i] = gradient
    springs[i] = wp.vec2(stiffness, distance)


class TestPressureTangentCPU(unittest.TestCase):
    def _evaluate_cpu(self, slopes):
        wp.init()
        rows = wp.array(np.asarray(slopes, dtype=np.float32), dtype=wp.vec2, device="cpu")
        gradients = wp.zeros(len(slopes), dtype=float, device="cpu")
        springs = wp.zeros(len(slopes), dtype=wp.vec2, device="cpu")
        wp.launch(
            _probe_affine_springs,
            dim=len(slopes),
            inputs=[rows, gradients, springs],
            device="cpu",
        )
        return gradients.numpy(), springs.numpy()

    def test_affine_series_tangent_and_preserved_force_cpu(self):
        gradients, springs = self._evaluate_cpu(
            [(10000.0, 10000.0), (10000.0, 20000.0), (5000.0, 20000.0)]
        )
        expected = (
            (5000.0, 5000.0, -0.020),
            (6666.6666667, 6666.6666667, -0.015),
            (4000.0, 4000.0, -0.025),
        )
        for i, (expected_g, expected_k, expected_phi) in enumerate(expected):
            with self.subTest(case=i):
                self.assertAlmostEqual(float(gradients[i]), expected_g, delta=0.05)
                self.assertAlmostEqual(float(springs[i][0]), expected_k, delta=0.05)
                self.assertAlmostEqual(float(springs[i][1]), expected_phi, delta=1.0e-6)
                self.assertAlmostEqual(
                    -float(springs[i][0]) * float(springs[i][1]), 100.0, delta=0.02
                )

    def test_zero_gradient_retains_geometric_secant_cpu(self):
        gradients, springs = self._evaluate_cpu([(10000.0, 0.0)])
        self.assertEqual(float(gradients[0]), 0.0)
        self.assertAlmostEqual(float(springs[0][0]), 5000.0, delta=0.05)
        self.assertAlmostEqual(float(springs[0][1]), -0.02, delta=1.0e-6)

    def test_new_option_is_off_by_default(self):
        self.assertFalse(HydroelasticSDF.Config().use_pressure_gradient)
        self.assertTrue(HydroelasticSDF.Config(use_pressure_gradient=True).use_pressure_gradient)


if __name__ == "__main__":
    unittest.main()
