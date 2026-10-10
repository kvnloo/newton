# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check cancellation and representable Boltzmann gradient boundaries."""

import unittest

import numpy as np
import warp as wp

from newton.tests.test_math_boltzmann import _boltzmann_kernel, _decimal_reference
from newton.tests.unittest_utils import get_test_devices


def _evaluate(cases, device):
    """Evaluate values and native reverse-mode gradients for float32 inputs."""
    cases = np.asarray(cases, dtype=np.float32)
    inputs = [wp.array(cases[:, i], dtype=float, device=device, requires_grad=True) for i in range(3)]
    result = wp.empty(len(cases), dtype=float, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(_boltzmann_kernel, dim=len(cases), inputs=inputs, outputs=[result], device=device)
    tape.backward(grads={result: wp.ones(len(cases), dtype=float, device=device)})
    return result.numpy(), np.column_stack([tape.gradients[array].numpy() for array in inputs])


class TestMathBoltzmannLimits(unittest.TestCase):
    """Preserve means, saturation limits, and representable small derivatives."""

    def test_zero_sharpness_mean_and_symmetry(self):
        """Preserve exactly representable opposite-sign means in both input orders."""
        cases = [[1, np.nextafter(np.float32(-1), np.float32(0)), 0], [16777216, -16777215, 0]]
        cases += [[b, a, k] for a, b, k in cases]
        expected = np.array([(float(a) + float(b)) / 2 for a, b, _ in cases], dtype=np.float32)
        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                actual, _ = _evaluate(cases, device)
                np.testing.assert_array_equal(actual, expected)

    def test_zero_sharpness_extreme_input_gradients(self):
        """Retain finite input gradients when only the alpha derivative exceeds float32."""
        cases = [[3e38, -3e38, 0], [-3e38, 3e38, 0]]
        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                _, gradients = _evaluate(cases, device)
                np.testing.assert_array_equal(gradients[:, :2], np.full((2, 2), 0.5, dtype=np.float32))
                self.assertTrue(np.isposinf(gradients[:, 2]).all())

    def test_extreme_saturated_gradients(self):
        """Retain representable dominant-input gradients after a difference overflow."""
        cases = [[3e38, -3e38, 1], [-3e38, 3e38, 1], [3e38, -3e38, -1], [-3e38, 3e38, -1]]
        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                _, gradients = _evaluate(cases, device)
                np.testing.assert_array_equal(gradients, [[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]])

    def test_underflowed_weight_retains_alpha_derivative(self):
        """Retain an alpha derivative whose small weight is below float32 range."""
        cases = np.array([[1e20, 0, 1.1e-18], [1e20, 0, -1.1e-18]], dtype=np.float32)
        cases = np.concatenate((cases, cases[:, [1, 0, 2]]))
        expected = np.array([_decimal_reference(*case)[1] for case in cases], dtype=np.float32)
        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                _, gradients = _evaluate(cases, device)
                self.assertTrue(np.isfinite(gradients).all())
                self.assertTrue(np.all(gradients[:, 2] > 0))
                np.testing.assert_allclose(gradients, expected, rtol=2e-5, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
