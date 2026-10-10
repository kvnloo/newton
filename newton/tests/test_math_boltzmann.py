# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the public Boltzmann average against high-precision scalar math."""

import unittest
from decimal import Decimal, localcontext

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_test_devices


@wp.kernel
def _boltzmann_kernel(a: wp.array[float], b: wp.array[float], alpha: wp.array[float], result: wp.array[float]):
    """Evaluate the public weighted average with its native float32 inputs."""
    i = wp.tid()
    result[i] = newton.math.boltzmann(a[i], b[i], alpha[i])


def _decimal_reference(a, b, alpha):
    """Evaluate the unshifted definition and analytic derivatives at high precision."""
    with localcontext() as context:
        context.prec = 80
        context.Emax = 10000000
        context.Emin = -10000000
        a, b, alpha = (Decimal.from_float(float(value)) for value in (a, b, alpha))
        ea = (alpha * a).exp()
        eb = (alpha * b).exp()
        total = ea + eb
        value = (a * ea + b * eb) / total
        pa, pb = ea / total, eb / total
        gradients = (
            pa * (1 + alpha * (a - value)),
            pb * (1 + alpha * (b - value)),
            pa * pb * (a - b) ** 2,
        )
        return float(value), [float(component) for component in gradients]


class TestMathBoltzmann(unittest.TestCase):
    """Cover finite scalar values and differentiable limiting controls."""

    def test_boltzmann_finite_values(self):
        """Match the defining average after common shifts and with either alpha sign."""
        cases = np.array(
            [
                [1.0, 2.0, 0.75],
                [100.0, 99.0, 1.0],
                [-1000.0, -999.0, 1.0],
                [-100.0, -99.0, -1.0],
                [1000.0, 999.0, -1.0],
                [100000.0, 100000.03125, 32.0],
                [100.0, 100.0, 1.0],
                [-1000.0, -1000.0, 1.0],
                [2.0, -3.0, 2.0],
                [2.0, -3.0, -2.0],
                [100.0, 99.0, 0.0],
                [3.0e38, 3.0e38, 0.0],
                [-3.0e38, 3.0e38, 0.0],
            ],
            dtype=np.float32,
        )
        expected = [_decimal_reference(*case)[0] for case in cases]
        for device in get_test_devices(mode="basic"):
            for swap in (False, True):
                with self.subTest(device=device, swap=swap):
                    a, b = (cases[:, 1], cases[:, 0]) if swap else (cases[:, 0], cases[:, 1])
                    result = wp.empty(len(cases), dtype=float, device=device)
                    wp.launch(
                        _boltzmann_kernel,
                        dim=len(cases),
                        inputs=[
                            wp.array(a, dtype=float, device=device),
                            wp.array(b, dtype=float, device=device),
                            wp.array(cases[:, 2], dtype=float, device=device),
                        ],
                        outputs=[result],
                        device=device,
                    )
                    actual = result.numpy()
                    self.assertTrue(np.isfinite(actual).all(), msg=f"Non-finite weighted averages: {actual}")
                    np.testing.assert_allclose(actual, expected, rtol=2.0e-7, atol=1.0e-6)
                    self.assertTrue(np.all(actual >= np.minimum(a, b)))
                    self.assertTrue(np.all(actual <= np.maximum(a, b)))

    def test_boltzmann_extreme_exponents(self):
        """Reach the dominant-input limit when finite exponent products overflow."""
        cases = np.array(
            [[2.0, 3.0, 2.0e38], [-2.0, -3.0, 2.0e38], [2.0, 3.0, -2.0e38], [-2.0, -3.0, -2.0e38]],
            dtype=np.float32,
        )
        # Each losing weight is smaller than exp(-2e38), below float32 resolution.
        expected = [3.0, -2.0, 2.0, -3.0]
        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                result = wp.empty(len(cases), dtype=float, device=device)
                wp.launch(
                    _boltzmann_kernel,
                    dim=len(cases),
                    inputs=[wp.array(cases[:, column], dtype=float, device=device) for column in range(3)],
                    outputs=[result],
                    device=device,
                )
                np.testing.assert_array_equal(result.numpy(), expected)

    def test_boltzmann_gradients(self):
        """Preserve analytic derivatives across shifts, equal inputs, and alpha zero."""
        cases = np.array(
            [
                [1.0, 2.0, 0.75],
                [100.0, 99.0, 1.0],
                [-100.0, -99.0, 1.0],
                [-100.0, -99.0, -1.0],
                [100.0, 100.0, 1.0],
                [100.0, 99.0, 0.0],
                [2.0, -3.0, -2.0],
                [2.0, -3.0, 2.0],
                [10000.0, 10000.25, 4.0],
            ],
            dtype=np.float32,
        )
        expected = np.array([_decimal_reference(*case)[1] for case in cases])
        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                inputs = [
                    wp.array(cases[:, column], dtype=float, device=device, requires_grad=True) for column in range(3)
                ]
                result = wp.empty(len(cases), dtype=float, device=device, requires_grad=True)
                with wp.Tape() as tape:
                    wp.launch(_boltzmann_kernel, dim=len(cases), inputs=inputs, outputs=[result], device=device)
                tape.backward(grads={result: wp.ones(len(cases), dtype=float, device=device)})
                actual = np.column_stack([tape.gradients[array].numpy() for array in inputs])
                self.assertTrue(np.isfinite(actual).all(), msg=f"Non-finite derivatives: {actual}")
                np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
