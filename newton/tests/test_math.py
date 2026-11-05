# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for math utilities in ``newton.math``."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_test_devices


@wp.kernel(enable_backward=False)
def _allclose_kernel(a: wp.array[wp.vec3], b: wp.array[wp.vec3], result: wp.array[bool]):
    """Evaluate vector tolerance comparisons on a Warp device."""
    i = wp.tid()
    result[i] = newton.math.vec_allclose(a[i], b[i], rtol=0.1, atol=0.25)


@wp.kernel(enable_backward=False)
def _inside_limits_kernel(
    a: wp.array[wp.vec3], lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], result: wp.array[bool]
):
    """Evaluate inclusive vector bounds on a Warp device."""
    i = wp.tid()
    result[i] = newton.math.vec_inside_limits(a[i], lower[i], upper[i])


@wp.kernel
def _quat_velocity_kernel(
    now: wp.array[wp.quat], previous: wp.array[wp.quat], dt: float, result: wp.array[wp.vec3]
):
    i = wp.tid()
    result[i] = newton.math.quat_velocity(now[i], previous[i], dt)


@wp.kernel
def _quat_velocity_x_kernel(now: wp.array[wp.quat], previous: wp.array[wp.quat], result: wp.array[float]):
    velocity = newton.math.quat_velocity(now[0], previous[0], 0.5)
    result[0] = velocity[0]


class TestMathVectorPredicates(unittest.TestCase):
    """Check finite boundaries and unordered values in vector predicates."""

    def test_vec_allclose_rejects_nan(self):
        """Reject NaN operands while accepting the tolerance boundary."""
        a = np.array(
            [[11.25, 0.0, 2.0], [11.5, 0.0, 2.0], [np.nan, 0.0, 2.0], [10.0, 0.0, 2.0]],
            dtype=np.float32,
        )
        b = np.array(
            [[10.0, 0.0, 2.0], [10.0, 0.0, 2.0], [10.0, 0.0, 2.0], [np.nan, 0.0, 2.0]],
            dtype=np.float32,
        )
        expected = [True, False, False, False]

        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                result = wp.empty(len(a), dtype=bool, device=device)
                wp.launch(
                    _allclose_kernel,
                    dim=len(a),
                    inputs=[wp.array(a, dtype=wp.vec3, device=device), wp.array(b, dtype=wp.vec3, device=device)],
                    outputs=[result],
                    device=device,
                )
                np.testing.assert_array_equal(result.numpy(), expected)

    def test_vec_allclose_handles_infinities(self):
        """Match NumPy for infinite operands without accepting other invalid elements."""
        values = (0.0, np.inf, -np.inf, np.nan)
        pairs = [(left, right) for left in values for right in values]
        a = np.array(
            [[left, 0.0, 2.0] for left, _ in pairs] + [[np.inf, 0.0, 3.0], [np.inf, 0.0, np.nan]],
            dtype=np.float32,
        )
        b = np.array(
            [[right, 0.0, 2.0] for _, right in pairs] + [[np.inf, 0.0, 2.0], [np.inf, 0.0, 2.0]],
            dtype=np.float32,
        )
        expected = [np.allclose(left, right, rtol=0.1, atol=0.25) for left, right in zip(a, b, strict=True)]

        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                result = wp.empty(len(a), dtype=bool, device=device)
                wp.launch(
                    _allclose_kernel,
                    dim=len(a),
                    inputs=[wp.array(a, dtype=wp.vec3, device=device), wp.array(b, dtype=wp.vec3, device=device)],
                    outputs=[result],
                    device=device,
                )
                np.testing.assert_array_equal(result.numpy(), expected)

    def test_vec_inside_limits_rejects_nan(self):
        """Reject NaN values and bounds while accepting inclusive endpoints."""
        a = np.array(
            [[0.0, 1.0, 2.0], [-2.0, 1.0, 2.0], [2.0, 1.0, 2.0], [np.nan, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            dtype=np.float32,
        )
        lower = np.array(
            [
                [-1.0, 0.0, 2.0],
                [-1.0, 0.0, 2.0],
                [-1.0, 0.0, 2.0],
                [-1.0, 0.0, 2.0],
                [np.nan, 0.0, 2.0],
                [-1.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        upper = np.array(
            [[0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [np.nan, 1.0, 3.0]],
            dtype=np.float32,
        )
        expected = [True, False, False, False, False, False]

        for device in get_test_devices(mode="basic"):
            with self.subTest(device=device):
                result = wp.empty(len(a), dtype=bool, device=device)
                wp.launch(
                    _inside_limits_kernel,
                    dim=len(a),
                    inputs=[
                        wp.array(a, dtype=wp.vec3, device=device),
                        wp.array(lower, dtype=wp.vec3, device=device),
                        wp.array(upper, dtype=wp.vec3, device=device),
                    ],
                    outputs=[result],
                    device=device,
                )
                np.testing.assert_array_equal(result.numpy(), expected)


class TestMathQuaternionVelocity(unittest.TestCase):
    """Check small-angle quaternion velocities and gradients on CPU."""

    def test_zero_tiny_and_quarter_turn_cpu(self):
        """Preserve a subnormal angle without disturbing finite rotations."""
        half_sqrt = np.sqrt(0.5)
        now = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, -1.0],
                [1.0e-24, 0.0, 0.0, 1.0],
                [0.0, 0.0, half_sqrt, half_sqrt],
            ],
            dtype=np.float32,
        )
        previous = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (len(now), 1))
        result = wp.empty(len(now), dtype=wp.vec3, device="cpu")
        wp.launch(
            _quat_velocity_kernel,
            dim=len(now),
            inputs=[wp.array(now, dtype=wp.quat, device="cpu"), wp.array(previous, dtype=wp.quat, device="cpu"), 0.5],
            outputs=[result],
            device="cpu",
        )
        actual = result.numpy()
        np.testing.assert_allclose(actual[[0, 1]], 0.0, rtol=0.0, atol=1.0e-7)
        # 2 * (1e-24) / 0.5 = 4e-24 rad/s. The baseline's vector norm squares
        # 1e-24 to 1e-48, which underflows float32 and returns zero instead.
        np.testing.assert_allclose(actual[2], [4.0e-24, 0.0, 0.0], rtol=1.0e-5, atol=1.0e-27)
        np.testing.assert_allclose(actual[3], [0.0, 0.0, np.pi], rtol=1.0e-5, atol=1.0e-6)

    def test_tiny_rotation_gradient_cpu(self):
        """The angular-velocity derivative must stay finite near identity."""
        now = wp.array([[5.0e-14, 0.0, 0.0, 1.0]], dtype=wp.quat, requires_grad=True, device="cpu")
        previous = wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quat, device="cpu")
        result = wp.zeros(1, dtype=wp.float32, requires_grad=True, device="cpu")
        with wp.Tape() as tape:
            wp.launch(_quat_velocity_x_kernel, dim=1, inputs=[now, previous], outputs=[result], device="cpu")
        tape.backward(loss=result)

        # 2 * (5e-14) / 0.5 = 2e-13; d(omega_x)/d(q_now.x) = 2 / 0.5 = 4.
        np.testing.assert_allclose(result.numpy(), [2.0e-13], rtol=1.0e-5, atol=1.0e-17)
        gradient = now.grad.numpy()
        self.assertTrue(np.isfinite(gradient).all())
        np.testing.assert_allclose(gradient[0, 0], 4.0, rtol=1.0e-4, atol=1.0e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
