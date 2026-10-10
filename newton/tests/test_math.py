# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for math utilities in ``newton.math``."""

import math
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


@wp.kernel(enable_backward=False)
def _smooth_extrema_kernel(
    a: wp.array[wp.float32],
    b: wp.array[wp.float32],
    eps: wp.array[wp.float32],
    smooth_max: wp.array[wp.float32],
    smooth_min: wp.array[wp.float32],
    leaky_max: wp.array[wp.float32],
    leaky_min: wp.array[wp.float32],
):
    """Evaluate the four public smooth-extremum functions on the CPU."""
    i = wp.tid()
    smooth_max[i] = newton.math.smooth_max(a[i], b[i], eps[i])
    smooth_min[i] = newton.math.smooth_min(a[i], b[i], eps[i])
    leaky_max[i] = newton.math.leaky_max(a[i], b[i])
    leaky_min[i] = newton.math.leaky_min(a[i], b[i])


@wp.kernel
def _smooth_max_grad_kernel(
    a: wp.array[wp.float32], b: wp.array[wp.float32], eps: wp.array[wp.float32], result: wp.array[wp.float32]
):
    """Evaluate smooth max for reverse-mode differentiation."""
    result[0] = newton.math.smooth_max(a[0], b[0], eps[0])


@wp.kernel
def _smooth_min_grad_kernel(
    a: wp.array[wp.float32], b: wp.array[wp.float32], eps: wp.array[wp.float32], result: wp.array[wp.float32]
):
    """Evaluate smooth min for reverse-mode differentiation."""
    result[0] = newton.math.smooth_min(a[0], b[0], eps[0])


class TestMathSmoothExtrema(unittest.TestCase):
    """Check finite smooth extrema and their CPU reverse-mode gradients."""

    def _evaluate(self, a, b, eps):
        """Evaluate public smooth extrema for Float32 arrays on the CPU."""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        eps = np.asarray(eps, dtype=np.float32)
        outputs = [wp.empty(len(a), dtype=wp.float32, device="cpu") for _ in range(4)]
        wp.launch(
            _smooth_extrema_kernel,
            dim=len(a),
            inputs=[
                wp.array(a, dtype=wp.float32, device="cpu"),
                wp.array(b, dtype=wp.float32, device="cpu"),
                wp.array(eps, dtype=wp.float32, device="cpu"),
            ],
            outputs=outputs,
            device="cpu",
        )
        return np.stack([output.numpy() for output in outputs])

    def test_finite_float32_inputs(self):
        """Avoid intermediate overflow when the smooth result is finite."""
        maximum = np.finfo(np.float32).max
        quarter = np.float32(2.0**125)
        a = np.array([2.0, 1.0e20, 0.0, 2.0e38, 2.0e38, maximum, quarter, -maximum], dtype=np.float32)
        b = np.array([1.0, 0.0, 1.0e20, 2.0e38, -2.0e38, quarter, maximum, -quarter], dtype=np.float32)
        result = self._evaluate(a, b, np.full(len(a), 1.0e-5, dtype=np.float32))

        self.assertTrue(np.all(np.isfinite(result)))
        for actual, expected in zip(result, (np.maximum(a, b), np.minimum(a, b)) * 2, strict=True):
            # The smooth correction can be lost to rounding at wide separation.
            np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=1.0e-5)

    def test_smallest_positive_float32_eps(self):
        """Taking a quarter of a subnormal eps must not erase smoothing."""
        smallest = np.nextafter(np.float32(0.0), np.float32(1.0))
        a = np.array([0.0, 1.0e-22], dtype=np.float32)
        b = np.zeros_like(a)
        result = self._evaluate(a, b, np.full(len(a), smallest, dtype=np.float32))

        expected = []
        for x, y in zip(a, b, strict=True):
            center = 0.5 * float(x) + 0.5 * float(y)
            radius = math.hypot(0.5 * float(x) - 0.5 * float(y), 0.5 * math.sqrt(float(smallest)))
            expected.append((center + radius, center - radius))
        np.testing.assert_allclose(result[:2], np.asarray(expected, dtype=np.float32).T, rtol=2.0e-5, atol=1.0e-30)

    def test_fast_path_boundaries(self):
        """Keep finite values across the legacy-safe input and eps seams."""
        bound = np.float32(2.0**62)
        below = np.nextafter(bound, np.float32(0.0))
        above = np.nextafter(bound, np.float32(np.inf))
        low = np.float32(2.0**-126)
        high = np.float32(2.0**126)
        cases = (
            (below, 0.0, 1.0e-5),
            (bound, 0.0, 1.0e-5),
            (above, 0.0, 1.0e-5),
            (0.0, -bound, 1.0e-5),
            (bound, -bound, 1.0e-5),
            (-bound, bound, 1.0e-5),
            (0.0, 0.0, np.nextafter(low, np.float32(0.0))),
            (0.0, 0.0, low),
            (0.0, 0.0, np.nextafter(low, np.float32(np.inf))),
            (0.0, 0.0, np.nextafter(high, np.float32(0.0))),
            (0.0, 0.0, high),
            (0.0, 0.0, np.nextafter(high, np.float32(np.inf))),
            (1.0e10, 0.0, 1.0e-5),
        )
        a, b, eps = (np.asarray(values, dtype=np.float32) for values in zip(*cases, strict=True))
        result = self._evaluate(a, b, eps)
        self.assertTrue(np.all(np.isfinite(result)))
        for i, (x, y, width) in enumerate(zip(a, b, eps, strict=True)):
            center = 0.5 * float(x) + 0.5 * float(y)
            radius = math.hypot(0.5 * float(x) - 0.5 * float(y), 0.5 * math.sqrt(float(width)))
            scale = max(abs(float(x)), abs(float(y)), math.sqrt(float(width)))
            tolerance = 8.0 * np.finfo(np.float32).eps * scale
            np.testing.assert_allclose(result[:2, i], (center + radius, center - radius), rtol=0.0, atol=tolerance)

    def test_zero_eps_forward_compatibility(self):
        """Keep the established finite forward values at zero smoothing."""
        a = np.array([0.0, 2.0, 2.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        result = self._evaluate(a, b, np.zeros(len(a), dtype=np.float32))
        np.testing.assert_array_equal(result[0], np.maximum(a, b))
        np.testing.assert_array_equal(result[1], np.minimum(a, b))

    def test_positive_eps_cpu_gradients(self):
        """Match analytic derivatives at ties, crossover, and wide separation."""
        smallest = np.nextafter(np.float32(0.0), np.float32(1.0))
        bound = np.float32(2.0**62)
        low = np.float32(2.0**-126)
        high = np.float32(2.0**126)
        cases = (
            (0.0, 0.0, 1.0e-5),
            (0.0, 0.0, smallest),
            (math.sqrt(1.0e-5), 0.0, 1.0e-5),
            (1.0e20, 0.0, 1.0e-5),
            (0.0, 1.0e20, 1.0e-5),
            (np.nextafter(bound, np.float32(0.0)), 0.0, 1.0e-5),
            (bound, 0.0, 1.0e-5),
            (np.nextafter(bound, np.float32(np.inf)), 0.0, 1.0e-5),
            (0.0, -bound, 1.0e-5),
            (0.0, 0.0, np.nextafter(low, np.float32(0.0))),
            (0.0, 0.0, low),
            (0.0, 0.0, np.nextafter(low, np.float32(np.inf))),
            (0.0, 0.0, np.nextafter(high, np.float32(0.0))),
            (0.0, 0.0, high),
            (0.0, 0.0, np.nextafter(high, np.float32(np.inf))),
        )
        for case in cases:
            x, y, width = (float(np.float32(value)) for value in case)
            d = x - y
            radius = math.hypot(d, math.sqrt(width))
            for kernel, sign in ((_smooth_max_grad_kernel, 1.0), (_smooth_min_grad_kernel, -1.0)):
                with self.subTest(x=x, y=y, eps=width, sign=sign):
                    a_wp = wp.array([x], dtype=wp.float32, device="cpu", requires_grad=True)
                    b_wp = wp.array([y], dtype=wp.float32, device="cpu", requires_grad=True)
                    eps_wp = wp.array([width], dtype=wp.float32, device="cpu", requires_grad=True)
                    result = wp.empty(1, dtype=wp.float32, device="cpu", requires_grad=True)
                    with wp.Tape() as tape:
                        wp.launch(kernel, dim=1, inputs=[a_wp, b_wp, eps_wp], outputs=[result], device="cpu")
                    tape.backward(grads={result: wp.ones_like(result)})
                    actual = (a_wp.grad.numpy()[0], b_wp.grad.numpy()[0], eps_wp.grad.numpy()[0])
                    expected = (
                        0.5 * (1.0 + sign * d / radius),
                        0.5 * (1.0 - sign * d / radius),
                        sign / (4.0 * radius),
                    )
                    np.testing.assert_allclose(actual[:2], expected[:2], rtol=1.0e-5, atol=1.0e-6)
                    np.testing.assert_allclose(actual[2], expected[2], rtol=2.0e-5, atol=0.0)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
