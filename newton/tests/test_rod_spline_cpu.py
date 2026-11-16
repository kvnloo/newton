# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for Rod.create_spline and builder.add_rod."""

import math
import unittest

import numpy as np
import warp as wp

import newton


class TestRodSplineCPU(unittest.TestCase):
    def test_open_two_point_and_duplicate_controls_have_half_meter_chords(self):
        """Verify two-point and repeated-endpoint controls yield four 0.5 m chords."""
        start = wp.vec3(0.0, 0.0, 0.0)
        end = wp.vec3(2.0, 0.0, 0.0)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        reference_points = None
        reference_quaternions = None
        for name, controls in (
            ("two_points", (start, end)),
            ("duplicate_start", (start, start, end)),
            ("duplicate_end", (start, end, end)),
        ):
            with self.subTest(controls=name):
                rod = newton.Rod.create_spline(controls, segment_count=4, radius=0.02)
                self.assertFalse(rod.closed)
                self.assertEqual((rod.point_count, rod.segment_count), (5, 4))
                np.testing.assert_array_equal(
                    rod.edges, [[0, 1], [1, 2], [2, 3], [3, 4]]
                )
                np.testing.assert_array_equal(rod.points[[0, -1]], expected[[0, -1]])
                np.testing.assert_allclose(rod.points, expected, rtol=0.0, atol=5.0e-3)
                np.testing.assert_allclose(
                    np.linalg.norm(np.diff(rod.points, axis=0), axis=1),
                    [0.5, 0.5, 0.5, 0.5],
                    rtol=0.0,
                    atol=5.0e-3,
                )
                self.assertEqual(rod.quaternions.shape, (4, 4))
                self.assertTrue(np.isfinite(rod.quaternions).all())
                if reference_points is None:
                    reference_points = rod.points.copy()
                    reference_quaternions = rod.quaternions.copy()
                else:
                    np.testing.assert_array_equal(rod.points, reference_points)
                    np.testing.assert_array_equal(rod.quaternions, reference_quaternions)

    def test_curved_duplicate_control_assembles_with_straight_rest(self):
        """Verify a curved duplicate-control spline assembles as a Rod."""
        controls = [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.5, 0.5, 0.0),
            wp.vec3(0.5, 0.5, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
        ]
        rod = newton.Rod.create_spline(
            controls, segment_count=8, radius=0.02, rest_straight=True
        )
        self.assertEqual((rod.point_count, rod.segment_count), (9, 8))
        self.assertEqual(rod.quaternions.shape, (8, 4))
        self.assertTrue(rod.rest_straight)
        self.assertIsNone(rod.rest_points)
        np.testing.assert_allclose(rod.points[0], (0.0, 0.0, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(rod.points[-1], (1.0, 0.0, 0.0), atol=1.0e-6)
        self.assertGreater(float(rod.points[:, 1].max()), 0.4)
        lengths = np.linalg.norm(np.diff(rod.points, axis=0), axis=1)
        self.assertTrue(np.all(np.isfinite(lengths)))
        self.assertTrue(np.all(lengths > 0.0))
        self.assertLess(float(np.ptp(lengths) / np.mean(lengths)), 0.10)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        bodies, joints = builder.add_rod(
            rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=10.0), label="spline"
        )
        self.assertEqual((len(bodies), len(joints)), (8, 7))
        self.assertEqual(builder.body_count, 8)

    def test_closed_spline_integer_twist_assembles_loop(self):
        """Verify a whole-turn closed spline creates a closed rod topology."""
        controls = [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(1.0, 1.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ]
        with self.assertRaisesRegex(ValueError, "multiple of 2.pi"):
            newton.Rod.create_spline(
                controls, segment_count=8, closed=True, twist_total=math.pi
            )

        rod = newton.Rod.create_spline(
            controls, segment_count=8, closed=True,
            twist_total=2.0 * math.pi, radius=0.02,
        )
        self.assertTrue(rod.closed)
        self.assertEqual((rod.point_count, rod.segment_count), (9, 8))
        np.testing.assert_array_equal(rod.points[-1], rod.points[0])
        self.assertTrue(np.isfinite(rod.quaternions).all())
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        bodies, joints = builder.add_rod(
            rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=10.0), label="loop"
        )
        self.assertEqual((len(bodies), len(joints)), (8, 8))

    def test_spline_sampling_validation(self):
        """Reject inconsistent sampling options and degenerate loop inputs."""
        controls = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            newton.Rod.create_spline(controls)
        with self.assertRaisesRegex(ValueError, "integer >= 2"):
            newton.Rod.create_spline(controls, segment_count=2.5)
        with self.assertRaisesRegex(ValueError, "distinct control points"):
            newton.Rod.create_spline(controls, segment_count=8, closed=True)


if __name__ == "__main__":
    unittest.main()
