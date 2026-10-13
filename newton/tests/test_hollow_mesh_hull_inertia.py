# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.inertia import compute_inertia_mesh, compute_inertia_shape
from newton._src.geometry.types import GeoType
from newton.tests.unittest_utils import assert_np_equal


class TestHollowMeshHullInertia(unittest.TestCase):
    def test_real_hull_preserves_generated_inertia_provenance(self):
        """Respect hollow thickness after real convex-hull remeshing."""
        vertices = [
            [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0],
        ]
        indices = [
            [1, 2, 3], [7, 6, 5], [4, 5, 1], [5, 6, 2],
            [2, 6, 7], [0, 3, 7], [0, 1, 3], [4, 7, 5],
            [0, 4, 1], [1, 5, 2], [3, 2, 7], [4, 0, 7],
        ]
        source = newton.Mesh(vertices, indices, is_solid=False, compute_inertia=True)
        hull = source.compute_convex_hull(replace=False)
        self.assertEqual(len(hull.vertices), 8)
        self.assertTrue(hull.has_inertia)
        self.assertFalse(hull.is_solid)

        scale = wp.vec3(1.0, 1.0, 1.0)
        for thickness, expected_mass in ((0.0, 0.0), (0.001, 0.012), (0.01, 0.12)):
            with self.subTest(thickness=thickness):
                expected = compute_inertia_mesh(
                    1.0, hull.vertices, hull.indices, is_solid=False, thickness=thickness
                )
                self.assertAlmostEqual(expected[0], expected_mass, delta=2e-6)
                actual = compute_inertia_shape(GeoType.CONVEX_MESH, scale, hull, 1.0, False, thickness)
                self.assertAlmostEqual(actual[0], expected[0], delta=2e-6)
                assert_np_equal(np.array(actual[1]), np.array(expected[1]), tol=1e-5)
                assert_np_equal(np.array(actual[2]), np.array(expected[2]), tol=1e-5)

        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_convex_hull(
            body, mesh=hull, cfg=newton.ModelBuilder.ShapeConfig(density=1.0, is_solid=False, margin=0.01)
        )
        self.assertAlmostEqual(builder.body_mass[body], 0.12, delta=2e-6)
        assert_np_equal(np.array(builder.body_com[body]), np.array([0.5, 0.5, 0.5]), tol=1e-5)
        direct_hull = compute_inertia_mesh(1.0, hull.vertices, hull.indices, is_solid=False, thickness=0.01)
        assert_np_equal(np.array(builder.body_inertia[body]), np.array(direct_hull[2]), tol=1e-5)

        source.compute_convex_hull(replace=True)
        replaced = compute_inertia_shape(GeoType.MESH, scale, source, 1.0, False, 0.01)
        self.assertAlmostEqual(replaced[0], 0.12, delta=2e-6)

        hull.mass = 0.7
        hull.com = wp.vec3(0.25, 0.5, 0.75)
        hull.inertia = wp.mat33(np.eye(3) * 0.2)
        authored = compute_inertia_shape(GeoType.CONVEX_MESH, scale, hull, 1.0, False, 0.01)
        self.assertAlmostEqual(authored[0], 0.7, delta=1e-6)
        assert_np_equal(np.array(authored[1]), np.array([0.25, 0.5, 0.75]), tol=1e-6)
        assert_np_equal(np.array(authored[2]), np.eye(3) * 0.2, tol=1e-6)
