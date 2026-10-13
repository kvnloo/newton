# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

import newton
from newton._src.geometry.utils import remesh_mesh


class TestRemeshMeshTopology(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "requires SciPy")
    def test_inplace_convex_hull_can_reduce_vertex_count(self):
        """Update remeshed vertices and indices together when a hull shrinks."""
        mesh = newton.Mesh.create_box(0.5, duplicate_vertices=True, compute_inertia=False)
        self.assertEqual(len(mesh.vertices), 24)
        old_id = mesh.finalize(device="cpu")

        result = remesh_mesh(mesh, method="convex_hull", inplace=True)

        self.assertIs(result, mesh)
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.indices), 36)
        self.assertLess(int(mesh.indices.max()), len(mesh.vertices))
        self.assertNotEqual(mesh.finalize(device="cpu"), old_id)
        self.assertEqual(len(mesh.mesh.points.numpy()), 8)

    @unittest.skipUnless(importlib.util.find_spec("fast_simplification") is not None, "requires fast-simplification")
    def test_inplace_quadratic_can_reduce_vertex_count(self):
        """Apply real quadratic remeshing with fewer vertices atomically."""
        mesh = newton.Mesh.create_box(0.5, duplicate_vertices=True, compute_inertia=False)
        result = remesh_mesh(mesh, method="quadratic", inplace=True, target_reduction=0.5)

        self.assertIs(result, mesh)
        self.assertLess(len(mesh.vertices), 24)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertEqual(len(mesh.indices) % 3, 0)
        self.assertLess(int(mesh.indices.max()), len(mesh.vertices))

    def test_invalid_output_leaves_mesh_unchanged(self):
        """Reject invalid remesher indices before mutating source geometry."""
        mesh = newton.Mesh.create_box(0.5, duplicate_vertices=False, compute_inertia=False)
        original_vertices = mesh.vertices.copy()
        original_indices = mesh.indices.copy()
        original_id = mesh.finalize(device="cpu")
        invalid_vertices = original_vertices + 0.25
        invalid_faces = np.array([[0, 1, len(invalid_vertices)]], dtype=np.int32)

        with patch("newton._src.geometry.utils.remesh", return_value=(invalid_vertices, invalid_faces)):
            with self.assertRaisesRegex(ValueError, "exceeds vertex count"):
                remesh_mesh(mesh, method="quadratic", inplace=True)

        np.testing.assert_array_equal(mesh.vertices, original_vertices)
        np.testing.assert_array_equal(mesh.indices, original_indices)
        self.assertEqual(mesh.finalize(device="cpu"), original_id)

    @unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "requires SciPy")
    def test_non_inplace_convex_hull_preserves_source(self):
        """Keep the source mesh unchanged for a copied convex hull."""
        mesh = newton.Mesh.create_box(0.5, duplicate_vertices=True, compute_inertia=False)
        original_vertices = mesh.vertices.copy()
        original_indices = mesh.indices.copy()

        hull = remesh_mesh(mesh, method="convex_hull", inplace=False)

        self.assertIsNot(hull, mesh)
        self.assertEqual(len(hull.vertices), 8)
        self.assertLess(int(hull.indices.max()), len(hull.vertices))
        np.testing.assert_array_equal(mesh.vertices, original_vertices)
        np.testing.assert_array_equal(mesh.indices, original_indices)
