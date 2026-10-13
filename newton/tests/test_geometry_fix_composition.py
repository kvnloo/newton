# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.inertia import compute_inertia_mesh, compute_inertia_shape
from newton._src.geometry.types import GeoType
from newton._src.geometry.utils import remesh_mesh
from newton.tests.unittest_utils import assert_np_equal


@unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "requires SciPy")
class TestGeometryFixComposition(unittest.TestCase):
    @staticmethod
    def _duplicated_box(is_solid: bool) -> newton.Mesh:
        box = newton.Mesh.create_box(
            0.5, duplicate_vertices=True, compute_normals=False, compute_uvs=False, compute_inertia=False
        )
        return newton.Mesh(box.vertices, box.indices, is_solid=is_solid, compute_inertia=True)

    def test_hollow_inertia_after_shrinking_topology(self):
        """Recompute hollow inertia after atomic 24-to-8 vertex remeshing."""
        mesh = self._duplicated_box(is_solid=False)
        self.assertEqual(len(mesh.vertices), 24)
        old_id = mesh.finalize(device="cpu")

        self.assertIs(remesh_mesh(mesh, method="convex_hull", inplace=True, recompute_inertia=True), mesh)
        self.assertEqual(len(mesh.vertices), 8)
        self.assertNotEqual(mesh.finalize(device="cpu"), old_id)
        self.assertEqual(len(mesh.mesh.points.numpy()), 8)

        direct = compute_inertia_mesh(1.0, mesh.vertices, mesh.indices, is_solid=False, thickness=0.01)
        actual = compute_inertia_shape(GeoType.MESH, wp.vec3(1.0, 1.0, 1.0), mesh, 1.0, False, 0.01)
        self.assertAlmostEqual(direct[0], 0.12, delta=2e-6)
        self.assertAlmostEqual(actual[0], direct[0], delta=2e-6)
        assert_np_equal(np.array(actual[1]), np.array(direct[1]), tol=1e-5)
        assert_np_equal(np.array(actual[2]), np.array(direct[2]), tol=1e-5)

        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_mesh(
            body, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(density=1.0, is_solid=False, margin=0.01)
        )
        self.assertAlmostEqual(builder.body_mass[body], direct[0], delta=2e-6)
        assert_np_equal(np.array(builder.body_com[body]), np.array(direct[1]), tol=1e-5)
        assert_np_equal(np.array(builder.body_inertia[body]), np.array(direct[2]), tol=1e-5)

    def test_authored_hollow_inertia_survives_unrequested_recompute(self):
        """Retain authored mass properties when remeshing does not recompute."""
        mesh = self._duplicated_box(is_solid=False)
        mesh.mass = 0.7
        mesh.com = wp.vec3(0.25, 0.5, 0.75)
        mesh.inertia = wp.mat33(np.eye(3) * 0.2)

        remesh_mesh(mesh, method="convex_hull", inplace=True, recompute_inertia=False)
        actual = compute_inertia_shape(GeoType.MESH, wp.vec3(1.0, 1.0, 1.0), mesh, 1.0, False, 0.01)
        self.assertAlmostEqual(actual[0], 0.7, delta=1e-6)
        assert_np_equal(np.array(actual[1]), np.array([0.25, 0.5, 0.75]), tol=1e-6)
        assert_np_equal(np.array(actual[2]), np.eye(3) * 0.2, tol=1e-6)

    def test_invalid_output_preserves_hollow_geometry_and_inertia(self):
        """Reject invalid topology without changing cached hollow mass data."""
        mesh = self._duplicated_box(is_solid=False)
        original_vertices = mesh.vertices.copy()
        original_indices = mesh.indices.copy()
        original_mass = mesh.mass
        original_com = np.array(mesh.com).copy()
        original_inertia = np.array(mesh.inertia).copy()
        original_computed_inertia = (
            mesh._computed_inertia[0],
            mesh._computed_inertia[1].copy(),
            mesh._computed_inertia[2].copy(),
        )
        old_id = mesh.finalize(device="cpu")
        invalid_vertices = original_vertices + 0.25
        invalid_faces = np.array([[0, 1, len(invalid_vertices)]], dtype=np.int32)

        with patch("newton._src.geometry.utils.remesh", return_value=(invalid_vertices, invalid_faces)):
            with self.assertRaisesRegex(ValueError, "exceeds vertex count"):
                remesh_mesh(mesh, method="quadratic", inplace=True, recompute_inertia=True)

        np.testing.assert_array_equal(mesh.vertices, original_vertices)
        np.testing.assert_array_equal(mesh.indices, original_indices)
        self.assertEqual(mesh.mass, original_mass)
        np.testing.assert_array_equal(np.array(mesh.com), original_com)
        np.testing.assert_array_equal(np.array(mesh.inertia), original_inertia)
        self.assertEqual(mesh._computed_inertia[0], original_computed_inertia[0])
        np.testing.assert_array_equal(mesh._computed_inertia[1], original_computed_inertia[1])
        np.testing.assert_array_equal(mesh._computed_inertia[2], original_computed_inertia[2])
        self.assertEqual(mesh.finalize(device="cpu"), old_id)

    def test_solid_inertia_after_shrinking_topology(self):
        """Preserve solid-mesh inertia semantics through atomic remeshing."""
        mesh = self._duplicated_box(is_solid=True)
        remesh_mesh(mesh, method="convex_hull", inplace=True, recompute_inertia=True)
        self.assertEqual(len(mesh.vertices), 8)

        direct = compute_inertia_mesh(1.0, mesh.vertices, mesh.indices, is_solid=True)
        actual = compute_inertia_shape(GeoType.MESH, wp.vec3(1.0, 1.0, 1.0), mesh, 1.0, True, 0.01)
        self.assertAlmostEqual(actual[0], direct[0], delta=2e-6)
        assert_np_equal(np.array(actual[1]), np.array(direct[1]), tol=1e-5)
        assert_np_equal(np.array(actual[2]), np.array(direct[2]), tol=1e-5)
