# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU regression for the cloth Franka viewer's centimeter-to-meter conversion."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.cloth.example_cloth_franka import Example as ClothFrankaExample


class TestClothFrankaVizUnits(unittest.TestCase):
    def _make_example(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        builder.add_shape_box(body=body, hx=1.0, hy=1.0, hz=1.0)
        model = builder.finalize(device="cpu")

        # Explicit centimeter values make unconverted viewer fields observable.
        model.body_com = wp.array(
            np.array([[10.0, -20.0, 5.0]], dtype=np.float32), dtype=wp.vec3, device="cpu"
        )
        model.shape_collision_radius = wp.array([40.0], dtype=wp.float32, device="cpu")

        # Exercise the real example's visualization path without assets or solvers.
        example = ClothFrankaExample.__new__(ClothFrankaExample)
        example.model = model
        example.viz_scale = 0.01
        example.sim_shape_transform = model.shape_transform
        example.viz_shape_transform = model.shape_transform
        example.sim_shape_scale = model.shape_scale
        example.viz_shape_scale = model.shape_scale
        example.sim_particle_radius = model.particle_radius
        example.viz_particle_radius = model.particle_radius
        return example

    def test_prepares_scaled_overlay_metadata(self):
        example = self._make_example()
        model = example.model
        original_com = model.body_com
        original_radius = model.shape_collision_radius

        example._prepare_viz_overlay_data()

        np.testing.assert_allclose(example.viz_body_com.numpy(), [[0.1, -0.2, 0.05]], atol=1.0e-6)
        np.testing.assert_allclose(example.viz_shape_collision_radius.numpy(), [0.4], atol=1.0e-6)
        self.assertIs(model.body_com, original_com)
        self.assertIs(model.shape_collision_radius, original_radius)

    def test_swaps_overlay_metadata_and_restores_after_viewer_error(self):
        example = self._make_example()
        model = example.model
        example.sim_body_com = model.body_com
        example.sim_shape_collision_radius = model.shape_collision_radius
        example.viz_body_com = wp.array(
            np.array([[0.1, -0.2, 0.05]], dtype=np.float32), dtype=wp.vec3, device="cpu"
        )
        example.viz_shape_collision_radius = wp.array([0.4], dtype=wp.float32, device="cpu")

        with self.assertRaisesRegex(RuntimeError, "viewer failed"):
            with example.viz_units():
                np.testing.assert_allclose(model.body_com.numpy(), [[0.1, -0.2, 0.05]], atol=1.0e-6)
                np.testing.assert_allclose(model.shape_collision_radius.numpy(), [0.4], atol=1.0e-6)
                raise RuntimeError("viewer failed")

        self.assertIs(model.body_com, example.sim_body_com)
        self.assertIs(model.shape_collision_radius, example.sim_shape_collision_radius)


if __name__ == "__main__":
    unittest.main()
