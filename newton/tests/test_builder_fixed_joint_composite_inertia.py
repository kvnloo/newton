# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for composite inertia after fixed-joint collapse."""

import math
import unittest

import numpy as np
import warp as wp

import newton


class TestBuilderFixedJointCompositeInertia(unittest.TestCase):
    @staticmethod
    def _collapsed_model(child_rotation: wp.quat):
        builder = newton.ModelBuilder()
        parent = builder.add_link(
            mass=2.0,
            com=wp.vec3(1.0, 0.0, 0.0),
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0),
        )
        child_xform = wp.transform(wp.vec3(3.0, 0.0, 0.0), child_rotation)
        child = builder.add_link(
            xform=child_xform,
            mass=3.0,
            com=wp.vec3(0.0, 1.0, 0.0),
            inertia=wp.mat33(4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0),
        )
        root_joint = builder.add_joint_revolute(-1, parent, axis=(0.0, 0.0, 1.0))
        fixed_joint = builder.add_joint_fixed(parent, child, parent_xform=child_xform)
        builder.add_articulation([root_joint, fixed_joint])
        builder.collapse_fixed_joints()
        return builder.finalize(device="cpu")

    def test_offset_centers_shift_both_inertias_cpu(self):
        """Two offset centers must contribute about the new center, not the joint origin."""
        model = self._collapsed_model(wp.quat_identity())
        self.assertEqual(model.body_count, 1)
        self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 5.0)
        np.testing.assert_allclose(model.body_com.numpy()[0], [2.2, 0.6, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            model.body_inertia.numpy()[0],
            [[6.2, -2.4, 0.0], [-2.4, 11.8, 0.0], [0.0, 0.0, 15.0]],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_rotated_child_inertia_uses_merged_center_cpu(self):
        """Child inertia must rotate before both bodies are shifted to their shared center."""
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
        model = self._collapsed_model(rotation)
        self.assertEqual(model.body_count, 1)
        self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 5.0)
        np.testing.assert_allclose(model.body_com.numpy()[0], [1.6, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            model.body_inertia.numpy()[0],
            [[6.0, 0.0, 0.0], [0.0, 7.2, 0.0], [0.0, 0.0, 10.2]],
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
