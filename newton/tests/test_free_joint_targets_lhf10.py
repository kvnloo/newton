# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton


class TestFreeJointInitialTargets(unittest.TestCase):
    def test_world_free_joint_target_matches_authored_pose(self):
        """Initialize world-root FREE targets from the authored body pose."""
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
        builder = newton.ModelBuilder()
        builder.add_body(xform=wp.transform(wp.vec3(2.0, -1.0, 4.0), rotation))
        model = builder.finalize(device="cpu")

        half_sqrt_two = math.sqrt(0.5)
        expected = np.array([2.0, -1.0, 4.0, 0.0, 0.0, half_sqrt_two, half_sqrt_two])
        self.assertEqual(model.joint_count, 1)
        np.testing.assert_allclose(model.joint_q.numpy()[:7], expected, rtol=0, atol=1.0e-6)
        np.testing.assert_allclose(model.joint_target_q.numpy()[:7], expected, rtol=0, atol=1.0e-6)
        np.testing.assert_allclose(model.control().joint_target_q.numpy()[:7], expected, rtol=0, atol=1.0e-6)

    def test_parented_free_joint_target_respects_both_anchors(self):
        """Use both rotated joint anchors for a parented FREE control target."""
        quarter_turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
        minus_quarter_turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -math.pi / 2.0)
        builder = newton.ModelBuilder()
        parent = builder.add_link(xform=wp.transform(wp.vec3(2.0, 1.0, 0.0), quarter_turn))
        child = builder.add_link(xform=wp.transform(wp.vec3(5.0, 4.0, 1.0), quarter_turn))
        joint = builder.add_joint_free(
            parent=parent,
            child=child,
            parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), minus_quarter_turn),
            child_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), quarter_turn),
        )
        builder.add_articulation([joint])

        # Parent and child anchors are at (2, 2, 0) and (5, 5, 1).
        # Their world rotations are identity and 180 degrees about Z.
        expected = np.array([3.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        start = builder.joint_q_start[joint]
        np.testing.assert_allclose(builder.joint_q[start : start + 7], expected, rtol=0, atol=1.0e-6)
        np.testing.assert_allclose(builder.joint_target_q[start : start + 7], expected, rtol=0, atol=1.0e-6)

        model = builder.finalize(device="cpu")
        start = int(model.joint_q_start.numpy()[joint])
        np.testing.assert_allclose(
            model.control().joint_target_q.numpy()[start : start + 7], expected, rtol=0, atol=1.0e-6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
