# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverKamino


class TestKaminoAxisNotifications(unittest.TestCase):
    def test_joint_properties_rejects_parallel_universal_axes(self):
        builder = newton.ModelBuilder()
        body = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32)),
            label="link",
        )
        joint = builder.add_joint_d6(
            -1,
            body,
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(axis=wp.vec3(1.0, 0.0, 0.0)),
                newton.ModelBuilder.JointDofConfig(axis=wp.vec3(0.0, 1.0, 0.0)),
            ],
            label="universal",
        )
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        solver = SolverKamino(model)

        # Two unit axes with dot product zero form a valid universal joint.
        axes = model.joint_axis.numpy().copy()
        np.testing.assert_array_equal(axes, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)

        # An axis edit reported with JOINT_PROPERTIES must be validated, too.
        axes[1] = axes[0]
        self.assertEqual(float(np.dot(axes[0], axes[1])), 1.0)
        model.joint_axis.assign(wp.array(axes, dtype=wp.vec3, device="cpu"))
        with self.assertRaisesRegex(ValueError, "axes must be unit length and orthogonal"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
