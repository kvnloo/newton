# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView


class TestArticulationViewStaleArray(unittest.TestCase):
    def build(self):
        world = newton.ModelBuilder()
        body = world.add_link()
        joint = world.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        world.add_articulation([joint], label="robot")
        builder = newton.ModelBuilder()
        builder.replicate(world, 1)
        model = builder.finalize()
        return model, ArticulationView(model, "robot", verbose=False)

    def test_replaced_joint_q_is_not_aliased(self):
        model, view = self.build()
        state = model.state()
        state.joint_q.fill_(1.0)
        before = view.get_dof_positions(state).numpy().copy()
        self.assertTrue(np.allclose(before, 1.0))

        old = state.joint_q
        replacement = wp.zeros_like(old)
        replacement.fill_(2.0)
        state.joint_q = replacement

        after = view.get_dof_positions(state).numpy()
        self.assertTrue(np.allclose(after, 2.0))
        self.assertTrue(np.allclose(old.numpy(), 1.0))

        written = wp.zeros_like(view.get_dof_positions(state))
        written.fill_(3.0)
        view.set_dof_positions(state, written)
        self.assertTrue(np.allclose(state.joint_q.numpy(), 3.0))
        self.assertTrue(np.allclose(old.numpy(), 1.0))


if __name__ == "__main__":
    unittest.main()
