# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for persistent external forces in SolverSemiImplicit."""

import unittest

import numpy as np
import warp as wp

import newton


class TestSemiImplicitForceBuffers(unittest.TestCase):
    def test_particle_spring_forces_are_input_only_cpu(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_particles(
            pos=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)],
            vel=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            mass=[1.0, 1.0],
            radius=[0.01, 0.01],
        )
        builder.add_spring(0, 1, ke=10.0, kd=0.0, control=0.0)
        model = builder.finalize(device="cpu")
        model.particle_grid = None  # Isolate the spring from particle collisions.

        state_in, state_out = model.state(), model.state()
        state_in.particle_q.assign(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        )
        external = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        output_sentinel = np.array([[-7.0, 1.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
        state_in.particle_f.assign(external)
        state_out.particle_f.assign(output_sentinel)
        input_buffer = state_in.particle_f

        solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)
        solver.step(state_in, state_out, None, None, 0.1)

        self.assertIs(state_in.particle_f, input_buffer)
        np.testing.assert_array_equal(state_in.particle_f.numpy(), external)
        np.testing.assert_array_equal(state_out.particle_f.numpy(), output_sentinel)
        np.testing.assert_allclose(
            state_out.particle_qd.numpy()[:, 0],
            [1.1, -1.0],
            rtol=0.0,
            atol=1.0e-5,
        )

    def test_rigid_contact_forces_are_input_only_cpu(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        builder.default_shape_cfg.ke = 1000.0
        builder.default_shape_cfg.kd = 0.0
        body = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.45), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            lock_inertia=True,
        )
        builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        self.assertEqual(model.joint_count, 0)

        state_in, state_out = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

        external = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        output_sentinel = np.array([[-7.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        state_in.body_f.assign(external)
        state_out.body_f.assign(output_sentinel)
        input_buffer = state_in.body_f

        solver = newton.solvers.SolverSemiImplicit(model, angular_damping=0.0)
        solver.step(state_in, state_out, None, contacts, 0.001)

        self.assertIs(state_in.body_f, input_buffer)
        np.testing.assert_array_equal(state_in.body_f.numpy(), external)
        np.testing.assert_array_equal(state_out.body_f.numpy(), output_sentinel)
        self.assertGreater(state_out.body_qd.numpy()[body, 2], 0.0)
