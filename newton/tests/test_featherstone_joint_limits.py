# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.featherstone.kernels import apply_implicit_joint_limit_terms
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestFeatherstoneImplicitJointLimits(unittest.TestCase):
    pass


def test_implicit_limit_terms(test: TestFeatherstoneImplicitJointLimits, device):
    """Verify active limit terms add the expected RHS and diagonal corrections."""
    joint_type = wp.array([newton.JointType.REVOLUTE], dtype=wp.int32, device=device)
    joint_q_start = wp.array([0], dtype=wp.int32, device=device)
    joint_qd_start = wp.array([0, 1], dtype=wp.int32, device=device)
    joint_dof_dim = wp.array([[0, 1]], dtype=wp.int32, device=device)
    joint_q = wp.array([0.6], dtype=wp.float32, device=device)
    joint_qd = wp.array([2.0], dtype=wp.float32, device=device)
    lower = wp.array([-0.5], dtype=wp.float32, device=device)
    upper = wp.array([0.5], dtype=wp.float32, device=device)
    ke = wp.array([10000.0], dtype=wp.float32, device=device)
    kd = wp.array([10.0], dtype=wp.float32, device=device)
    armature = wp.array([0.01], dtype=wp.float32, device=device)
    tau = wp.array([-1020.0], dtype=wp.float32, device=device)
    solve_armature = wp.zeros(1, dtype=wp.float32, device=device)
    dt = 0.01

    wp.launch(
        apply_implicit_joint_limit_terms,
        dim=1,
        inputs=[
            joint_type,
            joint_q_start,
            joint_qd_start,
            joint_dof_dim,
            joint_q,
            joint_qd,
            lower,
            upper,
            ke,
            kd,
            armature,
            dt,
        ],
        outputs=[tau, solve_armature],
        device=device,
    )

    test.assertAlmostEqual(float(tau.numpy()[0]), -1220.0, delta=1.0e-4)
    test.assertAlmostEqual(float(solve_armature.numpy()[0]), 1.11, delta=1.0e-5)


def _make_limit_model(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(
        1.0e-5,
        0.0,
        0.0,
        0.0,
        1.0e-5,
        0.0,
        0.0,
        0.0,
        1.0e-5,
    )
    body = builder.add_link(mass=0.05, inertia=inertia, com=wp.vec3())
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 1.0, 0.0),
        target_ke=0.0,
        target_kd=0.0,
        damping=0.0,
        limit_lower=-0.5,
        limit_upper=0.5,
        limit_ke=10000.0,
        limit_kd=10.0,
        armature=0.0,
    )
    builder.add_articulation([joint])

    q_start = builder.joint_q_start[joint]
    qd_start = builder.joint_qd_start[joint]
    builder.joint_q[q_start] = 0.5272407
    builder.joint_qd[qd_start] = 6.897536
    return builder.finalize(device=device)


def _initial_state(model):
    state = model.state()
    state.joint_q.assign(model.joint_q)
    state.joint_qd.assign(model.joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def test_implicit_limit_stabilizes_low_inertia_hinge(test: TestFeatherstoneImplicitJointLimits, device):
    """Verify the opt-in limit solve avoids the explicit low-inertia hinge blow-up."""
    model = _make_limit_model(device)
    control = model.control()
    dt = 1.0 / 240.0

    explicit_in = _initial_state(model)
    explicit_out = model.state()
    explicit = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    explicit.step(explicit_in, explicit_out, control, None, dt)

    implicit_in = _initial_state(model)
    implicit_out = model.state()
    implicit = newton.solvers.SolverFeatherstone(
        model,
        angular_damping=0.0,
        implicit_joint_limits=True,
    )
    implicit.step(implicit_in, implicit_out, control, None, dt)

    explicit_q = float(explicit_out.joint_q.numpy()[0])
    implicit_q = float(implicit_out.joint_q.numpy()[0])
    implicit_qd = float(implicit_out.joint_qd.numpy()[0])

    test.assertTrue(np.isfinite(implicit_q))
    test.assertTrue(np.isfinite(implicit_qd))
    test.assertLess(abs(implicit_q - 0.5), abs(explicit_q - 0.5))
    test.assertLess(abs(implicit_q), 1.0)


devices = get_test_devices()
add_function_test(
    TestFeatherstoneImplicitJointLimits,
    "test_implicit_limit_terms",
    test_implicit_limit_terms,
    devices=devices,
)
add_function_test(
    TestFeatherstoneImplicitJointLimits,
    "test_implicit_limit_stabilizes_low_inertia_hinge",
    test_implicit_limit_stabilizes_low_inertia_hinge,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
