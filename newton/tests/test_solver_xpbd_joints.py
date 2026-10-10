# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverXPBD revolute limits that cross the ±π branch cut."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _hinge_link(device, limit_lower, limit_upper, q0=0.0, qd0=0.0, ke=0.0, kd=0.0, target=0.0):
    """One link on a world revolute about +Y. Shape mass is included; gravity is off."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(link, xform=wp.transform((0.15, 0.0, 0.0), wp.quat_identity()), hx=0.15, hy=0.03, hz=0.03)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
        axis=(0.0, 1.0, 0.0),
        limit_lower=limit_lower,
        limit_upper=limit_upper,
        target_ke=ke,
        target_kd=kd,
        target_pos=target,
    )
    builder.add_articulation([joint])
    builder.joint_q = [q0]
    builder.joint_qd = [qd0]
    return builder.finalize(device=device)


def _pivot_inertia(model) -> float:
    """Inertia of the link about the hinge (body-frame Y through the joint anchor)."""
    inertia = model.body_inertia.numpy()[0]
    mass = float(model.body_mass.numpy()[0])
    com_x = float(model.body_com.numpy()[0][0])
    return float(inertia[1, 1]) + mass * com_x * com_x


def _spin_into_limit(model, device, q0, qd0, steps):
    """Integrate and return peak |qd|, peak kinetic energy, peak limit-violation energy, max |Δq|, final q."""
    solver = newton.solvers.SolverXPBD(model, iterations=4, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    joint_q = wp.zeros(1, dtype=float, device=device)
    joint_qd = wp.zeros(1, dtype=float, device=device)
    dt = 1.25e-3
    inertia = _pivot_inertia(model)
    lower = float(model.joint_limit_lower.numpy()[0])
    upper = float(model.joint_limit_upper.numpy()[0])
    peak_qd = 0.0
    ke_peak = 0.0
    # 0.5 * violation^2: the squared principal-value limit error that used to be ~2π.
    violation_energy_peak = 0.0
    max_step = 0.0
    previous = None
    final_q = q0
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        newton.eval_ik(model, state_0, joint_q, joint_qd)
        angle = float(joint_q.numpy()[0])
        speed = float(joint_qd.numpy()[0])
        peak_qd = max(peak_qd, abs(speed))
        ke_peak = max(ke_peak, 0.5 * inertia * speed * speed)
        if angle < lower:
            violation = angle - lower
        elif angle > upper:
            violation = angle - upper
        else:
            violation = 0.0
        violation_energy_peak = max(violation_energy_peak, 0.5 * violation * violation)
        if previous is not None:
            max_step = max(max_step, abs(angle - previous))
        previous = angle
        final_q = angle
    return peak_qd, ke_peak, violation_energy_peak, max_step, final_q


def test_revolute_limit_branch_cut_positive_and_negative(test, device):
    """Spin a limited hinge across +π and across -π. Neither crossing may inject a ~2π correction."""
    # Unfixed solver, same fixture, 1200 steps, CPU: upper 3.4 rad jumps by 9.44 rad
    # (3.139 -> -3.164, a 2π branch error) and kinetic energy peaks at 9.12e9.
    # The negative limit mirrors it (jump -9.44 rad, kinetic energy 9.65e9).
    cases = ((-0.5, 3.4, 0.0, 3.0), (-3.4, 0.5, 0.0, -3.0))
    for lower, upper, q0, qd0 in cases:
        model = _hinge_link(device, lower, upper, q0=q0, qd0=qd0)
        ke0 = 0.5 * _pivot_inertia(model) * qd0 * qd0
        peak_qd, ke_peak, violation_energy_peak, max_step, final_q = _spin_into_limit(model, device, q0, qd0, 1200)
        # Continuous coordinate: one step at the initial rate is |qd| * dt ≈ 0.00375 rad.
        test.assertLess(max_step, 0.5)
        test.assertLessEqual(peak_qd, abs(qd0) * 1.05)
        test.assertLess(ke_peak, 2.0 * ke0)
        # A 2π mis-read has violation energy 0.5*(2π)^2 ≈ 19.7. Stay far below that.
        test.assertLess(violation_energy_peak, 0.5)
        if qd0 > 0.0:
            test.assertGreater(final_q, upper - 0.6)
            test.assertLessEqual(final_q, upper + 0.05)
        else:
            test.assertLess(final_q, lower + 0.6)
            test.assertGreaterEqual(final_q, lower - 0.05)
        # Traveling 1200 steps without a limit would pass π; the coordinate must have reached it.
        test.assertGreater(abs(final_q), np.pi - 0.05)


def test_revolute_eval_ik_angle_beyond_pi(test, device):
    """eval_ik stays on the limit branch, including the opposite quaternion hemisphere."""
    model = _hinge_link(device, -0.227, 3.421, q0=3.3)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    joint_q = wp.zeros(1, dtype=float, device=device)
    joint_qd = wp.zeros(1, dtype=float, device=device)
    newton.eval_ik(model, state, joint_q, joint_qd)
    test.assertAlmostEqual(float(joint_q.numpy()[0]), 3.3, places=4)
    body_q = state.body_q.numpy()
    body_q[0, 3:] *= -1.0
    state.body_q.assign(body_q)
    newton.eval_ik(model, state, joint_q, joint_qd)
    test.assertAlmostEqual(float(joint_q.numpy()[0]), 3.3, places=4)


def test_revolute_hold_beyond_pi(test, device):
    """A hinge posed beyond π inside its limits is not treated as a violation."""
    model = _hinge_link(device, -0.227, 3.421, q0=3.3)
    solver = newton.solvers.SolverXPBD(model, iterations=4, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    for _ in range(200):
        solver.step(state_0, state_1, control, None, 1.25e-3)
        state_0, state_1 = state_1, state_0
    joint_q = wp.zeros(1, dtype=float, device=device)
    joint_qd = wp.zeros(1, dtype=float, device=device)
    newton.eval_ik(model, state_0, joint_q, joint_qd)
    test.assertAlmostEqual(float(joint_q.numpy()[0]), 3.3, places=3)
    test.assertLess(abs(float(joint_qd.numpy()[0])), 1.0e-3)


def test_revolute_unlimited_drive_takes_short_way(test, device):
    """An unlimited hinge driven across the branch uses the short arc to the target."""
    model = _hinge_link(device, -newton.MAXVAL, newton.MAXVAL, q0=3.0, ke=100.0, kd=5.0, target=-3.0)
    solver = newton.solvers.SolverXPBD(model, iterations=4, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    for _ in range(800):
        solver.step(state_0, state_1, control, None, 1.25e-3)
        state_0, state_1 = state_1, state_0
    rel = state_0.body_q.numpy()[0, 3:]
    angle = 2.0 * np.arctan2(rel[1], rel[3])
    err = (angle - (-3.0) + np.pi) % (2.0 * np.pi) - np.pi
    test.assertLess(abs(err), 0.05)


devices = get_test_devices()


class TestSolverXPBDJoints(unittest.TestCase):
    pass


add_function_test(
    TestSolverXPBDJoints,
    "test_revolute_limit_branch_cut_positive_and_negative",
    test_revolute_limit_branch_cut_positive_and_negative,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_revolute_eval_ik_angle_beyond_pi",
    test_revolute_eval_ik_angle_beyond_pi,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_revolute_hold_beyond_pi",
    test_revolute_hold_beyond_pi,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestSolverXPBDJoints,
    "test_revolute_unlimited_drive_takes_short_way",
    test_revolute_unlimited_drive_takes_short_way,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
