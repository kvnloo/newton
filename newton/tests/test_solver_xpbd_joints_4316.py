# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for XPBD hinge angle branches and differentiable joint anchors."""

import unittest

import numpy as np
import warp as wp

import newton


@wp.kernel
def _body_x_loss(body_q: wp.array[wp.transform], loss: wp.array[float]):
    loss[0] = wp.transform_get_translation(body_q[0])[0]


class TestXPBDHingeReferenceCPU(unittest.TestCase):
    def _hinge(self, axis, lower, upper, angle=0.0, d6=False):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        if d6:
            joint = builder.add_joint_d6(
                -1,
                link,
                angular_axes=[
                    newton.ModelBuilder.JointDofConfig(
                        axis=axis, limit_lower=lower, limit_upper=upper
                    )
                ],
            )
        else:
            joint = builder.add_joint_revolute(
                -1, link, axis=axis, limit_lower=lower, limit_upper=upper
            )
        builder.add_articulation([joint])
        builder.joint_q = [angle]
        builder.joint_qd = [0.0]
        return builder.finalize(device="cpu")

    def test_limited_hinge_ik_beyond_pi(self):
        """Verify revolute and one-axis D6 angles unwrap near finite limit midpoints."""
        for d6 in (False, True):
            with self.subTest(d6=d6):
                model = self._hinge((0.0, 1.0, 0.0), -0.227, 3.421, 3.3, d6=d6)
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                q = wp.zeros(1, dtype=float, device="cpu")
                qd = wp.zeros(1, dtype=float, device="cpu")
                newton.eval_ik(model, state, q, qd)
                self.assertAlmostEqual(float(q.numpy()[0]), 3.3, delta=1.0e-4)

                # Quaternions q and -q represent the same orientation.
                body_q = state.body_q.numpy()
                body_q[0, 3:] *= -1.0
                state.body_q.assign(body_q)
                newton.eval_ik(model, state, q, qd)
                self.assertAlmostEqual(float(q.numpy()[0]), 3.3, delta=1.0e-4)

    def test_oblique_hinge_remains_at_rest(self):
        """Verify reference selection never accelerates oblique resting hinges."""
        for d6 in (False, True):
            with self.subTest(d6=d6):
                model = self._hinge((1.0, 1.0, 0.0), -0.5, 1.5, d6=d6)
                solver = newton.solvers.SolverXPBD(
                    model,
                    iterations=4,
                    joint_linear_relaxation=0.4,
                    joint_angular_relaxation=0.4,
                )
                before, after = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, before)
                solver.step(before, after, model.control(), None, 1.25e-3)
                np.testing.assert_allclose(
                    after.body_qd.numpy()[0], np.zeros(6), rtol=0.0, atol=1.0e-3
                )

    def test_limited_hinge_beyond_pi_stays_at_rest(self):
        """Verify a resting hinge beyond pi does not receive a false limit impulse."""
        model = self._hinge((0.0, 1.0, 0.0), -0.227, 3.421, angle=3.3)
        solver = newton.solvers.SolverXPBD(
            model, iterations=4, joint_linear_relaxation=0.4, joint_angular_relaxation=0.4
        )
        before, after = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, before)
        solver.step(before, after, model.control(), None, 1.25e-3)
        np.testing.assert_allclose(
            after.body_qd.numpy()[0], np.zeros(6), rtol=0.0, atol=1.0e-3
        )

    @staticmethod
    def _fixed_model(child_anchor, requires_grad):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_fixed(
            -1,
            link,
            child_xform=wp.transform((child_anchor, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([joint])
        return builder.finalize(device="cpu", requires_grad=requires_grad)

    @staticmethod
    def _fixed_step(model, record_grad=False):
        solver = newton.solvers.SolverXPBD(model, iterations=1, joint_linear_relaxation=0.7)
        before = model.state(requires_grad=record_grad)
        after = model.state(requires_grad=record_grad)
        before.body_q.assign([wp.transform_identity()])
        if not record_grad:
            solver.step(before, after, model.control(), None, 1.25e-3)
            return float(after.body_q.numpy()[0, 0]), None

        loss = wp.zeros(1, dtype=float, device="cpu", requires_grad=True)
        with wp.Tape() as tape:
            solver.step(before, after, model.control(), None, 1.25e-3)
            wp.launch(_body_x_loss, dim=1, inputs=[after.body_q], outputs=[loss], device="cpu")
        tape.backward(loss=loss)
        anchor_grad = tape.gradients.get(model.joint_X_c)
        return float(after.body_q.numpy()[0, 0]), (
            None if anchor_grad is None else float(anchor_grad.numpy()[0, 0])
        )

    def test_fixed_child_anchor_gradient_matches_finite_difference(self):
        """Verify fixed-anchor gradients pass through the original model array."""
        model = self._fixed_model(0.1, requires_grad=True)
        self.assertTrue(model.joint_X_c.requires_grad)
        x, grad = self._fixed_step(model, record_grad=True)
        self.assertAlmostEqual(x, -0.07, delta=2.0e-3)
        self.assertIsNotNone(grad, "The joint anchor must remain connected to the Warp tape")

        eps = 1.0e-3
        xp, _ = self._fixed_step(self._fixed_model(0.1 + eps, False))
        xm, _ = self._fixed_step(self._fixed_model(0.1 - eps, False))
        numerical = (xp - xm) / (2.0 * eps)
        self.assertAlmostEqual(numerical, -0.7, delta=1.0e-2)
        self.assertAlmostEqual(grad, numerical, delta=1.0e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
