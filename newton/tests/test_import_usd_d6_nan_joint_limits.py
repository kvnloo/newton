# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise all six USD D6 limit axes with invalid and valid CPU imports."""

import math
import unittest

import newton


D6_AXES = ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")


class TestImportUsdD6NanJointLimits(unittest.TestCase):
    @staticmethod
    def make_stage(axis, lower, upper, *, other_free_axis=None):
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")

        root = UsdGeom.Xform.Define(stage, "/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        for name in ("Parent", "Child"):
            body = UsdGeom.Cube.Define(stage, f"/Robot/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        joint = UsdPhysics.Joint.Define(stage, "/Robot/Joint")
        joint.CreateBody0Rel().SetTargets(["/Robot/Parent"])
        joint.CreateBody1Rel().SetTargets(["/Robot/Child"])

        # Explicitly lock all other axes so each control has one known DOF.
        for name in D6_AXES:
            axis_lower, axis_upper = (1.0, -1.0)
            if name == axis:
                axis_lower, axis_upper = lower, upper
            elif name == other_free_axis:
                axis_lower, axis_upper = -1.0, 1.0
            limit_api = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), name)
            limit_api.CreateLowAttr(axis_lower)
            limit_api.CreateHighAttr(axis_upper)
        return stage

    def import_axis_on_cpu(self, axis, lower, upper):
        builder = newton.ModelBuilder()
        builder.add_usd(self.make_stage(axis, lower, upper), load_visual_shapes=False)
        joint_index = builder.joint_label.index("/Robot/Joint")
        expected_dim = (1, 0) if axis.startswith("trans") else (0, 1)
        self.assertEqual(builder.joint_dof_dim[joint_index], expected_dim)

        model = builder.finalize(device="cpu")
        dof = int(model.joint_qd_start.numpy()[joint_index])
        return (
            float(model.joint_limit_lower.numpy()[dof]),
            float(model.joint_limit_upper.numpy()[dof]),
        )

    def test_nan_limits_rejected_for_every_d6_axis_during_cpu_import(self):
        """Reject NaN limits on both bounds of every USD D6 axis."""
        for axis in D6_AXES:
            for bound in ("lower", "upper"):
                with self.subTest(axis=axis, bound=bound):
                    lower = float("nan") if bound == "lower" else -1.0
                    upper = float("nan") if bound == "upper" else 1.0
                    # Retain a different free axis to avoid a zero-DOF fallback.
                    other_free_axis = "rotX" if axis == "transX" else "transX"
                    stage = self.make_stage(axis, lower, upper, other_free_axis=other_free_axis)

                    with self.assertRaisesRegex(
                        ValueError,
                        rf"USD joint /Robot/Joint axis {axis} has NaN {bound} limit",
                    ):
                        builder = newton.ModelBuilder()
                        builder.add_usd(stage, load_visual_shapes=False)
                        builder.finalize(device="cpu")

    def test_finite_d6_limits_preserve_axis_units_on_cpu(self):
        """Preserve translational meters and rotational radians."""
        for axis in D6_AXES:
            with self.subTest(axis=axis):
                if axis.startswith("trans"):
                    lower, upper = -0.25, 0.5
                    expected_lower, expected_upper = -0.25, 0.5
                else:
                    lower, upper = -45.0, 90.0
                    expected_lower, expected_upper = -math.pi / 4.0, math.pi / 2.0
                actual_lower, actual_upper = self.import_axis_on_cpu(axis, lower, upper)
                self.assertAlmostEqual(actual_lower, expected_lower, places=6)
                self.assertAlmostEqual(actual_upper, expected_upper, places=6)

    def test_infinite_d6_limits_use_newton_unbounded_sentinels_on_cpu(self):
        """Map USD infinities to finite Newton unbounded-limit sentinels."""
        for axis in D6_AXES:
            with self.subTest(axis=axis):
                lower, upper = self.import_axis_on_cpu(axis, -math.inf, math.inf)
                self.assertEqual(lower, -newton.MAXVAL)
                self.assertLess(lower, 0.0)
                self.assertEqual(upper, newton.MAXVAL)
                self.assertGreater(upper, 0.0)
