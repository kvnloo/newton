# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject NaN USD scalar-joint limits before they enter model state."""

import math
import unittest

import newton


class TestImportUsdNanJointLimits(unittest.TestCase):
    @staticmethod
    def make_stage(joint_type, lower, upper):
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root = UsdGeom.Xform.Define(stage, "/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        for name in ("Parent", "Child"):
            body = UsdGeom.Cube.Define(stage, f"/Robot/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        joint_cls = UsdPhysics.RevoluteJoint if joint_type == "revolute" else UsdPhysics.PrismaticJoint
        joint = joint_cls.Define(stage, "/Robot/Joint")
        joint.CreateBody0Rel().SetTargets(["/Robot/Parent"])
        joint.CreateBody1Rel().SetTargets(["/Robot/Child"])
        joint.CreateLowerLimitAttr(lower)
        joint.CreateUpperLimitAttr(upper)
        return stage

    def test_nan_scalar_joint_limits_raise_during_import(self):
        for joint_type in ("revolute", "prismatic"):
            for bound in ("lower", "upper"):
                with self.subTest(joint_type=joint_type, bound=bound):
                    lower = float("nan") if bound == "lower" else -45.0
                    upper = float("nan") if bound == "upper" else 45.0
                    stage = self.make_stage(joint_type, lower, upper)

                    with self.assertRaisesRegex(
                        ValueError, rf"USD joint /Robot/Joint has NaN {bound} limit"
                    ):
                        newton.ModelBuilder().add_usd(stage)

    def test_finite_scalar_joint_limits_preserve_units_on_cpu(self):
        cases = (
            ("revolute", -45.0, 45.0, -math.pi / 4.0, math.pi / 4.0),
            ("prismatic", -0.25, 0.25, -0.25, 0.25),
        )
        for joint_type, lower, upper, expected_lower, expected_upper in cases:
            with self.subTest(joint_type=joint_type):
                builder = newton.ModelBuilder()
                builder.add_usd(self.make_stage(joint_type, lower, upper))
                dof = builder.joint_qd_start[builder.joint_label.index("/Robot/Joint")]
                model = builder.finalize(device="cpu")

                self.assertAlmostEqual(float(model.joint_limit_lower.numpy()[dof]), expected_lower, places=6)
                self.assertAlmostEqual(float(model.joint_limit_upper.numpy()[dof]), expected_upper, places=6)
