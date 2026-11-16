# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate USD distance-joint limits through CPU model construction."""

import math
import unittest

import newton


class TestImportUsdDistanceNanJointLimits(unittest.TestCase):
    """Exercise distance-joint bounds authored in a real in-memory USD stage."""

    @staticmethod
    def make_stage(min_distance=None, max_distance=None):
        """Create an articulated pair of bodies with an authored distance joint."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")

        root = UsdGeom.Xform.Define(stage, "/World")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        for name in ("Body0", "Body1"):
            body = UsdGeom.Cube.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        joint = UsdPhysics.DistanceJoint.Define(stage, "/World/DistanceJoint")
        joint.CreateBody0Rel().SetTargets(["/World/Body0"])
        joint.CreateBody1Rel().SetTargets(["/World/Body1"])
        if min_distance is not None:
            joint.CreateMinDistanceAttr(min_distance)
        if max_distance is not None:
            joint.CreateMaxDistanceAttr(max_distance)
        return stage

    def test_nan_distance_bounds_raise_during_cpu_import(self):
        """Reject either NaN bound even when USD marks that bound disabled."""
        for bound in ("min", "max"):
            with self.subTest(bound=bound):
                min_distance = math.nan if bound == "min" else 0.25
                max_distance = math.nan if bound == "max" else 1.5
                stage = self.make_stage(min_distance, max_distance)

                with self.assertRaisesRegex(
                    ValueError,
                    rf"USD joint /World/DistanceJoint has NaN {bound} distance limit",
                ):
                    builder = newton.ModelBuilder()
                    builder.add_usd(stage, load_visual_shapes=False)
                    builder.finalize(device="cpu")

    def test_finite_disabled_and_infinite_distance_limits_on_cpu(self):
        """Preserve finite meters, negative disabled bounds, and infinities."""
        cases = (
            ("no limits", None, None, -1.0, -1.0),
            ("minimum only", 0.25, None, 0.25, -1.0),
            ("maximum only", None, 1.5, -1.0, 1.5),
            ("both finite", 0.25, 1.5, 0.25, 1.5),
            ("disabled minimum", -0.25, None, -1.0, -1.0),
            ("disabled maximum", None, -0.25, -1.0, -1.0),
            ("negative infinity disables minimum", -math.inf, None, -1.0, -1.0),
            ("positive infinity maximum", None, math.inf, -1.0, math.inf),
            ("infinite interval", -math.inf, math.inf, -1.0, math.inf),
        )
        for name, min_distance, max_distance, expected_min, expected_max in cases:
            with self.subTest(name=name):
                builder = newton.ModelBuilder()
                builder.add_usd(
                    self.make_stage(min_distance, max_distance),
                    load_visual_shapes=False,
                )
                joint_index = builder.joint_label.index("/World/DistanceJoint")
                dof_index = builder.joint_qd_start[joint_index]
                model = builder.finalize(device="cpu")
                actual_min = float(model.joint_limit_lower.numpy()[dof_index])
                actual_max = float(model.joint_limit_upper.numpy()[dof_index])
                self.assertEqual((actual_min, actual_max), (expected_min, expected_max))
