# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check MuJoCo USD joint effort ranges, NaN rejection and CPU force precedence."""

import math
import unittest

import newton
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolverMjc, SchemaResolverNewton


class TestImportUsdEffortNan(unittest.TestCase):
    """Exercise authored MuJoCo joint effort limits with and without USD drives."""

    @staticmethod
    def make_stage(kind, *, drive_force=None, merged=False):
        """Construct two articulated bodies with a scalar joint and optional drive."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        root = UsdGeom.Xform.Define(stage, "/World")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        for name in ("Parent", "Child"):
            body = UsdGeom.Cube.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        if kind == "prismatic":
            joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joint")
            joint.CreateAxisAttr("X")
            joint.CreateLowerLimitAttr(-1.0)
            joint.CreateUpperLimitAttr(1.0)
            drive_name = "linear"
        else:
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
            joint.CreateAxisAttr("Z")
            joint.CreateLowerLimitAttr(-45.0)
            joint.CreateUpperLimitAttr(45.0)
            drive_name = "angular"
        joint.CreateBody0Rel().SetTargets(["/World/Parent"])
        joint.CreateBody1Rel().SetTargets(["/World/Child"])
        if drive_force is not None:
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_name)
            drive.CreateStiffnessAttr(8.0)
            drive.CreateMaxForceAttr(drive_force)

        if merged:
            hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge")
            hinge.CreateAxisAttr("Z")
            hinge.CreateLowerLimitAttr(-45.0)
            hinge.CreateUpperLimitAttr(45.0)
            hinge.CreateBody0Rel().SetTargets(["/World/Parent"])
            hinge.CreateBody1Rel().SetTargets(["/World/Child"])

        return stage, joint

    @staticmethod
    def set_joint_effort_range(joint, lower, upper, *, limited="auto"):
        """Author MuJoCo namespace fields without requiring a registered USD schema."""
        from pxr import Sdf

        prim = joint.GetPrim()
        if lower is not None:
            prim.CreateAttribute("mjc:actuatorfrcrange:min", Sdf.ValueTypeNames.Float, custom=True).Set(lower)
        if upper is not None:
            prim.CreateAttribute("mjc:actuatorfrcrange:max", Sdf.ValueTypeNames.Float, custom=True).Set(upper)
        prim.CreateAttribute("mjc:actuatorfrclimited", Sdf.ValueTypeNames.Token, custom=True).Set(limited)

    @staticmethod
    def import_cpu_effort_limit(stage, joint):
        """Resolve the joint limit using the opt-in MuJoCo schema and CPU model."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        paths = builder.add_usd(
            stage,
            load_visual_shapes=False,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()],
        )["path_joint_map"]
        joint_index = paths[joint.GetPath().pathString]
        dof = builder.joint_qd_start[joint_index]
        model = builder.finalize(device="cpu")
        return float(model.joint_effort_limit.numpy()[dof])

    def test_nan_mjc_effort_bounds_are_rejected(self):
        """Reject NaN on either range endpoint before auto or true limit checks."""
        for kind in ("prismatic", "revolute"):
            for field in ("min", "max"):
                for limited in ("auto", "true"):
                    with self.subTest(kind=kind, field=field, limited=limited):
                        stage, joint = self.make_stage(kind, drive_force=10.0)
                        lower = math.nan if field == "min" else -3.0
                        upper = math.nan if field == "max" else 6.0
                        self.set_joint_effort_range(joint, lower, upper, limited=limited)
                        with self.assertRaisesRegex(
                            ValueError, rf"/World/Joint.*NaN.*mjc:actuatorfrcrange:{field}"
                        ):
                            self.import_cpu_effort_limit(stage, joint)

    def test_nan_mjc_effort_bound_is_rejected_on_merged_joint(self):
        """Reject an invalid scalar joint limit before constructing a merged D6."""
        stage, joint = self.make_stage("prismatic", drive_force=10.0, merged=True)
        self.set_joint_effort_range(joint, -3.0, math.nan)
        with self.assertRaisesRegex(ValueError, r"/World/Joint.*NaN.*mjc:actuatorfrcrange:max"):
            self.import_cpu_effort_limit(stage, joint)

    def test_joint_and_drive_force_limit_precedence_on_cpu(self):
        """Choose the tighter finite bound and preserve unlimited force sentinels."""
        cases = (
            (12.0, -3.0, 5.0, "auto", 5.0),
            (3.0, -4.0, 7.0, "true", 3.0),
            (math.inf, -2.0, 6.0, "auto", 6.0),
            (4.0, -math.inf, math.inf, "true", 4.0),
            (math.inf, -math.inf, math.inf, "auto", math.inf),
            (9.0, -2.0, 5.0, "false", 9.0),
            (9.0, 0.0, 0.0, "auto", 9.0),
        )
        for kind in ("prismatic", "revolute"):
            for drive_force, lower, upper, limited, expected in cases:
                with self.subTest(kind=kind, drive_force=drive_force, lower=lower, upper=upper, limited=limited):
                    stage, joint = self.make_stage(kind, drive_force=drive_force)
                    self.set_joint_effort_range(joint, lower, upper, limited=limited)
                    self.assertEqual(self.import_cpu_effort_limit(stage, joint), expected)

    def test_joint_only_effort_limit_on_cpu(self):
        """Honor an authored joint limit without an enabled USD drive."""
        stage, joint = self.make_stage("prismatic")
        self.set_joint_effort_range(joint, -1.5, 4.0, limited="true")
        self.assertEqual(self.import_cpu_effort_limit(stage, joint), 4.0)

    def test_disabled_mjc_effort_range_retains_drive_limit(self):
        """Ignore unused MuJoCo range attributes when force limiting is false."""
        stage, joint = self.make_stage("prismatic", drive_force=9.0)
        self.set_joint_effort_range(joint, math.nan, 5.0, limited="false")
        self.assertEqual(self.import_cpu_effort_limit(stage, joint), 9.0)


if __name__ == "__main__":
    unittest.main()
