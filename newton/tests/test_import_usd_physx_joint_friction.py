# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for unsupported PhysX joint friction imports."""

import unittest
import warnings

import newton
from newton.exceptions import NewtonWarning
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx


class TestPhysxJointFrictionWarning(unittest.TestCase):
    @staticmethod
    def make_stage(joint_type: str, *, physx_friction: float | None = None, newton_friction: float | None = None):
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/World")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        for name in ("Parent", "Child"):
            body = UsdGeom.Cube.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        schema = {"revolute": UsdPhysics.RevoluteJoint, "prismatic": UsdPhysics.PrismaticJoint}[joint_type]
        joint = schema.Define(stage, "/World/Joint")
        joint.CreateBody0Rel().SetTargets(["/World/Parent"])
        joint.CreateBody1Rel().SetTargets(["/World/Child"])
        if physx_friction is not None:
            joint.GetPrim().CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(physx_friction)
        if newton_friction is not None:
            joint.GetPrim().CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(newton_friction)
        return stage

    @staticmethod
    def import_friction(stage, resolvers) -> float:
        builder = newton.ModelBuilder()
        builder.default_joint_cfg.friction = 0.125
        builder.add_usd(stage, schema_resolvers=resolvers)
        index = builder.joint_label.index("/World/Joint")
        return float(builder.joint_friction[builder.joint_qd_start[index]])

    def test_authored_physx_joint_friction_warns_without_false_conversion(self):
        """A dimensionless PhysX coefficient cannot become an absolute N or Nm limit."""
        for kind in ("revolute", "prismatic"):
            with self.subTest(kind=kind):
                stage = self.make_stage(kind, physx_friction=0.75)
                with self.assertWarnsRegex(NewtonWarning, r"/World/Joint.*physxJoint:jointFriction.*load-dependent"):
                    friction = self.import_friction(stage, [SchemaResolverPhysx()])
                self.assertAlmostEqual(friction, 0.125)
                self.assertNotAlmostEqual(friction, 0.75)

    def test_unset_physx_joint_friction_does_not_warn(self):
        stage = self.make_stage("revolute")
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", NewtonWarning)
            friction = self.import_friction(stage, [SchemaResolverPhysx()])
        self.assertAlmostEqual(friction, 0.125)
        self.assertFalse(
            any(
                issubclass(item.category, NewtonWarning) and "physxJoint:jointFriction" in str(item.message)
                for item in seen
            )
        )

    def test_explicit_newton_friction_takes_precedence_without_warning(self):
        stage = self.make_stage("revolute", physx_friction=0.75, newton_friction=0.25)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", NewtonWarning)
            friction = self.import_friction(stage, [SchemaResolverNewton(), SchemaResolverPhysx()])
        self.assertAlmostEqual(friction, 0.25)
        self.assertFalse(
            any(
                issubclass(item.category, NewtonWarning) and "physxJoint:jointFriction" in str(item.message)
                for item in seen
            )
        )


    def test_merged_d6_joint_friction_warning_and_newton_precedence(self):
        """Merged siblings must warn once, without overriding explicit Newton friction."""
        from pxr import Sdf, UsdPhysics

        for newton_friction in (None, 0.25):
            with self.subTest(newton_friction=newton_friction):
                stage = self.make_stage("revolute")
                slide = UsdPhysics.PrismaticJoint.Define(stage, "/World/Slide")
                slide.CreateBody0Rel().SetTargets(["/World/Parent"])
                slide.CreateBody1Rel().SetTargets(["/World/Child"])
                slide_prim = slide.GetPrim()
                slide_prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.75)
                if newton_friction is not None:
                    slide_prim.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(newton_friction)

                builder = newton.ModelBuilder()
                builder.default_joint_cfg.friction = 0.125
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always", NewtonWarning)
                    result = builder.add_usd(
                        stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()]
                    )

                merged = result["path_joint_map"]["/World/Joint"]
                self.assertEqual(result["path_joint_map"]["/World/Slide"], merged)
                self.assertEqual(builder.joint_type[merged], newton.JointType.D6)
                self.assertEqual(builder.joint_dof_dim[merged], (1, 1))

                # Merged D6 DOFs are linear first, then angular.
                qd_start = builder.joint_qd_start[merged]
                expected_linear_friction = 0.125 if newton_friction is None else 0.25
                self.assertAlmostEqual(builder.joint_friction[qd_start], expected_linear_friction)
                self.assertAlmostEqual(builder.joint_friction[qd_start + 1], 0.125)

                friction_warnings = [
                    str(item.message)
                    for item in seen
                    if issubclass(item.category, NewtonWarning)
                    and "physxJoint:jointFriction" in str(item.message)
                ]
                self.assertEqual(len(friction_warnings), 1 if newton_friction is None else 0)
                if newton_friction is None:
                    self.assertRegex(
                        friction_warnings[0],
                        r"/World/Slide.*physxJoint:jointFriction.*load-dependent",
                    )


if __name__ == "__main__":
    unittest.main()
