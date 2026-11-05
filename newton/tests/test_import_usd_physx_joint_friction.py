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
    def make_stage(
        joint_type: str, *, physx_friction: float | None = None, newton_friction: float | None = None
    ):
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
                with self.assertWarnsRegex(
                    NewtonWarning, r"/World/Joint.*physxJoint:jointFriction.*load-dependent"
                ):
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
        stage = self.make_stage(
            "revolute",
            physx_friction=0.75,
            newton_friction=0.25,
        )
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


if __name__ == "__main__":
    unittest.main()
