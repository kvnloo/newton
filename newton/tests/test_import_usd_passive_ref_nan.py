# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject non-finite MuJoCo USD spring references before CPU finalization."""

import math
import unittest

import newton
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolverMjc, SchemaResolverNewton


class TestImportUsdPassiveRefNan(unittest.TestCase):
    """Exercise MuJoCo scalar reference resolution and merged USD joints."""

    @staticmethod
    def make_stage(kind, *, angle="degree", merged=False):
        """Build a small in-memory articulated USD scene with one scalar joint."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        scene.GetPrim().CreateAttribute("mjc:compiler:angle", Sdf.ValueTypeNames.Token, custom=True).Set(angle)
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
        elif kind == "revolute":
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
            joint.CreateAxisAttr("X")
            joint.CreateLowerLimitAttr(-45.0)
            joint.CreateUpperLimitAttr(45.0)
        else:
            raise ValueError(f"Unsupported scalar joint kind: {kind}")

        joint.CreateBody0Rel().SetTargets(["/World/Parent"])
        joint.CreateBody1Rel().SetTargets(["/World/Child"])

        if merged:
            hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge")
            hinge.CreateBody0Rel().SetTargets(["/World/Parent"])
            hinge.CreateBody1Rel().SetTargets(["/World/Child"])
            hinge.CreateAxisAttr("Z")
            hinge.CreateLowerLimitAttr(-45.0)
            hinge.CreateUpperLimitAttr(45.0)

        return stage, joint.GetPrim()

    @staticmethod
    def author_reference(prim, name, value):
        """Author a real MuJoCo namespaced scalar USD attribute."""
        from pxr import Sdf

        prim.CreateAttribute(f"mjc:{name}", Sdf.ValueTypeNames.Float, custom=True).Set(float(value))

    @staticmethod
    def import_cpu(stage, *, register):
        """Import with an explicit MuJoCo resolver and finalize on CPU."""
        builder = newton.ModelBuilder()
        if register:
            SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(
            stage,
            load_visual_shapes=False,
            schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
        )
        return builder, builder.finalize(device="cpu")

    def test_nonfinite_scalar_references_are_rejected(self):
        """Reject NaN and both infinities for each active scalar reference."""
        for kind in ("prismatic", "revolute"):
            for name in ("ref", "springref"):
                for value in (math.nan, math.inf, -math.inf):
                    for register in (False, True):
                        with self.subTest(kind=kind, name=name, value=value, register=register):
                            stage, prim = self.make_stage(kind)
                            self.author_reference(prim, name, value)
                            with self.assertRaisesRegex(
                                ValueError, rf"USD joint /World/Joint has non-finite mjc:{name}"
                            ):
                                self.import_cpu(stage, register=register)

    def test_merged_joint_references_are_rejected(self):
        """Reject NaN references on a scalar joint merged into a D6 joint."""
        for name in ("ref", "springref"):
            for register in (False, True):
                with self.subTest(name=name, register=register):
                    stage, prim = self.make_stage("prismatic", merged=True)
                    self.author_reference(prim, name, math.nan)
                    with self.assertRaisesRegex(
                        ValueError, rf"USD joint /World/Joint has non-finite mjc:{name}"
                    ):
                        self.import_cpu(stage, register=register)

    def test_finite_reference_offsets_and_units_on_cpu(self):
        """Preserve linear offsets and angular degree/radian conversions."""
        cases = (
            ("prismatic", "degree", 0.1, 0.35, 0.25, -1.1, 0.9),
            (
                "revolute",
                "degree",
                30.0,
                45.0,
                math.pi / 12.0,
                -5.0 * math.pi / 12.0,
                math.pi / 12.0,
            ),
            (
                "revolute",
                "radian",
                0.4,
                0.7,
                0.3,
                -math.pi / 4.0 - 0.4,
                math.pi / 4.0 - 0.4,
            ),
        )
        for kind, angle, ref, springref, rest, lower, upper in cases:
            for register in (False, True):
                with self.subTest(kind=kind, angle=angle, register=register):
                    stage, prim = self.make_stage(kind, angle=angle)
                    self.author_reference(prim, "ref", ref)
                    self.author_reference(prim, "springref", springref)
                    builder, model = self.import_cpu(stage, register=register)
                    joint = builder.joint_label.index("/World/Joint")
                    q_index = builder.joint_q_start[joint]
                    dof = builder.joint_qd_start[joint]
                    self.assertAlmostEqual(float(model.joint_rest_q.numpy()[q_index]), rest, places=5)
                    self.assertAlmostEqual(float(model.joint_limit_lower.numpy()[dof]), lower, places=5)
                    self.assertAlmostEqual(float(model.joint_limit_upper.numpy()[dof]), upper, places=5)

    def test_omitted_reference_defaults_on_cpu(self):
        """Preserve zero defaults and a missing spring reference with nonzero ref."""
        cases = (
            (0.1, None, -0.1, -1.1, 0.9),
            (None, 0.35, 0.35, -1.0, 1.0),
            (None, None, 0.0, -1.0, 1.0),
        )
        for ref, springref, rest, lower, upper in cases:
            with self.subTest(ref=ref, springref=springref):
                stage, prim = self.make_stage("prismatic")
                if ref is not None:
                    self.author_reference(prim, "ref", ref)
                if springref is not None:
                    self.author_reference(prim, "springref", springref)
                builder, model = self.import_cpu(stage, register=True)
                joint = builder.joint_label.index("/World/Joint")
                q_index = builder.joint_q_start[joint]
                dof = builder.joint_qd_start[joint]
                self.assertAlmostEqual(float(model.joint_rest_q.numpy()[q_index]), rest, places=5)
                self.assertAlmostEqual(float(model.joint_limit_lower.numpy()[dof]), lower, places=5)
                self.assertAlmostEqual(float(model.joint_limit_upper.numpy()[dof]), upper, places=5)
