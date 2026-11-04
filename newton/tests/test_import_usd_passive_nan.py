# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject NaN passive USD joint parameters and initial state on CPU."""

import math
import unittest

import newton
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx


class TestImportUsdPassiveNan(unittest.TestCase):
    """Cover resolved passive joint parameters and authored initial DOF state."""

    @staticmethod
    def make_stage(kind):
        """Create a minimal articulated USD scene with an undriven joint."""
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
        elif kind == "revolute":
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
            joint.CreateAxisAttr("X")
            joint.CreateLowerLimitAttr(-45.0)
            joint.CreateUpperLimitAttr(45.0)
        elif kind == "spherical":
            joint = UsdPhysics.SphericalJoint.Define(stage, "/World/Joint")
        elif kind == "d6":
            joint = UsdPhysics.Joint.Define(stage, "/World/Joint")
            for axis in ("transX", "rotX"):
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                bound = 1.0 if axis == "transX" else 45.0
                limit.CreateLowAttr(-bound)
                limit.CreateHighAttr(bound)
        else:
            raise ValueError(f"Unsupported joint kind: {kind}")

        joint.CreateBody0Rel().SetTargets(["/World/Parent"])
        joint.CreateBody1Rel().SetTargets(["/World/Child"])
        return stage, joint.GetPrim()

    @staticmethod
    def author_float(prim, name, value):
        """Author a real namespaced USD float attribute."""
        from pxr import Sdf

        prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=True).Set(value)

    @staticmethod
    def import_cpu(stage):
        """Import with the supported schema resolvers and finalize on CPU."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(
            stage,
            load_visual_shapes=False,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
        )
        return builder, builder.finalize(device="cpu")

    def test_nan_passive_damping_rejected(self):
        """Reject authored NaN passive damping on scalar, spherical and D6 joints."""
        for kind in ("prismatic", "revolute", "spherical", "d6"):
            with self.subTest(kind=kind):
                stage, prim = self.make_stage(kind)
                self.author_float(prim, "newton:damping", math.nan)
                field = "linearDamping" if kind == "d6" else "damping"
                with self.assertRaisesRegex(ValueError, rf"USD joint /World/Joint has NaN {field}"):
                    self.import_cpu(stage)

    def test_nan_passive_stiffness_rejected(self):
        """Reject NaN MuJoCo passive stiffness on every supported joint kind."""
        for kind in ("prismatic", "revolute", "spherical", "d6"):
            with self.subTest(kind=kind):
                stage, prim = self.make_stage(kind)
                self.author_float(prim, "mjc:stiffness", math.nan)
                with self.assertRaisesRegex(ValueError, r"USD joint /World/Joint has NaN stiffness"):
                    self.import_cpu(stage)

    def test_nan_initial_state_rejected(self):
        """Reject NaN position and velocity on active scalar and D6 DOFs."""
        cases = (
            ("prismatic", "linear", "initialPosition", "position", ""),
            ("prismatic", "linear", "initialVelocity", "velocity", ""),
            ("revolute", "angular", "initialPosition", "position", ""),
            ("revolute", "angular", "initialVelocity", "velocity", ""),
            ("d6", "rotX", "initialPosition", "position", " axis rotX"),
            ("d6", "rotX", "initialVelocity", "velocity", " axis rotX"),
        )
        for kind, axis, field, component, axis_suffix in cases:
            with self.subTest(kind=kind, field=field):
                stage, prim = self.make_stage(kind)
                self.author_float(prim, f"state:{axis}:physics:{component}", math.nan)
                with self.assertRaisesRegex(
                    ValueError, rf"USD joint /World/Joint{axis_suffix} has NaN {field}"
                ):
                    self.import_cpu(stage)

    def test_merged_joint_rejects_nan_initial_state(self):
        """Reject invalid state on a scalar joint merged into a D6 joint."""
        from pxr import UsdPhysics

        stage, prim = self.make_stage("prismatic")
        self.author_float(prim, "state:linear:physics:velocity", math.nan)
        hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge")
        hinge.CreateBody0Rel().SetTargets(["/World/Parent"])
        hinge.CreateBody1Rel().SetTargets(["/World/Child"])
        hinge.CreateAxisAttr("Z")
        hinge.CreateLowerLimitAttr(-45.0)
        hinge.CreateUpperLimitAttr(45.0)
        with self.assertRaisesRegex(ValueError, r"USD joint /World/Joint has NaN initialVelocity"):
            self.import_cpu(stage)

    def test_finite_passive_values_and_initial_state_on_cpu(self):
        """Preserve passive gains and degree-to-radian state conversion on CPU."""
        for kind in ("prismatic", "revolute", "d6"):
            with self.subTest(kind=kind):
                stage, prim = self.make_stage(kind)
                self.author_float(prim, "newton:damping", 2.0)
                self.author_float(prim, "mjc:stiffness", 3.0)
                axis = "linear" if kind == "prismatic" else "angular" if kind == "revolute" else "rotX"
                position = 0.25 if kind == "prismatic" else 30.0
                velocity = 0.5 if kind == "prismatic" else 60.0
                self.author_float(prim, f"state:{axis}:physics:position", position)
                self.author_float(prim, f"state:{axis}:physics:velocity", velocity)

                builder, model = self.import_cpu(stage)
                joint = builder.joint_label.index("/World/Joint")
                dof = builder.joint_qd_start[joint]
                coord = builder.joint_q_start[joint]
                if kind == "d6":
                    self.assertAlmostEqual(float(model.joint_stiffness.numpy()[dof]), 3.0, places=5)
                    self.assertAlmostEqual(float(model.joint_damping.numpy()[dof]), 2.0, places=5)
                    dof += 1
                    coord += 1

                self.assertAlmostEqual(float(model.joint_stiffness.numpy()[dof]), 3.0, places=5)
                expected_damping = 2.0 if kind == "prismatic" else 2.0 * 180.0 / math.pi
                self.assertAlmostEqual(float(model.joint_damping.numpy()[dof]), expected_damping, places=4)
                expected_position = 0.25 if kind == "prismatic" else math.pi / 6.0
                expected_velocity = 0.5 if kind == "prismatic" else math.pi / 3.0
                self.assertAlmostEqual(float(model.joint_q.numpy()[coord]), expected_position, places=5)
                self.assertAlmostEqual(float(model.joint_qd.numpy()[dof]), expected_velocity, places=5)

    def test_inactive_d6_initial_state_is_not_applied(self):
        """Ignore a NaN on a D6 axis that has no active degree of freedom."""
        stage, prim = self.make_stage("d6")
        self.author_float(prim, "state:rotY:physics:position", math.nan)
        builder, model = self.import_cpu(stage)
        joint = builder.joint_label.index("/World/Joint")
        q_start = builder.joint_q_start[joint]
        self.assertTrue(math.isfinite(float(model.joint_q.numpy()[q_start])))
        self.assertTrue(math.isfinite(float(model.joint_q.numpy()[q_start + 1])))
