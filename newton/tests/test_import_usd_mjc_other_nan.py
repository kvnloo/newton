# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate authored MuJoCo joint numeric data at USD import and CPU finalization."""

import math
import unittest

import newton
from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolverMjc, SchemaResolverNewton


class TestImportUsdMjcOtherNan(unittest.TestCase):
    """Exercise non-effort MuJoCo joint values through the real USD and CPU model paths."""

    CORE_FIELDS = ("mjc:armature", "mjc:frictionloss", "mjc:ref", "mjc:springref")
    RESPONSE_FIELDS = (
        ("mjc:margin", (0.25,)),
        ("mjc:solimplimit", (0.9, 0.95, 0.001, 0.5, 2.0)),
        ("mjc:solreflimit", (0.02, 1.0)),
        ("mjc:solreffriction", (0.02, 1.0)),
        ("mjc:solimpfriction", (0.9, 0.95, 0.001, 0.5, 2.0)),
    )

    @staticmethod
    def make_stage(kind="prismatic", *, merged=False):
        """Build two articulated bodies and a concrete USD joint."""
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
        else:
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
            joint.CreateAxisAttr("Z")
            joint.CreateLowerLimitAttr(-45.0)
            joint.CreateUpperLimitAttr(45.0)
        joint.CreateBody0Rel().SetTargets(["/World/Parent"])
        joint.CreateBody1Rel().SetTargets(["/World/Child"])

        if merged:
            hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge")
            hinge.CreateAxisAttr("Z")
            hinge.CreateLowerLimitAttr(-45.0)
            hinge.CreateUpperLimitAttr(45.0)
            hinge.CreateBody0Rel().SetTargets(["/World/Parent"])
            hinge.CreateBody1Rel().SetTargets(["/World/Child"])
        return stage, joint

    @staticmethod
    def author(joint, name, value):
        """Author a scalar or vector MuJoCo numeric USD attribute."""
        from pxr import Sdf

        dtype = Sdf.ValueTypeNames.FloatArray if isinstance(value, (list, tuple)) else Sdf.ValueTypeNames.Float
        joint.GetPrim().CreateAttribute(name, dtype, custom=True).Set(value)

    @staticmethod
    def import_cpu(stage, joint):
        """Import with the opted-in MuJoCo schema and finalize a real CPU model."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        joints = builder.add_usd(
            stage,
            load_visual_shapes=False,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverMjc()],
        )["path_joint_map"]
        joint_index = joints[joint.GetPath().pathString]
        dof = builder.joint_qd_start[joint_index]
        coord = builder.joint_q_start[joint_index]
        model = builder.finalize(device="cpu")
        return model, dof, coord

    def test_nonfinite_core_joint_scalars_reject_nan(self):
        """Reject authored NaN armature, friction, and reference coordinates for scalar DOFs."""
        for kind in ("prismatic", "revolute"):
            for field in self.CORE_FIELDS:
                with self.subTest(kind=kind, field=field):
                    stage, joint = self.make_stage(kind)
                    self.author(joint, field, math.nan)
                    with self.assertRaisesRegex(ValueError, rf"/World/Joint.*NaN.*{field}"):
                        self.import_cpu(stage, joint)

    def test_nan_reference_rejected_before_merged_joint_creation(self):
        """Reject a NaN spring reference while merging scalar USD joints into a D6."""
        stage, joint = self.make_stage("prismatic", merged=True)
        self.author(joint, "mjc:springref", math.nan)
        with self.assertRaisesRegex(ValueError, r"/World/Joint.*NaN.*mjc:springref"):
            self.import_cpu(stage, joint)

    def test_nan_raw_joint_response_scalars_and_vectors(self):
        """Reject NaN in authored MuJoCo limit response fields before Warp conversion."""
        for field, valid in self.RESPONSE_FIELDS:
            positions = (0,) if len(valid) == 1 else (0, len(valid) - 1)
            for position in positions:
                with self.subTest(field=field, component=position):
                    stage, joint = self.make_stage()
                    values = list(valid)
                    values[position] = math.nan
                    authored = values[0] if len(values) == 1 else values
                    self.author(joint, field, authored)
                    with self.assertRaisesRegex(ValueError, rf"/World/Joint.*NaN.*{field}"):
                        self.import_cpu(stage, joint)

    def test_finite_joint_values_keep_cpu_units_and_reference_shift(self):
        """Preserve finite MuJoCo joint data, angular conversion and shifted limits on CPU."""
        for kind in ("prismatic", "revolute"):
            with self.subTest(kind=kind):
                stage, joint = self.make_stage(kind)
                springref, reference = (0.6, 0.1) if kind == "prismatic" else (45.0, 15.0)
                for field, value in (
                    ("mjc:armature", 0.75),
                    ("mjc:frictionloss", 0.4),
                    ("mjc:damping", 2.0),
                    ("mjc:stiffness", 3.0),
                    ("mjc:springref", springref),
                    ("mjc:ref", reference),
                ):
                    self.author(joint, field, value)
                for field, values in self.RESPONSE_FIELDS:
                    self.author(joint, field, values[0] if len(values) == 1 else list(values))

                model, dof, coord = self.import_cpu(stage, joint)
                self.assertAlmostEqual(float(model.joint_armature.numpy()[dof]), 0.75)
                self.assertAlmostEqual(float(model.joint_friction.numpy()[dof]), 0.4)
                self.assertAlmostEqual(float(model.joint_damping.numpy()[dof]), 2.0)
                self.assertAlmostEqual(float(model.joint_stiffness.numpy()[dof]), 3.0)
                expected_rest = 0.5 if kind == "prismatic" else math.pi / 6.0
                expected_low = -1.1 if kind == "prismatic" else -math.pi / 3.0
                self.assertAlmostEqual(float(model.joint_rest_q.numpy()[coord]), expected_rest, places=5)
                self.assertAlmostEqual(float(model.joint_limit_lower.numpy()[dof]), expected_low, places=5)
                self.assertAlmostEqual(float(model.mujoco.limit_margin.numpy()[dof]), 0.25)
                self.assertAlmostEqual(float(model.mujoco.solreflimit.numpy()[dof][0]), 0.02, places=5)

    def test_authored_zero_solref_sentinel_and_inactive_mjc_value(self):
        """Preserve raw zero solref mode and ignore a lower-priority NaN armature."""
        stage, joint = self.make_stage()
        self.author(joint, "mjc:solreflimit", [0.0, 0.0])
        self.author(joint, "mjc:armature", math.nan)
        self.author(joint, "newton:armature", 1.25)
        model, dof, _ = self.import_cpu(stage, joint)
        self.assertAlmostEqual(float(model.joint_armature.numpy()[dof]), 1.25)
        self.assertEqual(int(model.mujoco.solreflimit_mode.numpy()[dof]), int(SOLREF_MODE_RAW))
        self.assertEqual(list(model.mujoco.solreflimit.numpy()[dof]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
