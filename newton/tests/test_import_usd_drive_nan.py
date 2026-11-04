# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject NaN USD joint-drive values and preserve valid CPU drive controls."""

import math
import unittest

import newton


class TestImportUsdDriveNan(unittest.TestCase):
    """Exercise scalar, native D6, and merged joint drive parsing."""

    @staticmethod
    def make_stage(kind, *, with_drive=True):
        """Create a two-body articulated USD scene with one driven joint."""
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

        axis = None
        if kind == "prismatic":
            joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joint")
            joint.CreateAxisAttr("X")
            joint.CreateLowerLimitAttr(-1.0)
            joint.CreateUpperLimitAttr(1.0)
            drive_name = "linear"
        elif kind == "revolute":
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
            joint.CreateAxisAttr("X")
            joint.CreateLowerLimitAttr(-45.0)
            joint.CreateUpperLimitAttr(45.0)
            drive_name = "angular"
        else:
            axis = kind.removeprefix("d6-")
            if axis not in ("transX", "rotX"):
                raise ValueError(f"Unsupported D6 test axis: {axis}")
            joint = UsdPhysics.Joint.Define(stage, "/World/Joint")
            for name in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
                lower, upper = (-1.0, 1.0) if name == axis else (1.0, -1.0)
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), name)
                limit.CreateLowAttr(lower)
                limit.CreateHighAttr(upper)
            drive_name = axis

        joint.CreateBody0Rel().SetTargets(["/World/Parent"])
        joint.CreateBody1Rel().SetTargets(["/World/Child"])
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_name) if with_drive else None
        return stage, drive, axis

    def test_nan_drive_fields_rejected_on_scalar_joints(self):
        """Reject NaN gains, force limit and targets on scalar joint drives."""
        for kind in ("prismatic", "revolute"):
            for attribute in ("Stiffness", "Damping", "MaxForce", "TargetPosition", "TargetVelocity"):
                with self.subTest(kind=kind, attribute=attribute):
                    stage, drive, _ = self.make_stage(kind)
                    getattr(drive, f"Create{attribute}Attr")(math.nan)
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"USD joint /World/Joint has NaN drive {attribute[0].lower() + attribute[1:]}",
                    ):
                        builder = newton.ModelBuilder()
                        builder.add_usd(stage, load_visual_shapes=False)
                        builder.finalize(device="cpu")

    def test_nan_drive_fields_rejected_on_d6_axes(self):
        """Reject NaN fields on translation and rotation D6 drives."""
        for axis in ("transX", "rotX"):
            for attribute in ("Stiffness", "Damping", "MaxForce", "TargetPosition", "TargetVelocity"):
                with self.subTest(axis=axis, attribute=attribute):
                    stage, drive, _ = self.make_stage(f"d6-{axis}")
                    getattr(drive, f"Create{attribute}Attr")(math.nan)
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"USD joint /World/Joint axis {axis} has NaN drive "
                        rf"{attribute[0].lower() + attribute[1:]}",
                    ):
                        builder = newton.ModelBuilder()
                        builder.add_usd(stage, load_visual_shapes=False)
                        builder.finalize(device="cpu")

    def test_merged_joint_rejects_nan_drive(self):
        """Validate scalar drives before merging two joints into a D6 joint."""
        from pxr import UsdPhysics

        stage, drive, _ = self.make_stage("prismatic")
        drive.CreateTargetVelocityAttr(math.nan)
        hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge")
        hinge.CreateBody0Rel().SetTargets(["/World/Parent"])
        hinge.CreateBody1Rel().SetTargets(["/World/Child"])
        hinge.CreateAxisAttr("Z")
        hinge.CreateLowerLimitAttr(-45.0)
        hinge.CreateUpperLimitAttr(45.0)

        with self.assertRaisesRegex(ValueError, r"USD joint /World/Joint has NaN drive targetVelocity"):
            builder = newton.ModelBuilder()
            builder.add_usd(stage, load_visual_shapes=False)
            builder.finalize(device="cpu")

    def test_finite_drives_and_unlimited_force_on_cpu(self):
        """Preserve finite drive inputs and the standard unlimited-force sentinel."""
        for kind in ("prismatic", "revolute", "d6-transX", "d6-rotX"):
            for maximum in (12.0, math.inf):
                with self.subTest(kind=kind, maximum=maximum):
                    stage, drive, _ = self.make_stage(kind)
                    drive.CreateStiffnessAttr(8.0)
                    drive.CreateDampingAttr(2.0)
                    drive.CreateTargetPositionAttr(0.25)
                    drive.CreateTargetVelocityAttr(0.5)
                    drive.CreateMaxForceAttr(maximum)

                    builder = newton.ModelBuilder()
                    builder.add_usd(stage, joint_drive_gains_scaling=1.0, load_visual_shapes=False)
                    joint_index = builder.joint_label.index("/World/Joint")
                    dof = builder.joint_qd_start[joint_index]
                    coord = builder.joint_q_start[joint_index]
                    model = builder.finalize(device="cpu")

                    angular = kind in ("revolute", "d6-rotX")
                    gain_scale = 180.0 / math.pi if angular else 1.0
                    target_scale = math.pi / 180.0 if angular else 1.0
                    self.assertAlmostEqual(float(model.joint_target_ke.numpy()[dof]), 8.0 * gain_scale, places=4)
                    self.assertAlmostEqual(float(model.joint_target_kd.numpy()[dof]), 2.0 * gain_scale, places=4)
                    self.assertAlmostEqual(float(model.joint_target_q.numpy()[coord]), 0.25 * target_scale, places=6)
                    self.assertAlmostEqual(float(model.joint_target_qd.numpy()[dof]), 0.5 * target_scale, places=6)
                    self.assertEqual(float(model.joint_effort_limit.numpy()[dof]), maximum)

    def test_undriven_joint_is_unchanged(self):
        """Leave an unauthored drive passive and finalize it on the CPU."""
        stage, _, _ = self.make_stage("prismatic", with_drive=False)
        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)
        joint_index = builder.joint_label.index("/World/Joint")
        dof = builder.joint_qd_start[joint_index]
        model = builder.finalize(device="cpu")
        self.assertEqual(float(model.joint_target_ke.numpy()[dof]), 0.0)
        self.assertEqual(float(model.joint_target_kd.numpy()[dof]), 0.0)
