# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject NaN in active MuJoCo USD equalities while preserving inactive and finite data."""

import math
import unittest

import newton
from newton.solvers import SolverMuJoCo


class TestImportUsdMjcEqualityNan(unittest.TestCase):
    """Use real in-memory USD scenes and CPU-finalized equality constraint arrays."""

    @staticmethod
    def make_stage(kind):
        """Create a joint coupling, connect, or weld with valid body references."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")

        if kind == "joint":
            articulation = UsdGeom.Xform.Define(stage, "/World/Articulation")
            UsdPhysics.ArticulationRootAPI.Apply(articulation.GetPrim())
            links = []
            for name in ("Root", "Link1", "Link2"):
                link = UsdGeom.Xform.Define(stage, f"/World/Articulation/{name}")
                UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
                links.append(link)

            fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Articulation/RootToWorld")
            fixed.CreateBody0Rel().SetTargets([links[0].GetPath()])
            fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

            for index in (1, 2):
                joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/Articulation/Joint{index}")
                joint.CreateBody0Rel().SetTargets([links[index - 1].GetPath()])
                joint.CreateBody1Rel().SetTargets([links[index].GetPath()])
                joint.CreateAxisAttr().Set("Z")
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

            follower = stage.GetPrimAtPath("/World/Articulation/Joint2")
            follower.SetMetadata(
                "apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityJointAPI", "NewtonMimicAPI"])
            )
            follower.CreateRelationship("newton:mimicJoint").SetTargets(["/World/Articulation/Joint1"])
            return stage, follower

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        bodies = []
        for name in ("Body0", "Body1"):
            body = UsdGeom.Cube.Define(stage, f"/World/{name}")
            body.CreateSizeAttr(0.2)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
            bodies.append(body)

        path = "/World/EqualityConnect" if kind == "connect" else "/World/EqualityWeld"
        if kind == "connect":
            joint = UsdPhysics.SphericalJoint.Define(stage, path)
            api = "MjcEqualityConnectAPI"
        elif kind == "weld":
            joint = UsdPhysics.FixedJoint.Define(stage, path)
            api = "MjcEqualityWeldAPI"
        else:
            raise ValueError(f"Unknown equality kind: {kind}")

        joint.CreateBody0Rel().SetTargets([bodies[0].GetPath()])
        joint.CreateBody1Rel().SetTargets([bodies[1].GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.25, -0.1, 0.3))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.1, 0.2, 0.3))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateExcludeFromArticulationAttr().Set(True)
        prim = joint.GetPrim()
        prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[api]))
        return stage, prim

    @staticmethod
    def author(prim, attr_name, value):
        """Author scalar or vector numeric attributes with the same USD types as existing tests."""
        from pxr import Sdf

        if isinstance(value, (list, tuple)):
            dtype = Sdf.ValueTypeNames.DoubleArray
        elif attr_name in ("newton:mimicCoef0", "newton:mimicCoef1", "mjc:torqueScale"):
            dtype = Sdf.ValueTypeNames.Float
        else:
            dtype = Sdf.ValueTypeNames.Double
        prim.CreateAttribute(attr_name, dtype, custom=True).Set(value)

    @staticmethod
    def import_cpu(stage, prim, *, converted=False, include_disabled=False):
        """Import MuJoCo equality properties and finalize the real CPU model."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(
            stage,
            load_visual_shapes=False,
            convert_mjc_equality_constraints=converted,
            only_load_enabled_joints=not include_disabled,
        )
        model = builder.finalize(device="cpu")
        index = model.mujoco.equality_constraint_label.index(str(prim.GetPath()))
        return model, index

    def test_enabled_equality_rejects_nan_solver_response_components(self):
        """Each equality kind must reject NaN in either solref or solimp vector."""
        for kind in ("joint", "connect", "weld"):
            for attr_name, values in (
                ("mjc:solref", [0.03, 0.8]),
                ("mjc:solimp", [0.8, 0.9, 0.002, 0.6, 3.0]),
            ):
                for component in (0, len(values) - 1):
                    with self.subTest(kind=kind, field=attr_name, component=component):
                        stage, prim = self.make_stage(kind)
                        invalid = values.copy()
                        invalid[component] = math.nan
                        self.author(prim, attr_name, invalid)
                        with self.assertRaisesRegex(ValueError, rf"NaN in {attr_name}"):
                            self.import_cpu(stage, prim)

    def test_enabled_joint_rejects_nan_in_all_five_polynomial_terms(self):
        """Both Newton and legacy offsets/slopes plus MuJoCo higher terms are checked."""
        for attr_name in (
            "newton:mimicCoef0",
            "newton:mimicCoef1",
            "mjc:coef0",
            "mjc:coef1",
            "mjc:coef2",
            "mjc:coef3",
            "mjc:coef4",
        ):
            with self.subTest(field=attr_name):
                stage, prim = self.make_stage("joint")
                self.author(prim, attr_name, math.nan)
                with self.assertRaisesRegex(ValueError, rf"NaN in {attr_name}"):
                    self.import_cpu(stage, prim)

    def test_enabled_weld_and_connect_reject_nan_in_pose_and_torque(self):
        """Derived anchors and weld relative poses cannot retain authored NaN."""
        from pxr import Gf

        for kind, field, label in (
            ("connect", "physics:localPos0", "equality anchor"),
            ("weld", "physics:localPos1", "equality anchor"),
            ("weld", "physics:localPos0", "equality relative position"),
            ("weld", "physics:localRot0", "equality relative orientation"),
        ):
            with self.subTest(kind=kind, field=field):
                stage, prim = self.make_stage(kind)
                invalid = (
                    Gf.Quatf(1.0, math.nan, 0.0, 0.0)
                    if field == "physics:localRot0"
                    else Gf.Vec3f(math.nan, 0.0, 0.0)
                )
                prim.GetAttribute(field).Set(invalid)
                with self.assertRaisesRegex(ValueError, rf"NaN in {label}"):
                    self.import_cpu(stage, prim)

        stage, prim = self.make_stage("weld")
        self.author(prim, "mjc:torqueScale", math.nan)
        with self.assertRaisesRegex(ValueError, r"NaN in mjc:torqueScale"):
            self.import_cpu(stage, prim)

    def test_finite_equality_values_survive_cpu_import(self):
        """Preserve joint angle conversion, weld scaling, anchors, and native responses."""
        for kind in ("joint", "connect", "weld"):
            with self.subTest(kind=kind):
                stage, prim = self.make_stage(kind)
                self.author(prim, "mjc:solref", [0.03, 0.8])
                self.author(prim, "mjc:solimp", [0.8, 0.9, 0.002, 0.6, 3.0])
                if kind == "joint":
                    self.author(prim, "newton:mimicCoef0", 90.0)
                    self.author(prim, "newton:mimicCoef1", 1.5)
                    self.author(prim, "mjc:coef2", 0.125)
                elif kind == "weld":
                    self.author(prim, "mjc:torqueScale", 2.5)

                model, index = self.import_cpu(stage, prim)
                self.assertTrue(bool(model.mujoco.equality_constraint_enabled.numpy()[index]))
                for actual, expected in zip(model.mujoco.eq_solref.numpy()[index], (0.03, 0.8), strict=True):
                    self.assertAlmostEqual(float(actual), expected, places=5)
                for actual, expected in zip(
                    model.mujoco.eq_solimp.numpy()[index],
                    (0.8, 0.9, 0.002, 0.6, 3.0),
                    strict=True,
                ):
                    self.assertAlmostEqual(float(actual), expected, places=5)

                if kind == "joint":
                    values = model.mujoco.equality_constraint_polycoef.numpy()[index]
                    for actual, expected in zip(values, (math.pi / 2, 1.5, 0.125, 0.0, 0.0), strict=True):
                        self.assertAlmostEqual(float(actual), expected, places=5)
                else:
                    anchor = model.mujoco.equality_constraint_anchor.numpy()[index]
                    expected_anchor = (0.25, -0.1, 0.3) if kind == "connect" else (0.1, 0.2, 0.3)
                    for actual, expected in zip(anchor, expected_anchor, strict=True):
                        self.assertAlmostEqual(float(actual), expected, places=5)
                    if kind == "weld":
                        torquescale = model.mujoco.equality_constraint_torquescale.numpy()[index]
                        self.assertAlmostEqual(float(torquescale), 2.5, places=5)

    def test_disabled_equality_does_not_reject_unused_nan(self):
        """A disabled equality row retains the existing inactive-data policy."""
        from pxr import Sdf

        for kind in ("connect", "weld", "joint"):
            with self.subTest(kind=kind):
                stage, prim = self.make_stage(kind)
                self.author(prim, "mjc:solref", [math.nan, 0.8])
                if kind == "joint":
                    prim.CreateAttribute("newton:mimicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                    self.author(prim, "mjc:coef2", math.nan)
                else:
                    prim.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                    if kind == "weld":
                        self.author(prim, "mjc:torqueScale", math.nan)
                model, index = self.import_cpu(stage, prim, include_disabled=(kind != "joint"))
                self.assertFalse(bool(model.mujoco.equality_constraint_enabled.numpy()[index]))

    def test_current_mimic_properties_override_unused_legacy_nan(self):
        """Do not reject a lower-priority deprecated alias that is not consumed."""
        stage, prim = self.make_stage("joint")
        self.author(prim, "newton:mimicCoef0", 90.0)
        self.author(prim, "newton:mimicCoef1", 1.5)
        self.author(prim, "mjc:coef0", math.nan)
        self.author(prim, "mjc:coef1", math.nan)
        model, index = self.import_cpu(stage, prim, converted=True)
        polycoef = model.mujoco.equality_constraint_polycoef.numpy()[index]
        self.assertAlmostEqual(float(polycoef[0]), math.pi / 2.0, places=5)
        self.assertAlmostEqual(float(polycoef[1]), 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
