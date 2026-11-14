# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU regression cases for USD equality anchors under scale and shear."""

import math
import unittest

import numpy as np

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE


class TestUsdEqualityAnchorFrame(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_precision_and_imported_body_frame(self):
        """Resolve authored site positions in the rigid body's imported pose."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        cases = (
            ("far_origin", 128.0, (1.0, 1.0, 1.0), 0.0, 0.013579135),
            ("milliscale", 0.0, (1.0e-3, 1.0e-3, 1.0e-3), 0.0, 0.01),
            ("tiny_scale", 0.0, (1.0e-4, 1.0e-4, 1.0e-4), 0.0, 0.01),
            ("sheared_rotation", 0.0, (2.0, 1.0, 1.0), 30.0, 0.02),
        )
        for label, world_x, scale, angle, offset in cases:
            with self.subTest(case=label):
                stage = Usd.Stage.CreateInMemory()
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                UsdGeom.SetStageMetersPerUnit(stage, 1.0)
                UsdPhysics.Scene.Define(stage, "/physicsScene")
                root = UsdGeom.Xform.Define(stage, "/World")
                stage.SetDefaultPrim(root.GetPrim())
                root.AddTranslateOp().Set(Gf.Vec3d(world_x, 0.0, 0.0))
                root.AddScaleOp().Set(Gf.Vec3d(*scale))

                body = UsdGeom.Xform.Define(stage, "/World/Body")
                body.AddRotateZOp().Set(angle)
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
                mass.CreateMassAttr(1.0)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
                site = UsdGeom.Xform.Define(stage, "/World/Body/Site")
                site.AddTranslateOp().Set(Gf.Vec3d(offset, 0.0, 0.0))
                site.GetPrim().SetMetadata(
                    "apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcSiteAPI"])
                )
                connect = UsdPhysics.SphericalJoint.Define(stage, "/World/Connect")
                connect.CreateBody0Rel().SetTargets([site.GetPath()])
                connect.CreateExcludeFromArticulationAttr(True)
                connect.GetPrim().SetMetadata(
                    "apiSchemas",
                    Sdf.TokenListOp.Create(prependedItems=["MjcEqualityConnectAPI"]),
                )

                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                result = builder.add_usd(stage, convert_mjc_equality_constraints=False)
                model = builder.finalize(device="cpu")
                self.assertEqual(model.mujoco.equality_constraint_count, 1)
                body_index = result["path_body_map"]["/World/Body"]
                self.assertEqual(int(model.mujoco.equality_constraint_body1.numpy()[0]), body_index)
                anchor = model.mujoco.equality_constraint_anchor.numpy()[0]

                # Unrotated cases have an exact, scale-adjusted local oracle.
                if angle == 0.0:
                    np.testing.assert_array_equal(
                        anchor,
                        np.asarray([scale[0] * offset, 0.0, 0.0], dtype=np.float32),
                    )

                # Independently rotate the imported body pose's local anchor.
                # This catches mismatches between Gf and Warp shear decompositions.
                pose = np.asarray(builder.body_q[body_index], dtype=np.float64).reshape(7)
                quat_xyz = pose[3:6]
                local = np.asarray(anchor, dtype=np.float64)
                world_delta = (
                    local
                    + 2.0 * pose[6] * np.cross(quat_xyz, local)
                    + 2.0 * np.cross(quat_xyz, np.cross(quat_xyz, local))
                )
                radians = math.radians(angle)
                authored_delta = np.asarray(
                    [
                        scale[0] * offset * math.cos(radians),
                        scale[1] * offset * math.sin(radians),
                        0.0,
                    ],
                    dtype=np.float64,
                )
                np.testing.assert_allclose(world_delta, authored_delta, rtol=2.0e-6, atol=2.0e-7)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_sheared_matrix_anchor_oracle(self):
        """Match double-precision shear offsets in the imported body frame."""
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        angle = math.radians(30.0)
        sine, cosine = math.sin(angle), math.cos(angle)
        # USD's parent scale precedes the child's rotation: M = S @ Rz.
        parent_scale = np.diag([2.0, 1.0, 1.0])
        authored_rotation = np.asarray(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cases = (
            (
                "mixed_sign_offset",
                (0.0, 0.0, 0.0),
                (0.02, -0.03, 0.04),
                (0.06464101615137755, -0.01598076211353316, 0.04),
            ),
            (
                "far_origin_oblique_offset",
                (128.0, -32.0, 16.0),
                (-0.01, 0.04, -0.02),
                (-0.05732050807568877, 0.02964101615137754, -0.02),
            ),
        )
        for label, root_translation, site_offset, expected_world in cases:
            with self.subTest(case=label):
                stage = Usd.Stage.CreateInMemory()
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                UsdGeom.SetStageMetersPerUnit(stage, 1.0)
                UsdPhysics.Scene.Define(stage, "/physicsScene")
                root = UsdGeom.Xform.Define(stage, "/World")
                stage.SetDefaultPrim(root.GetPrim())
                root.AddTranslateOp().Set(Gf.Vec3d(*root_translation))
                root.AddScaleOp().Set(Gf.Vec3d(2.0, 1.0, 1.0))

                body = UsdGeom.Xform.Define(stage, "/World/Body")
                body.AddRotateZOp().Set(30.0)
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
                mass.CreateMassAttr(1.0)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
                site = UsdGeom.Xform.Define(stage, "/World/Body/Site")
                site.AddTranslateOp().Set(Gf.Vec3d(*site_offset))
                site.GetPrim().SetMetadata(
                    "apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcSiteAPI"])
                )
                connect = UsdPhysics.SphericalJoint.Define(stage, "/World/Connect")
                connect.CreateBody0Rel().SetTargets([site.GetPath()])
                connect.CreateExcludeFromArticulationAttr(True)
                connect.GetPrim().SetMetadata(
                    "apiSchemas",
                    Sdf.TokenListOp.Create(prependedItems=["MjcEqualityConnectAPI"]),
                )

                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                result = builder.add_usd(stage, convert_mjc_equality_constraints=False)
                model = builder.finalize(device="cpu")
                self.assertEqual(model.mujoco.equality_constraint_count, 1)
                body_index = result["path_body_map"]["/World/Body"]
                self.assertEqual(int(model.mujoco.equality_constraint_body1.numpy()[0]), body_index)
                actual_local = model.mujoco.equality_constraint_anchor.numpy()[0]

                # Independent, authored double-precision world displacement:
                # no Gf.RemoveScaleShear or USD importer transform helper.
                world_delta = parent_scale @ authored_rotation @ np.asarray(site_offset, dtype=np.float64)
                np.testing.assert_allclose(world_delta, expected_world, rtol=0.0, atol=1.0e-14)

                # The builder's actual quaternion defines the rigid frame in
                # which the equality anchor must be stored. Form its rotation
                # matrix independently of the importer's Gf rotation path.
                pose = np.asarray(builder.body_q[body_index], dtype=np.float64).reshape(7)
                x, y, z, w = pose[3:]
                np.testing.assert_allclose(x * x + y * y + z * z + w * w, 1.0, rtol=0.0, atol=2.0e-6)
                imported_rotation = np.asarray(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ],
                    dtype=np.float64,
                )
                np.testing.assert_allclose(
                    imported_rotation.T @ imported_rotation, np.eye(3), rtol=0.0, atol=3.0e-6
                )
                expected_local = imported_rotation.T @ world_delta
                np.testing.assert_allclose(
                    actual_local, expected_local.astype(np.float32), rtol=2.0e-6, atol=2.0e-7
                )
