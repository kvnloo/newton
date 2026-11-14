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
