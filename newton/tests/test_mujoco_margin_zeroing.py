# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for SolverMuJoCo contact margin preservation (#2106, #5377).

``SolverMuJoCo`` forwards authored geom and contact-pair margins unchanged on
both the MuJoCo CPU and MuJoCo Warp backends across NATIVECCD and MULTICCD
configurations, and updates them on ``ModelFlags.SHAPE_PROPERTIES``.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton._src.core.types import vec5
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import MUJOCO_MODULES, requires


@requires(*MUJOCO_MODULES)
class TestMuJoCoMarginZeroing(unittest.TestCase):
    """Verify SolverMuJoCo preserves geom and pair margins on CPU and MuJoCo Warp."""

    @staticmethod
    def _build_model_with_margin(
        margin: float = 1e-5, *, nativeccd: bool = True, device: str | None = None
    ) -> newton.Model:
        """Build a minimal model with two boxes that have non-zero margin."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.custom_attributes["mujoco:flag_nativeccd"].values[0] = nativeccd
        builder.add_shape_box(
            body=-1,
            hx=1.0,
            hy=1.0,
            hz=0.01,
            cfg=newton.ModelBuilder.ShapeConfig(margin=margin),
        )
        b = builder.add_body(label="box")
        builder.add_shape_box(
            body=b,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(margin=margin),
        )
        return builder.finalize(device=device)

    def test_geom_margin_preserved_throughout_lifecycle(self):
        """Preserve and update geom margins under NATIVECCD and MULTICCD without warnings."""
        for nativeccd, multiccd in ((True, False), (False, False), (False, True), (True, True)):
            with self.subTest(nativeccd=nativeccd, multiccd=multiccd):
                model = self._build_model_with_margin(margin=0.01, nativeccd=nativeccd)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    solver = SolverMuJoCo(model, enable_multiccd=multiccd)
                np.testing.assert_allclose(solver.mj_model.geom_margin, 0.01)
                np.testing.assert_allclose(solver.mjw_model.geom_margin.numpy(), 0.01)
                model.shape_margin.assign(wp.array([0.015, 0.02], dtype=wp.float32, device=model.device))
                solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
                np.testing.assert_allclose(solver.mjw_model.geom_margin.numpy(), [[0.015, 0.02]], atol=1e-7)

    def test_backends_keep_margins_like_native_mujoco(self):
        """Both CPU and MuJoCo Warp backends keep authored margins and generate contacts."""
        mujoco, mjw = SolverMuJoCo.import_mujoco()
        mesh = '<mesh name="tetra" vertex="-0.1 -0.1 0  0.1 -0.1 0  0 0.1 0  0 0 0.1"/>'
        devices = ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]
        backends = [("cpu", True, "cpu"), *[("warp", False, dev) for dev in devices]]
        for backend_name, use_cpu, device in backends:
            for multiccd in ("enable", "disable"):
                with self.subTest(backend=backend_name, device=device, multiccd=multiccd):
                    mjcf = f"""<mujoco><option><flag multiccd="{multiccd}"/></option><asset>{mesh}</asset>
                      <worldbody><geom name="floor" type="box" size="1 1 0.01" margin="0.004"/>
                        <body name="box" pos="0 0 0.062"><freejoint/>
                          <geom name="box" type="box" size="0.05 0.05 0.05" margin="0.003"/></body>
                        <body name="mesh" pos="0.5 0 0.012"><freejoint/>
                          <geom name="mesh" type="mesh" mesh="tetra" margin="0.002"/></body>
                      </worldbody>
                      <contact><pair geom1="floor" geom2="mesh" margin="0.005"/></contact></mujoco>"""
                    native = mujoco.MjModel.from_xml_string(mjcf)
                    builder = newton.ModelBuilder()
                    builder.add_mjcf(mjcf)
                    model = builder.finalize(device=device)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        solver = SolverMuJoCo(
                            model,
                            use_mujoco_cpu=use_cpu,
                            enable_multiccd=(multiccd == "enable"),
                        )
                    self.assertEqual(solver.mj_model.opt.disableflags, native.opt.disableflags)
                    geom_names = [
                        model.shape_label[shape].rsplit("/", 1)[-1]
                        for shape in solver.mjc_geom_to_newton_shape.numpy()[0]
                    ]
                    native_margins = [native.geom(name).margin[0] for name in geom_names]
                    np.testing.assert_allclose(solver.mj_model.geom_margin, native_margins, atol=1e-7)
                    np.testing.assert_allclose(solver.mjw_model.geom_margin.numpy()[0], native_margins, atol=1e-7)
                    np.testing.assert_allclose(solver.mj_model.pair_margin, native.pair_margin, atol=1e-7)
                    np.testing.assert_allclose(solver.mjw_model.pair_margin.numpy()[0], native.pair_margin, atol=1e-7)
                    expected = mujoco.MjData(native)
                    mujoco.mj_forward(native, expected)
                    expected_active = expected.contact.exclude[: expected.ncon] == 0
                    expected_dists = np.sort(expected.contact.dist[: expected.ncon][expected_active])
                    expected_margins = np.sort(expected.contact.includemargin[: expected.ncon][expected_active])
                    self.assertGreater(len(expected_dists), 0)
                    if use_cpu:
                        actual = mujoco.MjData(solver.mj_model)
                        mujoco.mj_forward(solver.mj_model, actual)
                        actual_active = actual.contact.exclude[: actual.ncon] == 0
                        actual_dists = np.sort(actual.contact.dist[: actual.ncon][actual_active])
                        actual_margins = np.sort(actual.contact.includemargin[: actual.ncon][actual_active])
                    else:
                        with wp.ScopedDevice(device):
                            mjw.forward(solver.mjw_model, solver.mjw_data)
                        nacon = int(solver.mjw_data.nacon.numpy()[0])
                        actual_dists = np.sort(solver.mjw_data.contact.dist.numpy()[:nacon])
                        actual_margins = np.sort(solver.mjw_data.contact.includemargin.numpy()[:nacon])
                    self.assertEqual(len(actual_dists), len(expected_dists))
                    np.testing.assert_allclose(actual_dists, expected_dists, atol=1e-6)
                    np.testing.assert_allclose(actual_margins, expected_margins, atol=1e-6)


    def test_skipped_pair_source_row_cpu_warp(self):
        """Copy the valid source row for every runtime pair property on CPU Warp."""
        skipped_row = {
            "pair_solref": (0.06, 1.1),
            "pair_solreffriction": (0.07, 1.2),
            "pair_solimp": (0.85, 0.93, 0.002, 0.5, 2.0),
            "pair_margin": 0.01,
            "pair_gap": 0.001,
            "pair_friction": (1.0, 1.0, 0.005, 0.0001, 0.0001),
        }
        valid_row = {
            "pair_solref": (0.04, 1.3),
            "pair_solreffriction": (0.05, 1.4),
            "pair_solimp": (0.88, 0.96, 0.003, 0.6, 2.0),
            "pair_margin": 0.05,
            "pair_gap": 0.004,
            "pair_friction": (1.2, 1.2, 0.007, 0.0002, 0.0002),
        }
        updated_skipped = {
            "pair_solref": (0.09, 1.8),
            "pair_solreffriction": (0.08, 1.7),
            "pair_solimp": (0.84, 0.92, 0.004, 0.5, 2.0),
            "pair_margin": 0.02,
            "pair_gap": 0.002,
            "pair_friction": (1.1, 1.1, 0.006, 0.0003, 0.0003),
        }
        updated_valid = {
            "pair_solref": (0.03, 1.2),
            "pair_solreffriction": (0.035, 1.25),
            "pair_solimp": (0.89, 0.97, 0.005, 0.7, 2.5),
            "pair_margin": 0.06,
            "pair_gap": 0.006,
            "pair_friction": (1.4, 1.3, 0.008, 0.0004, 0.0004),
        }
        dtypes = {
            "pair_solref": wp.vec2,
            "pair_solreffriction": wp.vec2,
            "pair_solimp": vec5,
            "pair_margin": wp.float32,
            "pair_gap": wp.float32,
            "pair_friction": vec5,
        }

        # The no-skip variant protects the ordinary identity mapping.
        for skipped in (True, False):
            with self.subTest(skipped=skipped):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                floor = builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=0.01)
                body = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
                box = builder.add_shape_box(body=body, hx=0.05, hy=0.05, hz=0.05)
                joint = builder.add_joint_free(body)
                builder.add_articulation([joint])

                rows = [skipped_row, valid_row] if skipped else [valid_row]
                for index, properties in enumerate(rows):
                    attributes = {
                        "mujoco:pair_world": 0,
                        "mujoco:pair_geom1": -1 if skipped and index == 0 else floor,
                        "mujoco:pair_geom2": box,
                        "mujoco:pair_condim": 3,
                    }
                    for name, value in properties.items():
                        attributes[f"mujoco:{name}"] = dtypes[name](*value) if isinstance(value, tuple) else value
                    builder.add_custom_values(**attributes)

                model = builder.finalize(device="cpu")
                solver = SolverMuJoCo(model, use_mujoco_cpu=False, enable_multiccd=True)
                self.assertEqual(solver.mj_model.npair, 1)
                self.assertEqual(model.custom_frequency_counts["mujoco:pair"], len(rows))

                for name, expected in valid_row.items():
                    np.testing.assert_allclose(
                        getattr(solver.mjw_model, name).numpy()[0, 0],
                        expected,
                        rtol=0,
                        atol=1e-7,
                        err_msg=f"Initial {name}, skipped={skipped}",
                    )

                changed_rows = [updated_skipped, updated_valid] if skipped else [updated_valid]
                for name, dtype in dtypes.items():
                    values = np.asarray([row[name] for row in changed_rows], dtype=np.float32)
                    getattr(model.mujoco, name).assign(wp.array(values, dtype=dtype, device=model.device))

                solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
                for name, expected in updated_valid.items():
                    np.testing.assert_allclose(
                        getattr(solver.mjw_model, name).numpy()[0, 0],
                        expected,
                        rtol=0,
                        atol=1e-7,
                        err_msg=f"Updated {name}, skipped={skipped}",
                    )

if __name__ == "__main__":
    unittest.main()
