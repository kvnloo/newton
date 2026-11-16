# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for MuJoCo joint springdamper import."""

import unittest

import numpy as np

import newton
from newton.solvers import SolverMuJoCo


def springdamper_xml(joint_type="hinge", springdamper='springdamper="0.2 1.5"'):
    """Make a one-body mechanism with analytically known joint inertia."""
    return f"""<mujoco>
        <worldbody><body name="body">
            <joint name="joint" type="{joint_type}" {springdamper}/>
            <geom type="sphere" size="0.1" mass="2"/>
        </body></worldbody>
    </mujoco>"""


class TestMjcfJointSpringdamper(unittest.TestCase):
    def test_matches_native_mujoco_and_core_gains(self):
        """Resolve hinge, slide, and ball springs on both CPU backends."""
        mujoco, _ = SolverMuJoCo.import_mujoco()
        for joint_type in ("hinge", "slide", "ball"):
            xml = springdamper_xml(joint_type)
            native = mujoco.MjModel.from_xml_string(xml)
            # Solid sphere: rotational inertia = 2/5 * 2 * 0.1**2 = 0.008.
            inertia = 2.0 if joint_type == "slide" else 0.008
            expected_k = inertia / (0.2**2 * 1.5**2)
            expected_d = 2.0 * inertia / 0.2
            np.testing.assert_allclose(native.jnt_stiffness, [expected_k], rtol=1e-5)
            np.testing.assert_allclose(native.dof_damping, expected_d, rtol=1e-5)
            for native_cpu in (True, False):
                with self.subTest(joint_type=joint_type, native_cpu=native_cpu):
                    builder = newton.ModelBuilder()
                    builder.add_mjcf(xml)
                    model = builder.finalize(device="cpu")
                    solver = SolverMuJoCo(model, use_mujoco_cpu=native_cpu, disable_contacts=True)
                    np.testing.assert_allclose(model.joint_stiffness.numpy(), expected_k, rtol=1e-5)
                    np.testing.assert_allclose(model.joint_damping.numpy(), expected_d, rtol=1e-5)
                    np.testing.assert_allclose(solver.mj_model.jnt_stiffness, native.jnt_stiffness, rtol=1e-5)
                    np.testing.assert_allclose(solver.mj_model.dof_damping, native.dof_damping, rtol=1e-5)
                    np.testing.assert_allclose(model.mujoco.dof_springdamper.numpy(), 0.0)
                    if joint_type == "hinge" and native_cpu:
                        model.joint_damping.assign([0.3])
                        rebuilt = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                        np.testing.assert_allclose(model.joint_damping.numpy(), [0.3], rtol=1e-5)
                        np.testing.assert_allclose(rebuilt.mj_model.dof_damping, [0.3], rtol=1e-5)

    def test_inherited_and_disabled_springdamper(self):
        """Inherit time constants from defaults and preserve explicit gains if disabled."""
        inherited = springdamper_xml(springdamper="")
        inherited = inherited.replace("<worldbody>", '<default><joint springdamper="0.2 1.5"/></default><worldbody>')
        for xml, expected_k, expected_d in (
            (inherited, 0.008 / 0.09, 0.08),
            (springdamper_xml(springdamper='springdamper="0 0" stiffness="0.5" damping="0.7"'), 0.5, 0.7),
        ):
            with self.subTest(xml=xml):
                builder = newton.ModelBuilder()
                builder.add_mjcf(xml)
                model = builder.finalize(device="cpu")
                SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                np.testing.assert_allclose(model.joint_stiffness.numpy(), [expected_k], rtol=1e-5)
                np.testing.assert_allclose(model.joint_damping.numpy(), [expected_d], rtol=1e-5)

    def test_per_world_inertia_and_reconstruction(self):
        """Use each world's mass and inertia; do not recompute after an explicit edit."""
        template = newton.ModelBuilder()
        template.add_mjcf(springdamper_xml())
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize(device="cpu")
        body_per_world = model.body_count // model.world_count
        masses = model.body_mass.numpy()
        inertias = model.body_inertia.numpy()
        masses[body_per_world:] *= 2.0
        inertias[body_per_world:] *= 2.0
        model.body_mass.assign(masses)
        model.body_inertia.assign(inertias)
        SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True)
        np.testing.assert_allclose(model.joint_stiffness.numpy(), [0.008 / 0.09, 0.016 / 0.09], rtol=1e-5)
        np.testing.assert_allclose(model.joint_damping.numpy(), [0.08, 0.16], rtol=1e-5)
        np.testing.assert_allclose(model.mujoco.dof_springdamper.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
