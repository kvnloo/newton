# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU witness for cross-world Nesterov restart behavior in implicit MPM."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.implicit_mpm.rheology_solver_kernels import vec6
from newton._src.solvers.implicit_mpm.solve_rheology import (
    _NESTEROV_RESTART_INTERVAL,
    _nesterov_check_restart,
    _nesterov_extrapolate,
    _nesterov_reset,
)


class TestRheologyNesterovRestart(unittest.TestCase):
    def test_opposing_residuals_restart_other_world_cpu(self):
        """Witness a worsening world changing an improving world's velocity iterate."""
        device = wp.get_device("cpu")
        interval = _NESTEROV_RESTART_INTERVAL

        def extrapolated_velocity(world_zero_next_residual):
            state = wp.empty(2, dtype=float, device=device)
            wp.launch(_nesterov_reset, dim=1, inputs=[state], device=device)

            def check(world_zero, world_one):
                # One residual DOF per world: squared L2 and Linf norms coincide.
                residual = wp.array(
                    np.array([[world_zero, world_one], [world_zero, world_one]], dtype=np.float32),
                    dtype=float,
                    device=device,
                )
                wp.launch(
                    _nesterov_check_restart,
                    dim=1,
                    inputs=[residual, interval, state],
                    device=device,
                )

            check(4.0, 16.0)
            self.assertEqual(float(state.numpy()[0]), float(interval))
            check(world_zero_next_residual, 8.0)

            # One velocity/stress DOF per world, with no collider contacts.
            velocity = wp.array([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
            velocity_prev = wp.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
            stress = wp.zeros(2, dtype=vec6, device=device)
            stress_prev = wp.zeros(2, dtype=vec6, device=device)
            impulse = wp.empty(0, dtype=wp.vec3, device=device)
            impulse_prev = wp.empty(0, dtype=wp.vec3, device=device)
            friction = wp.empty(0, dtype=float, device=device)
            collider_velocity = wp.empty(0, dtype=wp.vec3, device=device)
            collider_velocity_prev = wp.empty(0, dtype=wp.vec3, device=device)

            wp.launch(
                _nesterov_extrapolate,
                dim=2,
                inputs=[
                    0,
                    1.0,
                    state,
                    stress,
                    stress_prev,
                    velocity,
                    velocity_prev,
                    impulse,
                    impulse_prev,
                    friction,
                    collider_velocity,
                    collider_velocity_prev,
                ],
                device=device,
            )
            return float(state.numpy()[0]), velocity.numpy()

        # World 1 improves identically (16 -> 8) in both runs.
        # World 0 worsens (4 -> 24), or stays flat (4 -> 4) as a control.
        restarted_k, restarted = extrapolated_velocity(24.0)
        continuing_k, continuing = extrapolated_velocity(4.0)

        self.assertEqual(restarted_k, 0.0)
        self.assertEqual(continuing_k, float(2 * interval))
        # Shared restart: beta=0 in both worlds; without restart beta=10/13.
        np.testing.assert_allclose(restarted, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            continuing,
            [[0.0, 0.0, 0.0], [2.0 + 2 * interval / (2 * interval + 3.0), 0.0, 0.0]],
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertGreater(float(continuing[1, 0] - restarted[1, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
