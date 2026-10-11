"""Independently check SemiImplicit D6 drive rest state on CPU for six axis orders."""
import sys

import numpy as np
import warp as wp

import newton


AXIS_PAIRS = (
    (newton.Axis.X, newton.Axis.Y),
    (newton.Axis.Y, newton.Axis.X),
    (newton.Axis.Y, newton.Axis.Z),
    (newton.Axis.Z, newton.Axis.Y),
    (newton.Axis.Z, newton.Axis.X),
    (newton.Axis.X, newton.Axis.Z),
)


def run_pair(axes):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    joint = builder.add_joint_d6(
        -1,
        body,
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=axes[0], target_pos=0.2, target_ke=2.0),
            newton.ModelBuilder.JointDofConfig(axis=axes[1], target_pos=0.4, target_ke=2.0),
        ],
    )
    builder.add_articulation([joint])
    builder.joint_q[:] = [0.2, 0.4]
    model = builder.finalize(device="cpu")
    solver = newton.solvers.SolverSemiImplicit(
        model, angular_damping=0.0, joint_attach_ke=0.0, joint_attach_kd=0.0
    )
    state, result = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    state.clear_forces()
    solver.step(state, result, model.control(), None, 0.001)
    newton.eval_ik(model, result, result.joint_q, result.joint_qd)
    return result.joint_qd.numpy().tolist()


def main():
    failures = 0
    for axes in AXIS_PAIRS:
        actual = run_pair(axes)
        passed = np.allclose(actual, [0.0, 0.0], rtol=0.0, atol=1.0e-8)
        print(f"{axes[0].name} {axes[1].name}: qd={actual} {'PASS' if passed else 'FAIL'}", flush=True)
        failures += not passed
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
