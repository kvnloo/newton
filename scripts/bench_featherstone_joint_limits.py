#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare explicit and semi-implicit Featherstone joint-limit behavior."""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

import newton


def build_serial_arm(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    inertia = wp.mat33(
        [[1.0e-5, 0.0, 0.0], [0.0, 1.0e-5, 0.0], [0.0, 0.0, 1.0e-5]]
    )
    parent = -1
    joints = []

    for index in range(6):
        x = float(index) * 0.1
        body = builder.add_link(
            xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
            com=wp.vec3(0.05, 0.0, 0.0),
            mass=0.05,
            inertia=inertia,
        )
        joint = builder.add_joint_revolute(
            parent=parent,
            child=body,
            axis=(0.0, 1.0, 0.0),
            parent_xform=(
                wp.transform_identity()
                if parent < 0
                else wp.transform((0.1, 0.0, 0.0), wp.quat_identity())
            ),
            target_ke=0.0,
            target_kd=0.0,
            damping=0.0,
            limit_lower=-0.5,
            limit_upper=0.5,
            limit_ke=10000.0,
            limit_kd=10.0,
            armature=0.0,
        )
        joints.append(joint)
        parent = body

    builder.add_articulation(joints)
    return builder.finalize(device=device)


def run(model, implicit, steps, dt):
    solver = newton.solvers.SolverFeatherstone(
        model,
        angular_damping=0.0,
        implicit_joint_limits=implicit,
    )
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    control = model.control()

    rows = []
    for step in range(steps):
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
        q = state_in.joint_q.numpy()
        qd = state_in.joint_qd.numpy()
        row = {
            "step": step,
            "time_s": (step + 1) * dt,
            "q_max_abs": float(np.max(np.abs(q))),
            "qd_max_abs": float(np.max(np.abs(qd))),
            "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
            "q0": float(q[0]),
            "qd0": float(qd[0]),
        }
        rows.append(row)
        if not row["finite"] or row["q_max_abs"] > 1.0e3:
            break

    return {
        "implicit_joint_limits": implicit,
        "rows": rows,
        "final": rows[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    model = build_serial_arm(device)

    explicit = run(model, False, args.steps, args.dt)
    implicit = run(model, True, args.steps, args.dt)

    payload = {
        "device": {
            "alias": device.alias,
            "name": device.name,
            "arch": str(device.arch),
        },
        "dt": args.dt,
        "steps_requested": args.steps,
        "explicit": explicit,
        "implicit": implicit,
    }
    raw = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(raw + "\n")
    print(raw)


if __name__ == "__main__":
    main()
