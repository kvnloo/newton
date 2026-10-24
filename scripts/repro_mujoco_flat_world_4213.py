#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current-main reproducer for flat multi-articulation SolverMuJoCo layouts (#4213)."""

from __future__ import annotations

import argparse
import json
import traceback

import numpy as np
import warp as wp

import newton


def _add_env(builder: newton.ModelBuilder, x: float, label: str) -> None:
    inertia = wp.mat33([[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02]])

    root = builder.add_link(
        xform=wp.transform((x, 0.0, 0.55), wp.quat_identity()),
        com=wp.vec3(),
        mass=1.0,
        inertia=inertia,
        label=f"{label}/root",
    )
    builder.add_shape_box(root, hx=0.25, hy=0.2, hz=0.2, label=f"{label}/root_box")

    child = builder.add_link(
        xform=wp.transform((x + 0.45, 0.0, 0.55), wp.quat_identity()),
        com=wp.vec3(),
        mass=0.5,
        inertia=inertia,
        label=f"{label}/child",
    )
    builder.add_shape_box(child, hx=0.2, hy=0.15, hz=0.15, label=f"{label}/child_box")

    free = builder.add_joint_free(parent=-1, child=root, label=f"{label}/free")
    hinge = builder.add_joint_revolute(
        parent=root,
        child=child,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform((0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform((-0.2, 0.0, 0.0), wp.quat_identity()),
        target_ke=25.0,
        target_kd=2.0,
        limit_lower=-1.2,
        limit_upper=1.2,
        label=f"{label}/hinge",
    )
    builder.add_articulation([free, hinge], label=label)


def _build_flat(num_envs: int, spacing: float, device) -> newton.Model:
    builder = newton.ModelBuilder()
    for env in range(num_envs):
        _add_env(builder, env * spacing, f"env_{env}")
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _build_replicated(num_envs: int, spacing: float, device) -> newton.Model:
    prototype = newton.ModelBuilder()
    _add_env(prototype, 0.0, "robot")
    prototype.add_ground_plane()

    scene = newton.ModelBuilder()
    scene.replicate(prototype, world_count=num_envs, spacing=(spacing, 0.0, 0.0))
    return scene.finalize(device=device)


def _array_summary(value):
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    result = {"shape": None if shape is None else list(shape)}
    if hasattr(value, "numpy"):
        try:
            host = np.asarray(value.numpy())
            if host.size <= 32:
                result["values"] = host.tolist()
            else:
                result["min"] = float(np.min(host))
                result["max"] = float(np.max(host))
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic script
            result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _solver_receipt(solver):
    m = solver.mjw_model
    d = solver.mjw_data
    return {
        "m": {
            "nv": int(m.nv),
            "nq": int(m.nq),
            "nbody": int(m.nbody),
            "ntree": int(m.ntree),
            "njnt": int(m.njnt),
            "ngeom": int(m.ngeom),
        },
        "d": {
            "nworld": int(d.nworld),
            "nconmax": int(d.nconmax),
            "naconmax": int(d.naconmax),
            "njmax": int(d.njmax),
            "njmax_nnz": int(getattr(d, "njmax_nnz", 0)),
            "ncon": _array_summary(getattr(d, "ncon", None)),
            "nacon": _array_summary(getattr(d, "nacon", None)),
            "nefc": _array_summary(getattr(d, "nefc", None)),
            "qfrc_constraint": _array_summary(getattr(d, "qfrc_constraint", None)),
            "efc_J": _array_summary(getattr(d, "efc_J", None)),
            "efc_J_rownnz": _array_summary(getattr(d, "efc_J_rownnz", None)),
        },
    }


def _run_case(name: str, model: newton.Model, separate_worlds: bool, steps: int, dt: float):
    solver = newton.solvers.SolverMuJoCo(
        model,
        separate_worlds=separate_worlds,
        use_mujoco_contacts=True,
        update_data_interval=0,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    result = {
        "name": name,
        "model": {
            "world_count": int(model.world_count),
            "articulation_count": int(model.articulation_count),
            "body_count": int(model.body_count),
            "joint_count": int(model.joint_count),
            "dof_count": int(model.joint_dof_count),
        },
        "before": _solver_receipt(solver),
        "steps": [],
        "status": "ok",
    }

    for step in range(steps):
        try:
            solver.step(state_in, state_out, control, None, dt)
            wp.synchronize_device(model.device)
            state_in, state_out = state_out, state_in

            q = state_in.joint_q.numpy()
            qd = state_in.joint_qd.numpy()
            result["steps"].append(
                {
                    "step": step,
                    "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
                    "q_max_abs": float(np.max(np.abs(q), initial=0.0)),
                    "qd_max_abs": float(np.max(np.abs(qd), initial=0.0)),
                    "layout": _solver_receipt(solver),
                }
            )
        except Exception as exc:  # noqa: BLE001 - retain the exact first failing step
            result["status"] = "error"
            result["error_step"] = step
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
            try:
                result["at_error"] = _solver_receipt(solver)
            except Exception as receipt_exc:  # noqa: BLE001
                result["receipt_error"] = f"{type(receipt_exc).__name__}: {receipt_exc}"
            break

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)

    flat = _build_flat(args.envs, args.spacing, device)
    replicated = _build_replicated(args.envs, args.spacing, device)

    payload = {
        "issue": "https://github.com/newton-physics/newton/issues/4213",
        "device": {
            "alias": device.alias,
            "name": device.name,
            "arch": str(device.arch),
        },
        "config": {
            "envs": args.envs,
            "spacing": args.spacing,
            "steps": args.steps,
            "dt": args.dt,
        },
        "flat_single_world": _run_case("flat_single_world", flat, False, args.steps, args.dt),
        "replicated_worlds": _run_case("replicated_worlds", replicated, True, args.steps, args.dt),
    }

    raw = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(raw + "\n")
    print(raw)


if __name__ == "__main__":
    main()
