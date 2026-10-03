#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark integrated Featherstone serial-vs-auto dense GEMM dispatch."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import warp as wp

import newton


def _percentile(values, fraction):
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def _build_model(articulations, dofs, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]])

    for articulation in range(articulations):
        parent = -1
        joints = []
        for index in range(dofs):
            x = float(articulation) * 2.0 + float(index) * 0.1
            body = builder.add_link(
                xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
                com=wp.vec3(0.05, 0.0, 0.0),
                mass=1.0,
                inertia=inertia,
            )
            joint = builder.add_joint_revolute(
                parent=parent,
                child=body,
                axis=(0.0, 1.0, 0.0),
                parent_xform=(
                    wp.transform((float(articulation) * 2.0, 0.0, 0.0), wp.quat_identity())
                    if parent < 0
                    else wp.transform((0.1, 0.0, 0.0), wp.quat_identity())
                ),
                damping=0.0,
            )
            joints.append(joint)
            parent = body
        builder.add_articulation(joints)

    return builder.finalize(device=device)


def _time_case(model, mode, warmup, samples):
    solver = newton.solvers.SolverFeatherstone(
        model,
        angular_damping=0.0,
        update_mass_matrix_interval=1,
        use_tile_gemm=False,
    )
    if mode == "serial":
        solver._use_elementwise_P = False
        solver._use_elementwise_H = False

    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    control = model.control()
    dt = 1.0 / 600.0

    def step():
        nonlocal state_in, state_out
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in

    for _ in range(warmup):
        step()
    wp.synchronize_device(model.device)

    timings = []
    for _ in range(samples):
        wp.synchronize_device(model.device)
        start = time.perf_counter()
        step()
        wp.synchronize_device(model.device)
        timings.append((time.perf_counter() - start) * 1.0e6)

    return {
        "mode": mode,
        "use_elementwise_P": bool(solver._use_elementwise_P),
        "use_elementwise_H": bool(solver._use_elementwise_H),
        "median_us": statistics.median(timings),
        "p95_us": _percentile(timings, 0.95),
        "min_us": min(timings),
        "max_us": max(timings),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    rows = []
    for dofs in (1, 2, 7, 18, 23):
        for articulations in (1, 8, 64):
            model = _build_model(articulations, dofs, device)
            serial = _time_case(model, "serial", args.warmup, args.samples)
            auto = _time_case(model, "auto", args.warmup, args.samples)
            row = {
                "dofs": dofs,
                "articulations": articulations,
                "P_elements_per_articulation": 6 * dofs * dofs,
                "H_elements_per_articulation": dofs * dofs,
                "serial": serial,
                "auto": auto,
                "auto_over_serial_median": auto["median_us"] / serial["median_us"],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "device": {
            "alias": device.alias,
            "name": device.name,
            "arch": str(device.arch),
            "is_cuda": bool(device.is_cuda),
        },
        "warmup": args.warmup,
        "samples": args.samples,
        "rows": rows,
    }
    raw = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(raw + "\n")
    print(raw)


if __name__ == "__main__":
    main()
