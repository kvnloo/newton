#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Profile current VBD cloth/volume phase costs for #4141 follow-up evidence."""

from __future__ import annotations

import argparse
import collections
import json
import statistics

import numpy as np
import warp as wp

import newton


def _prototype(kind: str) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    if kind == "cloth":
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.22),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=6,
            dim_y=6,
            cell_x=0.08,
            cell_y=0.08,
            mass=0.1,
            fix_left=False,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e2,
            edge_ke=1.0e1,
            edge_kd=0.0,
            particle_radius=0.03,
        )
        builder.color(include_bending=True)
    elif kind == "volume":
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.08),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=3,
            dim_y=3,
            dim_z=3,
            cell_x=0.08,
            cell_y=0.08,
            cell_z=0.08,
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1.0e2,
            fix_left=False,
        )
        builder.color()
    else:
        raise ValueError(kind)

    return builder


def _build(kind: str, worlds: int, device) -> newton.Model:
    prototype = _prototype(kind)
    scene = newton.ModelBuilder()
    scene.replicate(prototype, world_count=worlds, spacing=(0.7, 0.7, 0.0))
    model = scene.finalize(device=device)
    model.soft_contact_ke = 1.0e2
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 1.0
    return model


def _category(name: str) -> str:
    n = name.lower()
    if "build_particle_body_contact_adjacency_active" in n:
        return "contact_adjacency"
    if "gather_particle_body_contact_force_and_hessian" in n:
        return "contact_gather"
    if "init_body_particle_contacts" in n or "update_duals_body_particle_contacts" in n:
        return "contact_state"
    if "solve_elasticity" in n or "elasticity" in n:
        return "elasticity"
    if (
        "soft_contact" in n
        or "collision" in n
        or "contact_pair" in n
        or "shape_contact" in n
        or "sdf" in n
    ):
        return "collision_other"
    if "forward_step" in n or "update_velocity" in n:
        return "integration"
    return "other"


def _summarize(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    p95_idx = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "n": len(values),
        "p50_ms": float(statistics.median(values)),
        "p95_ms": float(ordered[p95_idx]),
        "max_ms": float(max(values)),
    }


def _run(kind: str, worlds: int, device, warmup: int, samples: int, iterations: int):
    model = _build(kind, worlds, device)
    pipeline = newton.CollisionPipeline(model)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        particle_enable_self_contact=False,
        particle_enable_tile_solve=True,
        rigid_compliant_alm=False,
        collision_pipeline=pipeline,
    )
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    dt = 1.0 / 60.0

    def step():
        nonlocal state_in, state_out
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in

    for _ in range(warmup):
        step()
    wp.synchronize_device(device)

    wall_ms = []
    kernel_ms = []
    category_samples = collections.defaultdict(list)
    kernel_totals = collections.Counter()
    kernel_counts = collections.Counter()
    active_contacts = []

    for _ in range(samples):
        with wp.ScopedTimer(
            "vbd-step",
            print=False,
            synchronize=True,
            cuda_filter=wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN,
        ) as timer:
            step()

        wall_ms.append(float(timer.elapsed))
        kernel_total = 0.0
        categories = collections.Counter()
        for result in timer.timing_results:
            elapsed = float(result.elapsed)
            kernel_total += elapsed
            category = _category(result.name)
            categories[category] += elapsed
            kernel_totals[result.name] += elapsed
            kernel_counts[result.name] += 1

        kernel_ms.append(kernel_total)
        for category in (
            "contact_adjacency",
            "contact_gather",
            "contact_state",
            "elasticity",
            "collision_other",
            "integration",
            "other",
        ):
            category_samples[category].append(float(categories[category]))

        contacts = solver.contacts
        if contacts is not None and contacts.soft_contact_count is not None:
            active_contacts.append(int(contacts.soft_contact_count.numpy()[0]))

    top_kernels = [
        {
            "name": name,
            "total_ms": float(total),
            "count": int(kernel_counts[name]),
            "mean_ms_per_launch": float(total / kernel_counts[name]),
        }
        for name, total in kernel_totals.most_common(25)
    ]

    result = {
        "kind": kind,
        "worlds": worlds,
        "particles": int(model.particle_count),
        "triangles": int(model.tri_count),
        "tetrahedra": int(model.tet_count),
        "iterations": iterations,
        "wall": _summarize(wall_ms),
        "kernel_total": _summarize(kernel_ms),
        "categories": {name: _summarize(values) for name, values in sorted(category_samples.items())},
        "top_kernels": top_kernels,
        "soft_contacts": {
            "min": min(active_contacts) if active_contacts else None,
            "median": statistics.median(active_contacts) if active_contacts else None,
            "max": max(active_contacts) if active_contacts else None,
        },
    }

    median_ms = result["wall"]["p50_ms"]
    result["env_steps_per_s"] = float(worlds * 1000.0 / median_ms) if median_ms else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--kinds", nargs="+", choices=("cloth", "volume"), default=["cloth", "volume"])
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device(args.device)
    rows = []
    for kind in args.kinds:
        for worlds in args.worlds:
            row = _run(kind, worlds, device, args.warmup, args.samples, args.iterations)
            rows.append(row)
            print(
                f"{kind} worlds={worlds} "
                f"p50={row['wall']['p50_ms']:.3f}ms "
                f"env_steps/s={row['env_steps_per_s']:.1f}",
                flush=True,
            )

    payload = {
        "issue": "https://github.com/newton-physics/newton/issues/4141",
        "related_pr": "https://github.com/newton-physics/newton/pull/3995",
        "method": {
            "note": (
                "Current-main reduced workload, not the Isaac Lab Franka benchmark. "
                "Each sampled step uses Warp CUDA activity timing; instrumentation adds overhead. "
                "Use category fractions/top-kernel ordering for diagnosis and the wall p50 only as a local control."
            ),
            "warmup": args.warmup,
            "samples": args.samples,
            "iterations": args.iterations,
        },
        "device": {
            "alias": device.alias,
            "name": device.name,
            "arch": str(device.arch),
        },
        "rows": rows,
    }
    raw = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(raw + "\n")
    print(raw)


if __name__ == "__main__":
    main()
