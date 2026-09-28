# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Profile tri-mesh self-contact storage: fixed per-element capacity vs detected contacts.

Prints a JSON object of measurements taken on this machine. Does not change allocation.
Warm up detection, synchronize the CUDA device, and report medians.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import statistics
import subprocess
import time

import numpy as np
import warp as wp

import newton
from newton._src.geometry.tri_mesh_collision import (
    TriMeshCollisionDetector,
    build_tri_mesh_collision_info,
)


def _sync(device: wp.Device) -> None:
    wp.synchronize_device(device)


def _used_bytes(device: wp.Device) -> int:
    _sync(device)
    return int(device.total_memory) - int(device.free_memory)


def _median_int(samples: list[int]) -> int:
    return int(statistics.median(samples))


def _array_nbytes(array: wp.array | None) -> int:
    if array is None:
        return 0
    # warp.array.capacity is the allocation size in bytes.
    return int(array.capacity)


def _info_nbytes(info) -> dict[str, int]:
    names = (
        "vertex_colliding_triangles",
        "vertex_colliding_triangles_offsets",
        "vertex_colliding_triangles_buffer_sizes",
        "vertex_colliding_triangles_count",
        "vertex_colliding_triangles_min_dist",
        "triangle_colliding_vertices",
        "triangle_colliding_vertices_offsets",
        "triangle_colliding_vertices_buffer_sizes",
        "triangle_colliding_vertices_count",
        "triangle_colliding_vertices_min_dist",
        "edge_colliding_edges",
        "edge_colliding_edges_offsets",
        "edge_colliding_edges_buffer_sizes",
        "edge_colliding_edges_count",
        "edge_colliding_edges_min_dist",
    )
    parts = {name: _array_nbytes(getattr(info, name)) for name in names}
    parts["total"] = int(sum(parts.values()))
    return parts


def _build_grid(dim: int, device: str):
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim,
        dim_y=dim,
        cell_x=0.02,
        cell_y=0.02,
        mass=0.01,
        fix_left=True,
        tri_ke=100.0,
        tri_ka=100.0,
        tri_kd=0.01,
        edge_ke=10.0,
        edge_kd=0.01,
    )
    model = builder.finalize(device=device)
    return model


def _compress_positions(model, scale: float) -> None:
    positions = model.particle_q.numpy()
    positions[:, 0] *= scale
    positions[:, 2] *= scale
    model.particle_q.assign(positions)


def _contact_stats(info) -> dict[str, int]:
    vertex_count = info.vertex_colliding_triangles_count.numpy()
    vertex_capacity = info.vertex_colliding_triangles_buffer_sizes.numpy()
    edge_count = info.edge_colliding_edges_count.numpy()
    edge_capacity = info.edge_colliding_edges_buffer_sizes.numpy()
    vertex_stored = np.minimum(vertex_count, vertex_capacity)
    edge_stored = np.minimum(edge_count, edge_capacity)
    return {
        "vertex_capacity_slots": int(vertex_capacity.sum()),
        "vertex_detected": int(vertex_count.sum()),
        "vertex_stored": int(vertex_stored.sum()),
        "vertex_overflow_elements": int(np.count_nonzero(vertex_count > vertex_capacity)),
        "edge_capacity_slots": int(edge_capacity.sum()),
        "edge_detected": int(edge_count.sum()),
        "edge_stored": int(edge_stored.sum()),
        "edge_overflow_elements": int(np.count_nonzero(edge_count > edge_capacity)),
    }


def _detect(detector: TriMeshCollisionDetector, radius: float) -> None:
    detector.vertex_triangle_collision_detection(radius)
    detector.edge_edge_collision_detection(radius)


def _measure_allocation_deltas(model, device: wp.Device, vertex_pre: int, edge_pre: int, repeats: int) -> dict:
    """Incremental cuda free-memory drop for repeated result-buffer allocations."""
    retained = []
    # Warm the mempool with one allocation that is not part of the median.
    warmup = build_tri_mesh_collision_info(
        model.particle_count,
        model.tri_count,
        model.edge_count,
        vertex_collision_buffer_pre_alloc=vertex_pre,
        edge_collision_buffer_pre_alloc=edge_pre,
        record_triangle_contacting_vertices=False,
        device=device,
    )
    retained.append(warmup)
    _sync(device)
    deltas = []
    declared = None
    for _ in range(repeats):
        before = _used_bytes(device)
        info = build_tri_mesh_collision_info(
            model.particle_count,
            model.tri_count,
            model.edge_count,
            vertex_collision_buffer_pre_alloc=vertex_pre,
            edge_collision_buffer_pre_alloc=edge_pre,
            record_triangle_contacting_vertices=False,
            device=device,
        )
        after = _used_bytes(device)
        deltas.append(after - before)
        if declared is None:
            declared = _info_nbytes(info)
        retained.append(info)
    return {
        "cuda_used_delta_bytes_samples": deltas,
        "cuda_used_delta_bytes_median": _median_int(deltas),
        "declared_nbytes": declared,
    }


def _measure_scene(
    *,
    name: str,
    dim: int,
    radius: float,
    compress: float | None,
    vertex_pre: int,
    edge_pre: int,
    device: wp.Device,
    warmup: int,
    repeats: int,
) -> dict:
    model = _build_grid(dim, str(device))
    if compress is not None:
        _compress_positions(model, compress)
    allocation = _measure_allocation_deltas(model, device, vertex_pre, edge_pre, repeats)
    info = build_tri_mesh_collision_info(
        model.particle_count,
        model.tri_count,
        model.edge_count,
        vertex_collision_buffer_pre_alloc=vertex_pre,
        edge_collision_buffer_pre_alloc=edge_pre,
        record_triangle_contacting_vertices=False,
        device=device,
    )
    before_detector = _used_bytes(device)
    detector = TriMeshCollisionDetector(
        model=model,
        vertex_collision_buffer_pre_alloc=vertex_pre,
        edge_collision_buffer_pre_alloc=edge_pre,
        collision_info=info,
    )
    if compress is not None:
        detector.refit()
    after_detector = _used_bytes(device)

    for _ in range(warmup):
        _detect(detector, radius)
        _sync(device)

    used_samples = []
    elapsed_ms = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        _detect(detector, radius)
        _sync(device)
        elapsed_ms.append((time.perf_counter() - start) * 1e3)
        used_samples.append(_used_bytes(device))

    stats = _contact_stats(info)
    index_pair_bytes = 8  # two int32 ids per stored vertex-triangle or edge-edge pair
    proportional_stored = (stats["vertex_stored"] + stats["edge_stored"]) * index_pair_bytes
    proportional_detected = (stats["vertex_detected"] + stats["edge_detected"]) * index_pair_bytes
    fixed_index = allocation["declared_nbytes"]["vertex_colliding_triangles"] + allocation["declared_nbytes"][
        "edge_colliding_edges"
    ]
    resize_flags = [int(v) for v in detector.resize_flags.numpy().tolist()]
    return {
        "name": name,
        "dim_cells": dim,
        "query_radius_m": radius,
        "position_xz_scale": compress,
        "vertex_buffer_pre_alloc": vertex_pre,
        "edge_buffer_pre_alloc": edge_pre,
        "particle_count": int(model.particle_count),
        "tri_count": int(model.tri_count),
        "edge_count": int(model.edge_count),
        "contacts": stats,
        "resize_flags": resize_flags,
        "fixed_index_buffer_bytes": int(fixed_index),
        "proportional_stored_index_bytes": int(proportional_stored),
        "proportional_detected_index_bytes": int(proportional_detected),
        "allocation": allocation,
        "detector_cuda_used_delta_bytes": int(after_detector - before_detector),
        "cuda_used_bytes_after_detection_samples": used_samples,
        "cuda_used_bytes_after_detection_median": _median_int(used_samples),
        "detection_ms_samples": elapsed_ms,
        "detection_ms_median": float(statistics.median(elapsed_ms)),
    }


def _mempool_off_allocation_probe(device_name: str) -> list[dict]:
    """Child process: CUDA mempool disabled so free-memory deltas track allocations."""
    if os.environ.get("NEWTON_SELF_CONTACT_MEMPOOL_OFF") == "1":
        cases = [
            ("sparse_default_buffers", 1681, 3200, 4880, 16, 32),
            ("dense_default_buffers", 1681, 3200, 4880, 16, 32),
            ("sparse_large_mesh_high_capacity", 25921, 51200, 77120, 256, 256),
            ("dense_small_mesh_high_capacity", 625, 1152, 1776, 256, 256),
        ]
        device = wp.get_device(device_name)
        _sync(device)
        rows = []
        for name, particles, tris, edges, vertex_pre, edge_pre in cases:
            # One unmeasured allocation warms the driver before the median.
            warmup = build_tri_mesh_collision_info(
                particles,
                tris,
                edges,
                vertex_collision_buffer_pre_alloc=vertex_pre,
                edge_collision_buffer_pre_alloc=edge_pre,
                record_triangle_contacting_vertices=False,
                device=device,
            )
            del warmup
            _sync(device)
            deltas = []
            declared = None
            for _ in range(7):
                before = _used_bytes(device)
                info = build_tri_mesh_collision_info(
                    particles,
                    tris,
                    edges,
                    vertex_collision_buffer_pre_alloc=vertex_pre,
                    edge_collision_buffer_pre_alloc=edge_pre,
                    record_triangle_contacting_vertices=False,
                    device=device,
                )
                after = _used_bytes(device)
                deltas.append(int(after - before))
                if declared is None:
                    declared = _info_nbytes(info)
                del info
            rows.append(
                {
                    "name": name,
                    "cuda_used_delta_bytes_samples": deltas,
                    "cuda_used_delta_bytes_median": _median_int(deltas),
                    "declared_nbytes_total": declared["total"],
                    "vertex_index_bytes": declared["vertex_colliding_triangles"],
                    "edge_index_bytes": declared["edge_colliding_edges"],
                }
            )
        return rows

    env = os.environ.copy()
    env["NEWTON_SELF_CONTACT_MEMPOOL_OFF"] = "1"
    # Warp reads this before init; the child imports warp after the variable would be too late
    # if we only set it here. The child entry below sets the config flag before wp.init().
    completed = subprocess.run(
        [sys.executable, __file__, "--mempool-off-probe", "--device", device_name],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    start = completed.stdout.find("[")
    if start < 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return json.loads(completed.stdout[start:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="")
    parser.add_argument("--mempool-off-probe", action="store_true")
    args = parser.parse_args()

    if args.mempool_off_probe:
        wp.config.enable_mempools_at_init = False
        wp.init()
        rows = _mempool_off_allocation_probe(args.device)
        print(json.dumps(rows))
        return

    wp.init()
    device = wp.get_device(args.device)
    if device.is_cpu:
        raise SystemExit("self-contact storage profile requires a CUDA device")

    scenes = [
        _measure_scene(
            name="sparse_default_buffers",
            dim=40,
            radius=1.0e-6,
            compress=None,
            vertex_pre=16,
            edge_pre=32,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        _measure_scene(
            name="dense_default_buffers",
            dim=40,
            radius=0.05,
            compress=0.02,
            vertex_pre=16,
            edge_pre=32,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        _measure_scene(
            name="sparse_large_mesh_high_capacity",
            dim=160,
            radius=1.0e-6,
            compress=None,
            vertex_pre=256,
            edge_pre=256,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        _measure_scene(
            name="dense_small_mesh_high_capacity",
            dim=24,
            radius=0.05,
            compress=0.02,
            vertex_pre=256,
            edge_pre=256,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    ]
    payload = {
        "device_name": device.name,
        "device_arch": device.arch,
        "sm_count": int(device.sm_count),
        "total_memory_bytes": int(device.total_memory),
        "mempool_enabled": bool(device.is_mempool_enabled),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scenes": scenes,
        "mempool_disabled_allocation": _mempool_off_allocation_probe(args.device),
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
