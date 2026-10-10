"""Frozen, independent analytic witnesses for Newton's public cone raycast.

The expected answers come from cone cross-sections, not Newton's quadratic.
Run against any checkout by setting PYTHONPATH to that checkout. Nothing here
edits that checkout. Public rays are normalized and nonzero; dimensions positive.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import warp as wp
import newton
from newton._src.geometry.raycast import ray_intersect_cone

SQRT5 = math.sqrt(5.0)
SIDE_X = (2.0 / SQRT5, 0.0, 1.0 / SQRT5)
BASE = (0.0, 0.0, -1.0)


def case(name, origin, direction, expected, normal=None, radius=1.0, half_height=1.0, tier="core"):
    """Store a manually derived ray witness with a normalized direction."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return dict(name=name, origin=list(origin), direction=direction.tolist(), expected=expected,
                normal=normal, radius=radius, half_height=half_height, tier=tier)


CASES = [
    case("interior_center_side_x", (0, 0, 0), (1, 0, 0), 0.5, SIDE_X),
    case("interior_center_side_y", (0, 0, 0), (0, -1, 0), 0.5, (0, -2 / SQRT5, 1 / SQRT5)),
    case("interior_offset_outward", (0.2, 0, 0), (1, 0, 0), 0.3, SIDE_X),
    case("interior_offset_inward", (0.2, 0, 0), (-1, 0, 0), 0.7, (-2 / SQRT5, 0, 1 / SQRT5)),
    case("interior_upper_slice", (0, 0, 0.5), (1, 0, 0), 0.25, SIDE_X),
    case("interior_axis_base", (0, 0, 0), (0, 0, -1), 1.0, BASE),
    case("interior_offset_base", (0.25, 0, 0), (0, 0, -1), 1.0, BASE),
    case("interior_axis_apex", (0, 0, 0), (0, 0, 1), 1.0),
    case("interior_offset_ascending", (0.25, 0, 0), (0, 0, 1), 0.5, SIDE_X),
    case("exterior_side_entry", (-2, 0, 0), (1, 0, 0), 1.5, (-2 / SQRT5, 0, 1 / SQRT5)),
    case("exterior_base_entry", (0, 0, -2), (0, 0, 1), 1.0, BASE),
    case("exterior_base_before_side", (0, 0, -2), (3, 0, 4), 1.25, BASE),
    case("exterior_apex_entry", (0, 0, 2), (0, 0, -1), 1.0),
    case("exterior_side_after_clipped_root", (0.25, 0, 2), (0, 0, -1), 1.5, SIDE_X),
    case("side_tangent", (-1, 0.5, 0), (1, 0, 0), 1.0, (0, 2 / SQRT5, 1 / SQRT5)),
    case("miss_offset", (-2, 2, 0), (1, 0, 0), -1.0, (0, 0, 0)),
    case("miss_both_roots_behind", (2, 0, 0), (1, 0, 0), -1.0, (0, 0, 0)),
    case("miss_both_roots_axially_clipped", (-2, 0, 2), (1, 0, 0), -1.0, (0, 0, 0)),
    case("linear_interior_side", (0, 0, 0), (1, 0, 2), SQRT5 / 4, SIDE_X),
    case("linear_interior_base", (0, 0, 0), (1, 0, -2), SQRT5 / 2, BASE),
    case("linear_exterior_side", (1, 0, 0), (-1, 0, -2), SQRT5 / 4, SIDE_X),
    case("linear_exterior_away", (1, 0, 0), (1, 0, 2), -1.0, (0, 0, 0)),
    case("constant_nonzero_polynomial_miss", (0.5, 0.25, 0), (-1, 0, 2), -1.0, (0, 0, 0)),
    case("tall_cone_center_exit", (0, 0, 0), (1, 0, 0), 0.25, radius=0.5, half_height=2.0),
    case("squat_cone_center_exit", (0, 0, 0), (1, 0, 0), 1.0, radius=2.0, half_height=0.5),
    case("small_cone_center_exit", (0, 0, 0), (1, 0, 0), 0.005, radius=0.01, half_height=0.01),
    case("large_cone_center_exit", (0, 0, 0), (1, 0, 0), 5.0, radius=10.0, half_height=10.0),
    case("boundary_side_origin", (0.5, 0, 0), (1, 0, 0), 0.0, SIDE_X, tier="boundary"),
    case("boundary_base_origin", (0.25, 0, -1), (0, 0, -1), 0.0, BASE, tier="boundary"),
    case("generator_surface_origin", (0.5, 0, 0), (-1, 0, 2), 0.0, SIDE_X, tier="boundary"),
    case("generator_from_below", (1.5, 0, -2), (-1, 0, 2), SQRT5 / 2, tier="boundary"),
]


@wp.kernel
def direct_probe(origins: wp.array[wp.vec3], directions: wp.array[wp.vec3],
                 radii: wp.array[float], heights: wp.array[float],
                 distances: wp.array[float], normals: wp.array[wp.vec3]):
    """Probe the unchanged private helper to distinguish filtering from solving."""
    i = wp.tid()
    t, n = ray_intersect_cone(origins[i], directions[i], radii[i], heights[i])
    distances[i] = t
    normals[i] = n


def validate_witnesses():
    """Check hit witnesses independently against their closed-cone surfaces."""
    for c in CASES:
        d = np.asarray(c["direction"])
        assert abs(np.linalg.norm(d) - 1.0) < 1e-12
        if c["expected"] < 0:
            continue
        point = np.asarray(c["origin"]) + c["expected"] * d
        r, h = c["radius"], c["half_height"]
        radial = float(np.linalg.norm(point[:2]))
        assert -h - 1e-12 <= point[2] <= h + 1e-12, c["name"]
        side_radius = r * (h - point[2]) / (2 * h)
        on_side = abs(radial - side_radius) < 1e-12
        on_base = abs(point[2] + h) < 1e-12 and radial <= r + 1e-12
        assert on_side or on_base, (c["name"], point)
        # An interior-origin ray must stay strictly inside up to its first exit.
        o = np.asarray(c["origin"])
        inside = -h < o[2] < h and np.linalg.norm(o[:2]) < r * (h - o[2]) / (2 * h)
        if inside:
            for fraction in (0.25, 0.5, 0.75, 0.999):
                p = o + c["expected"] * fraction * d
                assert -h < p[2] < h and np.linalg.norm(p[:2]) < r * (h - p[2]) / (2 * h), c["name"]


def report(label, distances, normals, shape_ids=None, transformed=False):
    """Report every independent witness instead of aborting on the first error."""
    failures = []
    records = []
    for i, (c, t, n) in enumerate(zip(CASES, distances, normals, strict=True)):
        expected = c["expected"]
        tolerance = max(2e-6, abs(expected) * 2e-5)
        ok = bool(np.isfinite(t) and abs(float(t) - expected) <= tolerance)
        if c["normal"] is not None:
            target_normal = np.asarray(c["normal"])
            if transformed:
                target_normal = target_normal[[2, 0, 1]]
            ok = ok and bool(np.allclose(n, target_normal, atol=2e-5, rtol=0))
        if shape_ids is not None:
            ok = ok and int(shape_ids[i]) == (i if expected >= 0 else -1)
        record = dict(path=label, case=c["name"], tier=c["tier"], expected=expected,
                      actual=float(t), normal=[float(v) for v in n], passed=ok)
        records.append(record)
        if not ok:
            failures.append(record)
        print(f'{"PASS" if ok else "FAIL"} {label} {c["name"]}: expected={expected:.9g} actual={t:.9g} normal={n}')
    return records, failures


def main():
    """Exercise analytic witnesses through public BVH and private helper paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    validate_witnesses()
    wp.config.kernel_cache_dir = str(Path(__file__).parent / "warp-cache")
    wp.init()
    print("Newton:", newton.__file__, "Warp:", wp.__version__)
    print("Expected values were frozen before any cone candidate was inspected.")
    all_records, failures = [], []
    local_origins = np.asarray([c["origin"] for c in CASES], dtype=np.float32)
    local_directions = np.asarray([c["direction"] for c in CASES], dtype=np.float32)
    make_vectors = lambda a: wp.array(a, dtype=wp.vec3, device="cpu")
    origins, directions = make_vectors(local_origins), make_vectors(local_directions)
    count = len(CASES)
    out_t = wp.empty(count, dtype=float, device="cpu")
    out_n = wp.empty(count, dtype=wp.vec3, device="cpu")
    radii = wp.array([c["radius"] for c in CASES], dtype=float, device="cpu")
    heights = wp.array([c["half_height"] for c in CASES], dtype=float, device="cpu")
    wp.launch(direct_probe, dim=count, inputs=[origins, directions, radii, heights], outputs=[out_t, out_n], device="cpu")
    records, failed = report("helper", out_t.numpy(), out_n.numpy())
    all_records.extend(records)
    failures.extend(failed)
    for transformed in (False, True):
        builder = newton.ModelBuilder()
        for c in CASES:
            builder.begin_world()
            xform = wp.transform(wp.vec3(3, -2, 5), wp.quat(0.5, 0.5, 0.5, 0.5)) if transformed else wp.transform_identity()
            builder.add_shape_cone(body=-1, xform=xform, radius=c["radius"], half_height=c["half_height"])
            builder.end_world()
        model = builder.finalize(device="cpu")
        if transformed:
            origins = make_vectors(local_origins[:, [2, 0, 1]] + np.array([3, -2, 5], dtype=np.float32))
            directions = make_vectors(local_directions[:, [2, 0, 1]])
        else:
            origins, directions = make_vectors(local_origins), make_vectors(local_directions)
        worlds = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device="cpu")
        out_shape = wp.empty(count, dtype=wp.int32, device="cpu")
        for fast_math in (False, True):
            newton.intersect_ray(model, ray_origins=origins, ray_directions=directions, ray_worlds=worlds,
                                 out_dist=out_t, out_shape_id=out_shape, out_normal=out_n, fast_math=fast_math)
            label = f"public_transformed={transformed}_fast_math={fast_math}"
            records, failed = report(label, out_t.numpy(), out_n.numpy(), out_shape.numpy(), transformed)
            all_records.extend(records)
            failures.extend(failed)
    result = dict(newton_file=newton.__file__, warp_version=wp.__version__, cases=len(CASES),
                  checks=len(all_records), failures=len(failures), records=all_records)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    print(f'SUMMARY: {len(all_records) - len(failures)}/{len(all_records)} checks passed; {len(failures)} failed')
    if failures and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
