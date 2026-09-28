# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Evidence for newton-physics/newton#4212. Does not change solver behavior.

Reproduces two different linesearch warnings and checks the optimal-point
hypothesis against the bracket condition in the installed mujoco_warp:

* default ``ls_iterations`` (MuJoCo's 50) on a resting contact
* a deliberately tiny budget (``ls_iterations=2``) on the same fixture

The number printed by mujoco_warp is the budget that was exhausted
(``LS_ITERATIONS``), not a suggested doubled value. A warning that names 2 is
the expected small-budget warning. A warning that names 50 would be the
default-budget defect described on the issue.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
from collections import Counter

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

LS_OVERFLOW = 1 << 10
GTOL_FLOOR = 1.0e-6


def _installed_solver_lines() -> str:
    import mujoco
    import mujoco_warp

    path = os.path.join(os.path.dirname(mujoco_warp.__file__), "_src", "solver.py")
    lines = open(path, encoding="utf-8").read().splitlines()
    interesting = []
    for i, line in enumerate(lines, start=1):
        if "ls_done = (" in line or "increase ls_iterations" in line:
            interesting.append(f"{i}:{line.strip()}")
    return (
        f"mujoco {mujoco.__version__}\n"
        f"solver {path}\n" + "\n".join(interesting)
    )


def _resting_box(ls_iterations: int | None, steps: int = 30) -> dict:
    """Frictionless box already in contact with the ground, zero velocity."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.1005), q=wp.quat_identity()))
    builder.add_shape_box(
        body,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.0, ke=1.0e5, kd=0.0, restitution=0.0),
    )
    builder.add_articulation([builder.add_joint_free(body)])
    model = builder.finalize(device="cuda:0")
    kwargs = dict(iterations=20, njmax=40, nconmax=20)
    if ls_iterations is not None:
        kwargs["ls_iterations"] = ls_iterations
    solver = SolverMuJoCo(model, **kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    capture = os.open(tmp.name, os.O_WRONLY | os.O_TRUNC)
    saved = os.dup(1)
    os.dup2(capture, 1)
    hits = 0
    max_niter = 0
    try:
        for _ in range(steps):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            overflow = int(solver.mjw_data.overflow.numpy()[0])
            niter = int(solver.mjw_data.solver_niter.numpy()[0])
            max_niter = max(max_niter, niter)
            if overflow & LS_OVERFLOW:
                hits += 1
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        ctypes.CDLL(None).fflush(None)
    finally:
        os.dup2(saved, 1)
        os.close(capture)
        os.close(saved)
    text = open(tmp.name, encoding="utf-8", errors="replace").read()
    os.unlink(tmp.name)
    warnings = [ln for ln in text.splitlines() if "linesearch iterations limit" in ln]
    return {
        "configured_ls_iterations": int(solver.mj_model.opt.ls_iterations),
        "steps": steps,
        "ls_overflow_steps": hits,
        "max_solver_niter": max_niter,
        "warning_counts": Counter(warnings),
    }


def _safe_div(x, y):
    y = np.float32(y)
    denom = y if y != np.float32(0.0) else np.float32(1.0e-15)
    return np.float32(np.float32(x) / denom)


def _in_bracket(x1, y1) -> bool:
    return (x1 < y1 and y1 < 0.0) or (x1 > y1 and y1 > 0.0)


def _eval_row(jaref, jv, d, alpha) -> np.ndarray:
    alpha = np.float32(alpha)
    x = np.float32(jaref) + alpha * np.float32(jv)
    jv_d = np.float32(jv) * np.float32(d)
    return np.array(
        [np.float32(0.5) * np.float32(d) * x * x, jv_d * x, np.float32(jv) * jv_d],
        dtype=np.float32,
    )


def _bracket_iterations(rows, limit: int = 50, gtol: float = GTOL_FLOOR) -> dict:
    """Same ls_done disjuncts as mujoco_warp 3.12 _linesearch_iterative_kernel.

    ``rows`` are quadratic constraint terms whose float64 residual is zero at
    alpha=0 (the current point is optimal) before they are rounded to float32.
    """
    def evaluate(alpha):
        acc = np.zeros(3, dtype=np.float32)
        for jaref, jv, d in rows:
            acc += _eval_row(jaref, jv, d, alpha)
        return acc

    p0 = evaluate(0.0)
    p0_delta = np.array([np.float32(0.0), p0[1], p0[2]], dtype=np.float32)
    lo_alpha_in = float(-_safe_div(p0[1], p0[2]))
    lo_in = evaluate(lo_alpha_in)
    if abs(float(lo_in[1])) < gtol and float(lo_in[0]) < 0.0:
        return {"iterations_until_done": 0, "initial_converged": True, "exhausted": False}

    lo_less = float(lo_in[1]) < float(p0[1])
    lo = lo_in if lo_less else p0_delta
    lo_alpha = lo_alpha_in if lo_less else 0.0
    hi = p0_delta if lo_less else lo_in
    hi_alpha = 0.0 if lo_less else lo_alpha_in
    same_sign_seen = False
    for it in range(limit):
        lo_next_alpha = float(np.float32(lo_alpha) - _safe_div(lo[1], lo[2]))
        hi_next_alpha = float(np.float32(hi_alpha) - _safe_div(hi[1], hi[2]))
        mid_alpha = float(np.float32(0.5) * (np.float32(lo_alpha) + np.float32(hi_alpha)))
        lo_next = evaluate(lo_next_alpha)
        hi_next = evaluate(hi_next_alpha)
        mid = evaluate(mid_alpha)
        swap_lo = False
        for cand, cand_alpha in (
            (lo_next, lo_next_alpha),
            (mid, mid_alpha),
            (hi_next, hi_next_alpha),
        ):
            if _in_bracket(float(lo[1]), float(cand[1])):
                lo, lo_alpha, swap_lo = cand, cand_alpha, True
        swap_hi = False
        for cand, cand_alpha in (
            (hi_next, hi_next_alpha),
            (mid, mid_alpha),
            (lo_next, lo_next_alpha),
        ):
            if _in_bracket(float(hi[1]), float(cand[1])):
                hi, hi_alpha, swap_hi = cand, cand_alpha, True
        if float(lo[1]) * float(hi[1]) > 0.0:
            same_sign_seen = True
        ls_done = (not swap_lo and not swap_hi) or (
            float(lo[0]) < 0.0 and float(lo[1]) < 0.0 and float(lo[1]) > -gtol
        ) or (float(hi[0]) < 0.0 and float(hi[1]) > 0.0 and float(hi[1]) < gtol)
        if ls_done:
            return {
                "iterations_until_done": it + 1,
                "initial_converged": False,
                "exhausted": False,
                "same_sign_derivatives_seen": same_sign_seen,
                "final_lo_alpha": float(lo_alpha),
                "final_hi_alpha": float(hi_alpha),
                "final_lo_deriv": float(lo[1]),
                "final_hi_deriv": float(hi[1]),
                "final_lo_cost_delta": float(lo[0]),
                "final_hi_cost_delta": float(hi[0]),
            }
    return {
        "iterations_until_done": limit,
        "initial_converged": False,
        "exhausted": True,
        "same_sign_derivatives_seen": same_sign_seen,
        "final_lo_alpha": float(lo_alpha),
        "final_hi_alpha": float(hi_alpha),
        "final_lo_deriv": float(lo[1]),
        "final_hi_deriv": float(hi[1]),
        "final_lo_cost_delta": float(lo[0]),
        "final_hi_cost_delta": float(hi[0]),
    }


def _optimal_rows() -> list[tuple[np.float32, np.float32, np.float32]]:
    """Two opposing quadratic rows. Float64 derivative at alpha=0 is 0.

    Float32 evaluation does not cancel: the summed derivative is noise relative
    to the two huge terms, which is the optimal-point situation in the issue
    comment. The bracket still finishes before the default budget of 50.
    """
    d = np.float32(1.0e8)
    return [
        (np.float32(1.0e-6), np.float32(1.0), d),
        (np.float32(-1.0e-6), np.float32(-1.0), d),
    ]


def main() -> None:
    wp.init()
    print(_installed_solver_lines())
    print("--- resting contact, default budget ---")
    default = _resting_box(None)
    print(default)
    print("--- resting contact, ls_iterations=2 ---")
    tiny = _resting_box(2)
    print(tiny)
    print("--- float32 optimal-point bracket, limit 50 ---")
    bracket = _bracket_iterations(_optimal_rows(), limit=50)
    print(bracket)
    print("--- same point, limit 2 (tiny budget) ---")
    tiny_bracket = _bracket_iterations(_optimal_rows(), limit=2)
    print(tiny_bracket)


if __name__ == "__main__":
    main()
