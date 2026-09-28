# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""RTX evidence for Featherstone dense-GEMM dispatch (newton-physics/newton#4265).

Compares, without tuning to a single shape:

- ``eval_dense_gemm_batched`` (one thread per articulation; in-tree)
- an element-parallel GEMM with the same per-element reduction order (candidate; not in the solver)
- ``create_inertia_matrix_kernel`` tile GEMM when every articulation shares ``(n_joints, n_dofs)`` (in-tree)

Also times ``SolverFeatherstone.step`` for ``use_tile_gemm`` False/True where the solver's
hard-coded 18-DOF tile asserts allow it.

Each timed sample is host wall time around one launch with device sync before the stamp
and after the launch. Statistics are median and nearest-rank p95. No numbers are filled in
by hand; this script writes ``scripts/featherstone_gemm_rtx3080ti.json``.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.featherstone.kernels import (
    create_inertia_matrix_kernel,
    eval_dense_gemm_batched,
)

OUT_PATH = Path(__file__).resolve().parent / "featherstone_gemm_rtx3080ti.json"
DEVICE = "cuda:0"
WARMUP = 20
SAMPLES = 40
SOLVER_WARMUP = 10
SOLVER_SAMPLES = 30


@wp.kernel
def eval_dense_gemm_element(
    m: wp.array[int],
    n: wp.array[int],
    p: wp.array[int],
    transpose_A: bool,
    transpose_B: bool,
    A_start: wp.array[int],
    B_start: wp.array[int],
    C_start: wp.array[int],
    A: wp.array[float],
    B: wp.array[float],
    C: wp.array[float],
):
    """One thread per output element. Inner ``k`` loop matches ``dense_gemm``."""
    batch, elem = wp.tid()
    mm = m[batch]
    nn = n[batch]
    pp = p[batch]
    if elem >= mm * nn:
        return
    i = elem // nn
    j = elem - i * nn
    sum = float(0.0)
    for k in range(pp):
        if transpose_A:
            a_i = k * mm + i
        else:
            a_i = i * pp + k
        if transpose_B:
            b_j = j * pp + k
        else:
            b_j = k * nn + j
        sum += A[A_start[batch] + a_i] * B[B_start[batch] + b_j]
    C[C_start[batch] + i * nn + j] = sum


def _percentile_nearest(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        raise ValueError("empty sample")
    xs = np.sort(values)
    idx = int(np.ceil(q / 100.0 * xs.size) - 1)
    idx = min(max(idx, 0), xs.size - 1)
    return float(xs[idx])


def summarize(samples_s: list[float]) -> dict:
    arr = np.asarray(samples_s, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median_us": float(np.median(arr) * 1e6),
        "p95_us": _percentile_nearest(arr, 95.0) * 1e6,
        "min_us": float(arr.min() * 1e6),
        "max_us": float(arr.max() * 1e6),
        "samples_us": [float(v * 1e6) for v in arr],
    }


def sync() -> None:
    wp.synchronize_device(DEVICE)


def time_calls(fn, warmup: int, samples: int) -> dict:
    for _ in range(warmup):
        fn()
    sync()
    measured: list[float] = []
    for _ in range(samples):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        measured.append(time.perf_counter() - t0)
    return summarize(measured)


def gemm_numpy(A, B, m, n, p, transpose_A, transpose_B) -> np.ndarray:
    AA = A.reshape(p, m).T if transpose_A else A.reshape(m, p)
    BB = B.reshape(n, p).T if transpose_B else B.reshape(p, n)
    return AA @ BB


def make_batch(n_art: int, m: int, n: int, p: int, rng: np.random.Generator, transpose_A: bool, transpose_B: bool):
    a_count = (m * p) if not transpose_A else (p * m)
    b_count = (p * n) if not transpose_B else (n * p)
    # Storage layout matches dense_gemm indexing, so element count is m*p and p*n either way.
    a_count = m * p
    b_count = p * n
    c_count = m * n
    A = rng.standard_normal(n_art * a_count).astype(np.float32)
    B = rng.standard_normal(n_art * b_count).astype(np.float32)
    ms = np.full(n_art, m, dtype=np.int32)
    ns = np.full(n_art, n, dtype=np.int32)
    ps = np.full(n_art, p, dtype=np.int32)
    A_start = (np.arange(n_art, dtype=np.int32) * a_count)
    B_start = (np.arange(n_art, dtype=np.int32) * b_count)
    C_start = (np.arange(n_art, dtype=np.int32) * c_count)
    return {
        "A": A,
        "B": B,
        "m": ms,
        "n": ns,
        "p": ps,
        "A_start": A_start,
        "B_start": B_start,
        "C_start": C_start,
        "c_count": n_art * c_count,
        "m_i": m,
        "n_i": n,
        "p_i": p,
    }


def to_wp(host, dtype, requires_grad=False):
    return wp.array(host, dtype=dtype, device=DEVICE, requires_grad=requires_grad)


def launch_serial(batch, A, B, C, transpose_A, transpose_B):
    wp.launch(
        eval_dense_gemm_batched,
        dim=batch["m"].shape[0],
        inputs=[
            batch["m_wp"],
            batch["n_wp"],
            batch["p_wp"],
            transpose_A,
            transpose_B,
            batch["A_start_wp"],
            batch["B_start_wp"],
            batch["C_start_wp"],
            A,
            B,
        ],
        outputs=[C],
        device=DEVICE,
    )


def launch_element(batch, A, B, C, transpose_A, transpose_B):
    n_art = int(batch["m"].shape[0])
    # Uniform shapes in this sweep: exact output count, no masked threads.
    wp.launch(
        eval_dense_gemm_element,
        dim=(n_art, batch["m_i"] * batch["n_i"]),
        inputs=[
            batch["m_wp"],
            batch["n_wp"],
            batch["p_wp"],
            transpose_A,
            transpose_B,
            batch["A_start_wp"],
            batch["B_start_wp"],
            batch["C_start_wp"],
            A,
            B,
        ],
        outputs=[C],
        device=DEVICE,
    )


def forward_max_abs(batch, transpose_A, transpose_B) -> dict:
    A = to_wp(batch["A"], wp.float32)
    B = to_wp(batch["B"], wp.float32)
    C_s = wp.zeros(batch["c_count"], dtype=wp.float32, device=DEVICE)
    C_e = wp.zeros(batch["c_count"], dtype=wp.float32, device=DEVICE)
    launch_serial(batch, A, B, C_s, transpose_A, transpose_B)
    launch_element(batch, A, B, C_e, transpose_A, transpose_B)
    sync()
    hs = C_s.numpy()
    he = C_e.numpy()
    refs = []
    m, n, p = batch["m_i"], batch["n_i"], batch["p_i"]
    a_count = m * p
    b_count = p * n
    c_count = m * n
    for i in range(batch["m"].shape[0]):
        refs.append(
            gemm_numpy(
                batch["A"][i * a_count : (i + 1) * a_count],
                batch["B"][i * b_count : (i + 1) * b_count],
                m,
                n,
                p,
                transpose_A,
                transpose_B,
            ).reshape(-1)
        )
    ref = np.concatenate(refs).astype(np.float32)
    return {
        "serial_vs_numpy_max_abs": float(np.max(np.abs(hs - ref))),
        "element_vs_numpy_max_abs": float(np.max(np.abs(he - ref))),
        "serial_vs_element_max_abs": float(np.max(np.abs(hs - he))),
    }


def backward_max_abs(batch, transpose_A, transpose_B) -> dict:
    """Seed dC = 1 and compare input grads. One articulation to keep the tape small."""
    m, n, p = batch["m_i"], batch["n_i"], batch["p_i"]
    a_count = m * p
    b_count = p * n
    c_count = m * n
    A_h = batch["A"][:a_count].copy()
    B_h = batch["B"][:b_count].copy()
    one = {
        "m": np.array([m], dtype=np.int32),
        "n": np.array([n], dtype=np.int32),
        "p": np.array([p], dtype=np.int32),
        "A_start": np.array([0], dtype=np.int32),
        "B_start": np.array([0], dtype=np.int32),
        "C_start": np.array([0], dtype=np.int32),
        "m_i": m,
        "n_i": n,
        "p_i": p,
    }
    one["m_wp"] = to_wp(one["m"], wp.int32)
    one["n_wp"] = to_wp(one["n"], wp.int32)
    one["p_wp"] = to_wp(one["p"], wp.int32)
    one["A_start_wp"] = to_wp(one["A_start"], wp.int32)
    one["B_start_wp"] = to_wp(one["B_start"], wp.int32)
    one["C_start_wp"] = to_wp(one["C_start"], wp.int32)

    def run(launch):
        A = to_wp(A_h, wp.float32, requires_grad=True)
        B = to_wp(B_h, wp.float32, requires_grad=True)
        C = wp.zeros(c_count, dtype=wp.float32, device=DEVICE, requires_grad=True)
        tape = wp.Tape()
        with tape:
            launch(one, A, B, C, transpose_A, transpose_B)
        tape.backward(grads={C: wp.ones(c_count, dtype=wp.float32, device=DEVICE)})
        sync()
        gA = A.grad.numpy().copy()
        gB = B.grad.numpy().copy()
        tape.zero()
        return gA, gB

    gA_s, gB_s = run(launch_serial)
    gA_e, gB_e = run(launch_element)
    # Numeric Jacobian-vector via central differences on a random projection (forward-only check of grads).
    rng = np.random.default_rng(123)
    vA = rng.standard_normal(a_count).astype(np.float32)
    vB = rng.standard_normal(b_count).astype(np.float32)
    eps = np.float32(1e-3)

    def prod(Ah, Bh):
        return gemm_numpy(Ah, Bh, m, n, p, transpose_A, transpose_B).reshape(-1).astype(np.float32)

    base_plus = prod(A_h + eps * vA, B_h + eps * vB)
    base_minus = prod(A_h - eps * vA, B_h - eps * vB)
    fd = (base_plus - base_minus) / (np.float32(2.0) * eps)
    directional_serial = float(np.sum(gA_s * vA) + np.sum(gB_s * vB))
    directional_fd = float(np.sum(fd))
    return {
        "articulations_checked": 1,
        "grad_A_max_abs": float(np.max(np.abs(gA_s - gA_e))),
        "grad_B_max_abs": float(np.max(np.abs(gB_s - gB_e))),
        "grad_A_max_abs_value_serial": float(np.max(np.abs(gA_s))),
        "grad_B_max_abs_value_serial": float(np.max(np.abs(gB_s))),
        "directional_serial": directional_serial,
        "directional_fd": directional_fd,
        "directional_abs_diff": abs(directional_serial - directional_fd),
    }


def tile_vs_block_diag(n_art: int, n_joints: int, n_dofs: int, rng: np.random.Generator) -> dict:
    """Tile kernel assumes a 6x6 block-diagonal spatial mass. Compare to NumPy on that structure."""
    m = 6 * n_joints
    J = rng.standard_normal((n_art, m, n_dofs)).astype(np.float32)
    M = np.zeros((n_art, m, m), dtype=np.float32)
    for a in range(n_art):
        for j in range(n_joints):
            block = rng.standard_normal((6, 6)).astype(np.float32)
            block = block @ block.T + np.eye(6, dtype=np.float32)
            M[a, j * 6 : (j + 1) * 6, j * 6 : (j + 1) * 6] = block
    kernel = create_inertia_matrix_kernel(n_joints, n_dofs)
    J_wp = wp.array(J, dtype=wp.float32, device=DEVICE)
    M_wp = wp.array(M, dtype=wp.float32, device=DEVICE)
    H_wp = wp.zeros((n_art, n_dofs, n_dofs), dtype=wp.float32, device=DEVICE)
    wp.launch_tiled(
        kernel,
        dim=n_art,
        inputs=[J_wp, M_wp],
        outputs=[H_wp],
        device=DEVICE,
        block_dim=256,
    )
    sync()
    H = H_wp.numpy()
    max_abs = 0.0
    for a in range(n_art):
        ref = J[a].T @ M[a] @ J[a]
        max_abs = max(max_abs, float(np.max(np.abs(H[a] - ref))))
    return {"max_abs": max_abs, "kernel_block_dim": 256}


def time_tile(n_art: int, n_joints: int, n_dofs: int, rng: np.random.Generator) -> dict:
    m = 6 * n_joints
    J = rng.standard_normal((n_art, m, n_dofs)).astype(np.float32)
    M = rng.standard_normal((n_art, m, m)).astype(np.float32)
    kernel = create_inertia_matrix_kernel(n_joints, n_dofs)
    J_wp = wp.array(J, dtype=wp.float32, device=DEVICE)
    M_wp = wp.array(M, dtype=wp.float32, device=DEVICE)
    H_wp = wp.zeros((n_art, n_dofs, n_dofs), dtype=wp.float32, device=DEVICE)

    def launch():
        wp.launch_tiled(
            kernel,
            dim=n_art,
            inputs=[J_wp, M_wp],
            outputs=[H_wp],
            device=DEVICE,
            block_dim=256,
        )

    # Compile outside the timed window.
    launch()
    sync()
    stats = time_calls(launch, WARMUP, SAMPLES)
    stats["host_launches_per_call"] = 1
    stats["note"] = (
        "One tiled launch computes H = J^T (M_blockdiag J), not the two dense GEMMs. "
        "M is filled densely here so the timing includes full tile loads; correctness is separate."
    )
    return stats


def build_chain(n_art: int, n_joints: int, requires_grad: bool):
    builder = newton.ModelBuilder()
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.05)
    for a in range(n_art):
        parent = -1
        joints = []
        for j in range(n_joints):
            child = builder.add_link(
                xform=wp.transform((float(a) * 1.5, 0.0, float(j) * 0.4), wp.quat_identity()),
                com=wp.vec3(0.0, 0.0, 0.0),
                inertia=inertia,
                mass=1.0,
            )
            jid = builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=wp.vec3(0.0, 1.0, 0.0),
                armature=0.01,
                damping=0.1,
            )
            joints.append(jid)
            parent = child
        builder.add_articulation(joints)
    model = builder.finalize(device=DEVICE, requires_grad=requires_grad)
    return model


def init_state(model, seed: int):
    rng = np.random.default_rng(seed)
    state = model.state()
    q = model.joint_q.numpy()
    qd = model.joint_qd.numpy()
    q[:] = rng.uniform(-0.2, 0.2, size=q.shape).astype(np.float32)
    qd[:] = rng.uniform(-0.2, 0.2, size=qd.shape).astype(np.float32)
    model.joint_q.assign(q)
    model.joint_qd.assign(qd)
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def time_solver(n_art: int, n_joints: int, use_tile: bool, fuse: bool) -> dict:
    model = build_chain(n_art, n_joints, requires_grad=False)
    solver = newton.solvers.SolverFeatherstone(
        model,
        angular_damping=0.05,
        update_mass_matrix_interval=1,
        use_tile_gemm=use_tile,
        fuse_cholesky=fuse,
    )
    state_in = init_state(model, seed=7)
    state_out = model.state()
    control = model.control()
    dt = 1.0 / 600.0

    def step():
        nonlocal state_in, state_out
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in

    # Compile / settle graphs outside the sample window's first-call tax: still included in warmup.
    stats = time_calls(step, SOLVER_WARMUP, SOLVER_SAMPLES)
    stats["dof_count"] = int(model.joint_dof_count)
    stats["joint_count"] = int(model.joint_count)
    stats["articulation_count"] = int(model.articulation_count)
    stats["use_tile_gemm"] = use_tile
    stats["fuse_cholesky"] = fuse
    stats["host_note"] = (
        "Each sample is one SolverFeatherstone.step including FK, RNEA, both GEMMs (or the tile kernel), "
        "Cholesky, and the solve. update_mass_matrix_interval=1."
    )
    q = state_in.joint_q.numpy().copy()
    qd = state_in.joint_qd.numpy().copy()
    return stats, q, qd


def solver_forward_and_backward() -> dict:
    report: dict = {"forward": [], "backward": []}
    # Tile solver path asserts H is (arts, 18, 18). 18 revolute joints => 18 DOF.
    configs = [
        (1, 2, False),
        (64, 2, False),
        (1, 7, False),
        (32, 7, False),
        (1, 23, False),
        (8, 23, False),
        (1, 18, False),
        (1, 18, True),
        (32, 18, False),
        (32, 18, True),
        (128, 18, False),
        (128, 18, True),
    ]
    trajectories = {}
    for n_art, n_j, tile in configs:
        key = f"arts={n_art},joints={n_j},tile={tile},fuse={True}"
        try:
            stats, q, qd = time_solver(n_art, n_j, tile, True)
            trajectories[(n_art, n_j, tile)] = (q, qd)
            report["forward"].append({"config": key, "status": "ok", "timing": stats})
        except Exception as exc:  # noqa: BLE001 — record the exact failure, keep going
            report["forward"].append({"config": key, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    for n_art, n_j in ((1, 18), (32, 18), (128, 18)):
        a = trajectories.get((n_art, n_j, False))
        b = trajectories.get((n_art, n_j, True))
        if a is None or b is None:
            report["forward"].append(
                {
                    "config": f"compare arts={n_art} joints={n_j}",
                    "status": "skipped",
                    "error": "one side missing",
                }
            )
            continue
        # Same number of steps, but independent RNG re-init. Re-run paired from identical state.
        try:
            diff = paired_solver_diff(n_art, n_j, steps=20)
            report["forward"].append({"config": f"paired_diff arts={n_art} joints={n_j} steps=20", **diff})
        except Exception as exc:  # noqa: BLE001
            report["forward"].append(
                {
                    "config": f"paired_diff arts={n_art} joints={n_j}",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    for n_art, n_j, tile in ((1, 4, False), (1, 18, False), (1, 18, True)):
        try:
            report["backward"].append(solver_backward_once(n_art, n_j, tile))
        except Exception as exc:  # noqa: BLE001
            report["backward"].append(
                {
                    "config": f"arts={n_art},joints={n_j},tile={tile}",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return report


def paired_solver_diff(n_art: int, n_joints: int, steps: int) -> dict:
    model = build_chain(n_art, n_joints, requires_grad=False)
    q0 = model.joint_q.numpy()
    qd0 = model.joint_qd.numpy()
    rng = np.random.default_rng(11)
    q0[:] = rng.uniform(-0.15, 0.15, size=q0.shape).astype(np.float32)
    qd0[:] = rng.uniform(-0.15, 0.15, size=qd0.shape).astype(np.float32)

    def run(tile: bool):
        model.joint_q.assign(q0)
        model.joint_qd.assign(qd0)
        solver = newton.solvers.SolverFeatherstone(model, use_tile_gemm=tile, fuse_cholesky=True)
        s0 = model.state()
        s1 = model.state()
        s0.joint_q.assign(q0)
        s0.joint_qd.assign(qd0)
        newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
        control = model.control()
        dt = 1.0 / 600.0
        for _ in range(steps):
            solver.step(s0, s1, control, None, dt)
            s0, s1 = s1, s0
        sync()
        return s0.joint_q.numpy().copy(), s0.joint_qd.numpy().copy()

    q_s, qd_s = run(False)
    q_t, qd_t = run(True)
    return {
        "status": "ok",
        "q_max_abs": float(np.max(np.abs(q_s - q_t))),
        "qd_max_abs": float(np.max(np.abs(qd_s - qd_t))),
    }


def solver_backward_once(n_art: int, n_joints: int, tile: bool) -> dict:
    model = build_chain(n_art, n_joints, requires_grad=True)
    solver = newton.solvers.SolverFeatherstone(model, use_tile_gemm=tile, fuse_cholesky=True)
    rng = np.random.default_rng(3)
    q = rng.uniform(-0.1, 0.1, size=model.joint_dof_count).astype(np.float32)
    # joint_q may be longer than dof count (not for revolute).
    q_full = np.zeros(model.joint_coord_count, dtype=np.float32)
    qd_full = np.zeros(model.joint_dof_count, dtype=np.float32)
    q_full[: q.shape[0]] = q[: q_full.shape[0]]
    qd_full[:] = rng.uniform(-0.1, 0.1, size=qd_full.shape).astype(np.float32)
    model.joint_q.assign(q_full)
    model.joint_qd.assign(qd_full)
    s0 = model.state(requires_grad=True)
    s1 = model.state(requires_grad=True)
    s0.joint_q.assign(q_full)
    s0.joint_qd.assign(qd_full)
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    control = model.control()
    dt = 1.0 / 600.0
    tape = wp.Tape()
    with tape:
        solver.step(s0, s1, control, None, dt)
        # Scalar loss on the outgoing generalized velocity.
        loss = wp.zeros(1, dtype=wp.float32, device=DEVICE, requires_grad=True)
    # Manual loss: sum of joint_qd via numpy-side launch of a tiny kernel would need a kernel.
    # Use tape.backward on joint_qd directly.
    tape.backward(grads={s1.joint_qd: wp.ones_like(s1.joint_qd)})
    sync()
    gq = s0.joint_q.grad
    gqd = s0.joint_qd.grad
    gq_np = None if gq is None else gq.numpy()
    gqd_np = None if gqd is None else gqd.numpy()
    tape.zero()
    return {
        "config": f"arts={n_art},joints={n_joints},tile={tile}",
        "status": "ok",
        "loss_unused_buffer": True,
        "grad_q_max_abs": None if gq_np is None else float(np.max(np.abs(gq_np))),
        "grad_qd_max_abs": None if gqd_np is None else float(np.max(np.abs(gqd_np))),
        "grad_q_finite": None if gq_np is None else bool(np.isfinite(gq_np).all()),
        "grad_qd_finite": None if gqd_np is None else bool(np.isfinite(gqd_np).all()),
    }


def gpu_util_during(fn_seconds: float, body) -> dict:
    """Sample nvidia-smi while ``body`` runs. CUPTI is not used."""
    stop = threading.Event()
    samples: list[str] = []

    def reader():
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "dmon",
                "-s",
                "u",
                "-d",
                "1",
                "-c",
                str(max(3, int(fn_seconds) + 2)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            samples.append(line.rstrip("\n"))
            if stop.is_set():
                break
        proc.kill()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    time.sleep(0.3)
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < fn_seconds:
        body()
        iters += 1
    sync()
    stop.set()
    th.join(timeout=3.0)
    utils = []
    for line in samples:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                utils.append(int(parts[1]))
            except ValueError:
                pass
    return {
        "seconds": fn_seconds,
        "iterations": iters,
        "nvidia_smi_dmon_lines": samples,
        "sm_util_samples_percent": utils,
        "sm_util_median_percent": None if not utils else float(np.median(utils)),
        "sm_util_max_percent": None if not utils else int(max(utils)),
        "cupti": "unavailable",
        "nsys": "unavailable",
    }


def main() -> None:
    wp.init()
    rng = np.random.default_rng(0)
    device = wp.get_device(DEVICE)
    props = {
        "name": device.name,
        "arch": str(device.arch),
        "sm_count": int(getattr(device, "sm_count", -1)),
        "warp": wp.config.version if hasattr(wp.config, "version") else "unknown",
    }
    try:
        import importlib.metadata

        props["warp"] = importlib.metadata.version("warp-lang")
    except Exception:
        props["warp"] = "1.17.0-runtime"

    shapes = [
        # (label, m, n, p, transpose_A, transpose_B, n_joints or None, n_dofs or None)
        ("gemm1_2dof", 12, 2, 12, False, False, 2, 2),
        ("gemm2_2dof", 2, 2, 12, True, False, None, None),
        ("gemm1_7dof", 42, 7, 42, False, False, 7, 7),
        ("gemm2_7dof", 7, 7, 42, True, False, None, None),
        ("gemm1_18dof", 108, 18, 108, False, False, 18, 18),
        ("gemm2_18dof", 18, 18, 108, True, False, None, None),
        ("gemm1_issue_23dof", 150, 23, 150, False, False, 25, 23),
        ("gemm2_issue_23dof", 23, 23, 150, True, False, None, None),
        ("gemm1_1dof", 6, 1, 6, False, False, 1, 1),
        ("gemm2_1dof", 1, 1, 6, True, False, None, None),
    ]
    art_counts = [1, 8, 64, 256]

    isolated = []
    for label, m, n, p, tA, tB, n_joints, n_dofs in shapes:
        for n_art in art_counts:
            case = {
                "label": label,
                "articulations": n_art,
                "m": m,
                "n": n,
                "p": p,
                "transpose_A": tA,
                "transpose_B": tB,
                "output_elements_total": n_art * m * n,
            }
            try:
                host = make_batch(n_art, m, n, p, rng, tA, tB)
                host["m_wp"] = to_wp(host["m"], wp.int32)
                host["n_wp"] = to_wp(host["n"], wp.int32)
                host["p_wp"] = to_wp(host["p"], wp.int32)
                host["A_start_wp"] = to_wp(host["A_start"], wp.int32)
                host["B_start_wp"] = to_wp(host["B_start"], wp.int32)
                host["C_start_wp"] = to_wp(host["C_start"], wp.int32)
                A = to_wp(host["A"], wp.float32)
                B = to_wp(host["B"], wp.float32)
                C_s = wp.zeros(host["c_count"], dtype=wp.float32, device=DEVICE)
                C_e = wp.zeros(host["c_count"], dtype=wp.float32, device=DEVICE)
                # Correctness on this same buffer (first few arts if huge — full buffer is fine).
                case["forward"] = forward_max_abs(host, tA, tB)
                if n_art == 1:
                    case["backward"] = backward_max_abs(host, tA, tB)
                else:
                    case["backward"] = {"status": "skipped_non_unit_batch"}

                def run_s(A=A, B=B, C_s=C_s, host=host, tA=tA, tB=tB):
                    launch_serial(host, A, B, C_s, tA, tB)

                def run_e(A=A, B=B, C_e=C_e, host=host, tA=tA, tB=tB):
                    launch_element(host, A, B, C_e, tA, tB)

                # Prime both kernels before timed section.
                run_s()
                run_e()
                sync()
                case["serial"] = time_calls(run_s, WARMUP, SAMPLES)
                case["serial"]["host_launches_per_call"] = 1
                case["element"] = time_calls(run_e, WARMUP, SAMPLES)
                case["element"]["host_launches_per_call"] = 1
                med_s = case["serial"]["median_us"]
                med_e = case["element"]["median_us"]
                case["element_over_serial_median_ratio"] = None if med_s == 0 else med_e / med_s
                case["faster_median"] = "element" if med_e < med_s else "serial"
                if n_joints is not None:
                    case["tile"] = time_tile(n_art, n_joints, n_dofs, rng)
                    if n_art <= 4:
                        case["tile_forward"] = tile_vs_block_diag(n_art, n_joints, n_dofs, rng)
                case["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                case["status"] = "error"
                case["error"] = f"{type(exc).__name__}: {exc}"
            isolated.append(case)
            print(
                f"{label} arts={n_art} {case['status']}"
                + (
                    f" serial_med={case['serial']['median_us']:.2f}us elem_med={case['element']['median_us']:.2f}us"
                    if case.get("status") == "ok"
                    else f" {case.get('error')}"
                ),
                flush=True,
            )

    print("solver...", flush=True)
    solver_report = solver_forward_and_backward()

    # Sustained util on the issue shape and on a wide small-batch shape.
    print("util...", flush=True)
    util = {}
    try:
        host = make_batch(1, 150, 23, 150, rng, False, False)
        host["m_wp"] = to_wp(host["m"], wp.int32)
        host["n_wp"] = to_wp(host["n"], wp.int32)
        host["p_wp"] = to_wp(host["p"], wp.int32)
        host["A_start_wp"] = to_wp(host["A_start"], wp.int32)
        host["B_start_wp"] = to_wp(host["B_start"], wp.int32)
        host["C_start_wp"] = to_wp(host["C_start"], wp.int32)
        A = to_wp(host["A"], wp.float32)
        B = to_wp(host["B"], wp.float32)
        C = wp.zeros(host["c_count"], dtype=wp.float32, device=DEVICE)

        def s():
            launch_serial(host, A, B, C, False, False)

        def e():
            launch_element(host, A, B, C, False, False)

        s()
        e()
        sync()
        util["issue_gemm1_arts1_serial"] = gpu_util_during(4.0, s)
        util["issue_gemm1_arts1_element"] = gpu_util_during(4.0, e)

        host2 = make_batch(256, 12, 2, 12, rng, False, False)
        host2["m_wp"] = to_wp(host2["m"], wp.int32)
        host2["n_wp"] = to_wp(host2["n"], wp.int32)
        host2["p_wp"] = to_wp(host2["p"], wp.int32)
        host2["A_start_wp"] = to_wp(host2["A_start"], wp.int32)
        host2["B_start_wp"] = to_wp(host2["B_start"], wp.int32)
        host2["C_start_wp"] = to_wp(host2["C_start"], wp.int32)
        A2 = to_wp(host2["A"], wp.float32)
        B2 = to_wp(host2["B"], wp.float32)
        C2 = wp.zeros(host2["c_count"], dtype=wp.float32, device=DEVICE)

        def s2():
            launch_serial(host2, A2, B2, C2, False, False)

        def e2():
            launch_element(host2, A2, B2, C2, False, False)

        s2()
        e2()
        sync()
        util["gemm1_2dof_arts256_serial"] = gpu_util_during(4.0, s2)
        util["gemm1_2dof_arts256_element"] = gpu_util_during(4.0, e2)
    except Exception as exc:  # noqa: BLE001
        util["error"] = f"{type(exc).__name__}: {exc}"

    # Crossover: among ok GEMM1 rows, smallest output-element count where element median < serial median,
    # and any art-count where serial wins, per shape family.
    crossover = []
    families: dict[str, list] = {}
    for case in isolated:
        if case.get("status") != "ok":
            continue
        families.setdefault(case["label"], []).append(case)
    for label, rows in families.items():
        rows_sorted = sorted(rows, key=lambda r: r["output_elements_total"])
        serial_wins = [r for r in rows_sorted if r["faster_median"] == "serial"]
        element_wins = [r for r in rows_sorted if r["faster_median"] == "element"]
        crossover.append(
            {
                "label": label,
                "serial_faster_at_articulations": [r["articulations"] for r in serial_wins],
                "element_faster_at_articulations": [r["articulations"] for r in element_wins],
                "ratios_element_over_serial": [
                    {"articulations": r["articulations"], "ratio": r["element_over_serial_median_ratio"]}
                    for r in rows_sorted
                ],
            }
        )

    payload = {
        "issue": "https://github.com/newton-physics/newton/issues/4265",
        "device": props,
        "method": {
            "warmup_isolated": WARMUP,
            "samples_isolated": SAMPLES,
            "warmup_solver": SOLVER_WARMUP,
            "samples_solver": SOLVER_SAMPLES,
            "sync": "wp.synchronize_device before perf_counter and after each launch/step",
            "p95": "nearest-rank on the sorted samples",
            "launch_count": "host wp.launch count only; nsys and cupti python module are not installed",
        },
        "isolated_gemm": isolated,
        "crossover": crossover,
        "solver": solver_report,
        "gpu_util": util,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
