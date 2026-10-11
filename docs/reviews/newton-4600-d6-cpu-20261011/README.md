# Newton #4600: independent CPU evidence

## Owner handoff

This packet supports Volodymyr Borysenko's existing [Newton PR #4600](https://github.com/newton-physics/newton/pull/4600), which addresses [issue #4540](https://github.com/newton-physics/newton/issues/4540), reported by [gyeomannvidia](https://github.com/gyeomannvidia). Implementation and upstream submission remain with the owner. It contains reproduction and review evidence only, with no production patch, new defect claim, upstream review approval, or GPU claim.

The independently rerun CPU evidence confirms the scoped correction: a two-angular-axis D6 joint initialized at both drive targets should remain at rest. All six ordered canonical axis pairs pass on the owner head; five fail on the comparison baseline. An additional physical oracle passes 15/15 on the owner head versus 3/15 on the baseline. The owner's focused test and 26 adjacent tests also pass.

Upstream CLA authorization remains unresolved. The current EasyCLA comment says: “The commit (c7a91b485e33ec489684c79b0deda2dd121c439f) is not authorized under a signed CLA.” This does not establish that the author never signed one. [Exact bot comment](https://github.com/newton-physics/newton/pull/4600#issuecomment-6069503235). The existing [minor test-docstring finding](https://github.com/newton-physics/newton/pull/4600#discussion_r4224394193) is already raised and is not duplicated here. Discussion status was read on 2026-10-11 at 00:02 UTC; these local tests do not establish current complete CI or merge readiness.

## Exact source identities

| Role | Commit |
| --- | --- |
| Owner head | `c7a91b485e33ec489684c79b0deda2dd121c439f` |
| Owner commit parent | `2e1e6b17c208b63132727dba53439122617d08ff` |
| Comparison main baseline | `a6e1649b112e3d962b35f7dad66780dc55150588` |

The owner-parent-to-head change is exactly one commit and three files: `kernels_body.py`, `test_joint_drive.py`, and `changelog/4540.fixed.md`. The baseline and owner parent have the identical pre-fix solver-kernel blob `deb49f0252ce46d1937dadfc1c99f55f65b6cda8`. The comparison main is not represented as the owner's PR merge-base; the snapshots differ elsewhere.

The patch changes the two-axis D6 relative quaternion to the declared joint-axis frame before XYZ decomposition, matching Newton's `invert_2d_rotational_dofs` convention. The correction fits the existing solver's responsibilities and adds no public API, configuration, dependency, or competing mechanism. No new material code defect was found within this scoped review. The original issue body and PR description were independently read on 2026-10-11 at 00:05 UTC. The issue's Y/Z CPU reproduction and the owner's stated scope match these frozen gates. The owner PR remains open and non-draft at the exact tested head; `primary-context.json` records the source metadata and verified contract.

## Frozen gates and results

| Gate | Baseline | Owner |
| --- | --- | --- |
| Original six ordered canonical pairs | 1 pass / 5 fail, exit 1 | 6 pass, exit 0 |
| Expanded physical oracle: 14 rest cases plus one live-drive control | 3 pass / 12 fail, exit 1 | 15 pass, exit 0 |
| Owner focused unittest | Not present in baseline | 1 passed |
| Owner joint-drive + joint-limit unittests | Not rerun on baseline | 26 passed |

Original witness: X/Y, Y/X, Y/Z, Z/Y, Z/X, and X/Z. Coordinates and targets `(0.2, 0.4)`; mass 1; identity inertia; gravity, attachment gains, and angular damping zero; stiffness 2; one `0.001`-second step; CPU; absolute speed tolerance `1e-8`, relative tolerance zero. The unchanged original witness SHA-256 is `ca0ba3a9770e051c48811e8f1314ff18d119c0bd45ddbbb8b9be7d5567d75e65`.

- Baseline maximum unwanted joint speed: `7.9999969e-4` rad/s.
- Owner maximum original-witness joint speed: `2.9802333e-10` rad/s.
- Rotated orthonormal pair `(1/sqrt(2), 1/sqrt(2), 0)` / `(0, 0, 1)`: baseline approximately `[0.00012788, 0.00051725]`; owner approximately `[-8.94e-11, -1.79e-10]` rad/s.

The expanded oracle freezes the same seven axis pairs with both identity and nontrivial parent/child anchor rotations. It asserts correct imports, normalized orthogonal axes, finalized axes, drive targets, and zero initial body velocity. It independently reconstructs the expected physical body orientation using NumPy quaternion products, without `eval_ik` or solver decomposition. After stepping it checks both generalized and physical body velocities at `1e-8`, and pose components at `1e-6`. A target offset of `+0.05` on the first X/Y coordinate must produce positive restoring velocity above `1e-6`; both snapshots give about `9.999994e-5` rad/s. No axis case is filtered out.

Across the expanded owner rest cases, maximum speed is `3.8743050e-10` rad/s, maximum initial orientation component error is `1.1567718e-7`, and maximum one-step pose component drift is `1.1920929e-7`. The latter checks allow float32 rounding using thresholds fixed before testing.

## Replay

Provide clean, already-available checkouts and an already-installed interpreter. The runner does not fetch source or install dependencies, and uses `uv` offline. Verified environment: CPython 3.12.14, uv 0.12.23, Warp 1.18.0, NumPy 2.5.3, MuJoCo 3.14.0, and mujoco-warp 3.14.0 on Linux CPU.

Set:

- `BASE_CHECKOUT`: clean checkout at the exact comparison baseline.
- `OWNER_CHECKOUT`: clean checkout at the exact owner head.
- `INTERPRETER`: existing Python executable with the required packages.
- Optional `UV_CACHE_DIR` and `WARP_CACHE_PATH`: existing caller-controlled cache locations.
- Optional `RESULTS_DIR`: caller-selected output directory.

Then run:

```bash
sha256sum -c SHA256SUMS
bash replay.sh
```

`replay.sh` verifies commits and clean sources; `run_verified.py` asserts that Newton imports from the intended checkout in the same Python process as each test. It runs the frozen scripts and focused unittest gates, checks expected exit codes and case counts, and never invokes the project-wide test runner.

## Scope and provenance

This is one-step, moderate-angle, normalized orthonormal-axis CPU evidence. It does not establish general correctness for arbitrary angles, non-orthogonal/degenerate axes, nonzero initial velocities or damping, nonzero attachment gains, long-horizon stability, autodiff, CUDA/GPU, every backend, or the full project suite. No performance claim is made. Adjacent MuJoCo tests emitted existing line-search warnings but passed.

The exact owner's AGENTS.md, Newton review skill, full coding and review guidelines, articulation/builder code, and changelog conventions were read. `git diff --check` passed for the exact owner commit; both source checkouts remained clean.

Published logs retain every result row but omit environment initialization, private cache locations/timings, and unrelated command diagnostics. `SOURCE_LOG_HASHES.json` preserves hashes of the original locally captured logs without publishing them. `SHA256SUMS` covers the distributed artifacts. Owner discussion status is recorded in `owner-status.json`, and the issue/PR primary context in `primary-context.json`. The portable replay itself was validated end-to-end; its gate outcomes are in `logs/replay-validation.log`.

No upstream post, source edit, cleanup, installation, new worktree, or push was performed for this evidence packet. The implementation credit belongs to Volodymyr Borysenko; the original report credit belongs to gyeomannvidia.
