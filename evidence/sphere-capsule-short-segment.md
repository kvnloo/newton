# Sphere–capsule short-segment CPU validation receipt

Date: 2026-10-13. This is downstream CPU evidence, not GPU or full-solver acceptance.

Tested upstream base: `1624cc70e2e030ca3dca6772b4243900202d24fb`.
Reviewed implementation commit: `8e135adef739488778acac7ceec50c98385c2101`.
Branch: `fix/sphere-capsule-segment-projection`.

## Witness and correction

For a capsule centered at `(0,0,0)`, axis `+Z`, half-length `0.0005`,
radius `0.00025`, and a sphere centered at `(0,0.00045,0)` with radius
`0.00025`, the signed contact distance should be `-0.00005`, midpoint
`(0,0.000225,0)`, and sphere-to-capsule normal `(0,-1,0)`. The exact base,
through the existing `test_sphere_capsule_kernel` on CPU, reported distance
`+1.4781486243009567e-05` and a tilted normal. Reversing the capsule axis
reversed the spurious normal's Z component.

`closest_segment_point` formerly added `1e-6` to the squared segment length,
biasing the projection for short capsules. The correction divides by the
exact nonzero squared length and returns endpoint `a` for a zero-length
segment. The helper signature, callers, and dependencies are unchanged.

## Regression and checks

The new registered unittest contains 15 analytic CPU subcases: 12 interior
cases across three scales, two axis signs, and two heights; two reversed-axis
endpoint cases; and one zero-length case. It checks signed distance, contact
midpoint, and normal using the existing public-contact test kernel. It does
not use the test helper whose formula repeats the old bias.

RED replay retained the final test while temporarily substituting the exact
base source, restoring the corrected source in a `finally` block: **one
unique test, 13 failing subcases of 15, exit 1**. GREEN: the complete
`newton.tests.test_collision_primitives` module passed **15 unique CPU tests**;
four targeted adjacent narrow-phase CPU tests passed. Full pre-commit passed
14 hooks, with two no-applicable-file skips. `git diff --check` passed.

The reproducible CPU test runner below assumes a compatible Python 3.12
environment with `warp-lang==1.18.0` and `numpy==2.5.4`, and runs from the
checkout root. The original validation used `uv run --no-project` with a
pre-existing CPU virtual environment and a separate Warp kernel cache.

```bash
uv run --no-project --python .validation/venv/bin/python python - <<'PY'
import unittest
import warp as wp

wp.config.kernel_cache_dir = '.validation/warp-cache'
suite = unittest.defaultTestLoader.loadTestsFromName(
    'newton.tests.test_collision_primitives'
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
uvx pre-commit run -a
git diff --check
```

For the adjacent run, replace `loadTestsFromName` with `loadTestsFromNames`
for these four methods of `newton.tests.test_narrow_phase.TestNarrowPhase`:
`test_sphere_capsule_cylinder_side`, `test_sphere_capsule_cap`,
`test_capsule_capsule_parallel`, and `test_capsule_capsule_crossed`.
For RED replay, select only
`newton.tests.test_collision_primitives.TestCollisionPrimitives.test_sphere_capsule_short_segment_analytic`
with the exact tested base's `collision_primitive.py` and the final test.
The published branch retains only the corrected source.

## Independent review and limits

An independent xhigh critic reviewed the complete frozen diff under Newton's
agent, coding, review, contribution, and changelog instructions. Verdict:
**Fits; no supported defects**. The critic independently passed the analytic
test, seven direct projection CPU probes and six segment-pair CPU probes.
The final reviewed SHA-256 hashes are:

```text
27d873949bed7f01f2b11753f278be9b590f26f64fcdc54e362ba2416578e66b  newton/_src/geometry/collision_primitive.py
ee690bdf405536e08a4bea14c8e306dcc28363d00cc6335ff7f4cd66e74c22de  newton/tests/test_collision_primitives.py
7951b859d96227832bd98c0d7e5e286ed483667fbf769ecf11c1898774784487  changelog/+sphere-capsule-short-segment-a37e1f94.fixed.md
```

Linux CPU, CPython 3.12.14, Warp 1.18.0, NumPy 2.5.4; CUDA driver unavailable.
The shared source file warns of MuJoCo-Warp use. `mujoco_warp` and `mujoco`
were absent from this environment, so that downstream runtime suite was not
run. GPU, full-solver compatibility, and performance remain unqualified.

Original primitive and MuJoCo-Warp contributors retain credit and ownership.
Open nvtw #4362 changes a separate capsule–cylinder path without changing
this helper; #4643 and #4227 are separate. This branch does not touch the
published ray-box or cylinder branches, other contributors' work, or canceled
evidence actions. Registration PR #56 remains pending, and factory #66/#14/#64
remain coordination records rather than registered automation routing.

Only source, test, changelog fragment, and this portable prose receipt are
published. Raw logs, caches, and sandbox-specific paths remain local. No
upstream PR/comment, main or force push, signoff, or CLA attestation is implied.
