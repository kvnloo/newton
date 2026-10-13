# Cylinder ray normal CPU evidence

Date: 2026-10-13. Tested upstream main:
`8157ac7e15374dbda4503eb1e2f7194af643e3f9`.
Implementation commit: `9c4f97b80676ea40065d7898eb89a57625a1aff2`.
Branch: `fix/cylinder-ray-normal-cap-selection`.

The existing `kernel_test_geom` in `newton/tests/test_raycast.py` uses
`GeoType.CYLINDER` and passes `(radius, half_height, barrel_radius)` in the
`size` vector. On CPU, radius and half-height 1, origin
`(2,0,1-2**-22)`, direction `(-1,0,0)` returned distance 1 and the
wrong `+Z` cap normal on the tested base. The correct curved-side normal is
`+X`. With both radius and half-height `2**-20`, origin `(2**-19,0,0)`
and direction `(-1,0,0)`, the base returned distance `2**-20` and zero
normal; the correct side normal is `+X`.

The fix records whether the winning cap is top or bottom when the intersection
distance is selected. A signed unit cap normal comes from that winner; a
straight side winner keeps its radial normal. When float32 rounds both cap
distances to an exact tie, the rule selects the cap approached by the ray.
For a barrel side winner, the preexisting barrel normal path, including its
near-cap proximity rule, is retained. Cap and side distance calculations,
barrel refinement and the public API are unchanged.

## RED and GREEN

Three new registered CPU unittest methods contain 25 subcases: 15 straight
cylinder cases (axis/sign, near-cap side, thin side/cap, inside exit, rotated
side, true caps and exact rim ties), eight barrel endpoint/near-rim controls,
and two public `newton.intersect_ray` settings (`fast_math=False/True`).

With these tests and the **exact base source** temporarily substituted, the
RED run executed three unique tests and failed **eight subcases**: six direct
straight-cylinder cases and both public-query modes. The other 17 subcases
passed. The base source was restored in a `finally` block after that run.
An intermediate provenance patch also failed barrel profile end and small
radius near-rim controls; those failures caused the deliberate preservation
of barrel-side normal behavior. No RED subcase count is represented as a
unique test count.

Final GREEN: the full raycast module passed **21 unique CPU tests**, including
all 25 new subcases. The adjacent BVH and collision primitive modules passed
**16 unique CPU tests**. Full `uvx pre-commit run -a`: 14 hooks passed, two
skipped for no applicable files; `git diff --check` passed. Earlier formatter
passes changed test layout; the final pass made no changes.

Independent xhigh review: **Fits; Requirements and Standards pass; no
remaining supported findings.** The critic independently passed five
cylinder-focused unittest methods, including the three new ones. Its
additional probes covered 14 direct thin/zero-height runs, eight public
thin/zero-height runs, 12 barrel rim runs and four public small-radius
near-rim runs in both fast-math modes. These probes are separate from the
registered test count. Exact reviewed SHA-256 file hashes:

```text
00d6565ad41a8c86e5e28bfaa5eba5d99aa5d5d58f20f632eadc5cb6aff929a3  newton/_src/geometry/raycast.py
9fb2c2129092d3734054a08cd8ae2a402b32ad656ded34cd3ea3b83d3e27ee50  newton/tests/test_raycast.py
d38086ae2499386a75080bed9a02ffe80e01f384667e5e32d87bc1034c9c1d5b  changelog/+cylinder-ray-surface-normal-1bc893af.fixed.md
```

## Environment and commands

CPU only; CPython 3.12.14, Warp 1.18.0 and NumPy 2.5.4 from the existing
wheel-only environment. No CUDA driver was available. No new dependencies,
OS/GPU/model installs or credentials were added. From the checkout root, the
same test loader can be run with local cache paths:

```bash
export UV_CACHE_DIR="$PWD/.validation/uv-cache"
export UV_TOOL_DIR="$PWD/.validation/uv-tools"
export PRE_COMMIT_HOME="$PWD/.validation/pre-commit"
uv venv --python 3.12 .validation/venv
uv pip install --python .validation/venv/bin/python --only-binary :all: warp-lang==1.18.0 numpy==2.5.4
uv run --no-project --python .validation/venv/bin/python python - <<'PY'
import unittest
import warp as wp
wp.config.kernel_cache_dir = '.validation/warp-cache'
suite = unittest.defaultTestLoader.loadTestsFromName('newton.tests.test_raycast')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
uvx pre-commit run -a
git diff --check
```

For adjacent tests, replace the loader argument with
`loadTestsFromNames(['newton.tests.test_bvh',
'newton.tests.test_collision_primitives'])`. For a RED replay, retain the
three new test methods and use `newton/_src/geometry/raycast.py` from the
tested base in a separate temporary checkout. The published branch contains
only the corrected source.

## Ownership and limits

Original raycast contributors nvtw, Daniela Hase, Mustafa H and Alain
Denzler retain credit; nvtw's merged #3842 is an attribution source. Open
#4624 opacity/BVH, #4463 heightfield cap, #4362 capsule-cylinder contact and
#4643 MPR/GJK changes are distinct. The published ray-box fix, cone work,
Hermes promotion work and held #4612 evidence were untouched.

The established barrel-side proximity normal remains as before. At very small
barrel end radius, float32 distance resolution may place a reported hit on
the axis; this change does not claim a more precise barrel normal or hit
distance. No CUDA, GPU, performance, full-solver or upstream acceptance claim
follows from CPU evidence. Registry PR #56 remains draft/pending; factory
#66/#14/#64 are coordination, not registered routing. No upstream PR/comment,
main/force push, cleanup, human signoff or CLA attestation is implied.
