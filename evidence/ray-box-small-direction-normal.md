# Box ray normal CPU validation receipt

Date: 2026-10-12. Downstream evidence only; no upstream or GPU acceptance.

Tested upstream base: `624005a4478dc385180dd28775ca3dda96e93a69`.
Reviewed implementation commit: `878ae1eb72016834ffb17cb62f11b8feb97b8f76`.
Branch: `fix/ray-box-small-direction-normal`.

## Witness and bounded correction

Half-extents `(1,1,1)`, origin `(1 + 2**-23,0,0)`, direction
`(-2**-22,1,0)` produce `t=0.5` and point `(1,0.5,0)`.
The exact base returns normal `(0,0,0)` through the existing
`kernel_test_geom`; the correction returns outward normal `(1,0,0)`.

Normal reconstruction now considers `abs(direction[i]) >= MINVAL`, matching
slab selection's complementary `< MINVAL` parallel branch. Solved distance,
face bounds, EPSILON time matching and normalized tied-face accumulation
remain unchanged. Existing large-distance inverse-multiply versus division
and absolute-time-tolerance limitations remain outside this bounded fix.

## Counts and coverage

Two new unique registered CPU unittest tests contain 52 subcases:

- `test_ray_intersect_box_normals_cpu`: 50 direct-kernel subcases through
  `kernel_test_geom`, which dispatches to the shape intersection routine.
  These comprise 24 face cases (three axes, two signs, exponents -22/-18,
  entry/inside exit), 24 ordinary edge/corner ties, and two small-component
  edge/corner grazing ties.
- `test_intersect_ray_box_small_direction_cpu`: two public
  `newton.intersect_ray` subcases, with `fast_math=False` and `True`, checking
  distance, shape ID and normal.

RED replay retained the new tests but temporarily substituted the exact base
raycast source, restoring the corrected source in a `finally` block.
Result: **two unique tests run; 16 failing subcases; process exit 1**.
The 16 failures are 12 face cases with exponent -22 (three axes, two signs,
entry/exit), two small-component ties, and two public-query modes.
The other 36 subcases passed. This is not 16 unique failing tests.

GREEN: the complete raycast module passed **20 unique CPU tests**, including
both new tests and all 52 new subcases; exit 0. Adjacent BVH and collision
primitive modules passed **16 unique CPU tests**; exit 0. These counts are
separate from the independent critic's probes below.

Full `uvx pre-commit run -a` passed 14 hooks, with two hooks skipped because
no applicable files existed. Earlier formatter runs changed only layout;
the final rerun passed. `git diff --check` passed. No lockfile changes.

## Environment and reproducible commands

Linux CPU, CPython 3.12.14, Warp 1.18.0, NumPy 2.5.4. CUDA driver unavailable.
Warp and NumPy installed from official wheels in about 10 seconds, with
approximately 158 MiB downloaded. Combined provisioned dependency, hook and
kernel caches occupied approximately 674 MiB, below the 1 GiB bound.
No OS, GPU or model installation was performed.

The original commands used sandbox-local virtual environment/cache paths.
The portable commands below replace those paths with checkout-local paths
without changing the interpreter, dependencies, test loader or test names.
Run from the checkout root; retain a separate checkout for RED replay.

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
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'newton.tests.test_raycast',
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
uvx pre-commit run -a
git diff --check
```

For the adjacent run, replace the loader's module list with
`['newton.tests.test_bvh', 'newton.tests.test_collision_primitives']`.
For the RED run, select only
`newton.tests.test_raycast.TestRaycast.test_ray_intersect_box_normals_cpu`
and
`newton.tests.test_raycast.TestRaycast.test_intersect_ray_box_small_direction_cpu`,
with the tested base's `newton/_src/geometry/raycast.py` and the final tests.
The published branch retains only the corrected source.

## Independent xhigh critic

An independent xhigh critic read the complete frozen diff and Newton's agent,
coding, review and applicable skill instructions. Verdict: **PASS; Fits**,
with no supported material findings. Both new registered CPU tests were
independently rerun successfully. An additional **72 direct-kernel CPU probe
cases** passed:

- 18 MINVAL controls: all axes/signs, just below, equal to and just above
  float32 MINVAL, using a representable thin box. Below-threshold rays miss;
  equality and above hit with the expected signed face normal.
- 48 small-component ties: all axes/signs, entry/exit, edge/corner,
  components `2**-22` and `2**-18`, expected normalized tied-face sum.
- Six zero-component, on-face, inward/outward and pointing-away controls.

These are additional probe cases, not 72 registered repository tests.
The review's SHA-256 hashes of the frozen files are:

```text
b1cd4494dbf484d9d29cbad3e05a8dfc71f40daf0f5dca46e7174417f080db10  newton/_src/geometry/raycast.py
1f12f66ce03f0932afda7a93e2d42d63fcd6834f186ce2a000deb0332ea8b1a6  newton/tests/test_raycast.py
ee75556ec83966acc98e925bf5299580fe0500d9d7fd7cd70cf9520725e18229  changelog/+ray-box-small-direction-72f6a93b.fixed.md
```

No CUDA execution, GPU acceptance, performance qualification or universal
small-direction robustness claim follows from this receipt.

## Ownership, privacy and routing

Credit original raycast contributors nvtw, Daniela Hase, Mustafa H and
Alain Denzler. This narrow correction does not transfer their ownership.
RoboX2020's upstream #4624 opacity/BVH work, eric-heiden's #4463 heightfield
caps and dot's downstream cone commit `05912514` remain separate.
Canceled #4612 evidence, disproven RK4 work and #4622/#4630 scopes were untouched.

Immediately before publication, Git remote heads still matched the tested
base on upstream and fork main. Bounded upstream open ray PR, exact
`ray_intersect_box` issue/PR, and downstream factory box-normal claim searches
found no competing matching item. Registration PR #56 remains pending;
#66/#14/#64 are coordination links, not registered automation routing.

The complete publication diff was reviewed for private paths, credentials,
secrets and generated artifacts. Only source, tests, changelog and this
portable prose receipt are published. Raw logs, sandbox paths, dependency
caches and generated patch exports remain local. Existing public Git author
metadata is retained. No origin PR/comment, main/force push, human signoff or
CLA attestation is authorized or implied.
