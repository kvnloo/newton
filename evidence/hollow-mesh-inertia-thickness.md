# Hollow mesh inertia thickness: CPU validation receipt

Date: 2026-10-13. Tested upstream base:
`71eeba16a717d4dcf0b9dd7056a7b6f863054dc3`.
Reviewed implementation commit: `1bb00d2ecd1648defb91954b2435c024c0bef227`.
Branch: `fix/hollow-mesh-inertia-thickness` on `kvnloo/newton`.

## Witness and bounded change

The closed unit cube from `test_cube_mesh_inertia` has six unit-area faces.
Newton's two-sided hollow triangle extrusion therefore gives volume and mass
`6 * 2 * thickness` at density 1: `0.12` at thickness `0.01`, `0.012` at the
default `0.001`, and zero at zero thickness. Direct CPU
`compute_inertia_mesh` returned approximately `0.1199999303`,
`0.01200010255`, and `0.0`, respectively.

On the pristine pinned base, `Mesh(is_solid=False, compute_inertia=True)`
cached the default-thickness mass. At requested thickness `0.01`, both
`compute_inertia_shape` and `ModelBuilder.add_shape_mesh` with hollow
`ShapeConfig(margin=0.01)` still reported `0.01200010255`. An otherwise
identical `compute_inertia=False` mesh reported `0.1199999303`. At requested
zero thickness, the cached mesh still reported `0.01200010255` while the
uncached path reported zero. The cached full tensor also differed by about
tenfold; the test compares all nine entries and center of mass with the
uncached path.

The fix records the automatically computed hollow mass properties and uses
them only when the requested thickness is the default and the scale is
identity. Otherwise it recomputes from mesh geometry. Explicitly replaced
mass properties retain the existing cache behavior. Provenance follows mesh
copies and internal mass-property transfers. The solid cached-inertia path,
including Mohit-Ak's nonuniform-scaling correction, is unchanged.

## Validation

The new analytic regression contains three thickness subcases (`0`, `0.001`,
`0.01`), actual builder mass/COM/tensor, nonuniform scale, solid control,
three copy forms, and authored mass-property controls. The new unittest on
the pristine source was RED: **one unique test, three failures** (zero
thickness, `0.01` thickness, and builder mass). A final available CPU run was
GREEN: **30 unique tests** across full `test_inertia` (18), full
`test_mesh_cache` (9), builder mesh lifetime (1), and two targeted model
tests. Full pre-commit passed 14 hooks with two no-file skips, and
`git diff --check` passed.

The exact local test command used an existing Python 3.12 virtual environment
with Warp 1.18.0 and NumPy 2.5.4. The equivalent portable command below runs
from the checkout root after provisioning those same packages in
`.validation/venv`:

```bash
uv run --no-project --python .validation/venv/bin/python python - <<'PY'
import unittest
import warp as wp

wp.config.kernel_cache_dir = '.validation/warp-cache'
names = [
    'newton.tests.test_inertia',
    'newton.tests.test_mesh_cache',
    'newton.tests.test_builder_replicate.TestReplicateMeshLifetime',
    'newton.tests.test_model.TestModelMesh.test_mesh_rejects_invalid_triangle_indices',
    'newton.tests.test_model.TestModelValidation.test_lock_inertia_on_shape_addition',
]
result = unittest.TextTestRunner(verbosity=1).run(
    unittest.defaultTestLoader.loadTestsFromNames(names)
)
raise SystemExit(not result.wasSuccessful())
PY
uvx pre-commit run -a
git diff --check
```

For RED, run only
`newton.tests.test_inertia.TestInertia.test_hollow_mesh_inertia_respects_thickness_with_cache`
with the pinned base's three source modules and the final test. The branch
retains only the corrected source.

## Independent review and limits

An independent xhigh critic reviewed the frozen diff and Newton instructions.
Verdict: **PASS / Fits; no remaining supported material defect**. The critic
passed 79 additional CPU property comparisons, covering thickness arrays,
scale, tensors/COM, manual overrides, copies, deduplication, and builder
mass properties. The critic first found a public `Mesh.copy` provenance gap;
it was fixed and included in the final regression. Reviewed file SHA-256:

```text
a3f0cc0a3b8fb1d11e1f240b99c5bb1af9e0f64eb0afb410dd5d832aba5f46f1  newton/_src/geometry/inertia.py
8a9465923b1c399ca96bdfa2c78fe9d04c1096a11ed2dc79dafc50cc261c34fb  newton/_src/geometry/types.py
e77fc76d721965dd9632910498ef5ea2bbddd8f7c156a1e2d24aa8a8c8c4649  newton/_src/sim/builder.py
81df90ac7d3e36d9933f3a44ccfa4b7c2f30f0cf951501404b4615e2d654d2b1  newton/tests/test_inertia.py
```

GPU execution was unavailable. SciPy was absent: the adjacent mesh
approximation test stopped in its convex-hull path, and the critic's hull
transfer probe used a stubbed remesher. Actual convex-hull remeshing and GPU
or native downstream acceptance remain untested. The other 30 CPU checks
passed. No performance qualification follows from these checks.

Original Newton geometry/inertia contributors, including nvtw and Han Xudong,
retain credit. Mohit-Ak's distinct cached solid-mesh scaling work (#4324)
retains its ownership. This branch leaves published ray-box, cylinder,
sphere-capsule, and other owners' branches untouched. The full published
diff consists only of source, tests, changelog, and this portable receipt;
raw logs, caches, and sandbox paths remain local. Registration PR #56 is
pending; factory #66/#14/#64 are coordination only. There is no upstream
PR/comment, main/force push, signoff, or CLA attestation.
