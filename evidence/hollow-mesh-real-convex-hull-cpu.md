# Real convex-hull CPU validation for hollow mesh inertia

Date: 2026-10-13. This follows the published hollow-mesh fix at
`4e73fd9e6c4694f068010061e73ef4a2a5e3ff01`; its tested upstream base is
`71eeba16a717d4dcf0b9dd7056a7b6f863054dc3`. This follow-up adds a
real-hull test and a narrow source correction for in-place remeshing. The
initial test commit is
`5c08a2f1dca3ebc88f5fb04c7f7f2d86a7fd8457`. The narrow source,
expanded regression, and changelog commit is
`2e3e12bff320e49cfc812407caed8b022fa7d040`.

## Environment and witness

The isolated Python 3.12.14 CPU environment already contained Warp 1.18.0
and NumPy 2.5.4. Newton declares `scipy>=1.11.0` for convex-hull approximation.
An ordinary wheel-only PyPI installation added SciPy 1.18.1; no OS, GPU,
model, credential, or policy change was made. Warp reported no CUDA driver.

The new registered test uses the closed unit cube from Newton's inertia test
and calls the real SciPy-backed `Mesh.compute_convex_hull(replace=False)`.
It compares mass, center of mass, and the entire inertia tensor against
direct integration of the returned hull at thickness `0`, `0.001`, and
`0.01`. At density 1, expected mass is respectively `0`, `0.012`, and
`0.12` within float tolerance. It also runs
`ModelBuilder.add_shape_convex_hull` with hollow margin `0.01` and checks
mass, COM, and tensor; exercises in-place hull replacement; and confirms
manually authored mass properties remain in use.

The independent review found a second cache-provenance gap in the public
`remesh_mesh(..., method="convex_hull", recompute_inertia=True, inplace=True)`
path. It recomputed hollow mass properties but left the old generated-value
snapshot behind. A deterministic test doubles the vertices of a closed
asymmetric cube before real in-place hull remeshing. Against the published
implementation head, its new regression was **RED: one unique test, one
failure**: cached shape mass `0.2488672882` versus direct requested-thickness
mass `2.4886846542`. The narrow correction refreshes the generated hollow
snapshot immediately after in-place recomputation. The new regression then
passed, comparing mass, COM, and the full tensor. It does not change authored
mass-property behavior or the existing solid path.

With the original real-hull test on pristine upstream base, the CPU run was
**RED: one unique test, three failures** (zero-thickness mass,
`0.01`-thickness mass, and builder mass). With both follow-up tests on the
pristine base, the run was **two unique tests, four failures**. The base
retained generated `0.001`-thickness mass of about `0.012` for both hull
requests. The original hull test passed on the published source fix, while
the new in-place regression required the follow-up correction.

The follow-up combined CPU run passed **51 unique tests**: the two new hull
tests, full inertia module (18), full mesh-cache module (9), builder
mesh-lifetime test (1), two model controls, the previously blocked model
mesh-approximation test (1), and the full convex-hull remeshing module (18).
Full pre-commit passed 14 hooks with two no-file skips; `git diff --check`
passed. The final SHA-256 hashes are:

```text
af5cb0346c97becaaf4df84f1da0671be38924329d16da1becf0fc6a7e0f9267  newton/_src/geometry/utils.py
71579b40478efdc2827f246d3b3dbca30002ebb17f8c34ae251707b34a4554bb  newton/tests/test_hollow_mesh_hull_inertia.py
```

An independent xhigh critic reviewed the final source and test under Newton's
coding and review rules: **PASS / Fits, no supported material defect**. The
critic independently reran the two-test and 51-test CPU sets and passed 368
additional real-hull property comparisons plus two in-place comparisons.
These are probe comparisons, not registered repository tests.

## Reproduction

Run from a checkout containing the follow-up test with Python 3.12,
`warp-lang==1.18.0`, `numpy==2.5.4`, and `scipy==1.18.1` installed in
`.validation/venv`. If that environment already has the first two packages,
install the declared SciPy dependency from its official wheel, then run:

```bash
uv pip install --python .validation/venv/bin/python --only-binary :all: 'scipy>=1.11.0'
uv run --no-project --python .validation/venv/bin/python python - <<'PY'
import unittest
import warp as wp

wp.config.kernel_cache_dir = '.validation/warp-cache'
names = [
    'newton.tests.test_hollow_mesh_hull_inertia',
    'newton.tests.test_inertia',
    'newton.tests.test_mesh_cache',
    'newton.tests.test_builder_replicate.TestReplicateMeshLifetime',
    'newton.tests.test_model.TestModelMesh.test_mesh_rejects_invalid_triangle_indices',
    'newton.tests.test_model.TestModelValidation.test_lock_inertia_on_shape_addition',
    'newton.tests.test_model.TestModelMesh.test_mesh_approximation',
    'newton.tests.test_remesh_convex_hull',
]
result = unittest.TextTestRunner(verbosity=1).run(
    unittest.defaultTestLoader.loadTestsFromNames(names)
)
raise SystemExit(not result.wasSuccessful())
PY
uvx pre-commit run -a
git diff --check
```

For the RED controls, run the complete new test module on the pinned pristine
base; run only
`newton.tests.test_hollow_mesh_hull_inertia.TestHollowMeshHullInertia.test_inplace_hull_recompute_refreshes_generated_inertia`
on the published implementation head before this follow-up correction.

## Scope and ownership

This validates actual CPU SciPy convex-hull remeshing and inertia provenance
transfer, resolving the stated missing-dependency gap in the original
receipt and repairing one newly demonstrated in-place recomputation gap.
It does not establish GPU, full-solver, performance, or native downstream
acceptance. The previously published implementation branch and every other
branch remain unchanged.
An unrelated pre-existing limitation remains: in-place remeshing of a default
box with 24 duplicated vertices to an 8-vertex hull can fail its intermediate
vertex setter because the old indices exceed the new vertex count. This was
reproduced on both the pristine base and follow-up source; the new closed
8-vertex regression exercises a supported in-place path.

Original Newton geometry/inertia contributors
retain credit; Mohit-Ak's separate solid-mesh scaling work retains its
ownership. Only the narrow utility correction, new tests, a changelog
fragment, and this portable receipt are published;
dependency caches, raw logs, and local paths remain local. There is no
upstream PR/comment, main or force push, signoff, or CLA attestation.
Registration PR #56 remains pending, and factory #66/#14/#64 remain
coordination records rather than registered routing.
