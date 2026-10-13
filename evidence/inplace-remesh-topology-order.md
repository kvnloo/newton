# In-place mesh remeshing topology: CPU validation receipt

Date: 2026-10-13. Exact tested upstream base:
`bb25fc0c679e7a81f5080c11962961f8cccb5d72`.
Reviewed implementation commit: `0528dda09096b8f4658aa5aba8d1a1a9df6bdbcd`.
Branch: `fix/inplace-remesh-topology-order` on `kvnloo/newton`.

## Witness and correction

The public `remesh_mesh(..., method="convex_hull", inplace=True)` receives
a default box with 24 duplicated vertices and a real SciPy convex hull with
eight vertices. The pristine base assigns the new vertices before the new
indices. `Mesh.vertices` validates the old indices (up to 23) against the
eight new vertices and raises `ValueError: indices contains index 23 which
exceeds vertex count 8`. Real fast-simplification quadratic remeshing has
the same failure with 14 returned vertices. Both methods use the shared
`remesh_mesh` in-place update path; all five remeshing methods dispatch into
that path.

For valid output, the fix calls the existing `Mesh._replace_geometry`, which
normalizes and validates both new arrays before assigning either and
invalidates geometry-derived caches once. A controlled invalid-output case
on the base demonstrated that sequential setters can leave vertices changed
when the new indices are invalid. The corrected path leaves vertices,
indices, and the previously finalized CPU mesh unchanged on rejection.
Non-in-place convex-hull behavior remains unchanged.

## Validation and commands

On pristine base with the final registered test module: **four unique CPU
tests; two errors, one failure, one pass; exit 1**. The errors are real
convex-hull and real quadratic shrinkage. The failure is partial mutation
after invalid remesher output. The non-in-place control passed. On the
reviewed fix: **69 unique CPU tests passed**, including the four new tests,
full convex-hull remeshing, mesh-cache and mesh-validation modules, plus
three selected model tests. Full pre-commit passed 14 hooks with two
no-applicable-file skips; `git diff --check` passed.

The saved sandbox used Linux CPU, Python 3.12.14, Warp 1.18.0, NumPy 2.5.4,
SciPy 1.18.1, and fast-simplification 0.2.0. SciPy and fast-simplification
were installed as official wheels of Newton's declared optional `importers`
dependencies, using ordinary permissions. No OS, GPU, model, credential, or
dependency-policy change was made. CUDA driver was unavailable. The new
tests skip the relevant real remesher if its optional dependency is absent;
all four ran without skips in this environment.

From a checkout with those packages in `.validation/venv`, run:

```bash
uv run --no-project --python .validation/venv/bin/python python - <<'PY'
import unittest
import warp as wp

wp.config.kernel_cache_dir = '.validation/warp-cache'
names = [
    'newton.tests.test_remesh_mesh_topology',
    'newton.tests.test_remesh_convex_hull',
    'newton.tests.test_mesh_cache',
    'newton.tests.test_mesh_validation',
    'newton.tests.test_model.TestModelMesh.test_mesh_approximation',
    'newton.tests.test_model.TestModelMesh.test_mesh_rejects_invalid_triangle_indices',
    'newton.tests.test_model.TestModelMesh.test_mesh_setters_preserve_valid_triangle_indices',
]
result = unittest.TextTestRunner(verbosity=1).run(
    unittest.defaultTestLoader.loadTestsFromNames(names)
)
raise SystemExit(not result.wasSuccessful())
PY
uvx pre-commit run -a
git diff --check
```

For RED, run only `newton.tests.test_remesh_mesh_topology` using the same
final test file on the exact pinned base. The published branch retains the
one-line source correction.

## Independent review, ownership, limits

An independent xhigh critic read Newton's agent, coding, contribution,
review, and changelog rules and reviewed the complete frozen diff. Verdict:
**Fits; no material findings**. The critic independently passed the four
new CPU tests and five additional probe methods spanning 38 cases: all five
dispatch paths, shrink/grow/reorder/empty outputs, invalid outputs, real
hulls, solid/hollow inertia, CPU finalization, and cache behavior. The three
other remeshing methods used controlled backend outputs; real SciPy hull and
fast-simplification quadratic paths were executed. Reviewed SHA-256 hashes:

```text
2b287c3481b1e9117d2b52ee379977b30b7afd8f5d34fe2e6c605ae3a4b5c355  newton/_src/geometry/utils.py
214cf8fad3ce5919f30cc18d28317d5daa8db6642a9b34237c20038d5eb5a271  newton/tests/test_remesh_mesh_topology.py
594300c57806458ada7d459418711b21ad78410eeed4faf963f2a447e1946caf  changelog/+inplace-remesh-topology-order-4eab7c29.fixed.md
```

Fresh open upstream issue/PR and factory #66/#14 ownership checks found no
matching active fix. Upstream #4428 USD convex-decomposition limits and
#4461 binary `.msh` loading are distinct and retain their owners. Original
Newton geometry and remeshing contributors, including nvtw, retain credit.
The previously published hollow-inertia, ray-box, cylinder, and
sphere-capsule branches remain untouched. Only source, tests, changelog,
and this portable receipt are published; raw logs, caches, and local paths
remain local.

This is CPU correctness evidence, not GPU, full-solver, native downstream,
or performance qualification. No upstream PR/comment, main/force push,
cleanup, signoff, CLA attestation, or previously denied/canceled action
occurred. Registration PR #56 remains pending; factory #66/#14/#64 remain
coordination rather than registered routing.
