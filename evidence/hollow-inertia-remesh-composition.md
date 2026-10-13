# Hollow inertia and in-place remesh composition (CPU)

This evidence branch composes two independently published, narrow Newton fixes. It does not propose a third source correction.

- Upstream base: `newton-physics/newton` `bb25fc0c679e7a81f5080c11962961f8cccb5d72`.
- Hollow inertia + real convex hull validation: `93c38ebbcc1550ecfa622709147c7793b3ae7263` (descendant of `4e73fd9e6c4694f068010061e73ef4a2a5e3ff01`).
- Atomic in-place remesh topology: `bf11c8f3ccd87573c1cb37849898458b64d6552e`.
- Merge `9f451c3c592c8dc2b11bed5955f7a8437951b3a8` joins the upstream base with the hollow branch. Merge `471015a8e3a6d703a537e7729e0af3d6c38401b6` joins that result with the remesh branch. Both original heads remain ancestors; neither was cherry-picked or changed. Git resolved the merges without conflict.

The interaction is in `remesh_mesh(inplace=True, recompute_inertia=True)`: a 24-vertex duplicated box can shrink to an 8-vertex convex hull before its hollow mass properties and generated-inertia provenance are refreshed. Four new tests cover this full sequence through the real CPU hull backend and builder, authored hollow properties when recomputation is omitted, invalid-output rollback of geometry/inertia/finalized mesh, and solid behavior after topology shrinkage.

Validation used Python 3.12.14, Warp 1.18.0, NumPy 2.5.4, SciPy 1.18.1, and fast-simplification 0.2.0. The four interaction tests passed. A single deduplicated CPU suite of 95 tests passed, spanning the new interaction tests, both branches' regressions, inertia, cache, mesh validation, convex hull, replication and relevant model tests. `uvx pre-commit run -a` and `git diff --check` passed. Warp reported no CUDA driver; no GPU or performance claim is made.

An independent xhigh review passed 10 focused tests and 192 real hull/quadratic property cases across solid/hollow, generated/authored inertia, scalar/list thickness, density, and nonuniform/reflected scaling. Four invalid-output probes preserved geometry, generated provenance, mass properties, finalized cache, hashes, edges, and watertightness by object identity. The reviewer found no supported composition defect.

This branch is downstream composition evidence. Original authors and distinct upstream owners retain credit and ownership. It does not modify Newton registration PR #56, merge upstream, or change factory routing.
