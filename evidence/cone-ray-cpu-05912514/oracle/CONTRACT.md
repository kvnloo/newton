# Frozen cone raycast contract

Frozen before any candidate implementation was inspected, 2026-10-10 UTC.
Upstream checkout: a6e1649b112e3d962b35f7dad66780dc55150588.
`newton/_src/geometry/raycast.py` blob: 09b7393e4aadd5c0521b2105d807b0b418ef5c61.
No production edits, commits, or publication are part of this review.

## Source-backed contract and scope

- Public `newton.intersect_ray` at `newton/_src/geometry/raycast.py:916` requires normalized, nonzero directions. Its output distance and shape ID are both `-1` on a miss. The dispatcher accepts nonnegative intersections and chooses the smallest one.
- Public `ModelBuilder.add_shape_cone` at `newton/_src/sim/builder.py:8331` and `docs/concepts/conventions.rst:489` define a closed collision cone with base radius r at z=-h and apex at z=+h. The ray helper explicitly promises distance and normal, or exactly -1 and zero normal on a miss.
- Existing raycast tests explicitly expect interior exits for sphere, box, ellipsoid, and barrel cylinder (`newton/tests/test_raycast.py`). Cone tests already require exterior base, side and apex hits. There is no documented origin-outside precondition. Interior exit behavior is therefore a strong family-contract inference backed by geometry, rather than an explicit cone-only sentence in public docs.
- For r>0 and h>0, the solid is -h<=z<=h and sqrt(x*x+y*y)<=r*(h-z)/(2*h). A forward ray intersects the boundary first at the smallest t>=0 satisfying either the side equation within the height interval or the base disk. Interior starts return the exit, not a negative entry and not a miss.
- Outward side normals away from singularities follow the gradient of radial_distance - r*(h-z)/(2*h). Base interior normal is (0,0,-1). Apex and base-rim normals are not uniquely determined geometrically; the harness deliberately imposes no particular normal there.
- `boundary`-tier cases adopt the natural t=0 closed-surface convention already used by neighboring primitives, but public docs do not spell it out. Report these separately from strict interior/exterior defects. Polynomial degeneracy on a generator is included in this tier where the ray lies on the surface.
- Zero-length directions are excluded by the public precondition. Zero/negative shape dimensions have insufficiently specified public semantics here. No demand to change those contracts is made.

## Oracle independence

The 31 fixed witnesses in `cone_contract.py` use manually derived distances from cone cross-sections and straight-line geometry; they do not evaluate Newton's quadratic or copy its root-selection algorithm. The harness checks every hit point against the independently written solid equation and samples the interval preceding every interior exit for strict interior membership. Expected values are frozen independently of candidate results.

For the unit cone, the z=0 cross-section radius is 1/2. Thus center +X exits at 1/2; starting at x=1/5 exits at 3/10 toward +X or 7/10 toward -X. At z=1/2 the radius is 1/4. Axial rays at x=1/4 leave the side at z=1/2 or the base at z=-1.

For d=(1,0,2)/sqrt(5), setting x=t/sqrt(5), z=2t/sqrt(5) gives the center exit t=sqrt(5)/4. This is a linear side equation, despite the general quadratic formulation. For d=(1,0,-2)/sqrt(5), a center ray remains within the expanding side until the base at t=sqrt(5)/2. Starting at (1,0,0), d=(-1,0,-2)/sqrt(5) enters the side at t=sqrt(5)/4. The offset ray (1/2,1/4,0)+t*(-1,0,2)/sqrt(5) has a constant nonzero side residual and no forward cap hit, so misses.

The side tangent (-1,1/2,0)+t*(1,0,0) touches at t=1. The generator (3/2,0,-2)+t*(-1,0,2)/sqrt(5) first reaches the finite cone at the base rim t=sqrt(5)/2. On-surface origins use t=0 in the separately identified boundary tier.

## Adversarial mechanisms

- For a positive quadratic coefficient, an inside ray's first algebraic root is negative; the positive root is its valid exit.
- For a negative quadratic coefficient, the expression with minus square root need not be the nearer root. Both roots require independent nonnegative and height checks; sorting by t must not assume coefficient sign.
- A zero quadratic coefficient can leave a linear equation with a valid side hit, or a constant nonzero equation with no side hit. A fully zero polynomial means the line lies on a cone generator, so finite-height clipping and boundary-origin handling determine the first hit.
- Base hits must remain candidates for rays starting inside the height slab, and must compete with side candidates for the nearest hit. Outside-origin-only cap gates cannot implement that contract.
- Behind-origin roots must not leak arbitrary negative distances from the private helper; its documented miss sentinel is exactly -1.

## Execution

The harness creates independent single-cone worlds through public `ModelBuilder`, finalizes the ordinary BVH, and calls public `newton.intersect_ray` with both fast-math settings. It repeats the public checks after a rigid cyclic-axis rotation and translation, and additionally probes the private helper to identify where misses are introduced. CPU with official Warp 1.18.0 is the available execution target; no CUDA claim is made.

Run from this packet directory with SOURCE pointing to Newton commit 059125141e71f1e72e294c047fffd06ca4b32d3e and a Python environment containing the checkout dependencies and official warp-lang==1.18.0:

    PYTHONPATH="$SOURCE" python oracle/cone_contract.py --json-output result.json

No expected-value changes after candidate inspection are permitted without a separately documented oracle revision and independent reason. The strict core and boundary-tier results must remain distinguishable.
