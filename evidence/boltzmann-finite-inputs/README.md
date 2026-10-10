# Boltzmann finite-input correctness candidate

**HOLD for upstream adoption pending maintainer acceptance of cost.**
The bounded ordinary-input CPU microbenchmark measured **2.10x forward** and
**1.25x backward-plus-gradient-reset** cost versus the original helper.
The private float64 intermediates repair numerical failures, but their CUDA
cost is unmeasured. No solver-workload or GPU performance claim is made.

## Scope

The existing public `newton.math.boltzmann()` helper keeps float32 inputs and
return value. A bounded exponential ratio and private float64 intermediates
avoid finite-input exponential overflow/underflow, preserve cancellation-sensitive
alpha-zero means, and keep representable reverse-mode gradients. There is no
custom adjoint, new public API, dependency/lockfile change, or solver change.
The inspected checkout has no internal call sites for this public helper.

Base: `a6e1649b112e3d962b35f7dad66780dc55150588`.

## Attribution

Credit Eric Shi for importing the existing `warp.sim` helper into Newton in
[1be9943](https://github.com/newton-physics/newton/commit/1be9943221dc0920c6f27ea86ef1d6bb8cfaa2ba),
including the original exponential-weight formula, and Eric Heiden for moving
the helper into the public `newton.math` API in
[958e363](https://github.com/newton-physics/newton/commit/958e363c0983c8af71765a3ba93e58db54e1d17f)
(#1481). This work builds on the collective Newton and Warp contributors.
These credits describe software provenance, not invention of the mathematical
Boltzmann-weighted average.

## Numerical validation

- Three frozen regression tests produced four assertion failures on unchanged
  production code, then passed after the repair.
- An initial candidate was rejected for mean/symmetry and gradient regressions.
  Four additional frozen controls failed that candidate and pass this revision,
  including a representable alpha derivative whose relative weight is exp(-110).
- All seven final shipping tests pass. Expected values and gradients use the
  unshifted defining formula in 80-digit Python Decimal and analytic derivatives;
  extreme exponents use the analytic dominant-input limit.
- Independent numerical re-review passes the seven tests, three independent
  regressions, and 34 probes. Exact opposite-sign alpha-zero means, finite
  extreme-input gradients, and saturated gradients are preserved.
- Thirty-seven adjacent CPU math/API/kinematics/axis tests pass.
- Repository-wide pre-commit and Towncrier 25.8.0 draft pass. Ruff changed only
  line wrapping of a test kernel signature; its AST is identical.

Validation used Python 3.12.14, Warp 1.18.0, and NumPy 2.5.3 on native Warp CPU
kernels and Tape autodiff. No CUDA device/driver was available. This is not a
full-suite or hosted-CI validation claim.

Reproduce the shipping tests using the documented development dependencies:

```sh
uv run --extra dev -m unittest \
  newton.tests.test_math_boltzmann newton.tests.test_math_boltzmann_limits -v
uv run --extra dev -m unittest \
  newton.tests.test_math newton.tests.test_api newton.tests.test_kinematics \
  newton.tests.test_up_axis.TestQuatBetweenAxes -v
uvx pre-commit run -a
uvx --from towncrier==25.8.0 towncrier build --draft \
  --version X.Y.Z --date 2026-10-10
```

## CPU cost measurement

`compare_cpu_cost.py` uses native CPU kernels with 65,536 float32 triples
uniformly sampled in [-2, 2], seed 20261010. Its original helper arithmetic
matches the base implementation. Ordinary values and gradients agree with the
baseline at 2e-5 relative / 2e-6 absolute tolerance. JIT/setup and four explicit
warmup launches per kernel are excluded. Sixteen alternating-order trials give
eight original-first and eight revised-first trials, with four launches per
sample. Measurement took place during a coordinated quiet CPU window.

Median forward launch cost: original 0.478542 ms, revised 1.004997 ms (2.100x).
Median backward plus gradient-reset cost: original 1.382616 ms, revised
1.722969 ms (1.246x). Every raw sample is in `cpu-cost-result.json`.
This is an ordinary-input CPU microbenchmark of a standalone helper, not a
performance improvement or a solver-workload estimate.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --extra dev python evidence/boltzmann-finite-inputs/compare_cpu_cost.py
```

## Remaining numerical limits

Uniform relative accuracy across the entire finite float32 domain is not
established. As on the base, `(1e-20, -1e-20, 1)` can return zero instead of an
approximately 1e-40 mean; the approximately 1e-40 alpha derivative is retained.
Nearly cancelling values can remain limited by intermediate precision. Some
true derivatives exceed float32 range; an infinite alpha derivative is retained
without contaminating finite input gradients in the covered extreme cases.
Non-finite operands and arbitrary higher-order derivatives are outside the
tested claim.

Simple float32 normalized expressions can address individual boundaries while
still overflowing generated weight adjoints at nearby extreme opposite inputs.
No unreviewed float32 replacement or custom adjoint is included.
