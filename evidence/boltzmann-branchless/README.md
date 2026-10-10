# Branchless Boltzmann bounded-ratio candidate

**HOLD for upstream adoption pending maintainer acceptance of residual cost and
CUDA validation.** On this bounded CPU helper benchmark, the candidate costs
**1.157x forward** and **1.122x backward plus gradient reset** versus the original
upstream helper. It is cheaper than the published float64 correctness candidate,
which measured 2.042x and 1.320x in the same coordinated window.
No GPU or solver-workload performance claim is made.

## Change and provenance

Keep the float32 public signature and every private float64 intermediate from
[58b2617](https://github.com/kvnloo/newton/commit/58b2617c50f4a489d26164c9c8b109de2617acc3).
Hoist the bounded ratio `exp(-abs(alpha * (a - b)))`, weight and correction ahead
of a final `wp.where` selection. There is no float32 fast path, custom adjoint,
new API, dependency change or solver change. Finite float32 differences and
products fit in float64; both unselected and selected output expressions fit
there too. Warp's absolute-value adjoint at zero agrees with the old nonnegative
branch. Native reverse mode remains responsible for the derivatives.

Original upstream pin: `a6e1649b112e3d962b35f7dad66780dc55150588`.
Published correctness pin: `58b2617c50f4a489d26164c9c8b109de2617acc3`.
The original evidence, shipping oracles and cost harness remain unchanged.

Credit Eric Shi for importing the helper from `warp.sim` in
[1be9943](https://github.com/newton-physics/newton/commit/1be9943221dc0920c6f27ea86ef1d6bb8cfaa2ba),
Eric Heiden for moving it into the public math API in
[958e363](https://github.com/newton-physics/newton/commit/958e363c0983c8af71765a3ba93e58db54e1d17f),
and the collective Newton and Warp contributors. These are software-provenance
credits, not a claim to invention of the mathematical weighted average.

## Fresh correctness evidence

- All seven unchanged shipping Boltzmann tests pass on native Warp CPU kernels
  and Tape autodiff, as do 37 adjacent math/API/kinematics/axis tests.
- A separately frozen **NEW independent suite** passes eight groups with
  214 recorded rows on both the published candidate and this version. The
  original upstream helper passes two groups and fails six. The two corrected
  versions have identical recorded values, value bits and reverse gradients.
- New controls include exact large and cancellation-sensitive means and both
  input orders; alpha +0/-0; signed-zero ties; extreme equal/opposite inputs;
  saturated reverse gradients; underflowed relative weights with representable
  alpha derivatives; and a fixed 200-digit Decimal-reference grid.
- The historical three independent regressions and 34 probes mentioned in the
  original receipt were unavailable after environment refresh. Their old pass
  was not reused or represented as a new pass. The new suite is distinct.

The frozen new suite and clearly labeled path-normalized publication copies of
the three-arm records are in `frozen-independent-evidence.tar.gz`. Only absolute
source and traceback directory prefixes are replaced with descriptive worktree
or evidence placeholders. Numerical fields and outcomes remain unchanged;
untouched original reports are preserved separately. `publication-manifest.json`
documents both original and normalized record hashes. Extract the archive to a
temporary directory to inspect or replay. Its Python source SHA256 is
`3ac0a2105dd3ef045e11516b0829e155c7bc4b39a97b862c3d14ec8f1c2c9327`.
The archive preserves the exact pre-replay source without formatter edits.

Validation environment: Python 3.12.14, Warp 1.18.0, NumPy 2.5.3. A CUDA driver
could not load; GPU execution was unavailable. This is not full-suite or hosted
CI validation. Repository-wide pre-commit, including the unchanged uv lockfile,
and the Towncrier 25.8.0 release-fragment draft pass.

## Bounded CPU cost

All three runs use the byte-identical existing
`../boltzmann-finite-inputs/compare_cpu_cost.py`, SHA256
`de611f587188b89115f00b7e811cc9086ac917218e9a757eb7059f23353039f7`.
It compares the literal original upstream arithmetic with the imported public
helper. The original-import run is an identity control. Actual source imports
were verified for the original, published and branchless worktrees.

Inputs, ordinary value/gradient grader, setup and timing boundaries are frozen:
65,536 float32 triples in [-2, 2], seed 20261010; four explicit warmup launches
per kernel; sixteen alternating-order trials with four launches per sample.
JIT/setup are excluded. Forward timings include host launches and end-of-sample
synchronization. Backward timings include Tape backward and gradient reset.
The coordinated exclusive quiet window was 2026-10-10 13:08:01–13:08:19 UTC.
Source, harness and independent-suite hashes matched before and after.

| Imported helper | Original forward (ms) | Helper forward (ms) | Ratio | Original backward + reset (ms) | Helper backward + reset (ms) | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original upstream identity | 0.475107 | 0.475477 | 1.001x | 1.321909 | 1.324014 | 1.002x |
| Published float64 repair | 0.480689 | 0.981422 | 2.042x | 1.321298 | 1.743547 | 1.320x |
| Branchless float64 repair | 0.479465 | 0.554943 | 1.157x | 1.326958 | 1.488748 | 1.122x |

The branchless median is 43.46% lower forward and 14.61% lower backward plus
reset than the published candidate's contemporaneous absolute medians.
Normalized by each run's original baseline, the corresponding ratios are
0.566890 and 0.850220. The candidate still carries 15.74% forward and 12.19%
backward-plus-reset overhead versus its same-run original baseline.
Every raw timing sample is retained in `cost-original.json`, `cost-stable.json`
and `cost-candidate.json`; derived comparisons are in `cost-summary.json`.
These are standalone helper host-launch costs on one CPU, not solver or GPU
performance estimates.

The three imported-helper arms ran sequentially; the baseline medians drifted
slightly between arms. Candidate samples range from 0.541331 to 0.793243 ms
forward and 1.464795 to 2.003244 ms backward plus reset, with corresponding
baseline tail variation. These observed medians do not establish a population
or tail-latency guarantee, statistical confidence interval, or repeatability
across hosts. All samples remain visible rather than discarding slow launches.

## Replay

Run the unchanged shipping tests using Newton's documented development setup:

```sh
uv run --extra dev -m unittest \
  newton.tests.test_math_boltzmann newton.tests.test_math_boltzmann_limits -v
uv run --extra dev -m unittest \
  newton.tests.test_math newton.tests.test_api newton.tests.test_kinematics \
  newton.tests.test_up_axis.TestQuatBetweenAxes -v
```

Extract the independent archive, then run its unchanged script against the
current public package (use separate pinned worktrees for the other arms):

```sh
mkdir -p /tmp/newton-independent-evidence
tar -xzf evidence/boltzmann-branchless/frozen-independent-evidence.tar.gz \
  -C /tmp/newton-independent-evidence
uv run --extra dev python /tmp/newton-independent-evidence/newton_counterexamples.py \
  --arm candidate --report /tmp/newton-independent-evidence/candidate.json
```

Run the frozen cost harness only in a quiet CPU window:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --extra dev python evidence/boltzmann-finite-inputs/compare_cpu_cost.py
```

## Remaining limits

The prior finite-input contract and limitations remain. Uniform relative
accuracy across the entire float32 domain is not established. The disclosed
`(1e-20, -1e-20, 1)` case can still return zero rather than approximately 1e-40,
while retaining its approximately 1e-40 alpha derivative. Nearly cancelling
values remain limited by intermediate precision. True derivatives exceeding
float32 range can remain infinite; covered finite input derivatives are not
contaminated. Non-finite operands and arbitrary higher-order derivatives are
outside the claim. Maintainer acceptance of residual CPU cost and CUDA
validation remain required before upstream adoption.
