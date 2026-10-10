# Newton smooth-extrema downstream evidence (HOLD)

Authored by dot (AI assistant). The connected Git-data publisher assigns
account author/committer metadata; those fields do not imply human authorship.
The smooth-extrema implementation originated with Eric Heiden; its public
docstrings were completed by Eric Shi.

This is the portable evidence for the CPU-only finite-Float32 experiment on
2026-10-10 UTC. It remains **HOLD for upstream promotion** pending owner
acceptance and review of CPU helper overhead and GPU/whole-solver cost.
No upstream posting or pull request is part of this publication.

## Published source and frozen archive

- Source carrier: [kvnloo/newton at e4a8db7](https://github.com/kvnloo/newton/commit/e4a8db7fd5a790a2e65ff80552411f7ba8c513a9)
  on `fix/smooth-extrema-fast-guard`.
- Source parent: `a6e1649b112e3d962b35f7dad66780dc55150588`.
- Source tree: `7fe555042a8be3b59559a79153b4543b5acf7797`.
- Archive: `smooth-extrema-HOLD-evidence.tar.gz`, 88,976 bytes; SHA256
  `4c248ea149bf486944f4e59d61d8a418108e8b36ad5bfcfb33d5492ae77a7666`.
- `publication-manifest.json` maps the captured local commit to the published
  commit, pins source-file hashes, and records the archive identity.

The frozen archive is unchanged. Its 42 regular text files include raw test
and timing results, oracle/harness inputs, reproduction notes, and an internal
per-file manifest. Fifteen text files have only disclosed absolute-path-prefix
normalization; numerical/result lines are preserved. No executable binary,
cache, private state, or source worktree is included inside the archive.

The archive was captured before downstream publication. Its references to
local commit `f5cedc5626a364c0228f921d9ce58ba33a6ada02` and statements that no
source had yet been pushed describe that capture-time state. For remote
checkout/reproduction, use **published commit e4a8db7fd5a790a2e65ff80552411f7ba8c513a9**.
Both commits have the identical source tree and sole parent. The publication
commit has a different message and author/committer metadata; no source bytes
were changed. The mapping is outside the archive so historical receipts are
not rewritten.

## What the recorded run establishes

- Shipping regressions: unmodified baseline ran 5 tests with 6 failing
  assertions/subtests; the final candidate passed all 5.
- Full math suite: 8/8 PASS. Real CPU XPBD prismatic caller: 1/1 PASS.
- Independent frozen 42-case oracle plus 77 guard controls passed, including
  real Warp Tape gradients. Fast-path values/gradients bit-match baseline;
  slow-path values/gradients bit-match candidate2 on the recorded controls.
- Required full `uvx --from pre-commit==4.3.0 pre-commit run -a` passed on
  byte-identical final source, tests, fragment, and unchanged `uv.lock`.
  Basedpyright was not run. No broad unrelated solver suite was run.

Candidate3's descriptive CPU median ratios were approximately 1.25241x for
scalar helpers and 1.22241x for the fixed-epsilon XPBD expression, compared to
candidate2's 1.59322x and 2.57103x. Raw ABBA-cycle ratios vary and are retained
in the archive; these samples are not statistical guarantees or CI bounds.
The XPBD expression microbenchmark is distinct from a whole-solver timing.
GPU correctness/performance and whole-solver speed were not measured.

The ordinary-range fast path retains baseline cancellation error. This is
an intermediate-overflow fix, not a correct-rounding guarantee. Nonpositive
epsilon retains baseline forward behavior without a derivative promise.

## Reconstructing the source siblings

Create the worktrees before extracting the archive, using a fresh work area:

```sh
git clone https://github.com/kvnloo/newton.git source
git -C source worktree add --detach ../red-test a6e1649b112e3d962b35f7dad66780dc55150588
git -C source worktree add --detach ../feature a6e1649b112e3d962b35f7dad66780dc55150588
git -C source worktree add --detach ../candidate3 e4a8db7fd5a790a2e65ff80552411f7ba8c513a9
sha256sum /path/to/smooth-extrema-HOLD-evidence.tar.gz
tar -xzf /path/to/smooth-extrema-HOLD-evidence.tar.gz
git -C feature apply ../receipts/candidate2-math.patch
```

The new archive extraction adds helpers to `red-test/` and `feature/` beside
`receipts/` and `critic/`. Verify the source hashes in the publication manifest
and extracted `receipts/smooth-extrema-HOLD-evidence.md` before replaying.
Read that report, `receipts/cpu-cost-plan-frozen.md`, and the archived harness
arguments/command receipts for the pinned runtime, oracle inputs, source
selectors, and exact workloads. Substitute local paths for `{NEWTON_NEXT}`
and `{PYTHON_RUNTIME}`. Historical local commit references use the mapping
above. CPU timings will depend on hardware and load.
