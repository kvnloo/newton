# Newton — fork-local canonical promotion queue

> Coordination document in `kvnloo/newton`; no upstream authorization, default-branch merge, or worker lease. This branch is documentation only. Historical factory issue #66 is preserved intact as the source archive; do not add new project-level issues there. Factory #14 is directory-only.

**Fork:** https://github.com/kvnloo/newton  ·  **Origin:** https://github.com/newton-physics/newton  ·  **Legacy complete record:** https://github.com/kvnloo/oss-factory/issues/66

## Current owner-first triage (verified 2026-10-17)

| Work | Current head and exact status | Action / provenance |
|---|---|---|
| [Origin #4775](https://github.com/newton-physics/newton/pull/4775); [fork #26](https://github.com/kvnloo/newton/pull/26) | **ORIGIN_LIVE / DRAFT** `6b83930d9d00ac91db3645b8ce682d35c60b5740`. EasyCLA / CodeRabbit / pre-commit all success. Three origin GHA workflows `action_required` (external approval); **native current-head Warp CPU/CUDA NOT_RUN**. CodeRabbit tolerance review addressed and latest review has no new actionable comments. | Exact-head native CPU raycast+adjacent test, then maintainer-owned Actions approval. No rerun bypass. Original source: nvtw, Daniela Hase, Mustafa H, Alain Denzler; existing downstream author and reviewers credited in original receipts. |
| [Origin #4762](https://github.com/newton-physics/newton/pull/4762) | **ORIGIN_LIVE / OWNER_COVERED** at head `971c8e7ea53b32e13c4d4e9c4e25011be22e8cbc`. Owner adenzler-nvidia; reviewer eric-heiden APPROVED. **New verified hosted native Warp CPU evidence (2026-10-17 audit):** the actual `test_tile_gemm_mixed_articulation_sizes_cpu` ran and returned `ok` in [Ubuntu AMD64](https://github.com/newton-physics/newton/actions/runs/38896219912/job/116957873971), [macOS](https://github.com/newton-physics/newton/actions/runs/38896219912/job/116957873955) and [Ubuntu ARM](https://github.com/newton-physics/newton/actions/runs/38896219912/job/116957874043) job logs. Runner checkout was **merge SHA** `e4068e15a8a67e57709bcdf22a55093b27df3b58` (head into base `2f6f67b3d25e715a9c7ecb46ea181a78a57fb5bd`); these are externally executed tests inspected through raw job logs, **not locally rerun**. Head's GHA, EasyCLA and pre-commit checks SUCCESS. **Verified hosted native Warp CUDA (2026-10-17 audit):** [GPU job 116958491664](https://github.com/newton-physics/newton/actions/runs/38896220000/job/116958491664) checked out exact PR head `971c8e7ea53b32e13c4d4e9c4e25011be22e8cbc` (not the CPU jobs' merge SHA), initialized Warp 1.18.0 on NVIDIA RTX PRO 6000 Blackwell (sm_120), and executed `test_tile_gemm_mixed_articulation_sizes_cuda_0` to **ok** (not skipped). Full native GPU job: **9,125 tests PASS, 166 skipped**. The named regression does *not* exercise CUDA graph capture/replay; that claim remains **NOT_VERIFIED**. Local CPU/CUDA **NOT_RUN**. #4488 CLOSED UNMERGED and maintainer-superseded. | No duplicate patch or PR; preserve #4487, performance request #4265 (bounceonme), #4488, fork #21/#25, and original author/reviewer credit. Remaining gate is owner review/merge decision; GPU-specific proof only when hardware available. |
| [Registration #56](https://github.com/kvnloo/oss-factory/pull/56) | **OPEN/DRAFT**, Newton absent from current `factory/projects.json`; do not claim registration or merged route. | Human policy review; no default-branch merge by worker. |
| Other Newton candidates | **CHECK/LIMITED**, historical CPU receipts only and source-head refresh missing: `a94decc4`, `274716c3`, `befeb3fd`, `e91951d9`; `bf11c8f3` overlaps the composition. | Use exact native current origin HEAD and independent requirement, dedupe, HITL; no automatic promotion. |

**Policy:** https://github.com/kvnloo/oss-factory/blob/main/factory/PROMOTION_QUEUES.md. #130 retired Linear; explicit scoped human approval still required for any origin-facing write. No GPU or native tests were executed to create this documentation snapshot. No new worker claim. Do not reuse the stale Linear gate language in the archived material below.

### #4775 — current-origin-main drift check (2026-10-17)

- Origin `newton-physics/newton:main` at [`e50f0d0e5807c3b1b9ee0848c18eee25e77673e7`](https://github.com/newton-physics/newton/commit/e50f0d0e5807c3b1b9ee0848c18eee25e77673e7); authored [#4775](https://github.com/newton-physics/newton/pull/4775) remains `6b83930d9d00ac91db3645b8ce682d35c60b5740`, open/draft, mergeable. Comparing PR-head → current main reports `ahead_by=2, behind_by=4` (compare direction matters), **not yet a current-base test result**.
- Exact blob readback: current main **and original PR base** `bd69b56dc00693976c2fea9bd89afd863178140c` share raycast implementation blob `413ad9daa4bda18c1fbe7686fec8d11080d356a4` and raycast test blob `12a20f5aee0022fb820a5b6fe3a36e031294a2b1`. The two relevant files did **not** change upstream since PR base; current main's other changes do not establish native head compatibility.
- GitHub check status at `6b83930`: EasyCLA, pre-commit.ci, CodeRabbit success; [three PR-triggered Actions runs](https://github.com/newton-physics/newton/actions?query=branch%3Apromotion%2Fray-box-normal-minval-20261016) remain `action_required`. CodeRabbit's sole original inline thread resolved. **NOT_RUN here:** current-base native Warp CPU raycast/adjacent; CUDA; negative controls. No worker lease, origin post, CI approval, or default-branch merge claimed. Required next owner action: run native CPU raycast tests on #4775 exact head and record raw output; external workflow approval is maintainer-controlled.
- Ownership remains with existing #4775 author; original Newton geometry authors nvtw, Daniela Hase, Mustafa H and Alain Denzler, previous donor and CodeRabbit reviewer retain credit.

## Lossless historical archive — original factory issue #66

Archived snapshot: https://github.com/kvnloo/oss-factory/issues/66 (original remains authoritative for pre-migration provenance and is not deleted).
Body copied unedited from the legacy issue; comment IDs and bodies preserved below in original order. This archive is for history, not current instructions.

<!-- BEGIN LEGACY ISSUE BODY -->
<!-- oss-factory:v1
{"id":"20261011-ayo-project-newton-001","kind":"state","project":"oss-factory","from":"ayo","to":["factory"],"status":"working","parent_id":"20261011-ayo-work-tracker-001","dedupe_key":"oss-factory-project-work-index-newton","civ":{"city_id":"oss:oss-factory"}}
-->

<!-- promotion-queue:canonical:v1 -->
## Canonical promotion queue — newton-physics/newton

**This issue is the only repo-level upstream promotion queue for `newton-physics/newton`.** Original fork/origin PRs, individual QA issues, proof and authorship stay in place and must be **linked here**, not re-posted. [Factory #14](https://github.com/kvnloo/oss-factory/issues/14) is a cross-repository index only; older work-order rows below are **historical** until exact-head requalification. Existing origin reviewer feedback and CI outrank new claims. **No automatic posting, default merges or proof upgrades.**

**Current triage:** [downstream Featherstone fix #21](https://github.com/kvnloo/newton/pull/21), [#4600 owner CPU evidence](https://github.com/kvnloo/oss-factory/issues/66), and [ray-box normal candidate](https://github.com/kvnloo/newton/commit/a94decc46cc28aa947baa9204ec043fffb0de330) are **CHECK**, not automatically origin-ready. Reconcile current owner PRs/reviewer feedback first. CPU suites never imply GPU, contact stability or performance. Preserve exact artifact SHA and author credit.

**For every item:** identify original reporter/author and origin URL; behavior **and separate requirement** evidence; exact fork/origin SHA, baseline→head commands/results and limitations; maintainer reviews/CI/duplicates; scoped human authorization; then record the posted origin URL or merged/closed/superseded disposition. This queue is an index, not fresh validation.

---


## Canonical per-repo promotion queue — 2026-10-16
This existing project issue is now the **one authoritative promotion queue** for `newton-physics/newton` (Newton); [#14](https://github.com/kvnloo/oss-factory/issues/14) is only the cross-repository index. REGISTRY_PENDING (reuse existing registration PR #56). Historical work-item tables and authorship below are preserved, **not automatically marked current or ready**. Do not create a second issue/queue for this repo.

**Promotion stages:** `READY_FOR_HITL` = exact-head requirement, duplication/ownership, human review/CI, native evidence, independent QA and policy checked; still requires correct Linear origin authorization. `IN_REVIEW` = published upstream, not a candidate for duplicate PR. `LANDED` = verified merge. Else `CHECK/LIMITED/HOLD/WAIT/DISCARDED` with one executable next gate. If the original source/head, permissions, or tests cannot be verified, retain historical rows in HOLD/CHECK. Do not infer readiness from old status or tests.

Keep one row per *contribution* referencing the original issue/PR, fork commit, exact tests/CI, reviewer feedback, named contributor attribution and current blocker. Update this issue as evidence changes, link final READY rows from #14, and avoid additional queue spam. No upstream write, default merge, or worker lease is authorized by this issue.

---



<!-- lane10-canonical-promotion-queue:2026-10-16 -->
## Canonical Newton promotion queue — 2026-10-16

**Authoritative per-repository queue:** this single issue, reusing the prior project index and its dated receipts below. Other Newton downstream PRs/branches are **code/evidence carriers, not extra promotion queues**. [Factory-wide dashboard #14](https://github.com/kvnloo/oss-factory/issues/14) links project readiness, but does not authorize upstream posting. **Confirmed new origin-ready publications: none.** `factory/projects.json` does **not** yet include Newton; [factory registration PR #56](https://github.com/kvnloo/oss-factory/pull/56) remains **open/draft**, so the factory issue is a provisional coordination fallback while fork Issues are disabled. Upstream action needs operation-specific HITL; downstream coordination is permitted. No worker claim or merge is implied.

### Highest priority: already live upstream; resolve CI and get real proof

| Candidate / code owner | Snapshot and evidence | Promotion disposition |
|---|---|---|
| [Newton origin #4488](https://github.com/newton-physics/newton/pull/4488) — Featherstone tile module loading; [fork original #21](https://github.com/kvnloo/newton/pull/21), [review fork #25](https://github.com/kvnloo/newton/pull/25) | **Origin head `f46449e5c928be440597e3065e134bfb32f71272`** on Oct 16. Both original CodeRabbit review threads **resolved and outdated** on this head (Oct 16 repairs); do **not** repeat older notes saying unresolved. EasyCLA + CodeRabbit success, **pre-commit.ci - pr FAIL**, external-GHA manual approval waiting, **CUDA capture/first-step not independently verified**. Older source/fixture checks in [dated receipt](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6179242498) are not Warp runtime evidence. | **LIVE_ORIGIN / REPAIR_CHECK.** Diagnose exact failing hook, capture with real GPU, then continue only the existing authorized PR. Fork #25 (head `59f57caf819efe4380b1fab2f2705c1f475955c6`) is an older staged mirror, not a second upstream PR or an instruction to merge it into the origin branch. |

### CPU-qualified downstream candidates — ready for promotion *triage*, not automatic upstream publishing

| Candidate / immutable fork carrier | Last evidenced qualification | Remaining gates |
|---|---|---|
| [Ray-box small-direction normal `a94decc4`](https://github.com/kvnloo/newton/commit/a94decc46cc28aa947baa9204ec043fffb0de330) | Warp CPU RED→GREEN, 20 raycast + 16 adjacent tests; [receipt](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6117660509) and [Oct 13 head refresh](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6148201065). | Recheck newest origin head, duplicates, risk/requirements, CUDA relevance, [registration #56](https://github.com/kvnloo/oss-factory/pull/56), HITL. Separate from cone work. |
| [Cylinder ray normal `274716c3`](https://github.com/kvnloo/newton/commit/274716c32ab454268b13b00d20ceada8f0d640f8) | 37 real CPU tests, pre-commit, independent review; [Oct 13 requalification](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6148201065). | Exact-current-base/ownership review, CUDA relevance and HITL. |
| [Sphere–capsule projection `befeb3fd`](https://github.com/kvnloo/newton/commit/befeb3fd1d4ef05674e472728c6c563059b78a35) | Analytic RED confirmed; 15 primitive + 4 narrow-phase CPU tests, pre-commit, independent review; [receipt](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6150120453). | Recheck current head and overlapping collision work; MJWarp/GPU not qualified. |
| [Hollow inertia + atomic remesh composition `e91951d9`](https://github.com/kvnloo/newton/tree/validation/hollow-inertia-remesh-composition) | **95/95 native CPU tests** on pinned Oct 13 upstream; preservation of independent commits and provenance. [Composition receipt](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157594627); original [inertia](https://github.com/kvnloo/newton/commit/4e73fd9e6c4694f068010061e73ef4a2a5e3ff01) + [remesh topology](https://github.com/kvnloo/newton/commit/bf11c8f3ccd87573c1cb37849898458b64d6552e) carried separately. | Recheck current head, scope/split decision, source/runtime validation and HITL; **do not double-promote both originals and composition**. |
| [Standalone remesh topology `bf11c8f3`](https://github.com/kvnloo/newton/commit/bf11c8f3ccd87573c1cb37849898458b64d6552e) | 69 native CPU tests, pre-commit and critic on pinned base; [receipt](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157514846). | **Alternative** to promoting the above combined branch, not an extra promotion for identical code. |

### Other recovered Newton work: NEEDS_RECHECK / HOLD, not ready
- [Inside-cone raycast `05912514`](https://github.com/kvnloo/newton/commit/059125141e71f1e72e294c047fffd06ca4b32d3e), [smooth extrema `e4a8db7f`](https://github.com/kvnloo/newton/commit/e4a8db7fd5a790a2e65ff80552411f7ba8c513a9), [branchless numerical candidate `1c6c6811`](https://github.com/kvnloo/newton/commit/1c6c6811abd99eec5e871098f52fc80f3e07e230), existing-owner #4600/#4559/#4629, and #3024 all retain distinct scope, ownership and evidence classification in the **preserved original index below**. Source-only/synthetic Lane 10 seam, centroid, overlap, high-valence and finalization benchmarks against now-merged #4673 are **CHECK/LOCAL_ONLY** unless a fresh supported-path native regression shows an independent remaining bug. Do not mint duplicate work.
- [#4612 / #4641](https://github.com/newton-physics/newton/issues/4612) **HOLD** under maintainer strategy instruction; no competing contribution.
- Historical fork PRs below are *not certified ready* from being open or having past tests. A dedicated owner must recheck exact head, accepted requirement, upstream duplicates/reviews, CI, and HITL.

#### Full still-open Newton fork PR inventory (25 total including #9/#10/#12/#21/#25 above)

| Fork PR | Scope | Triage |
|---|---|---|
| [#1](https://github.com/kvnloo/newton/pull/1) | fix(usd): convert rotational joint-state velocities to rad/s | CHECK: current-head/requirement/duplicate/HITL review |
| [#2](https://github.com/kvnloo/newton/pull/2) | Scale URDF prismatic mimic offsets with import scale | CHECK: current-head/requirement/duplicate/HITL review |
| [#3](https://github.com/kvnloo/newton/pull/3) | fix(selection): rebuild ArticulationView after the source array is replaced | CHECK: current-head/requirement/duplicate/HITL review |
| [#4](https://github.com/kvnloo/newton/pull/4) | Fix Kamino 2-D block-sparse GEMV kernel dispatch | CHECK: current-head/requirement/duplicate/HITL review |
| [#5](https://github.com/kvnloo/newton/pull/5) | Skip unused per-world pair bound for explicit broad phase | CHECK: current-head/requirement/duplicate/HITL review |
| [#6](https://github.com/kvnloo/newton/pull/6) | fix(geometry): scale cached mesh inertia for non-uniform scale | CHECK: current-head/requirement/duplicate/HITL review |
| [#7](https://github.com/kvnloo/newton/pull/7) | Fix KAMINO Jacobi diagonal kernel dtype capture | CHECK: current-head/requirement/duplicate/HITL review |
| [#8](https://github.com/kvnloo/newton/pull/8) | Fix XPBD positional joint relaxation consistency | CHECK: current-head/requirement/duplicate/HITL review |
| [#11](https://github.com/kvnloo/newton/pull/11) | Fix XPBD revolute limits that cross ±π | CHECK: current-head/requirement/duplicate/HITL review |
| [#13](https://github.com/kvnloo/newton/pull/13) | Fix XPBD joint drive stiffness vs iteration count | CHECK: current-head/requirement/duplicate/HITL review |
| [#14](https://github.com/kvnloo/newton/pull/14) | Evidence: tri-mesh self-contact storage is fixed per element | CHECK: current-head/requirement/duplicate/HITL review |
| [#15](https://github.com/kvnloo/newton/pull/15) | Evidence: Featherstone vs MuJoCo first divergence is the explicit joint-limit penalty | CHECK: current-head/requirement/duplicate/HITL review |
| [#16](https://github.com/kvnloo/newton/pull/16) | Evidence: MuJoCo linesearch exhaustion (#4212) | CHECK: current-head/requirement/duplicate/HITL review |
| [#17](https://github.com/kvnloo/newton/pull/17) | Evidence: VBD contact scaling on RTX 3080 Ti (#4141 / #3995) | CHECK: current-head/requirement/duplicate/HITL review |
| [#18](https://github.com/kvnloo/newton/pull/18) | Evidence: Featherstone GEMM utilization on RTX 3080 Ti | CHECK: current-head/requirement/duplicate/HITL review |
| [#19](https://github.com/kvnloo/newton/pull/19) | Experiment: dispatch Featherstone dense GEMM by output parallelism | CHECK: current-head/requirement/duplicate/HITL review |
| [#20](https://github.com/kvnloo/newton/pull/20) | Experiment: semi-implicit Featherstone joint limits | CHECK: current-head/requirement/duplicate/HITL review |
| [#22](https://github.com/kvnloo/newton/pull/22) | Evidence: reproduce MJWarp flat multi-articulation world on current main | CHECK: current-head/requirement/duplicate/HITL review |
| [#23](https://github.com/kvnloo/newton/pull/23) | Harden ArticulationView writes against invalid layouts | CHECK: current-head/requirement/duplicate/HITL review |
| [#24](https://github.com/kvnloo/newton/pull/24) | Evidence: decompose VBD contact scaling on current main | CHECK: current-head/requirement/duplicate/HITL review |

### Already upstream — remove from promotion frontier, retain credit
- [#4354](https://github.com/newton-physics/newton/pull/4354) rotated world-frame twists; [#4330](https://github.com/newton-physics/newton/pull/4330) KAMINO sparse pointer views; [#4355](https://github.com/newton-physics/newton/pull/4355) scalar sparse BLAS: **MERGED**. Corresponding still-open fork draft PRs #9, #10, #12 are historical artifacts, **not promotion targets**.
- [#4630](https://github.com/newton-physics/newton/pull/4630) Delassus overflow: **MERGED** by **arcusbuilds** (Oct 13), reported by **AntoineRichard**. Earlier independently reproduced int32 wrap and Kevin's overlapping fork work are not a competing PR.
- [#4673](https://github.com/newton-physics/newton/pull/4673) finite mesh boundaries/culling: **MERGED** Oct 15 by **eric-heiden**. Synthetic extra checks do not reopen it.

### Queue operating contract
1. Every hourly Lane 10 result/review/promotion requalification for **Newton** goes **as an append-only envelope receipt on this issue**, with source branch/commit, CPU vs GPU qualification, provenance and exact stage. Do not create more project queues. Project-specific origin-item QA threads and fork PRs remain valid evidence carriers per CONTROL_PLANE, and should link here.
2. Move a row to **READY_FOR_HITL** only after behavior, independent requirement/impact, fix, policy, current SHA/CI and non-duplication checks pass. **READY_FOR_HITL is not permission to post upstream**. The **origin gate remains Linear Todo / valid live-origin authority**. Human merges default branches only.
3. Before opening any fresh Newton upstream work, clear higher-priority live PR feedback and required checks. Honor shared six-worker claims; no new claim is made by this queue.
4. Credit original authors and independent reviewers, including CodeRabbit review of #4488 and bounceonme for its originating benchmark request, and retain existing provenance below.

---

## Historical work/evidence index (preserved verbatim below)


## Purpose
Project-level index for existing public **Newton** work, created at Kevin's request. [dot work index #64](https://github.com/kvnloo/oss-factory/issues/64) → this project → individual issue/PR or evidence record → [cross-repository index #14](https://github.com/kvnloo/oss-factory/issues/14).

This issue preserves historical coordination evidence under the single repo-level promotion queue. Preserve each contributor's ownership. Reuse existing downstream mirrors before creating a work-item issue.

## Registry and routing
**REGISTRY_PENDING.** This target is absent from the current main project registry. The existing work is indexed under OSS Factory coordination only; no per-project city ID or automation-routing registration is asserted.

Reuse existing [Newton registration PR #56](https://github.com/kvnloo/oss-factory/pull/56); preserve DigitalTwin ownership.

## Recovered work-item tracking
Snapshot: 2026-10-11. Evidence is historical unless a row explicitly reports a current requalification. **No unchecked candidate is automatically ready or authorized.** Exact commands remain in linked receipts; missing details are explicit backfill work.

| Work item / record | Known revision | Evidence | Status / blocker / next action | Owner / credit |
|---|---|---|---|---|
| [Inside-cone ray intersection](https://github.com/kvnloo/newton/commit/059125141e71f1e72e294c047fffd06ca4b32d3e) | `059125141e71f1e72e294c047fffd06ca4b32d3e` | 932 CPU checks;19 repository tests;pre-commit; evidence c8b28c09c608d6e853668938037f7398eb404bdc | **WAIT**. GPU conditional FP64 and cost untested. Next: verify current head, owner, canonical mirror and evidence before any promotion. | dot; original authors in receipt |
| [Smooth extrema](https://github.com/kvnloo/newton/commit/e4a8db7fd5a790a2e65ff80552411f7ba8c513a9) | `e4a8db7fd5a790a2e65ff80552411f7ba8c513a9` | Source/evidence f53f8d67fa8ae2417f7ae46279f57ca807f15fb2 remote verified | **WAIT**. Measured CPU overhead; GPU/whole solver unqualified; exact counts in receipt. Next: verify current head, owner, canonical mirror and evidence before any promotion. | Eric Heiden math; Eric Shi docs; dot correction |
| [Branchless numerical candidate](https://github.com/kvnloo/newton/commit/1c6c6811abd99eec5e871098f52fc80f3e07e230) | `1c6c6811abd99eec5e871098f52fc80f3e07e230` | 214 rows/8 groups; candidate8P/upstream2P;0.555ms forward/1.489ms backward | **WAIT**. Residual15.74%/12.19% original cost; upstream adoption HOLD, historical environment. Next: verify current head, owner, canonical mirror and evidence before any promotion. | dot; prior mathematical contributors |
| [#4600 owner CPU evidence](https://github.com/kvnloo/newton/commit/a5f6316ae19832fdb00064897188b6d02ba6d97e) | `a5f6316ae19832fdb00064897188b6d02ba6d97e` | Prior c7a91b48 CPU qualification;15-caseoracle6orderedcanonicalpairs;doc-only c9daf delta verified | **WAIT**. No current-head/GPU rerun;notuniversal axis proof;CLA nowclear,maintainer/GPUCI pending. Next: verify current head, owner, canonical mirror and evidence before any promotion. | Volodymyr Borysenko;gyeomannvidia;Kevin/dot evidence |
| [#4612 MuJoCo RK4](https://github.com/newton-physics/newton/issues/4612) | `79bbbe91e796f4e73a6ff7be69b6175c1261eb93` | Historical CPU evidence rejects naive poststep-RNE fix | **LOCAL_ONLY**. Local evidence publication withheld;no new permission or runtime;do not resume held scope. Next: resolve the recorded hold through its authorized owner; no retry implied. | existing owners;dot |
| [#4559 owner CPU acceptance](https://github.com/newton-physics/newton/pull/4559) | `b8ecfb40c2e793ed3540e0f830d6fd2f6ceef703` | 100steps mainjoint.761487/body2 vsownerboth.761487;focused1P/10subcases;critic reran witnesses | **LOCAL_ONLY**. Local portable packet only;CPU notCUDA;witness exits0 both and numerical mismatch constitutesRED. Next: resolve the recorded hold through its authorized owner; no retry implied. | cyanseek;Alain Denzler/adenzler-nvidia;dot |
| [#4629 heightfield](https://github.com/newton-physics/newton/pull/4629) | `cf2e3143a3a2c1f139b39cc3adb80169b94ea58f` | Heightfield base .0001→owner.10000000149;focused6P1skip;adjacent36P5skip;critic witness replay | **LOCAL_ONLY**. Local packet;no GPU acceptance. Next: resolve the recorded hold through its authorized owner; no retry implied. | HIROAKI GO/huster-wgm;dot |
| `#3024` helper (no published receipt) | Not recovered / local only | 4RED+2controls→6GREEN;29boundaryP | **WAIT**. Owner#3888 overlap;3baselineerrors250skips/no realUSD;no competing patch. Next: verify current head, owner, canonical mirror and evidence before any promotion. | existing#3888 owner;dot |
| [#4622 Delassus sizing](https://github.com/kvnloo/newton/commit/440d3ac45349cf6348bc20aa93cc91be47a4a39d) | `440d3ac45349cf6348bc20aa93cc91be47a4a39d` | Seven RED assertions to four GREEN tests; 63 adjacent CPU tests; pre-commit PASS. | **WAIT**. Later overlaps arcusbuilds `#4630`. Do not open a competing PR; owner integration and GPU acceptance remain. Next: verify current head, owner, canonical mirror and evidence before any promotion. | AntoineRichard; rubengrandia2949; nvtw4418; Kevin/dot |

## Per-item acceptance and update contract
- Link each origin issue/PR to exactly one canonical downstream work record where available.
- Record tested origin SHA, fork SHA, branch, exact commands, RED/GREEN or review evidence, independent QA, limits, owner, disposition and next action.
- Keep source-only, mocked, native CPU, native desktop and live-provider results distinct.
- Append timestamped result/blocker receipts; preserve earlier failures and author credit.
- Mirror only freshly qualified publication candidates to `#14`. Hermes handles authorized origin publication; this card grants no origin write, merge, signoff or permission recovery.
- Historical backfill is ongoing. Unknown branch names, exact commands, current origins or registry prerequisites remain gates.

<!-- END LEGACY ISSUE BODY -->

## Historical issue comments — immutable snapshots

### 6117582494 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6117582494

<!-- BEGIN LEGACY COMMENT 6117582494 -->
2026-10-12 bounded Newton correctness investigation, authorized by Kevin. Base/upstream main verified `624005a4478dc385180dd28775ca3dda96e93a69`; fork main `a6e1649b112e3d962b35f7dad66780dc55150588`. Isolated local branch `fix/ray-box-small-direction-normal`, worktree `/workspace/newton-raybox-work`. The selected sandbox has no recovered Newton clone or CPU Warp environment; a fresh clone was created. No existing work was edited.

Candidate (unverified): half-extents (1,1,1), origin (1+2^-23,0,0), direction (-2^-22,1,0) selects t=0.5 but normal reconstruction may skip the sole X face because slabs use MINVAL while normal selection uses EPSILON. Plan: existing kernel_test_geom CPU witness, RED regression, smallest justified fix, axis/sign/inside/control/tie checks, public intersect_ray if feasible, independent xhigh critic. CPU-only; no GPU acceptance. Patch remains local pending review.

Refreshed bounded GitHub searches: upstream issues and PRs for `ray box` and exact `ray_intersect_box`, downstream factory issues for `ray`, plus this card/comments. No matching active item found. RoboX2020 #4624 (opacity/BVH) and eric-heiden #4463 (heightfield cap) remain distinct and retain ownership. Original raycast contributors include Daniela Hase, Mustafa H, Alain Denzler and nvtw; no authorship transfer asserted. Existing dot cone05912514 is cone-only. #4612 and #4622/#4630 scopes untouched.

Routing remains #66 / #14 / #64 coordination only: registration PR #56 is open/unmerged, REGISTRY_PENDING. This receipt grants no origin publication, merge, signoff or CLA authority.

<!-- END LEGACY COMMENT 6117582494 -->

### 6117660509 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6117660509

<!-- BEGIN LEGACY COMMENT 6117660509 -->
2026-10-12 result for the [pre-test bounded concern](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6117582494): **LOCAL_CPU_REVIEW_PASS**, no publication/GPU acceptance.

Exact kernel_test_geom witness confirmed on upstream `624005a4478dc385180dd28775ca3dda96e93a69`:t=.5 normal(0,0,0). Smallest correction aligns normal reconstruction with slabs (`>=MINVAL`, previously `>EPSILON`); solved t, face checks, time tolerance and tie normalization unchanged. Public intersect_ray confirms correct(+1,0,0) in both CPU fast-math modes.

Local reviewed head `878ae1eb72016834ffb17cb62f11b8feb97b8f76`, branch `fix/ray-box-small-direction-normal`, base/merge-base `624005a4478dc385180dd28775ca3dda96e93a69`; clean worktree `/workspace/newton-raybox-work`. RED exact-base replay:2 tests16failed subcases. GREEN:20raycast tests including52new subcases (axes/signs,inside exits,above-threshold controls,ordinary and small-component ties,public modes); adjacent BVH/primitive16tests pass. Full pre-commit14PASS2inapplicableSKIP;diff check PASS. Independent xhigh critic:PASS/Fits,no material findings;72extra CPU probes acrossMINVAL boundary,tiny ties,miss/on-face controls.

Commands from worktree: `PYTHONPATH=. UV_CACHE_DIR=/workspace/uv-raybox-cache uv run --no-project --python /workspace/newton-raybox-venv/bin/python python /workspace/raybox-evidence/run-raybox-checks.py newton.tests.test_raycast` (adjacent replaces module argument with `newton.tests.test_bvh newton.tests.test_collision_primitives`); `UV_CACHE_DIR=/workspace/uv-raybox-cache UV_TOOL_DIR=/workspace/uv-raybox-tools PRE_COMMIT_HOME=/workspace/precommit-raybox-cache uvx pre-commit run -a`. Wheel-only Warp1.18.0/NumPy2.5.4 setup~10s;combined provisioned caches/env~674MiB. Evidence,review,logs,exact commands and portable fix.patch retained at `/workspace/raybox-evidence/`.

Final heads verified:upstream/main624005a;fork/main moved externally from initiala6e1649 to624005a (Git and GitHub API confirmation;this session did not push main). Feature branch absent remotely. Prepared feature-only fork push remains unexecuted. No upstream writes,main/force push,human signoff or CLA attestation. CPU-only;pre-existing large-t inverse-multiply/divide and absolute-tolerance reconstruction limits unchanged;no performance claim.

Original raycast authors and separate #4624/#4463/cone05912514 owners preserved;#4612 and#4622/#4630 untouched. Registration56 remainsopen/unmerged:REGISTRY_PENDING;coordination#66/#14/#64 only. Do not treat this receipt as GPU acceptance or registered automation routing.



## 2026-10-12 downstream publication and claim release

**DOWNSTREAM_PUBLISHED / BOUNDED_CLAIM_RELEASED.** Kevin authorized a normal, non-force feature-branch push. Published branch: `fix/ray-box-small-direction-normal` on `kvnloo/newton`. Verified remote head through Git and GitHub API: [`a94decc46cc28aa947baa9204ec043fffb0de330`](https://github.com/kvnloo/newton/commit/a94decc46cc28aa947baa9204ec043fffb0de330). Implementation remains [`878ae1eb72016834ffb17cb62f11b8feb97b8f76`](https://github.com/kvnloo/newton/commit/878ae1eb72016834ffb17cb62f11b8feb97b8f76); the final commit adds evidence only. Tested base and freshly checked upstream/fork main remain `624005a4478dc385180dd28775ca3dda96e93a69`. No tests are transferred to an untested origin revision.

Immutable privacy-clean [receipt at the published head](https://github.com/kvnloo/newton/blob/a94decc46cc28aa947baa9204ec043fffb0de330/evidence/ray-box-small-direction-normal.md) records commands, environment, reviewed file hashes, CPU limitation and ownership. RED is **16 failing subcases across two unique registered tests**, comprising 12 tiny-component face cases, two tiny-component ties and two public fast-math modes; 36 other subcases passed. GREEN is **20 unique raycast CPU tests**, including those two new tests and their **52 subcases** (50 direct-kernel, two public query modes), plus **16 adjacent CPU tests**. Independent xhigh critic PASS/Fits: **72 additional direct-kernel CPU probe cases**, comprising 18 MINVAL controls, 48 tiny-component ties and six zero/on-face/miss controls; these are not 72 registered tests. GPU acceptance and performance remain unqualified.

Pre-publication bounded ownership searches refreshed upstream open ray PRs, exact ray_intersect_box issues/PRs and factory box-normal claims; no competing matching item found. RoboX2020 #4624 BVH/opacity, eric-heiden #4463 heightfield caps and dot cone05912514 retain separate ownership, with original raycast authors credited. Complete final diff reviewed for private paths, secrets and generated artifacts: only source, tests, changelog and portable prose receipt published; raw logs, dependency caches and sandbox-path evidence stayed local.

The feature branch name was free and was created without overwriting any existing branch. Exact push: `git push fork HEAD:refs/heads/fix/ray-box-small-direction-normal`; remote SHA was read back with `git ls-remote` and the GitHub branches API. No upstream PR/comment, main/force push, registration merge, human signoff, CLA attestation or cleanup occurred. Existing promotion branches and canceled #4612 evidence remain untouched. Registration #56 remains pending; #66/#14/#64 are coordination only. This bounded investigation claim is released after verified downstream publication; downstream review and GPU qualification are future gates, not an active exclusive implementation claim.

<!-- END LEGACY COMMENT 6117660509 -->

### 6122355938 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6122355938

<!-- BEGIN LEGACY COMMENT 6122355938 -->
New current-base continuation, attributed to Hermes's [final #77 receipt](https://github.com/kvnloo/oss-factory/blob/8850cbbfc5a6fb41f48ce6de4b786f599b5ce7be/factory/evidence/promotion-wave-20261012.md): local successor `3f799f8694e1597b532a11e762ad18623eefa25f`, parent `2f154e45cf08ee2950c532f8523062ffabe2bdfb`, reports16 REDsubcases →36 CPUtests/113subcases,13 precommit hooksPASS/2skipped. This is not dot's rerun or a verified remote successor. Preserve publisheda94decc4/originalimplementation878ae1eb and originalcredit. Registry56,contributorCLA,authorization,GPU/fullsuite/hostedCIremain. Dashboard14 now separates originalpublishedhead from laterlocalqualification. Canceled #64 summary write remains stopped; this is distinct new continuation evidence.
<!-- END LEGACY COMMENT 6122355938 -->

### 6144643748 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6144643748

<!-- BEGIN LEGACY COMMENT 6144643748 -->
2026-10-13 bounded downstream Newton cylinder ray-normal correctness claim (Kevin-authorized investigation). Tested source candidate: upstream `newton-physics/newton` main `8157ac7e15374dbda4503eb1e2f7194af643e3f9`; fork main `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639`. Isolated clean worktree `/workspace/newton-cylinder-work`, local branch `fix/cylinder-ray-normal-cap-selection`; remote name was free when checked. No tests or source edits yet.

Candidate only, not executed proof: `ray_intersect_cylinder` currently classifies cap normals via absolute `EPSILON=1e-6` proximity after computing the hit. Unit cylinder (r=h=1), origin `(2,0,1-2^-22)`, direction `(-1,0,0)` should hit the curved side at `t=1` with `(+1,0,0)`, but source suggests `+Z`. A thin cylinder (`r=h=2^-20`, origin `(2^-19,0,0)`, direction `(-1,0,0)`) may yield a zero normal. Confirm with actual existing `kernel_test_geom` and Warp CPU before implementation.

Refreshed bounded searches of upstream open cylinder/raycast PRs and cylinder-normal issues, exact source claims and project #66 comments found no matching active cylinder ray-normal fix. nvtw #3842/#648 and original raycast contributors Daniela Hase, Mustafa H and Alain Denzler retain attribution. Open #4624 BVH opacity/empty path (RoboX2020), #4463 heightfield cap (eric-heiden), #4362 capsule/cylinder contact and #4643 MPR/GJK are distinct. Published ray-box branch `a94decc4`, cone and Hermes promotion work remain separate and untouched.

Scope: prove RED with existing kernel on CPU, implement smallest justified surface-normal correction, cover cap/side, thin scale, inside exits, rotations and deliberate rim ties, run focused/adjacent checks, and obtain independent xhigh review. Publish a non-force new branch to `kvnloo/newton` only after validation and fresh ownership/head checks, with exact receipt. CPU only; no GPU acceptance or upstream publication. Registry PR #56 is draft/open; #66/#14/#64 remain coordination only. This claim does not revive any canceled/denied action, including canceled #64 summary, #4612 evidence push or protected path reads. No upstream PR/comment, main/force push, signoff, CLA attestation or cleanup.



## 2026-10-13 result: downstream branch published; bounded claim released

**DOWNSTREAM_CPU_REVIEW_PASS / CLAIM_RELEASED.** Exact tested upstream/main and local merge base: `8157ac7e15374dbda4503eb1e2f7194af643e3f9`; refreshed upstream/main was unchanged immediately before push. Fork/main remained `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639`. Implementation commit: [`9c4f97b80676ea40065d7898eb89a57625a1aff2`](https://github.com/kvnloo/newton/commit/9c4f97b80676ea40065d7898eb89a57625a1aff2). Final privacy-clean evidence head: [`274716c32ab454268b13b00d20ceada8f0d640f8`](https://github.com/kvnloo/newton/commit/274716c32ab454268b13b00d20ceada8f0d640f8), branch `fix/cylinder-ray-normal-cap-selection`; exact remote SHA verified with `git ls-remote` and GitHub branch API. [Immutable receipt](https://github.com/kvnloo/newton/blob/274716c32ab454268b13b00d20ceada8f0d640f8/evidence/cylinder-ray-normal.md).

Actual existing `kernel_test_geom` on Warp1.18.0 CPU confirmed unit near-cap side t1/+Z instead of +X and tiny r=h=2^-20 t2^-20/zero instead of +X. Exact-base RED: **3 unique new tests, 25 subcases, 8 failing subcases** (six direct-kernel, two public `newton.intersect_ray` fast-math modes). Final GREEN: **21 unique raycast CPU tests**, including all25 new subcases; adjacent BVH/primitive **16 unique CPU tests**. Full `uvx pre-commit run -a`:14 applicable hooks pass,2 no-file skips; `git diff --check` passes. Independent xhigh critic: **Fits / Requirements and Standards pass**, five focused CPU methods pass and further thin/zero-height, barrel-rim and public small-radius probes pass. Intermediate cap tie and barrel profile concerns were found, corrected or bounded by preserving preexisting barrel-side normal behavior; exact resolution and CPU limits are in the receipt.

Exact commands used from the isolated worktree: `UV_CACHE_DIR=/workspace/uv-raybox-cache uv run --no-project --python /workspace/newton-raybox-venv/bin/python python -` with inline `unittest.defaultTestLoader.loadTestsFromName('newton.tests.test_raycast')`, and an analogous adjacent command with `loadTestsFromNames(['newton.tests.test_bvh','newton.tests.test_collision_primitives'])`; before importing tests, set `wp.config.kernel_cache_dir='/workspace/warp-cylinder-cache'`. Formatting command: `UV_CACHE_DIR=/workspace/uv-raybox-cache UV_TOOL_DIR=/workspace/uv-raybox-tools PRE_COMMIT_HOME=/workspace/precommit-raybox-cache uvx pre-commit run -a`. RED replay temporarily used `git show HEAD:newton/_src/geometry/raycast.py` from base, restoring patched source in `finally`; the exact artifact and portable commands are linked above.

Pre-push bounded ownership and exact source claim checks remained clear. Original raycast authors nvtw, Daniela Hase, Mustafa H and Alain Denzler retain credit. #4624/#4463/#4362/#4643, published ray-box, cone and Hermes promotion scopes remain separate; canceled #64 summary, #4612 evidence push and protected path reads were not retried. No main/force push, upstream PR/comment, registration merge, cleanup, signoff or CLA attestation. Registry #56 stays draft/open; #66/#14/#64 remain coordination only. CPU-only qualification; no CUDA/GPU, performance, full-solver or upstream acceptance claim. The bounded implementation claim is released after verified downstream publication.

<!-- END LEGACY COMMENT 6144643748 -->

### 6148201065 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6148201065

<!-- BEGIN LEGACY COMMENT 6148201065 -->
New CPU requalification against upstream `fc12268e82c589317866bea051462a882f736b58` (2026-10-13); this is new execution evidence, not a retry of the canceled #64 summary.

- **Cylinder prioritized:** existing remote `fix/cylinder-ray-normal-cap-selection` remains `274716c32ab454268b13b00d20ceada8f0d640f8`, implementation `9c4f97b80676ea40065d7898eb89a57625a1aff2`. Fresh exact-head replay: 37 registered CPU tests pass (21 raycast + 16 adjacent). Current upstream with the three frozen new tests: 8 failing subcases / 25 subcases. Unchanged implementation composed onto current upstream: the same 37 tests pass; tested uncommitted tree `b29c6dfd77c2d25ccecfd8ab4ac2621440fe0985` equals Git's clean merge tree.
- **Grazing box:** the remote remains `a94decc46cc28aa947baa9204ec043fffb0de330`; local successor `3f799f8694e1597b532a11e762ad18623eefa25f` is not advertised by any remote ref. Fresh exact-local replay: 36 registered CPU tests / 113 subcases pass. Current upstream with its two frozen new tests: 16 failing subcases / 52 subcases. The current-base composition passes the same 36-test selection; uncommitted tree `f196e54e9c9cfafb89f65a6969c288996db62318` equals Git's clean merge tree. These are separate candidates, not a tested combined stack.

Both current-base compositions passed native pre-commit: 14 passed, two inapplicable skips, source and lockfile unchanged. A separate read-only reviewer examined both substantive patches and relevant callers, found no supported correctness findings, and judged both to fit Newton. Fresh execution used CPython 3.12.13, Warp 1.18.0 and NumPy 2.5.4, with CUDA disabled before initialization. Original raycast authors nvtw, Daniela Hase, Mustafa H and Alain Denzler retain credit; Kevin Rajan's implementation metadata is preserved. Barrel-side proximity behavior remains unchanged. No CUDA, full-solver, performance or hosted candidate CI acceptance follows.

Admission clarification: current upstream #4488 carries an authoritative EasyCLA success for `kvnloo` (status description: “EasyCLA check passed. You are authorized to contribute.”); GitHub resolves both original implementation authors to that same account. This is positive account-level evidence, not a check on either proposed new candidate. Registry #56 remains draft/open and Newton remains absent from canonical `main` projects.json. Registry reconciliation and operation-specific publication permission remain gates. No origin post, branch update, merge, signoff or CLA attestation was performed. Protected #4612/#4641 and sibling work remain untouched.

<!-- END LEGACY COMMENT 6148201065 -->

### 6150120453 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6150120453

<!-- BEGIN LEGACY COMMENT 6150120453 -->
2026-10-13 bounded Newton sphere–capsule contact-sign investigation, authorized by Kevin. Exact upstream main `1624cc70e2e030ca3dca6772b4243900202d24fb`; fork main `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639`. Clean isolated worktree `/workspace/newton-sphere-capsule-work`, branch `fix/sphere-capsule-segment-projection`; remote name free when checked. No tests or source edits yet.

Candidate, not executed proof: `newton/_src/geometry/collision_primitive.py` `closest_segment_point` adds `1e-6` to the segment squared length. For capsule center(0,0,0), axis+Z, half-length0.0005, radius0.00025 and sphere center(0,0.00045,0), radius0.00025, analytic signed distance is -0.00005, normal sphere→capsule (0,-1,0), midpoint(0,0.000225,0). Source projection appears to use t=.25 instead of .5, possibly making a false positive gap. Axis reversal should preserve the geometry. Confirm with the existing real `test_sphere_capsule_kernel` and Warp CPU before claiming a runtime failure. Do not use `distance_point_to_capsule` test helper as oracle; it repeats the biased formula.

Refreshed source and ownership: existing open nvtw #4362 adds capsule–cylinder handling but retains this helper; #4643 MPR/GJK and #4227 deformable contacts are distinct. Closed #482/#693 attribute original primitive and MuJoCo-Warp integration to nvtw. No exact repair found in current #66/#14 records, bounded public PR search, or checked source diffs. Eric Heiden helper consolidation #4133 remains separate. Original and MuJoCo-Warp owners retain credit; no transfer asserted. Published ray-box/cylinder branches, cone, Hermes work and all prior canceled/denied actions remain untouched, especially canceled #64 summary, #4612 evidence push and protected path reads.

If real RED confirms the defect, make only the smallest justified segment-projection correction, cover axis reversal, scale, interior/endpoint and zero-length behavior, then run primitive/adjacent CPU gates and pre-commit with independent xhigh review. Inspect MJWarp compatibility and disclose unavailable downstream checks. Publish a new non-force `kvnloo/newton` feature branch only if reviewed, validated and still independent; preserve exact evidence, remote SHA and author credit. CPU evidence grants no GPU or full-solver acceptance. Registry #56 remains draft/open; #66/#14/#64 are coordination only. No upstream writes, main/force push, cleanup, new credentials, signoff or CLA attestation.

## 2026-10-13 downstream sphere–capsule publication and claim release

**DOWNSTREAM_PUBLISHED / BOUNDED_CLAIM_RELEASED.** The exact-base CPU witness confirmed a false positive gap: base `1624cc70e2e030ca3dca6772b4243900202d24fb` reported distance `+1.4781486243009567e-05` and a tilted normal where analytic distance is `-0.00005`, midpoint `(0,0.000225,0)`, and normal `(0,-1,0)`. The bounded correction removes the additive `1e-6` segment-length bias and handles exactly zero-length segments. The [implementation commit](https://github.com/kvnloo/newton/commit/8e135adef739488778acac7ceec50c98385c2101) changes only the helper, a registered analytic regression, and a changelog fragment.

Exact-base RED replay: **one unique test, 13 failing subcases of 15, exit 1**. GREEN: **15 collision-primitive CPU tests** and **four targeted narrow-phase CPU tests** passed. Full pre-commit passed 14 hooks with two inapplicable skips; `git diff --check` passed. Independent xhigh critic: Fits, no supported defects; independently passed the analytic test plus seven direct projection and six segment-pair CPU probes. Python 3.12.14, Warp 1.18.0, NumPy 2.5.4, CPU only. MuJoCo-Warp and mujoco are absent, so that downstream runtime suite, GPU execution, full-solver compatibility and performance remain unqualified.

Normal non-force feature-branch push to `kvnloo/newton`: `git push fork HEAD:refs/heads/fix/sphere-capsule-segment-projection`. Git and GitHub API verified [remote head `befeb3fd1d4ef05674e472728c6c563059b78a35`](https://github.com/kvnloo/newton/commit/befeb3fd1d4ef05674e472728c6c563059b78a35). The final commit adds only a [portable privacy-clean receipt](https://github.com/kvnloo/newton/blob/befeb3fd1d4ef05674e472728c6c563059b78a35/evidence/sphere-capsule-short-segment.md) with commands, counts, file hashes, review and limits. Tested base and current upstream main remain `1624cc70e2e030ca3dca6772b4243900202d24fb`; fork main was `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639` at push. The feature branch name was free before push. Raw logs, caches and local paths were not published.

Current open PR review found nvtw #4362 capsule–cylinder is separate and retains its owner. Original primitive/MuJoCo-Warp contributors retain credit; #4643/#4227 and previously published ray-box/cylinder work remain separate. No upstream PR/comment, main/force push, signoff, CLA attestation, cleanup, or previously canceled action occurred. Registration PR #56 remains draft/open; #66/#14/#64 are coordination only. This bounded implementation claim is released after verified downstream publication.

<!-- END LEGACY COMMENT 6150120453 -->

### 6156505799 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6156505799

<!-- BEGIN LEGACY COMMENT 6156505799 -->
2026-10-13 bounded Newton hollow-mesh inertia cache investigation, authorized by Kevin. Exact upstream main/base `71eeba16a717d4dcf0b9dd7056a7b6f863054dc3`; fork main `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639`. Isolated clean worktree `/workspace/newton-hollow-inertia-work`, branch `fix/hollow-mesh-inertia-thickness`; remote branch name free. The saved cloud executor is available. No tests or source edits yet.

Candidate only: Mesh(is_solid=False, compute_inertia=True) precomputes unit-density mass properties at default thickness 0.001. `compute_inertia_shape` appears to reuse that cache for a matching hollow mesh despite a requested different thickness, whereas the uncached path forwards thickness. Check the existing closed unit-cube fixture and source's two-sided triangle extrusion: density 1, identity scale, requested thickness 0.01 should give volume/mass 0.12 instead of default-cache 0.012. Compare full inertia tensor and COM, zero thickness, then `ModelBuilder.add_shape_mesh` with hollow margin 0.01. Cover default thickness, solids, and manually supplied cached properties. This is not yet a confirmed runtime defect; prove pristine-base CPU RED first.

Refreshed origin/fork heads, open PR titles and factory #66/#14 exact-keyword claims. No matching active exact claim found. Astra's broader changed-file dedupe found upstream #4442 warning categories separate, fork #6/merged #4324 nonuniform solid scaling separate; will preserve Mohit-Ak's ownership and not duplicate scaling. Existing ray-box, cylinder, sphere-capsule, cone, numerical extrema and all canceled/denied work stay untouched. Follow AGENTS, coding/review/contribution rules and changelog instructions. If validated, smallest compatible fix, focused/adjacent CPU tests, pre-commit, independent xhigh review, privacy-clean receipt, then only a new normal downstream feature push. No GPU/native acceptance, upstream PR/comment, main/force push, cleanup, credentials, signoff or CLA attestation. Registry #56 pending; #66/#14/#64 are coordination, not registered routing. Canceled #64 summary and #4612 evidence push remain blocked.

## 2026-10-13 hollow-mesh result and claim release

**DOWNSTREAM_PUBLISHED / BOUNDED_CLAIM_RELEASED.** On pinned upstream `71eeba16a717d4dcf0b9dd7056a7b6f863054dc3`, direct unit-cube hollow volume/mass at density 1 was ~0.12 for requested thickness 0.01, while `Mesh(compute_inertia=True)` and `ModelBuilder.add_shape_mesh` with hollow margin 0.01 reused default-thickness ~0.012. Zero thickness direct volume was zero but the cached path stayed ~0.012. The source-only fix records generated cache provenance, recomputes when hollow thickness or scale differs, and retains explicitly authored mass properties. It leaves Mohit-Ak's distinct solid scaling correction intact.

[Implementation `1bb00d2ecd1648defb91954b2435c024c0bef227`](https://github.com/kvnloo/newton/commit/1bb00d2ecd1648defb91954b2435c024c0bef227): source, analytic regression, and changelog. RED on pristine base: one unique test, three failures (zero, 0.01, builder). GREEN: 30 CPU tests (18 inertia, nine mesh cache, one builder replication mesh-lifetime, two model controls). Full pre-commit: 14 PASS, two inapplicable skips; diff check PASS. Independent xhigh critic: PASS/Fits, 79 additional CPU property comparisons; its public-copy finding was fixed before publication. [Portable receipt](https://github.com/kvnloo/newton/blob/4e73fd9e6c4694f068010061e73ef4a2a5e3ff01/evidence/hollow-mesh-inertia-thickness.md) records commands, exact file hashes, scope, and limits.

Normal non-force feature-only push: `git push fork HEAD:refs/heads/fix/hollow-mesh-inertia-thickness`. Git and GitHub API both verified [remote head `4e73fd9e6c4694f068010061e73ef4a2a5e3ff01`](https://github.com/kvnloo/newton/commit/4e73fd9e6c4694f068010061e73ef4a2a5e3ff01). The final commit adds only the privacy-clean receipt. At push, upstream main equaled the tested base, fork main was `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639`, and the branch name was free. Worktree clean. Raw logs, caches and local paths were not published.

GPU unavailable. SciPy absent: an adjacent mesh-approximation test stopped on its convex-hull path, so actual convex-hull remeshing is unqualified; the critic used a stub for the mass-property transfer probe. No GPU/native/performance acceptance. Original geometry/inertia contributors retain credit; Mohit-Ak #4324 and fork #6 solid scaling are separate. Published ray-box, cylinder, sphere-capsule and other work stayed untouched. No upstream PR/comment, main/force push, cleanup, signoff, CLA attestation, or canceled action occurred. Registration #56 remains draft/open; #66/#14/#64 are coordination only. This bounded claim is released after verified downstream publication.

<!-- END LEGACY COMMENT 6156505799 -->

### 6157411860 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157411860

<!-- BEGIN LEGACY COMMENT 6157411860 -->
2026-10-13 append-only CPU convex-hull validation handoff for the [published hollow-mesh inertia result](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6156505799). **DOWNSTREAM_VALIDATION_PUBLISHED / NO NEW EXCLUSIVE CLAIM.** This follows the previously published source head `4e73fd9e6c4694f068010061e73ef4a2a5e3ff01`; the original branch and all other branches remain unchanged.

SciPy 1.18.1 was installed from its official wheel into the existing isolated Python 3.12/Warp 1.18/NumPy 2.5 CPU environment under ordinary permissions. Real SciPy convex-hull remeshing now passes the original provenance transfer case, builder mass/COM/full tensor, zero/default/0.01 thickness, in-place replacement, and authored overrides. Exact pristine upstream base `71eeba16a717d4dcf0b9dd7056a7b6f863054dc3` with the final two-test module: **two tests, four failures**. Exact prior published head with that module: **two tests, one failure**. The newly discovered failure was `remesh_mesh(..., method="convex_hull", recompute_inertia=True, inplace=True)`: it recomputed hollow mass properties but retained the old generated-value snapshot, so a changed closed hull returned mass `0.2488672882` rather than direct thickness-0.01 mass `2.4886846542`.

The [narrow follow-up commit](https://github.com/kvnloo/newton/commit/2e3e12bff320e49cfc812407caed8b022fa7d040) refreshes that snapshot after in-place automatic hollow recomputation and adds the RED regression/changelog. **GREEN: 51 unique CPU tests**, including two new real-hull regressions, 18 inertia, nine mesh-cache, one builder mesh-lifetime, three selected model tests, and 18 convex-hull remeshing tests. Full pre-commit: 14 PASS/two inapplicable skips; diff check PASS. Independent xhigh critic: PASS/Fits, same RED/GREEN counts and 370 additional real-hull property comparisons. The exact commands, final source/test hashes, environment, and limits are in the [privacy-clean receipt](https://github.com/kvnloo/newton/blob/93c38ebbcc1550ecfa622709147c7793b3ae7263/evidence/hollow-mesh-real-convex-hull-cpu.md).

Normal non-force feature-only push: `git push fork HEAD:refs/heads/validation/hollow-mesh-convex-hull-cpu`. Git and GitHub API verified [remote head `93c38ebbcc1550ecfa622709147c7793b3ae7263`](https://github.com/kvnloo/newton/commit/93c38ebbcc1550ecfa622709147c7793b3ae7263). Merge base is the prior published head. Upstream main has since moved to `bb25fc0c679e7a81f5080c11962961f8cccb5d72`; these tests are explicitly pinned and were not transferred to that untested revision. Fork main remains `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639`. The branch name was free before push.

One unrelated pre-existing in-place remesh limitation remains on both pristine and patched source: reducing a default 24-vertex duplicated box to an eight-vertex hull can fail intermediate index validation. The new regression uses a closed eight-vertex geometry and exercises the supported path. GPU, full-solver, native downstream and performance acceptance remain untested. Original Newton geometry/inertia contributors and Mohit-Ak's distinct solid-mesh scaling work retain credit. No upstream PR/comment, main/force push, dashboard body overwrite, cleanup, signoff, CLA attestation, or canceled action. Registration PR #56 remains draft/open; #66/#14/#64 remain coordination only. No CLA/publication gate changed.

<!-- END LEGACY COMMENT 6157411860 -->

### 6157447469 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157447469

<!-- BEGIN LEGACY COMMENT 6157447469 -->
2026-10-13 bounded Newton in-place mesh remeshing topology-order investigation, authorized by Kevin. Current upstream main/base `bb25fc0c679e7a81f5080c11962961f8cccb5d72`; isolated clean downstream worktree/branch `fix/inplace-remesh-topology-order`, remote name free. Existing hollow inertia branch `4e73fd9e` and real-hull follow-up `93c38ebb` remain untouched.

Candidate only, before tests/source edits: `newton.utils.remesh_mesh(..., method="convex_hull", inplace=True)` receives a default box with 24 duplicated vertices and a real 8-vertex convex hull. It assigns `mesh.vertices` before `mesh.indices`; the vertex setter validates old indices (up to 23) against the new vertex count and raises. Confirm pristine-head CPU RED using actual SciPy hull, determine whether other remeshing paths share this update-order invariant, then make the smallest justified atomic topology update and add valid/invalid geometry controls. Run exact-head adjacent CPU tests, full pre-commit and independent xhigh review. CPU only; no GPU/full-solver/performance assertion.

Fresh ownership review: upstream open issues and PRs for remesh/topology/convex hull plus exact `remesh_mesh` search, fork PRs, and factory #66/#14 comments show no matching active repair or claim. Upstream #4428 concerns USD convex-decomposition limits in builder/USD code; #4461 binary .msh loading changes `newton/_src/utils/mesh.py`; both are separate and owners retained. Existing #14 handoff for hull-inertia validation remains distinct. GitHub search intermittently returned 502; direct open PR/issue API lists and source diffs were inspected. Follow Newton AGENTS, coding/review/contribution and changelog rules. No upstream PR/comment, main/force push, cleanup, credentials, registration/CLA/signoff change, or canceled/denied retry. Registry #56 remains draft/open; #66/#14/#64 are coordination only.

<!-- END LEGACY COMMENT 6157447469 -->

### 6157514846 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157514846

<!-- BEGIN LEGACY COMMENT 6157514846 -->
2026-10-13 append-only result for the [pre-test in-place remesh topology claim](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157447469): **DOWNSTREAM_PUBLISHED / BOUNDED_CLAIM_RELEASED**.

On exact upstream main/base `bb25fc0c679e7a81f5080c11962961f8cccb5d72`, a default 24-duplicated-vertex box remeshed by the real SciPy convex-hull path returns eight vertices, but sequential in-place setters raise while validating old index23 against the new count8. The real fast-simplification quadratic path likewise returns 14 vertices and raises against old index23. A controlled invalid-index output leaves vertices partially changed on base. RED with the final test module: **four unique CPU tests, two errors, one failure, one pass; exit1**.

[Implementation `0528dda09096b8f4658aa5aba8d1a1a9df6bdbcd`](https://github.com/kvnloo/newton/commit/0528dda09096b8f4658aa5aba8d1a1a9df6bdbcd) replaces the two sequential setters with Newton's existing atomic `_replace_geometry`, which validates both arrays before mutation and invalidates geometry caches once. GREEN: **69 unique CPU tests** across four new tests, full convex-hull remeshing, mesh cache, mesh validation, and three model controls. Both real remeshers, valid shrinkage, invalid-output rollback, CPU finalized-cache refresh and unchanged non-in-place behavior passed. Full `uvx pre-commit run -a`: 14 hooks PASS, two no-file skips; `git diff --check` PASS. Independent xhigh critic: Fits, no material findings; four tests independently rerun plus 38 additional cases across all five dispatch paths. The other three backend methods used controlled outputs. [Immutable privacy-clean receipt](https://github.com/kvnloo/newton/blob/bf11c8f3ccd87573c1cb37849898458b64d6552e/evidence/inplace-remesh-topology-order.md) records reproducible commands, hashes, environment, exact counts and limits.

Published new fork-only feature branch by normal non-force `git push fork HEAD:refs/heads/fix/inplace-remesh-topology-order`. Git and GitHub API verified [full remote head `bf11c8f3ccd87573c1cb37849898458b64d6552e`](https://github.com/kvnloo/newton/commit/bf11c8f3ccd87573c1cb37849898458b64d6552e). Merge base and freshly checked upstream main match the tested `bb25fc0c679e7a81f5080c11962961f8cccb5d72`; fork main remains `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639`. Remote branch name was free before push; worktree clean. Only source, tests, changelog and portable receipt were published. The optional repository-declared SciPy 1.18.1 and fast-simplification 0.2.0 were obtained as official wheels in the isolated CPU sandbox; raw logs, caches and local paths remain local.

Fresh upstream PR/issue and factory #66/#14 checks found no matching active owner. Upstream #4428 USD convex-decomposition limits and #4461 binary .msh loading are distinct; their owners and original geometry/remeshing contributors including nvtw retain credit. Earlier hollow-inertia, ray-box, cylinder, sphere-capsule, numerical and canceled scopes remain untouched. CPU correctness only; GPU, full-solver, native downstream and performance acceptance were not tested. No upstream PR/comment, main/force push, cleanup, dashboard overwrite, signoff, CLA attestation, or denied/canceled retry. Registration PR #56 remains draft/open; #66/#14/#64 remain coordination, not registered routing. No publication gate changed.

<!-- END LEGACY COMMENT 6157514846 -->

### 6157542814 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157542814

<!-- BEGIN LEGACY COMMENT 6157542814 -->
2026-10-13 bounded Newton CPU composition validation, authorized by Kevin. Exact fresh upstream main/base `bb25fc0c679e7a81f5080c11962961f8cccb5d72`; fork main `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639`. Clean separate local branch/worktree `validation/hollow-inertia-remesh-composition`, remote name free. No tests or edits yet on this branch.

Scope is interaction only: preserve and merge, rather than rewrite/cherry-pick, published hollow-inertia/real-hull branch `93c38ebbcc1550ecfa622709147c7793b3ae7263` and independent atomic-topology branch `bf11c8f3ccd87573c1cb37849898458b64d6552e` into the fresh base. Both touch `newton/_src/geometry/utils.py`; resolve any composition overlap without altering original branches/commits. Then test real shrink and in-place hollow inertia recomputation at requested thickness, authored overrides, invalid-geometry rollback, finalized CPU cache refresh and solid control, plus the union of relevant CPU suites. Independent xhigh critic, pre-commit, privacy-clean receipt. Publish only a separate non-force downstream composition/evidence branch if validated. If RED interaction appears, record it before a minimal isolated correction.

Refreshed factory #66/#14, upstream/fork heads and open upstream PR text; no exact competing composition claim found. #4428 USD convex-decomposition limits, #4461 binary mesh loading and prior geometry owners remain separate. Source work was already reviewed/published on its own branches; this claim is not a new competing implementation or merge authorization. Original authors and existing commit ancestry will be retained. CPU only; no GPU, full solver, performance claim. No upstream PR/comment, main/force push, cleanup, registration/CLA/signoff change, dashboard overwrite, denied-action retry, or OpenShell #67 action. #56 remains draft/open; #66/#14/#64 are coordination only.

<!-- END LEGACY COMMENT 6157542814 -->

### 6157594627 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157594627

<!-- BEGIN LEGACY COMMENT 6157594627 -->
2026-10-13 append-only result for the [bounded composition claim](https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157542814): **DOWNSTREAM_COMPOSITION_PUBLISHED / BOUNDED_CLAIM_RELEASED**.

On fresh upstream base `bb25fc0c679e7a81f5080c11962961f8cccb5d72`, merged the existing hollow-inertia/real-hull head `93c38ebbcc1550ecfa622709147c7793b3ae7263` and atomic remesh-topology head `bf11c8f3ccd87573c1cb37849898458b64d6552e`. Merge commits `9f451c3c592c8dc2b11bed5955f7a8437951b3a8` and `471015a8e3a6d703a537e7729e0af3d6c38401b6` retain both original heads as ancestors; Git merged without conflict. Neither original branch or source patch was rewritten. New composition-only tests and [privacy-clean evidence receipt](https://github.com/kvnloo/newton/blob/e91951d906ecf9b7444047ffe956b8e9d17a8022/evidence/hollow-inertia-remesh-composition.md) are on the separate branch.

Real CPU hull shrinkage (24 to 8 vertices), hollow recomputation at requested thickness through `ModelBuilder`, authored-property preservation, invalid-output rollback including generated-inertia provenance and finalized mesh identity, and solid control pass. The deduplicated adjacent CPU union is **95 unique tests, 95 passed**. Full `uvx pre-commit run -a` and `git diff --check` passed. Independent xhigh critic: PASS/Fits, 10 focused tests and 192 real hull/quadratic property cases plus four rollback probes; no supported composition defect. CPU Warp 1.18.0, Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1 and fast-simplification 0.2.0. CUDA driver unavailable; no GPU, full-solver or performance claim.

Published by normal non-force fork-only push; readback confirms [branch/head `e91951d906ecf9b7444047ffe956b8e9d17a8022`](https://github.com/kvnloo/newton/tree/validation/hollow-inertia-remesh-composition). Upstream main still equals tested base; fork main remains `4e1ef8c51d08042fadb01cc7c7a41278a2ea8639`. Worktree clean. Original Newton contributors and distinct scope owners retain credit. No upstream PR/comment/merge, main/force push, registry edit, CLA/signoff action, dashboard overwrite, or OpenShell #67 action. A fresh check now reports Newton PR #56 merged, a change since the claim; this receipt makes no inference of registered factory routing. #66/#14/#64 remain the agreed coordination path until separately established.

<!-- END LEGACY COMMENT 6157594627 -->

### 6157605907 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157605907

<!-- BEGIN LEGACY COMMENT 6157605907 -->
Correction to my composition receipt https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6157594627: its statement that “Newton PR #56 merged” is wrong for the registration gate. I queried https://github.com/newton-physics/newton/pull/56, an unrelated PR numbered 56 that merged in 2025. The canonical registration PR is https://github.com/kvnloo/oss-factory/pull/56; fresh direct GitHub API read reports state=open, draft=true, merged=false, merged_at=null. Factory registration remains pending. No routing or gate changed. The composition branch evidence file contains no merged-status assertion, so no evidence/source commit or test rerun is needed; branch head remains e91951d906ecf9b7444047ffe956b8e9d17a8022.
<!-- END LEGACY COMMENT 6157605907 -->

### 6179242498 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6179242498

<!-- BEGIN LEGACY COMMENT 6179242498 -->
<!-- oss-factory:v1
{"id":"20261014-lane10-newton-4488-cpu-contract-002","kind":"result","project":"oss-factory","from":"chatgpt","to":["factory"],"status":"done","parent_id":"20261011-ayo-project-newton-001","dedupe_key":"lane10:newton:4488:preload-launch-contract:20261014","civ":{"city_id":"oss:oss-factory"},"origin_url":"https://github.com/newton-physics/newton/pull/4488","evidence_urls":["https://github.com/kvnloo/newton/pull/25","https://github.com/kvnloo/oss-factory/pull/56"]}
-->

### Lane 10 review-validation wave — 2026-10-14 (CHECK / source-only)

**Existing open origin:** newton-physics/newton#4488 remains draft at `8f9c63f403def56d72fe8c0a7dcfe15a2fb58803`, with **two unresolved CodeRabbit threads** (tile preload `block_dim`, CUDA-only test false pass); `pre-commit.ci - pr` reports failure, while EasyCLA and CodeRabbit statuses report success. No maintainer-requested change from a human reviewer was established in the fetched review timeline. No origin update was made.

**Already-staged downstream remedy:** kvnloo/newton#25 at `9f082bfe9c206e41f935fcba3dea29276b51ecc8` supplies `block_dim=64` for fused/no-mimics, `256` otherwise, and registers CUDA-only tests with `get_cuda_test_devices()`. Exact inspected constructor and `wp.launch_tiled` code select the same generated module, model device, and block dimension. **Do not mint a duplicate patch or claim review threads resolved at origin**.

**Independent CPU-only check produced in ChatGPT Lane 10:** Python 3.13.5; **10/10** unittest checks on a source-shaped AST fixture transcribed from the inspected downstream head (4 fused/mimic combinations, CUDA registration source contract, and eight negative mutation controls; no Newton import). Preload/launch dimensions for `(fused,mimics)=(0,0),(0,1),(1,0),(1,1)` are **256/256, 256/256, 64/64, 256/256**. This does **not** execute production source, Newton, Warp, device kernels, graph capture or performance.

**New evidence gap:** existing #4488 regression only constructs the solver; it does not call `SolverFeatherstone.step()` or `wp.ScopedCapture()`. A separate runnable 18-DOF CUDA capture+replay handoff script was prepared locally for both fusion settings under conflicting CPU active-device context; Python syntax compilation passed, execution exited 2 with `SKIP_missing_dependency: No module named 'warp'`. Script and JSON report are available in this ChatGPT session but not committed to GitHub. No GPU claim or end-to-end correctness claim.

**Next action:** on an isolated downstream #25 checkout, run the CPU AST contract against actual files; run the CUDA capture/replay probe using native Warp, record GPU/Warp/SHA/exit/logs, and diagnose the existing required pre-commit failure. Then update the **existing** authorized #4488 PR, if allowed by its HITL/live-thread gate. Recheck #56 registry draft before routing Newton as a factory project. No new global worker slot, lease, origin action, merge, or source branch claimed.

**Credit:** CodeRabbit raised both #4488 review findings; the existing branch/fork is owned by kvnloo; bounceonme requested the #4265 benchmark; the original reported issue is #4487. This isolated source-contract harness and probe are additional ChatGPT Lane 10 test artifacts, not an upstream fix.
<!-- END LEGACY COMMENT 6179242498 -->

### 6226150094 — kvnloo
Source: https://github.com/kvnloo/oss-factory/issues/66#issuecomment-6226150094

<!-- BEGIN LEGACY COMMENT 6226150094 -->
<!-- oss-factory:v1
{"id":"20261016-newton-raybox-origin-4775","kind":"result","project":"newton","from":"chatgpt","to":["factory"],"status":"done","dedupe_key":"newton-physics/newton#4775:ray-box-minval:28795055744fe685ecb0de959d5d6cc01eca3c5f","origin_url":"https://github.com/newton-physics/newton/pull/4775","branch":"promotion/ray-box-normal-minval-20261016","commit":"28795055744fe685ecb0de959d5d6cc01eca3c5f","civ":{"city_id":"oss:newton"}}
-->

**ACTIVE_ORIGIN_REVIEW / DRAFT — promoted source into [newton-physics/newton #4775](https://github.com/newton-physics/newton/pull/4775), with supported [bug #4774](https://github.com/newton-physics/newton/issues/4774).** Fork [#26](https://github.com/kvnloo/newton/pull/26) carries the same exact source branch: [2879505](https://github.com/kvnloo/newton/commit/28795055744fe685ecb0de959d5d6cc01eca3c5f), current origin base `bd69b56dc00693976c2fea9bd89afd863178140c`, three files and three commits ahead/zero behind. The mismatch `slab >=MINVAL` vs `normal >EPSILON` causes an actual valid hit to report zero normal; the narrow correction aligns selection without changing distance or tie logic.

Prior donor [878ae1e](https://github.com/kvnloo/newton/commit/878ae1eb72016834ffb17cb62f11b8feb97b8f76): CPU Warp 1.18.0 20/20 raycast + 16/16 adjacent pass; 52 new subcases, RED 16 failing subcases across 2 unique tests before; 72 independent CPU probes and pre-commit pass on Oct12 source. **Fresh current-base CPU Warp/CUDA tests NOT_RUN**, no GPU/solver/performance claim. Remote source/test/changelog readback verified; upstream EasyCLA SUCCESS, pre-commit.ci pending, GitHub Actions jobs `action_required` (approval, NOT green), CodeRabbit pending. Original nvtw, Daniela Hase, Mustafa H, Alain Denzler retain code provenance. Upstream #4488 was closed in favor of another owner's #4762; this independent raycast issue does not revive Featherstone or misattribute that work. No default merge/reviewer ping.

**Next:** current-head native Warp CPU + pre-commit gate and maintainer review, then mark original draft ready if checks warrant. Source reference is already upstream; never create another competing PR.
<!-- END LEGACY COMMENT 6226150094 -->
