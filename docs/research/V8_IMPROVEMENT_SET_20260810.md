# V8 improvement set (2026-08-10)

Prioritized consolidation of three evidence streams: the reward/termination
audit ([V8_REWARD_TERMINATION_AUDIT.md](V8_REWARD_TERMINATION_AUDIT.md)), the
compact update-20k failure anatomy (development 546/720 exact, 163 of 174
failures on foundations), and the sampler rotation analysis (foundation family
pinned at depth 1). Each item names its decision evidence; nothing here
launches a job by itself.

## P0 — `continuous_banded_v2`: per-condition graduation (implemented here)

Defect: `continuous_banded_v1` rotates each family's 75% frontier as one block
and advances only when *every* condition at the active depth is mastered,
while preview conditions are barred from graduating at all. Three stubborn
depth-1 foundations (slab, irregular, bearing walls) therefore pinned the
whole family: ~6.45% assignment mass per depth-1 cell against ~0.617% per
depth-2 foundation. This rebuilds the staged-gate pathology the continuous
sampler was introduced to remove, and the D1 diagnostic may yet show the three
jailers are horizon-limited at 450 — i.e. the rotation condition was
potentially unsatisfiable.

Change (v1 behavior untouched, new named rule):

- 10% per-family floor unchanged; the remaining 90% covers **every unmastered
  condition** of the family, weighted `2**(2 - depth)` so shallow work leads
  without starving anyone;
- any condition with an eligible window (>=32 episodes, EMA >= 0.80) graduates
  regardless of depth; demotion (< 0.65) unchanged;
- a v2 sampler resumes a v1 checkpoint via an explicit one-way migration: the
  stored probabilities are validated against the v1 rule for the restored
  mastery, then recomputed under v2.

Application rule: both R2 children restore one common sampler state produced by
the explicit v1-to-v2 migration. The migration is a pre-fork prerequisite, not
an arm-level treatment. The before/after-v2 trunk comparison is deliberately
informal; the mechanistic mass evidence above is the justification, not a
paired run.

Files: `utils/pooled_sampler.py`, `utils/accepted_bank.py`, `train_mixed.py`,
`scripts/verify_continuous_sampler_checkpoint.py`,
`tests/test_continuous_banded_sampler.py`. The sampler suite passes 9/9 with
three new v2 tests (depth-weighted full support; straggler cannot pin +
any-depth graduation; v1-to-v2 migration and round-trip). Local failures in
`test_training_utils.py` / `test_reward_anneal.py` /
`test_v8_continuous_graph.py::test_real_graph_initial_mass...` are a
pre-existing terra-revision mismatch (`RewardStage`,
`jax.experimental.maps`) and occur identically without this change. A new
fork launcher must pass `--accepted-bank-sampler-profile continuous_banded_v2`
and sampler rule `continuous_banded_v2`; the existing named preset remains v1,
so R2 needs one explicit v2 preset rather than mismatched CLI overrides.

## P1 — R2-gating diagnostics and independent treatments (audit D0–D6)

Run as specified in the audit. Two sharpenings:

- D1/D2 also answer the *rotation feasibility* question: if 900-step
  inference solves the large slab/irregular/bearing-wall maps, depth-1
  mastery at 450 was unwinnable and v2's unpinning is not optional but
  required.
- D3: the runtime computes an action-mask helper but emits an all-zero
  informational mask to the policy; obstacle feasibility is a mechanics/action
  treatment, never a reward bonus.
- D4b: the provisional 34-map overlap query found only large depth-1 adjacent
  foundations (19 slab, 9 bearing walls, 3 irregular, 2 courtyard-pads, 1
  courtyard), with no `d12`, `d16`, or apron maps. Materialize the rows and
  dwell-cost grid before freezing constants; the supported mechanism is
  volume/per-map normalization, not excessive remote-haul reward magnitude.

## P2 — R2 normalized material-potential screen (accepted design, not implemented)

- Parent: the development-confirmed compact update-20k checkpoint.
- Common prepared fork: output-preserving carry-input expansion, one explicit
  v1-to-v2 sampler migration, one materialized v2 sampler state, fresh optimizer,
  parent-terminal entropy `0.02`, and identical short LR warmup. Existing
  warm-start and resume modes cannot express this combination without losing
  sampler history or retaining the old optimizer, so implement one narrow named
  fork initializer rather than another generic checkpoint mode. Retain absolute
  PPO update 20,000, but key the warmup from fresh optimizer-local step zero.
- Control: current dense reward and its frozen legacy carry ledger.
- Treatment: constant exact-success payment, fixed horizon-failure and step
  terms, globally normalized physical-distance ledger, and normalized
  excavation/relocation potential at fixed shaping weight.
- Both arms: identical map identities, v2 sampler state, horizon, PPO settings,
  seed, warmup, and 4,000--6,000-update budget. Their carry channel is labelled
  with the arm-specific exact ledger in every receipt; run, checkpoint, and
  evaluation receipts pin its protocol ID and sidecar hash and reject mismatch.
- Admission: D0 parity, durable D4a receipt, analytic exact-success dominance,
  implied dwell-cost grid, and reproduction of the scale/overlap findings.

The previously implemented whole-objective anneal, formerly R1, is retired
from the mainline. It changes action costs, timeout reward, success scale, and
efficiency bonuses simultaneously. Do not launch it as a prerequisite or
substitute for R2.

## P3 — progress-vector separation (audit accepted)

Loaded timeouts currently collapse `absolute_completion` to zero, erasing the
material progress of 30 of 174 failures. Expose
`dig / accepted / illegal / loaded` fractions, recompute them on the 174
failures first, and keep the combined scalar dashboard-only until loaded and
illegal handling is explicit. Never a promotion metric.

## P4 — conditional R3 shaping fade

Only if reward-v2 wins R2, fork the selected reward-v2 checkpoint into fixed
shaping and episode-latched fade children. Restore identical model, optimizer,
sampler state, source seed, and map identities; the sampler NumPy RNG restores,
while JAX rollout RNG and live environments restart and diverge. Only the frozen
shaping-weight schedule differs. Resolve the time-to-go observation decision
before calling the result a final fixed-horizon recipe. A later scratch run
must confirm that the selected recipe can teach rather than only preserve a
competent checkpoint.

## P5 — later, separate treatments

- Obstacle action-feasibility masking (D3-informed upper bound first).
- Workload-aware horizon (post-D1; changes the benchmark and needs its own
  comparability decision).
- Bank diversity 96 → 256+ layouts per condition: the staged epoch showed
  checkpoint churn consistent with 96-layout memorization; training compute is
  unchanged, the cost is a new frozen bank release and generation review.

## Non-goals (audit decision table, restated)

No loosening of exact success, cleanup, or unload; no topology- or
condition-specific reward bonuses; no legacy `Rewards.sparse()`; no further
10M-scale work.
