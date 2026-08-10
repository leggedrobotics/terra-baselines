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

Application rule: the dense trunk continuation and **both** R1 children use v2
identically, so reward stays the only difference inside the R1 pair. The
before/after-v2 trunk comparison is deliberately informal; the mechanistic
mass evidence above is the justification, not a paired run.

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
and sampler rule `continuous_banded_v2`; existing v1 launchers are untouched.

## P1 — diagnostics before the reward screen (audit D0–D5, endorsed)

Run as specified in the audit. Two sharpenings:

- D1/D2 also answer the *rotation feasibility* question: if 900-step
  inference solves the large slab/irregular/bearing-wall maps, depth-1
  mastery at 450 was unwinnable and v2's unpinning is not optional but
  required.
- D3: the runtime computes an action-mask helper but emits an all-zero
  informational mask to the policy; obstacle feasibility is a mechanics/action
  treatment, never a reward bonus.

## P2 — R1 matched fork (dense control vs annealed), amended

- Parent: the development-confirmed compact update-20k checkpoint.
- Both children: v2 sampler (identical), entropy coefficient pinned at its
  terminal 0.02 through the fade, optimizer/LR unchanged.
- Pre-registered freeze rule: any currently-mastered condition dropping more
  than 2/16 on two consecutive 1,000-update sweeps freezes the fade
  coefficient where it stands.
- Falsifiable prediction: the annealed child converts the conversion cluster
  (bearing-walls, side1, proc-ring3x, apron-c1p2, split, ring3x-obj1; graded
  0.65–0.81, exact 7–11/16) and reduces steps on jointly-solved identities;
  it does not move remote dumping (d12/d16) or obstacle stalls. Either
  deviation is a finding.
- The old online depth-2 reward trigger is superseded by an explicit
  fixed-evaluated parent receipt, per the audit's proposal.

## P3 — progress-vector separation (audit accepted)

Loaded timeouts currently collapse `absolute_completion` to zero, erasing the
material progress of 30 of 174 failures. Expose
`dig / accepted / illegal / loaded` fractions, recompute them on the 174
failures first, and keep the combined scalar dashboard-only until loaded and
illegal handling is explicit. Never a promotion metric.

## P4 — conditional R2 (normalized material potential)

Only if D4 confirms fragmented low-volume digging under the flat +1 dig
reward. Requires gamma-consistent potential shaping or a discounted-cycle
regression test. Never combined with a horizon, sampler, mask, bank, or
dynamics change.

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
