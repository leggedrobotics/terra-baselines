# V8 R2 reward-v2 implementation and experiment goal

Status: ready to activate

Authority:

- [V8 reward and termination audit](V8_REWARD_TERMINATION_AUDIT.md)
- [V8 improvement set](V8_IMPROVEMENT_SET_20260810.md)
- [V8 scale-up record](V8_10M_SCALEUP.md), current-successor amendment

## Outcome

Implement the smallest credible R2 path, launch a matched compact-policy
comparison on Euler, complete its frozen fixed evaluations, and record whether
the normalized material-potential reward improves held-out V8 performance over
the current dense reward.

R2 has exactly two arms:

1. control: current dense reward and frozen legacy relocation ledger;
2. treatment: reward-v2 from the audit, with flat exact-success payment,
   globally normalized excavation/relocation potential, fixed horizon-failure
   and step terms, and fixed `shaping_weight=1`.

This goal does not implement or launch R3 reward fading. R3 is eligible only if
reward-v2 wins R2.

## Baseline

- Selected compact checkpoint: update 20,000, SHA-256
  `0948a230a5c0929237a7adbdb6c1231691caab728238a600c0e819f02e200834`.
- Main development: 546/720 exact, macro graded completion 0.861.
- Capability development: 31/32 exact, macro graded completion 0.977.
- Parent sampler: `continuous_banded_v1`.
- Required child sampler: `continuous_banded_v2` from terra-baselines
  `60e7510`.
- Horizon: 450; full resets; 47 conditions x 96 train layouts; compact
  deep+xattn model; seed `20260807`.
- No reward-v2 implementation or valid R2 launch exists at activation.

## Frozen causal contract

Both arms must share:

- one output-preserving carry-input expansion of the compact parent;
- one prepared v1-to-v2 sampler migration and the exact same migrated sampler
  state;
- absolute PPO update 20,000, fresh optimizer-local step zero, identical short
  LR warmup, and entropy fixed at the parent endpoint `0.02`;
- map identities, graph, initial poses, horizon, reset mode, architecture,
  PPO settings, source seed, transition budget, checkpoint cadence, and fixed
  evaluation panels;
- source-disjoint promotion/development plus all-free capability evaluations.

Only the reward-plus-ledger bundle may differ. Each arm records and validates
its carry-channel protocol ID and distance-sidecar hash. No arm-specific
sampler, action-mask, horizon, observation, bank, architecture, or dynamics
change is allowed.

## Simple implementation boundary

Follow `$simple-research-code`:

- one named `continuous_banded_v2` preset;
- one named prepared-fork initializer;
- one canonical global distance routine;
- one carry-work scalar channel;
- one reward-v2 potential formula;
- no generic reward framework, compatibility matrix, or fallback modes;
- one reversible implementation commit per repository;
- one to four new claim-driving contract tests per repository unless a silent
  reward/termination error requires more.

The old R1 whole-objective anneal remains historical code and receives no
compute. If R2 loses, retain its receipts and revert or abandon the experiment
commits rather than hardening an unsuccessful design.

## Admission gates

R2 cannot launch until all blocking gates pass:

- **D0:** reproduce the selected checkpoint's frozen evaluation and emit the
  per-identity analysis receipt.
- **D4a:** materialize the exact targeted relocation replay receipt, including
  evaluator graph/batch shape and ledger parity.
- **D4b:** materialize the 4,512-map scale/overlap rows, the 34-map identity
  set, the proposed `(Q,P)` dwell-cost grid, and admitted potential extrema.
- **Dominance:** analytically prove, over every admitted potential and success
  step 1--450, that the minimum discounted exact-success return exceeds the
  maximum horizon-failure return.
- **Implementation:** dense endpoint parity, signed-cycle accounting,
  output-preserving carry expansion, prepared-fork state, v1-to-v2 migration,
  LR warmup, checkpoint/resume, and finite-value tests pass.
- **Runtime:** each arm independently completes a W&B-disabled update-1 smoke
  after CUDA convolution-backward and NCCL all-reduce preflight on an approved
  RTX 3090/4090 allocation.

D1--D3 and D5--D6 remain nonblocking diagnostics or independent treatments.
They must not be bundled into R2.

## Experiment

- Compact R2 screen: two matched arms, 6,000 additional PPO updates from the
  prepared update-20,000 parent state.
- Checkpoints: at least every 500 updates.
- Fixed evaluations: retained 1,000-update checkpoints on promotion,
  development, and both capability panels; sealed only after treatment
  selection.
- Run allocation: verified 4x RTX 3090/4090 with runtime GPU guard, artifacts
  on Euler scratch/work rather than home, and W&B in
  `aless-weber-eth/mixed-agents`.
- A one-seed difference is a screen. A material effect requires at least three
  paired seeds before a paper-level causal claim.

## Decision rule

Checkpoint selection uses promotion only. Development confirms the selected
checkpoint. Compare:

1. all-47 exact success and condition-balanced macro graded completion;
2. foundation/trench and depth slices;
3. p10 and worst condition;
4. all-free retention;
5. `d12`, `d16`, large adjacent foundations, and obstacle conjunctions;
6. steps and productive workspace cycles only on identities solved by both.

Never compare raw reward between arms. Reward-v2 wins the screen only if it
shows a material fixed-panel improvement without family, tail, or all-free
regression at the selected checkpoint. Ambiguous or checkpoint-unstable results
trigger paired-seed replication, not post-hoc gate changes.

## Iteration loop

1. Inspect the next unchecked gate and its strongest failure evidence.
2. Make one narrow change within the frozen contract.
3. Run the smallest deterministic verifier that can falsify the claim.
4. Record command, revision, artifact, and verdict in this file and the
   experiment ledger.
5. Revert or revise on failure; advance only after the gate passes.
6. After submission, reconcile Slurm, logs, W&B, checkpoints, and fixed
   evaluations rather than trusting one source.

## Anti-cheating and safety

- Do not loosen exact excavation, accepted dump mask, cleanup, unloaded final
  state, mass conservation, or fixed evaluation.
- Do not change the map bank, sampler between arms, horizon, architecture,
  seed, PPO shape, or action/observation contract beyond the common carry
  expansion.
- Do not select on development, sealed results, online success, or reward.
- Do not call a queued job, running job, finite checkpoint, or update-1 smoke a
  learning result.
- Do not update expected hashes to bless evidence generated by older source.
- Do not write checkpoints, W&B files, or large logs under Euler home.

## Completion proof

The goal is complete only when all of the following exist:

- clean committed Terra and terra-baselines revisions for the direct R2 path;
- checked D0, D4a, D4b, dominance, CPU, and GPU-smoke receipts;
- exact parent, prepared-fork, dataset, graph, sampler, reward-protocol, source,
  and sidecar identities in both run contracts;
- two Euler job IDs (plus continuation IDs if needed) that each advanced beyond
  update 1 and completed the declared 6,000-update screen;
- fixed promotion/development/capability results for both arms with integrity
  checks;
- a result table and causal interpretation in the experiment ledger;
- a recorded decision: reject reward-v2, replicate it, or make it eligible for
  R3.

## Status checklist

- [ ] G0 goal activated; clean implementation worktrees created
- [ ] G1 D0 receipt complete
- [ ] G2 D4a durable replay receipt complete
- [ ] G3 D4b scale/overlap/dwell receipt and dominance proof complete
- [ ] G4 Terra reward-v2 and carry-observation path implemented and committed
- [ ] G5 baselines prepared fork, v2 preset, warmup, receipts, and launcher implemented and committed
- [ ] G6 focused CPU tests and independent code review pass
- [ ] G7 both Euler update-1 smokes pass
- [ ] G8 matched 6,000-update R2 jobs submitted and verified beyond update 1
- [ ] G9 fixed evaluations complete and R2 decision recorded

## Worklog

- 2026-08-10: goal drafted from audit commit `9f34f6d`; no R2 code or job yet.
