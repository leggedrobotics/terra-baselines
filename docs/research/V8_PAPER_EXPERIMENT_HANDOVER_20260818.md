# Terra V8 paper-experiment handover

Snapshot: 2026-08-18 13:22 CEST

## Mandate for the receiving agent

Design the smallest credible paper experiment suite for two distinct questions:

1. Does actor recurrence reduce Terra's history-dependent no-op and limit-cycle
   failures?
2. Does relay-focused partial-start training teach staging, empty relocation,
   relifting, and final cleanup in a way that transfers to ordinary full-start
   tasks?

Return a preregistered experiment plan before implementing or launching
anything. The plan should contain the paper claim, arm matrix, exact frozen
code/environment contracts, seeds, transition budget, checkpoint schedule,
GPU-hour estimate, fixed panels, primary endpoint, secondary diagnostics,
statistical analysis, stopping rules, and literature support.

The scientific endpoint is strict exact completion from untouched full starts.
Online curves and success on synthetic partial starts are diagnostics, not
paper evidence.

## Executive state

Three lines of work exist, and they must not be conflated:

1. **Old-runtime capability line:** the v6.1 feed-forward policy with material
   stall age and Continuous Banded v3 was continued from u14 to u67. The matched
   held-out result improved from u40 `656/720` to u67 `672/720` exact. This is a
   useful capability benchmark, not a component ablation.
2. **Fresh relay-curriculum line:** a feed-forward policy is training from
   scratch with the new mechanics and a Backplay-inspired partial-start schedule.
3. **Fresh recurrent line:** an actor-only GRU64 with a current-observation skip
   is training from scratch on the relay curriculum. Its predecessor without the
   skip plateaued for an identifiable architectural reason.

The active fresh runs are practical capability experiments. They are not yet a
causal paper matrix because their seeds, GPU layouts, and effective Terra
runtime differ, and there is no full-start-only matched control.

## Established evidence

### Fixed-panel u40 to u67 result

The paired RTX 4090 evaluation used one process and one frozen 720-map promotion
panel. It covers 45 conditions x 16 maps; the current 47-condition training
distribution additionally contains `fnd-slab-allfree` and
`trn-straight-allfree`:

- u40 reference: `656/720` exact (91.11%);
- u67 candidate: `672/720` exact (93.33%);
- 38 failure-to-success conversions and 22 success-to-failure regressions;
- foundation: `323/384 -> 340/384` (+17);
- trench: `333/336 -> 332/336` (-1);
- mean successful length: `100.05 -> 95.13` steps;
- stall-saturated episodes: `76 -> 47`; and
- no-effect action fraction: `10.23% -> 12.12%`.

Thus further PPO training improved aggregate capability but did not eliminate
the structural no-op problem. The hardest condition,
`fnd-slab-side1-obj`, remained essentially flat. The historical Supercluster
u40 result was `657/720`; use the paired `656/720` reference for the u67
comparison because bfloat16/backend trajectories are not bit-identical across
hardware.

The u40/u67 result bundles stall age, final-v3 curriculum, and 27k additional
updates. It cannot attribute the gain to any one component. It also uses old
Terra `c2d2a94a` and does not contain the later singleton, partial-relift,
relay-reset, trench-ordering, or under-base fixes.

Primary artifact:

- `/home/lorenzo/moleworks/.artifacts/terra_v61_stall_age_continuation_20260814/final_eval/u67_fixed_panel/INTERPRETATION.md`
- paired JSON and map-level analysis are in the same directory.

### Failure mechanism evidence

- Exact full-policy-input fixed points and 2--18-step cycles are confirmed in
  targeted failures.
- Extending selected failures from 450 to 900 steps rescued none of them.
- Some near-finish cases are sampling-sensitive, while carry/obstacle cases are
  not reliably rescued by stochastic action selection.
- Off-terminal soil staging is intentional and necessary. A successful d16
  trace uses repeated staging, rehandling, and exact completion. Do not describe
  off-zone soil as intrinsically illegal or add a categorical staging penalty.
- Reward-v2 has no legacy `dump_wrong` penalty. Its point-geodesic potential can
  locally dislike a necessary detour or prefer a poorly serviceable pile, but
  the observed evidence does not justify broad reward replacement.
- Material `stall_age` breaks some repeated-input aliases, but saturates at 32
  and resets during material-changing ping-pong. It is not a general memory
  mechanism.
- Previous-action physical/material outcome bits remain an untrained compact
  hypothesis. Appending them splits none of the archived recurrence hashes, so
  they may help no-op recovery but cannot count or phase all cycles.
- The five-way current-DO affordance remains a conditional observation repair,
  not an established treatment. A large decodability audit should precede it.

Read the detailed evidence instead of reconstructing it:

- `docs/research/V8_V61_FAILURE_AUDIT_20260813.md`
- `docs/research/ORACLE_TERRA_STAGING_REVIEW_20260814.md`
- `docs/research/V8_REWARD_TERMINATION_AUDIT.md`
- `docs/research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md`

### Matched-update comparison: relay run versus prior v6.1 lineage (2026-08-19)

Recorded at Lorenzo's request as the paper's motivating evidence for
introducing the partial-reset (relay) curriculum. Online W&B training metrics,
medians over the listed update windows.

Runs:

- Current: `v8_relay_partial_2778766683_s20260815` (fresh init, relay partial
  resets + continuous_banded_v3 + reward-v2, no stall age; still running).
- Older lineage: `v8_v6_yolo_rv2_v61_9abf88eb60_phase1_10405296` (u1-13,991)
  continued as `v8_v61_stall_age_v3_dddc691c93_phase2_10625259`
  (u14,001-66,991, crashed at u67k; stall-age observation active in phase2).
- Closer control candidate for u<=14k only:
  `v8_rv2p1_02ccb62fb3_phase1_10480220` (parallel reward-v2 arm, finished at
  u13,991).

| window | success (relay / old) | completion (relay / old) | no-effect (relay / old) | return (relay / old) |
|---|---|---|---|---|
| u5-6k | 0.383 / 0.292 | 0.529 / 0.481 | 0.371 / 0.400 | -0.69 / -1.63 |
| u10-11k | 0.481 / 0.447 | 0.618 / 0.600 | 0.338 / 0.379 | 0.36 / -0.09 |
| u20-21k | 0.847 / 0.689 | 0.909 / 0.824 | 0.151 / 0.192 | 4.35 / 2.37 |
| u30-31k | 0.932 / 0.847 | 0.960 / 0.913 | 0.112 / 0.150 | 5.54 / 4.29 |
| u40-41k | 0.982 / 0.937 | 0.990 / 0.964 | 0.045 / 0.109 | 6.14 / 5.55 |

The relay run leads at every matched window; the gap is widest mid-run
(u20k: +0.158 success) and the terminal no-effect rate is about 2.4x lower
(0.045 versus 0.109 at u40k). The old lineage additionally died at u67k while
the relay run continues with a queued u200k extension.

Confounds and claim limits (all apply):

- Multi-variable difference: relay resets, absence of stall age, fresh
  initialization versus continuation, seed, and Terra-runtime revision differ
  together. This table motivates, it does not attribute.
- Online training metrics only; the fixed-panel evaluation contract in this
  document remains the promotion instrument.
- The older lineage never logged `full_start_episode_success_rate`; its
  success column is overall online success. The relay run's u5-6k window
  includes partial-reset episodes (share <=25% before annealing to zero by
  about u14k), which drag its overall success down, so the early-window lead
  is understated rather than inflated.
- The active partial bank is the d16-only 96-map treatment. Per the live-run
  section above, do not present this as evidence for a general relay
  curriculum; a causal paper claim requires paired arms differing only in the
  partial-reset treatment on a frozen multi-stratum bank, judged on the fixed
  panel.

## Live experiment snapshot

Refresh Slurm and W&B before relying on this section. The following was verified
at 2026-08-18 13:22 CEST.

### Feed-forward relay partial-reset run

- Slurm: `10777230`, RUNNING on eight RTX 4090 GPUs, approximately u30,645.
- W&B: `v8_relay_partial_2778766683_s20260815`.
- Latest complete checkpoint at the snapshot: u30,500, SHA-256
  `7ac15a35c8dfe616e8982a5c269e8afd2824edcf02b2e1e39b5f47e661486578`.
- Continuation `10777232` is PENDING on `afterok:10777230` and targets u200k.
- Baselines training source: `2778766683fb8a0a53a761385fae05cf9396dda9`.
- Terra base: `25f855db3d913fd638c4e56b1740437a2b7122ca` plus the effective
  pre-start patch `ebdc3ad7b0e7ef505bb6d442a97d18d986cced44`.
- Seed: `20260815`.
- Partial starts occupied at most 25% of lanes through u10k and have now
  annealed to zero. Partial episodes never update full-start v3 mastery.
- Current full-start success near 0.93 is an online diagnostic only.

The immutable partial bank currently covers 96 independent
`fnd-slab-apron-d16` maps at 50%, 75%, and 90% completion. The generator also
implements leaf/prong-first trench excavation, but the active sidecar is not a
broad 47-condition partial bank. Do not claim a general relay curriculum from
this d16-only treatment. If the paper needs that claim, design a frozen,
source-disjoint multi-stratum partial bank for the confirmatory runs rather
than changing the bank beneath the active pilot.

Relevant code and visual contracts:

- Baselines worktree:
  `/home/lorenzo/moleworks/.worktrees/terra_baselines_relay_main_integration_20260815`
- Terra worktree:
  `/home/lorenzo/moleworks/.worktrees/terra_v8_relay_corridor_20260814`
- Generator contract:
  `/home/lorenzo/moleworks/.worktrees/terra_v8_relay_corridor_20260814/terra/env_generation/PARTIAL_COMPLETION_RESETS.md`
- Final visual gallery:
  `/home/lorenzo/moleworks/.artifacts/terra_v8_relay_corridor_leaf_first_review_20260814_final/gallery/index.html`

These partial states are mass-conserving and geometrically screened. They are
not per-state demonstrated-success witnesses, and this is Backplay-inspired
rather than a literal implementation of Backplay.

### Actor-only GRU64 concat-skip run

- Slurm: `10991006`, RUNNING on four RTX 4090 GPUs, approximately u6,840.
- W&B: `v8_relay_gru64r_33d2621332_s20260817`.
- Latest complete checkpoint at the snapshot: u6,500, SHA-256
  `db21a843f4b269f5246ae9ba5ac1bc8b52f4e447858716e58369da100e5e71bd`.
- Baselines source: `33d26213327d66921b66753a5a6018a37d6f2e81`.
- Terra runtime: `25f855db3d913fd638c4e56b1740437a2b7122ca`.
- Seed: `20260817`.
- Architecture: spatial encoder -> Dense(160) -> GRU(64) -> concatenate the
  GRU output with the current Dense(160) features -> Dense(48) -> logits. The
  critic is feed-forward.

At the matched online window u6,201--u6,791, the concat-skip GRU v2 had median
full-start success 0.531 versus 0.451 for the feed-forward run, and comparable
no-effect rate. This shows that v2 escaped the pure-GRU v1 plateau. It is not
held-out evidence: the runs differ in seed, GPU count, and effective runtime.
Throughput was about 15.9k versus 26.6k transitions/s in total on four versus
eight GPUs.

The pure-GRU v1 was sequence-correct but forced all current-state information
through a 64-dimensional gated bottleneck and plateaued after about u800. The
concat skip is the only v2 architectural change.

Relevant files:

- `docs/research/V8_RECURRENT_ACTOR_GRU_20260817.md`
- `docs/research/V8_RECURRENT_GRU_PLATEAU_DIAGNOSIS_20260817.md`
- worktree:
  `/home/lorenzo/moleworks/.worktrees/terra_baselines_v8_recurrent_actor_20260817`

Repository warning: local commit `33d2621` is one commit ahead of its remote at
this snapshot. The exact source is staged on Euler, but push or otherwise
preserve the commit before treating the run as remotely reproducible.

### Stall-age continuation

- Training job `10752100` ended by normal wall-time at u67k.
- u67 checkpoint SHA-256:
  `c97138a5c0fc51eea0e7a74fb3568bf39fd95b5e72b450214c064e0031d21db5`.
- The paired fixed-panel job `11056514` completed cleanly.
- The evaluation results are local and durable; the u67 checkpoint itself is
  still only on Euler scratch and should be copied to persistent storage before
  paper artifact preparation.

## Current environment changes and issue status

The common fresh relay/GRU base contains the first four changes below; the old
u67 line contains none of them:

- singleton excavation/relift veto removed (`88c0099e`);
- positive staged-soil relift loads capacity-bounded soil instead of rejecting
  the whole over-capacity pile (`30ad500f`);
- natural relay-corridor partial resets (`67c72d09`);
- trench access preserved through leaf/prong-first completion (`794d4759`).

The feed-forward job additionally applies `ebdc3ad7`, which prevents
dig/relift beneath the active base and preserves underlying terrain blockers in
observations. The active GRU job records only the common `25f855db` Terra base,
so this is an effective-runtime confound in their current practical comparison.

Still unresolved or conditional:

- previous-action outcome observation;
- five-way DO affordance and eligible volume;
- accepted-first dump fallback, which had zero observed incidence in the
  selected lifecycle audit;
- hidden `last_dig_mask`, which changed counterfactual behavior but produced no
  observed exact input alias;
- serviceability-aware reward potential, supported by one local counterexample
  but not population-level causal evidence; and
- clipped global pile heights, which did not cause the measured target
  recurrence aliases.

Canonical status ledger:

- `docs/research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md`

Some volatile sections of that ledger and `docs/EXPERIMENTS_RUNNING.md` still
name u40 as the latest checkpoint. This handover and the u67 artifact supersede
those specific run-state statements; do not rewrite the historical evidence.

## Recommended core paper matrix

If the paper wants separate claims for recurrence and relay curriculum, use a
matched 2 x 2 design on one frozen fresh runtime:

| Arm | Actor | Training starts | Main effect |
| --- | --- | --- | --- |
| A | feed-forward | full only | matched baseline |
| B | GRU64 concat-skip | full only | recurrent-agent treatment |
| C | feed-forward | full + relay partial starts | curriculum |
| D | GRU64 concat-skip | full + relay partial starts | interaction / combined capability |

Run the feed-forward and GRU paths from one unified source where recurrence is
a static configuration, not from separate implementations. Match:

- Terra revision and every transition semantic;
- reward-v2, Continuous Banded v3, observations, bank, reset schedule, and seed;
- the reset-context observation in all four arms, including the full-only
  controls, so input dimensionality is matched;
- no stall-age scalar in the recommended recurrence matrix (or the identical
  scalar in every arm if the paper explicitly chooses a different claim);
- global environments, 32-step rollout, transitions/update, minibatch,
  optimizer updates, entropy/LR schedules, and absolute update budget;
- checkpoint/evaluation cadence; and
- GPU family and count when reporting wall-clock efficiency.

The existing fresh runs each process 65,536 transitions/update, but they are
not this matrix. Use them as capability pilots and to choose a reasonable
budget, not as clean recurrence attribution.

A feed-forward-versus-GRU difference alone does not isolate memory. GRU v2 has
46,336 additional actor parameters and trains whole environment sequences,
whereas the ordinary feed-forward path shuffles flat transitions. The receiving
agent must choose one of two honest routes:

1. call B/D a complete recurrent-agent treatment and make no memory-component
   claim; or
2. add a stateless-carry, sequence-minibatched control and, if capacity remains
   plausible, a parameter-matched feed-forward control before claiming a
   recurrence effect.

Evaluation-time hidden-state reset or shuffled-history probes can show that a
trained policy uses history, but they do not replace the matched training
control.

Start with one seed per arm as a go/no-go pilot. If the paper will make a
method claim, confirm all retained arms with at least three seeds. If only the
baseline and winning combination are run, claim only the combined treatment.
Match by environment transitions, not nominal wall time or episode count.

The receiving agent should estimate the cost of the full 2 x 2 design and also
offer a reduced matrix with explicit loss of attribution. It should not launch
the matrix before Lorenzo chooses the paper claim and budget.

## Conditional experiment arms

Do not add these to the core matrix unless the main readout justifies them:

1. Feed-forward plus previous physical-effect/material-change outcome bits.
2. Feed-forward plus five-way current-DO affordance, after a decodability audit.
3. Parameter-matched larger feed-forward actor if GRU wins and parameter count
   remains a plausible confound.
4. A matched stall-age-versus-no-stall-age comparison only if the paper
   explicitly contrasts handcrafted material age with learned recurrent memory.
5. LSTM64 only if GRU exhibits useful but insufficient memory. Defer ConvLSTM,
   recurrent map trunks, GTrXL, and linear recurrent units until the GRU result
   exposes a longer-memory or spatial-memory limitation.

Do not mix reward changes, action masking, time remaining, relaxed completion,
or loaded base movement into these arms.

## Evaluation contract for the plan

### Primary endpoint

- Greedy exact completion within 450 steps from ordinary full starts.

### Panels

1. The existing frozen 720-map promotion panel for continuity and paired
   development comparisons. It omits the two all-free conditions named above
   and must not be presented as complete 47-condition V8 coverage.
2. A frozen relay-mechanism panel stratified by geometry and relay difficulty.
3. A new untouched confirmatory full-start bank for the final paper table. The
   repeatedly used 720 panel is not a blind test set. Freeze the confirmatory
   sources and identities before model or checkpoint selection, keep them
   source-disjoint from training, and cover all 47 conditions or justify every
   omission.

### Required reporting

- exact completion overall, by family, and by condition;
- paired conversions and regressions;
- paired/hierarchical intervals over training seeds with maps nested in
  conditions, rather than a naive 720-map binomial interval;
- steps and productive workspace cycles for successes;
- stage -> empty relocation -> relift -> terminal-delivery lifecycle metrics;
- residual off-zone and loaded fractions;
- maximum no-effect streak, exact-input recurrence period/count, and stall
  saturation;
- recurrent hidden-state hash/norm and carry-reset events for recurrent arms;
- material vector: excavated, terminal, off-zone, and loaded fractions;
- parameter count, transitions, optimizer work, throughput, wall time, and GPU
  hours; and
- reset, mass-conservation, target/obstacle mutation, and finite-state gates.

Use Terra's authoritative `timestep.info["action_had_effect"]`; never infer
effect from a preprocessed observation comparison.

### Mechanistic tests

For a recurrent winner:

- replay the known fixed-point/cycle cases;
- compare normal hidden-state carry with hidden-state reset or shuffled history
  at evaluation only;
- show that memory changes actions when the instantaneous observation repeats;
  and
- retain the successful d16 relay as a negative control against disrupting
  legitimate long material-neutral relocation.

For a partial-reset winner:

- test held-out synthetic cleanup states and untouched full starts separately;
- require full-start relay conversions for a transfer claim; and
- treat partial-only gains as subtask learning without demonstrated integration.

## Nonclaims and paper safety

Do not claim:

- u67 proves stall age or final-v3 independently;
- online training success is held-out performance;
- GRU helps before a matched fixed-panel result;
- partial starts teach planning unless their benefit transfers to full starts;
- the reused 720 panel is an untouched paper test;
- static reset feasibility is action-level solvability;
- off-terminal staging is an error;
- broad traversability mismatch after the audit found sparse-height false
  blocking rare; the confirmed problem was under-base holes hidden by overlay;
- reward replacement is justified by one serviceability counterexample; or
- a one-seed pilot establishes statistical robustness.

## Deliverable requested from the receiving agent

Create a new paper-experiment plan that contains:

1. one primary paper claim and one optional secondary claim;
2. the minimal arm matrix needed for those claims;
3. exact source/runtime/config differences between every arm;
4. seed count, updates/transitions, checkpoint schedule, GPU hours, and stopping
   rule;
5. development, mechanism, and untouched confirmatory panels;
6. preregistered primary/secondary metrics and statistical tests;
7. checkpoint-selection rules that do not use the confirmatory set;
8. failure and nonpromotion conditions;
9. a table mapping each proposed claim to the experiment that supports it; and
10. primary-source literature support for recurrent PPO/POMDP memory,
    Backplay or reverse curricula, and action-outcome observability.

Begin by refreshing Slurm/W&B, checking whether current checkpoints have
already crossed the proposed readout updates, and deciding whether the paper is
about a practical combined system or separable causal mechanisms. Do not
launch anything until Lorenzo chooses that scope.

This note assumes access to Lorenzo's current workstation because several
artifact and worktree links are absolute local paths. The essential numerical
results and source hashes are repeated here so the scientific brief remains
useful in a clone; copy any selected raw artifact into a tracked or persistent
paper bundle before remote collaboration.
