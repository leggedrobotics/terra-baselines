# V8 deep+xattn curriculum campaign

> **Historical campaign record.** The hard Stage A/B/C launch sequence and
> commands below are superseded. The current map-curriculum authority is
> [`V8_10M_SCALEUP.md`](V8_10M_SCALEUP.md), section "Accepted decision:
> continuous all-47 bands (2026-08-07)": one uninterrupted all-47 run with
> continuously moving family bands. Depth remains map metadata, not a job
> boundary. Keep this file only for the provenance and results of the earlier
> staged campaign.

- Status: superseded staged campaign retained as historical evidence; its
  compact and 10M jobs are not the current map-curriculum launch path
- Date: 2026-08-06
- Dataset authority:
  [`V8_COMBINED_DISTRIBUTION_PLAN.md`](/home/lorenzo/moleworks/.worktrees/terra_v8_combined_20260803/V8_COMBINED_DISTRIBUTION_PLAN.md)
- Training-design authority:
  [`TRAINING_DESIGN.md`](/home/lorenzo/moleworks/.worktrees/terra_v8_combined_20260803/TRAINING_DESIGN.md)
- Reward authority:
  [`PROGRESSIVE_REWARD_CURRICULUM.md`](/home/lorenzo/moleworks/terra/PROGRESSIVE_REWARD_CURRICULUM.md)
  and
  [`PROGRESSIVE_REWARD_VALIDATION_PLAN.md`](/home/lorenzo/moleworks/terra/PROGRESSIVE_REWARD_VALIDATION_PLAN.md)
- Implementation style:
  [`simple-research-code`](/home/lorenzo/git/codex_skills/skills/simple-research-code/SKILL.md)

## Decision

Train the target network directly. The production comparison contains exactly
two policies:

| Arm | Encoder | Initialization |
|---|---|---|
| `G-DEEP-V8-DENSE-WARM` | deep SE `(2,2,3,3)` | P5c deep update 4,000 |
| `G-DEEP-XATTN-V8-DENSE-WARM` | the same deep SE plus E4-prime cross-attention | output-preserving graft from the same checkpoint |

Both policies use the same parent as their KL/value teacher, a fresh optimizer,
the same maps, seed, horizon, reward, PPO shape, entropy schedule, and fixed
evaluations. This isolates the value of cross-attention.

A smaller network may be used for a local or update-1 engineering smoke. It is
not an intermediate teacher and is not a third production arm. The target
deep+xattn network has about 2.857 million parameters versus 2.699 million for
the deep parent. In P5c, medium was about 22% faster over 4,000 updates but
ended below deep on development macro (`0.565` versus `0.586`) and exact
completion (`94/512` versus `143/512`). The trained deep parent was itself
grown from medium, so inserting another small V8 teacher saves little compute
while adding a capacity-transfer and checkpoint-selection confound.

If scaling efficiency is studied later, use a separate equal-transition
ablation: graft xattn onto the P5c medium policy, train it on V8, then grow only
the block depths `(1,2,2,2) -> (2,2,3,3)` and compare against direct
deep+xattn at matched GPU-hours. That is not a prerequisite for this campaign.

Do not add E7's self-attention token mixer or import historical attention
weights. The new cross-attention contribution is exactly zero at update zero,
which preserves the parent logits and value before PPO starts.

## Frozen inputs

| Input | Frozen value |
|---|---|
| V8 release | `terra_v8_v6_constraints_v7_adjacent_train96_v5` |
| V8 archive SHA-256 | `dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b` |
| V8 `dataset.json` SHA-256 | `715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798` |
| V8 training-mixture SHA-256 | `f2a2a33556d513b46193a8a3996d37e6989534eba9373f46f52d79f956ac128e` |
| Terra revision | `a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4` |
| Parent and teacher SHA-256 | `4d178c39443009cb4e57d83713421553689f6e3989da0be674184237c14d86cc` |
| Parent architecture | 2,699,117 parameters, channels `(24,48,64,96)`, blocks `(2,2,3,3)` |
| Parent fixed result | promotion/development macro `0.624/0.586`; exact `168/512` and `143/512` |
| Reward | `DENSE`; `apply_trench_rewards=false` |
| Horizon/reset | 450 steps; full resets |
| Completion | exact visible dump mask |

The P5c parent is descriptive rather than formally selected: its held-out
score was strongest at the final checkpoint, but the curve was unstable. That
is why the V8 campaign uses bounded stages and fixed promotion panels rather
than assuming the parent already solves the new distribution.

## Map curriculum

The dataset defines support. A fixed sampler defines exposure within each
checkpoint-bounded stage. The gate verifies the exact ordered condition list
and probability vector from every qualifying checkpoint's sampler state; a
vector that merely sums to one is insufficient. There is no per-environment
promotion or demotion.

The map archive remains immutable. The later-stage exposure rule is the named
training-protocol profile `bounded_replay25_v1`, which supersedes the archive's
original 50% replay suggestion without modifying any map, manifest, episode,
or dataset hash. Stage A's `bank_v4` vector is unchanged. Retention is enforced
by fixed panels and rollback, not by spending half of all rollouts forever on
already-mastered conditions.

### Stage A: capability

- support: `fnd-slab-allfree` and `trn-straight-allfree`;
- mass: 50% foundation, 50% trench;
- ordered sampler contract SHA-256:
  `a569e04eba1bc2ed7cff9d084ff75c7a09224df6d600a4ab647a7b28c15f8633`;
- bounded screen: at most 2,000 updates;
- promotion: each family condition reaches at least `12/16` exact on the
  promotion panel at the latest two scheduled checkpoints;
- development is reported but never used to promote.

### Stage B: nearby geometry

The original compact campaign below used a 4,000-update screen. The current
paired compact-versus-10M continuation is the separately named 20,000-update
treatment in [`V8_10M_SCALEUP.md`](V8_10M_SCALEUP.md); it keeps the same map
support, sampler vector, dense reward, horizon, and promotion contract.

- support: both capability controls plus the 13 V7 adjacent-generous cells;
- mass under `bounded_replay25_v1`: 25% capability replay and 75% nearby core;
- every slice remains 50/50 foundation/trench;
- V7 geometry mass within each family follows the frozen V8 mixture;
- ordered sampler contract SHA-256:
  `b6e9e5d4fd672b87b4b87252b630d3243355e5d10988772f9861f3ec0cf0f245`;
- bounded screen: at most 4,000 updates;
- promotion: foundation nearby core at least `78/96`, trench nearby core at
  least `91/112`, every cell at least `12/16`, at the latest two scheduled
  checkpoints;
- both capability controls must retain their Stage-A gate.

### Stage C: full V8

- support: all 47 training conditions;
- mass under `bounded_replay25_v1`: 6.25% capability, 18.75% nearby core,
  and 75% V6 constraints. This is 25% replay of the mastered Stage-B mixture
  plus 75% newly active constraints;
- every slice remains 50/50 foundation/trench;
- ordered sampler contract SHA-256:
  `989f379b038f71506a188ddf55e9789f79d94c1b537f76661e0d2d6af4653af3`;
- bounded first allocation: configure 8,000 updates, checkpoint every 500,
  and let the 24-hour allocation determine the reached update;
- continue a promising checkpoint with true optimizer/schedule/sampler resume
  on the 120-hour queue; never restart it as a nominal continuation.

Full-stage training and evaluation are separate jobs. An `afterany` evaluator
accepts a `COMPLETED` or wall-time `TIMEOUT` training job, discovers the longest
contiguous `500,1000,...,N` checkpoint prefix, and requires at least two
checkpoints. It rejects gaps, duplicate updates, OOM/node/cancellation failures,
and checkpoint or sampler-state mutation. Promotion, development, and both
capability panels are evaluated at exactly the same checkpoint paths and
hashes. Checkpoints are published by temporary-file plus atomic rename, so a
wall-time interruption cannot expose a truncated final-name pickle. Every
evaluated checkpoint is reloaded and verified, not just the latest pair.

For every previously mastered 16-map condition, retention requires

```text
successes >= max(11, lower_of_two_mastery_counts - 1)
```

Capability controls additionally retain the Stage-A mastery floor, so their
threshold is `max(12, lower_of_two_mastery_counts - 1)`. Thresholds are frozen
when the qualifying pair is recorded and never ratchet upward later.

At each scheduled checkpoint, the treatment has a retention failure if any
inherited condition falls below its frozen threshold. Any two adjacent
retention-failing checkpoints in the full stage history stop that treatment and
restore the last fully passing checkpoint, even if the final pair later
recovers. Training does not silently demote individual vector environments.

The permissive 120-hour compute gate compares the latest complete checkpoint
with the checkpoint exactly 1,000 updates earlier, or the preceding checkpoint
when only two exist. It requires on the 32 V6 constraint conditions of the
promotion panel either one additional exact completion or at least `0.001`
condition-macro graded gain. Foundation macro, trench macro, micro `p10`, and
worst-condition completion may each regress by at most five percentage points;
the same four guards also apply on development. All inherited retention and all
four-panel integrity checks must pass. Progress on capability/core replay alone
does not buy long compute.

A qualifying arm resumes the same full-stage checkpoint with optimizer, update
counter, entropy schedule, and fixed sampler state restored. The absolute
target is update 80,000 on `gpuhe.120h` for `119:45:00`; it is never interpreted
as 80,000 additional updates. If both arms qualify, continue the matched pair.
If only one qualifies, it may continue under an explicitly unpaired label and
cannot support a matched architecture conclusion. The resume uses the source
treatment name verbatim—without adding a second machine/timestamp suffix—so
source and continuation checkpoints remain one fixed-bank treatment.

## Architecture and PPO contract

- student model: `medium` widths with deep spatial blocks `(2,2,3,3)`;
- xattn treatment: `resnet_spatial_8x8_se_xattn`, attention compute in float32;
- control: `resnet_spatial_8x8_se`;
- encoder compute: bfloat16;
- critic: `512,256`;
- devices/environments: 4 RTX 4090 GPUs, 1,024 environments per GPU;
- rollout: 32 steps, 2 update epochs, 32 minibatches;
- learning rate: `3e-4`;
- entropy: `0.02 -> 0.005` over 10,000 updates;
- KL kickstart: `1.0 -> 0` over 1,500 updates;
- value kickstart: `0.5 -> 0` over 500 updates;
- learning-rate warmup: 100 updates;
- no value clipping; flat minibatch shuffle;
- fresh optimizer for both arms.

Map-stage transitions are parameter-only warm starts from each arm's own
promoted checkpoint. They reset the optimizer, update counter, entropy schedule,
environment trajectory, and fixed sampler; Stage B and C disable teacher KL and
value kickstart. This is a deliberate boundary between map distributions, not a
nominal continuation. Only a same-stage full-V8 continuation uses
`--resume_from` and restores the optimizer, update/schedule counter, and sampler
state.

The xattn graft must pass a real-checkpoint equality test before launch:
identical observations produce exactly identical logits and values. The
verified graft grows `2,699,117 -> 2,856,685` parameters and currently has
maximum absolute logit and value difference `0.0`.

## Reward curriculum

The first V8 architecture campaign remains entirely in `dense_skill`, which is
the current `DENSE` contract. Do not select the legacy `Rewards.sparse()` and
do not smoothly anneal reward weights inside a PPO run: that path is not a
clean sparse objective and would change the frozen V8 protocol.

The requested dense-to-sparse direction begins only when a full-stage dense
checkpoint passes all of these on the fixed promotion bank at three
consecutive scheduled evaluations:

- at least `576/720` exact overall;
- at least `308/384` foundations;
- at least `269/336` trenches;
- at least `10/16` in every main condition;
- capability and nearby-core retention still pass;
- no integrity failure.

That checkpoint becomes the common parent for a separate reward experiment:

```text
A: dense_skill continuation
B: dense_skill -> terminal_objective
C: dense_skill -> terminal_margin -> terminal_objective
```

The reward transition occurs at a checkpoint boundary, uses a fresh optimizer,
freezes the full V8 map mixture, disables teacher KL, and receives its own
derived reward-protocol identity. `terminal_margin` gets an initial 2,000-update
cap with fixed evaluation every 500 updates. No reward-return statistic may
promote a stage. Implementation of reward contract v2 remains a separate task;
the current dense run must not pretend the legacy sparse enum implements it.

## Evaluation and leaderboard

Every numbered screen checkpoint is evaluated deterministically on frozen
promotion and development panels. Capability panels remain separate from the
45-condition main macro.

Report at minimum:

- exact solved / total;
- macro graded completion, one equal vote per condition;
- foundation and trench macro and exact;
- every condition's graded completion and exact count;
- worst condition and lower-tail completion;
- transition-integrity failures;
- policy mode and checkpoint hash.

Online training completion, reward, finite losses, GPU utilization, or a job
being `RUNNING` are not behavioral outcomes.

## Launch and promotion sequence

- [x] V8 maps visually accepted.
- [x] V8 v5 protocol identities repaired without changing reset arrays.
- [x] V8 archive and immutable hashes frozen.
- [x] Deep→deep+xattn output-preserving graft implemented and verified.
- [x] Fixed stage-weight sampler and V8 loader implemented.
- [x] Capability-panel fixed evaluation added.
- [x] Focused CPU contract tests, full 331-test suite, launcher syntax, and
  ShellCheck pass.
- [x] Euler dry-run resolves only the two intended arms and immutable inputs.
- [x] Both update-1 jobs pass finite parameters/losses, transition integrity,
  four-GPU runtime, and checkpoint verification.
- [x] Submit the two-arm Stage-A bounded screen.
- [x] Implement and independently review Stage-C tail evaluation, true 120-hour
  resume, continuation leaderboard, and dense-reward qualification receipt.
- [ ] Evaluate every 500-update checkpoint and apply the Stage-A gate.
- [ ] Launch Stage B only from a checkpoint that passes Stage A twice.
- [ ] Launch Stage C only from a checkpoint that passes Stage B twice.
- [ ] Evaluate the full-stage tail in a separate job and issue a continuation
  receipt only after V6 progress, retention, tail, and integrity gates pass.
- [ ] Use 120-hour true resume only from that receipt; do not require the short
  run to have converged.
- [ ] Begin reward-v2 implementation/ablation only after dense full-V8 reward
  qualification.

The launch entry point is
[`scripts/euler_v8_deep_xattn_v1/submit.sh`](../../scripts/euler_v8_deep_xattn_v1/submit.sh).
It defaults to a non-mutating dry run and refuses to submit a screen without
passed update-1 receipts for both arms. Stage B and Stage C require the
prior-stage receipt and derive the exact promoted parent path and SHA-256 from
it; a later stage cannot restart from the original P5c parent. Full-stage and
120-hour training use separate tail-evaluation jobs so wall-time termination
cannot discard complete checkpoints.

Promotion receipts are per arm. The deep and deep+xattn policies each advance
from their own latest gate-passing checkpoint. Automatic matched submission
requires valid receipts for both arms; if only one arm passes, it is not copied
into the other architecture or silently continued as a matched comparison. A
single-arm feasibility continuation would require a separately named decision.

## Runtime admission log

- **Revision `b164d6b`, jobs `9546455`/`9546456`: failed admission, no policy
  conclusion.** The deep arm completed its first PPO update with the expected
  four RTX 4090 devices, CUDA/NCCL path, maps, parent, architecture, and finite
  training path. It then failed the bounded W&B schema because the declared V8
  branch depth `Nearby core` was emitted by `curriculum_metrics` but omitted
  from `TRAINING_SCALAR_KEYS`. The xattn arm was still pending and was cancelled
  before allocation because it shared the same deterministic instrumentation
  failure.
- **Recovery:** derive all allowed curriculum-population keys from the same
  canonical `FAMILIES` and `BRANCH_DEPTHS` tuples used to validate the bank,
  with a regression test requiring every declared label. This does not change
  PPO, reward, observations, maps, or architecture. Both matched update-1
  admissions must be rerun from the clean recovery revision.
- **Revision `97296e8`, jobs `9550441`/`9550442`: passed update-1 admission.**
  Both arms used four RTX 4090 devices, the frozen V8 capability sampler, the
  exact parent/teacher SHA, finite first-update training and checkpoint state,
  and passed periodic/final checkpoint verification. This is runtime admission,
  not a policy result.
- **Revision `97296e8`, jobs `9552543`/`9552544`: Stage-A behavioral screen
  running.** Each arm is configured for 2,000 updates and frozen evaluation
  every 500 updates. Deep started at `20:59:55 CEST`; deep+xattn started at
  `21:20:38 CEST`. Both passed four-GPU/CUDA/NCCL startup and loaded the frozen
  parent, teacher, bank, and sampler. This is runtime evidence only; no fixed
  behavioral checkpoint has returned yet.

## Decision log

- **V8-XA-01, accepted 2026-08-03:** use the same trained deep checkpoint as
  parent and teacher for both arms.
- **V8-XA-02, accepted 2026-08-03:** add E4-prime cross-attention with an
  exactly zero initial contribution; exclude E7's token mixer.
- **V8-XA-03, accepted 2026-08-03:** a small model may smoke plumbing but is
  not an intermediate production teacher.
- **V8-XA-04, accepted 2026-08-03:** run the map curriculum as capability,
  nearby, then full V8 with fixed stage weights and checkpoint gates.
- **V8-XA-05, accepted 2026-08-03:** keep the initial architecture campaign
  dense. Replace the requested informal fade with the specified
  checkpoint-bounded reward-v2 experiment only after high fixed completion.
- **V8-XA-06, accepted 2026-08-03:** require the latest two scheduled
  checkpoints, not a stale earlier pair, for map-stage promotion.
- **V8-XA-07, accepted 2026-08-03:** issue immutable per-arm promotion receipts;
  require both receipts for an automatic matched next-stage launch.
- **V8-XA-08, accepted 2026-08-03:** use a fresh optimizer and sampler and no
  teacher kickstart at map-stage boundaries; reserve true resume for unchanged
  full-V8 continuation.
- **V8-XA-09, accepted 2026-08-03:** train deep+xattn directly. Reserve the
  medium+xattn-to-deep+xattn route for a later equal-transition scaling
  ablation, not an intermediate production teacher.
- **V8-XA-10, accepted 2026-08-03:** award 120-hour compute per arm after any
  measurable V6 progress that passes promotion/development tail guards,
  inherited retention, and integrity; evaluate the wall-time tail separately.
- **V8-XA-11, accepted 2026-08-03:** automatically attach an `afterany`
  continuation evaluator. Freeze every complete checkpoint hash, evaluate the
  source plus every 2,000 updates and the latest complete diagnostic on all
  four panels, and issue—but never act on—the reward-qualification receipt.
- **V8-XA-12, accepted 2026-08-03:** harden the wall-time boundary and resume
  identity: atomically publish checkpoint pickles, preserve the exact source
  treatment name, reload every selected continuation checkpoint, validate its
  finite model/optimizer, update, architecture, fixed sampler, and resume
  source, and bind every parent receipt to the exact run directory and contract
  hash. An independent blocker re-review found no remaining correctness issue.
- **V8-XA-13, accepted 2026-08-03:** prepare, but do not yet launch, the
  [`V8 10M scale-up`](V8_10M_SCALEUP.md). Its teacher must be the qualified
  deep+xattn policy trained on the exact full-V8 distribution, passing three
  scheduled high-completion promotion gates plus the same development
  family/cell gate. The old P5c teacher is ineligible. Compare a 10M channel-
  grown student with a teacher-sized rewarm control at matched transitions.
- **V8-XA-14, accepted 2026-08-03:** postpone the actual 10M run until that
  teacher exists, but make the experiment launch-ready now. Revalidate main
  development plus both all-free capability controls, measure transplant damage
  on all 720 exact promotion resets, and use a true 24-hour matched screen with
  absolute target update 20,000. An `afterany` evaluator compares only the
  longest common complete checkpoint prefix on all four panels and writes the
  aggregate, family, and per-condition leaderboard. Preparation is not launch
  authorization.
- **V8-XA-15, accepted 2026-08-06:** keep active-stage online success as a
  diagnostic only. Primary success enumerates the complete source-disjoint V8
  fixed panel. Use `bounded_replay25_v1` after Stage A: 25% replay of the
  mastered previous-stage mixture and 75% active-stage exposure, with fixed
  retention gates and rollback. This changes the named sampler treatment, not
  the immutable V8 map archive.
