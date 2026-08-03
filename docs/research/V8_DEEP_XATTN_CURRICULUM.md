# V8 deep+xattn curriculum campaign

- Status: implementation and admission
- Date: 2026-08-03
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
the deep parent, so a smaller-teacher stage saves little compute while adding a
capacity-transfer confound.

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
checkpoint-bounded stage. There is no per-environment promotion or demotion.

### Stage A: capability

- support: `fnd-slab-allfree` and `trn-straight-allfree`;
- mass: 50% foundation, 50% trench;
- bounded screen: at most 2,000 updates;
- promotion: each family condition reaches at least `12/16` exact on the
  promotion panel at two consecutive scheduled checkpoints;
- development is reported but never used to promote.

### Stage B: nearby geometry

- support: both capability controls plus the 13 V7 adjacent-generous cells;
- mass: 50% capability replay and 50% nearby core;
- every slice remains 50/50 foundation/trench;
- V7 geometry mass within each family follows the frozen V8 mixture;
- bounded screen: at most 4,000 updates;
- promotion: foundation nearby core at least `78/96`, trench nearby core at
  least `91/112`, every cell at least `12/16`, twice consecutively;
- both capability controls must retain their Stage-A gate.

### Stage C: full V8

- support: all 47 training conditions;
- mass: 25% capability, 25% nearby core, 50% V6 constraints;
- every slice remains 50/50 foundation/trench;
- bounded first allocation: configure 8,000 updates, checkpoint every 500,
  and let the 24-hour allocation determine the reached update;
- continue a promising checkpoint with true optimizer/schedule/sampler resume
  on the 120-hour queue; never restart it as a nominal continuation.

For every previously mastered 16-map condition, retention requires

```text
successes >= max(11, lower_of_two_mastery_counts - 1)
```

Two consecutive retention failures stop that treatment and restore the last
fully passing checkpoint. Training does not silently demote individual vector
environments.

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
- [x] Focused CPU contract tests, full 307-test suite, launcher syntax, and
  ShellCheck pass.
- [ ] Euler dry-run resolves only the two intended arms and immutable inputs.
- [ ] Both update-1 jobs pass finite parameters/losses, transition integrity,
  four-GPU runtime, and checkpoint verification.
- [ ] Submit the two-arm Stage-A bounded screen.
- [ ] Evaluate every 500-update checkpoint and apply the Stage-A gate.
- [ ] Launch Stage B only from a checkpoint that passes Stage A twice.
- [ ] Launch Stage C only from a checkpoint that passes Stage B twice.
- [ ] Use 120-hour true resume only after fixed held-out evidence remains
  promising; do not require the short run to have converged.
- [ ] Begin reward-v2 implementation/ablation only after dense full-V8 reward
  qualification.

The launch entry point is
[`scripts/euler_v8_deep_xattn_v1/submit.sh`](../../scripts/euler_v8_deep_xattn_v1/submit.sh).
It defaults to a non-mutating dry run and refuses to submit a screen without
passed update-1 receipts for both arms. This revision accepts only Stage A.
Before Stage B or C is enabled, the launcher must require the prior-stage gate
receipt and the exact promoted parent path and SHA-256; a later stage must not
restart from the original P5c parent.

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
