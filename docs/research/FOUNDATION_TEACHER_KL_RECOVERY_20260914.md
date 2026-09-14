# Foundation recovery with a frozen policy teacher — September 14, 2026

The user requested using the strong old policies to restart learning, then
specified a foundation-only teacher-KL run on CSCS. Prepare one bounded
24-hour Daint allocation, four GH200 GPUs, using lterenzi. This replaces the
unsubmitted plain-recovery proposal; no new Euler recovery was submitted.
CSCS SSH currently rejects the certificate. Local implementation and validation
can proceed while authentication is renewed.

## Scientific question

Can PPO adapt an already capable foundation policy to corrected physical rules
while retaining its completion behavior? The old zero-added-cost control u15000
is both native student initialization and the frozen policy teacher. Its fresh
current-environment result is 62/64; old 2× scores 63/64. The zero-cost control
keeps the completion-first reward treatment consistent.

Teacher-guided RL with a policy-distribution loss is supported by the
[Kickstarting Deep Reinforcement Learning paper](https://arxiv.org/abs/1803.03835).
That supports trying the method, not this particular coefficient or a predicted
Terra improvement. A matched no-KL continuation would be needed to attribute
retention specifically to KL; this single run is a recovery screen.

The selected initial bank is the same 256 easy-foundation training maps, with
64 held-out validation resets. Full-foundation coverage was presented to the
user as an alternative scope; do not quietly change the bank after staging.
The teacher's 62/64 result applies to the easy bank, not to all foundation
categories in the full dataset.

## Run contract

| Setting | Value |
| --- | --- |
| Native student parent and frozen teacher | Old control u15000, SHA d1a6c07d9d8a40b7b7b60bd0b54313aa46a9b50fb09c789ed4ebffddb2b488b3 |
| Environment | Terra 46738cde; corrected swept movement and soil-free chassis protections |
| Initialization | Native weights, Adam state and absolute clocks; environment/RNG/action history restart as usual |
| Seed | 20260907 |
| GPU layout | 4 × 128 environments; global batch 512 |
| Rollout / PPO | 32 steps, two epochs, 32 minibatches; 16,384 transitions and 64 Adam steps per update |
| Advantage normalization | Global minibatch moments, matching merged one-GPU mathematics |
| Policy loss | PPO plus beta × KL(teacher || student) on student-visited observations |
| Teacher policy | Frozen parameters; same observation interface and current environment inputs |
| KL schedule | beta starts at 1 at absolute u15000 and cosine-decays to 0 at u35000 |
| Value distillation | Disabled; critic learns returns under the current physical rules |
| LR and entropy | Native 3e-4 constant LR; no fresh-optimizer warmup; existing absolute entropy schedule retained |
| Added behavior costs / cost ramp | All zero / disabled |
| Checkpoints / budget | Every 500 updates; absolute target u500000, bounded by one 24-hour allocation |
| Fixed evaluations | u17500, u20000 and u25000; same 64-map greedy 450-step panel |

Changing from the original one-GPU run to four GH200s changes arithmetic and RNG
streams. Global advantage moments preserve the intended global minibatch
normalization but do not make the trajectories bit-identical. Teacher guidance
may constrain useful departures from the teacher; its annealing schedule is an
explicit pilot choice. No automatic penalty promotion or further allocation is
part of this run.

Report exact completion, mean excavated/accepted material, productive base poses,
unique area per productive setup, retained-work travel and workspace adjacency.
Compare efficiency on common successful episodes and retain failures separately
so lost coverage cannot masquerade as more efficient digging.

## Why trench development can affect foundation learning

The inspected foundation launcher samples only train/all. It has one curriculum
level, no trench rewards, no mixed-task pool and no cross-process gradients.
The trench alignment gate is inactive on these maps. Shared soil/chassis,
dumping, relaxation and movement repairs still change exploration and can make
skills harder to discover from scratch. Successful frozen old-policy execution
proves completion remains possible; it does not establish equal learning
difficulty. See [the isolation audit](../../../../../.artifacts/terra_foundation_strong_recovery_20260914/foundation_isolation_audit.md).

## Implementation and validation

The existing teacher path required three fixes before this native continuation:

- Its original annealing clock used absolute update zero, which would disable
  the default KL term before a u15000 resume. An explicit checkpointed origin
  now supports a new teacher phase without resetting the PPO/Adam clock, and
  native continuation rejects dropping or changing the teacher schedule.
- Setting teacher LR warmup to zero still constructed a different Optax state
  tree. Zero now preserves the old constant-LR Adam structure; positive warmup
  keeps existing behavior.
- The teacher model's minimal environment stub omitted the executable-dig
  observation selector. It now preserves and checks that interface. Fixed-bank
  inference clears the new schedule origin when disabling the teacher.

Seven focused native-teacher tests and 47 existing training-utility tests pass.
The first actual-parent GPU attempt caught the observation-stub error before
training; that failure is retained under failed_smoke_observation_selector.
After correction, two local 1×32 updates pass finite model/Adam/loss/teacher,
R2/environment/bank/integrity and checkpoint checks. The second native checkpoint boundary also passes at u15003: Adam is 960192,
teacher KL is 0.0516, and its coefficient retains the u15000 origin. These are
functional gates, not full-size throughput or policy-quality evidence. The
second process still incurred about 206 seconds before its first update;
compilation-cache reuse across that boundary is not established by this smoke.

The CSCS allocation must independently pass four-GH200 identity, convolution
backward, NCCL and two finite native updates at the full 4×128 production shape
before continuing. A failure stops that allocation; the batch is not shrunk.
Source, inputs, immutable teacher and container image are hashed. A local
26-hour watcher can download and verify the three milestones and serialize
complete fixed evaluations on an idle GPU; it never submits jobs.

[Manifest](../../../../../.artifacts/terra_foundation_strong_recovery_20260914/manifest.json) · [Local startup checks](../../../../../.artifacts/terra_foundation_strong_recovery_20260914/local/startup_checks.json) ·
[Launch scripts](../../../../../.artifacts/terra_foundation_strong_recovery_20260914/launch/) · [Independent submission review](../../../../../.artifacts/terra_foundation_strong_recovery_20260914/submission_review.md).
