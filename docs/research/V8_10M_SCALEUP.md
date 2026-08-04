# V8 10M scale-up experiment

- Status: **STAGE-A LAUNCH AUTHORIZED; EVIDENCE PENDING**
- Date: 2026-08-04
- Parent campaign:
  [`V8_DEEP_XATTN_CURRICULUM.md`](V8_DEEP_XATTN_CURRICULUM.md)
- Implementation policy:
  [`simple-research-code`](/home/lorenzo/git/codex_skills/skills/simple-research-code/SKILL.md)

## Question

Starting from the same healthy full-V8 compact teacher, does increasing only
the spatial encoder width improve held-out V8 completion under the staged map
curriculum at the same number of PPO transitions?

On 2026-08-04 Lorenzo explicitly authorized starting the matched compact and
10M students without waiting for formal teacher mastery. This waives only the
teacher performance gate. The selected teacher must still be a finite,
same-distribution, full-V8 deep+xattn checkpoint with a valid optimizer and
full-bank sampler state.

The launched treatment is checkpoint-bounded on two independent axes:

```text
map:    capability -> nearby -> full
reward: dense_skill -> terminal_margin -> terminal_objective
```

Stage A starts under `dense_skill`. Reward progression cannot start from online
training success; it requires high fixed full-V8 completion after the full map
stage. Terra's legacy `SPARSE` enum is not an acceptable substitute for the
terminal reward contract described in the canonical progressive-reward spec.

## Teacher admission

### Provisional launch admission

The authorized Stage-A launch accepts the latest numbered checkpoint at or
after update 5,000 from `G-DEEP-XATTN-V8-DIRECT-FULL-TEACHER` when all of these
hold immediately inside Slurm:

- the source run contract names the exact V8 release, dataset hashes, dense
  reward, 450-step horizon, full resets, and all 47 conditions;
- checkpoint and run-contract hashes match the submitted identities;
- model and optimizer trees and optimizer step are finite;
- the compact deep+xattn architecture contains exactly 2,856,685 parameters;
- the checkpoint stores the frozen full-V8 accepted-bank identity; and
- the full ordered sampler state is valid for the paired seed.

The resulting inspection is explicitly marked
`performance_mastery_gate_waived_by_user=true`; it is not a mastery receipt.

### Formal reward-curriculum admission

The 10M experiment accepts exactly one teacher type:

- arm: `G-DEEP-XATTN-V8-DENSE-WARM`;
- map stage: full V8;
- dataset release:
  `terra_v8_v6_constraints_v7_adjacent_train96_v5`;
- dataset SHA-256:
  `715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798`;
- training-mixture SHA-256:
  `f2a2a33556d513b46193a8a3996d37e6989534eba9373f46f52d79f956ac128e`;
- full ordered-sampler SHA-256:
  `2a457be780e086c02e0474489b2060d6c577fac0ac429c48ad1a7e1e5e011357`;
- identical dense reward, horizon 450, actions, observations, dynamics, full
  resets, PPO shape, and deterministic evaluation protocol.

The selected teacher must be named by an immutable
`terra_v8_dense_reward_gate_v1` receipt. Its latest three scheduled promotion
evaluations must each pass:

- exact overall at least `576/720`;
- foundations at least `308/384`;
- trenches at least `269/336`;
- every main condition at least `10/16`;
- frozen capability/core retention and integrity.

The selected checkpoint must independently meet the same aggregate, family,
per-condition, and integrity thresholds on development. Each separate all-free
capability control must also reach at least `12/16` on capability development.
The receipt, main and capability development evaluations, checkpoint hash,
finite model/optimizer state, full-V8 configuration, and architecture are
revalidated immediately before growth. Failing any item stops the launch. A
merely `RUNNING` policy, high online success, or a P5c checkpoint cannot
qualify.

Admission reconstructs the checkpoint treatment fingerprint and sampler state,
re-enumerates every map against the current frozen manifests, recomputes the
promotion gate from the four hash-pinned panels, and then applies the selected
development and capability gates. There is no metadata-only or
`--skip-validation` path.

The executable gate is
[`scripts/v8_10m_student.py`](../../scripts/v8_10m_student.py).

## Frozen target architecture

The target retains the teacher's xattn readout, heads, block depth, observation
contract, and action heads. It changes one factor: spatial stage width.

| Field | Teacher/control | 10M treatment |
|---|---:|---:|
| Spatial channels | `(24,48,64,96)` | `(64,128,192,256)` |
| Residual blocks | `(2,2,3,3)` | `(2,2,3,3)` |
| Encoder | deep SE + E4-prime xattn | same |
| Critic | `(512,256)` | same |
| Parameters | `2,856,685` | `10,257,209` |

The treatment is 3.59 times the parameters of the compact teacher. It does not
add the E7 token mixer, change map resolution, change recurrence, or widen the
policy/value heads. `scripts/v8_10m_student.py probe` derives the count from the
real model and fails unless it is exactly `10,257,209`.

## Lux architecture-scaling review

The requested primary-source review supports the proportional-width target:

- Lux S1's winner used full-resolution, fixed-width 128-channel SE-ResNets and
  scaled depth from 8 to 16 to 24 blocks, ending near 20M parameters. Each new
  model learned from the previous smaller frozen teacher. See the
  [author's architecture writeup](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021#neural-network-architecture),
  [16-block teacher config](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/conf/conv_phase3_small_teacher.yaml),
  and [24-block final config](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/conf/conv_phase5%2B_final_model.yaml).
- Lux S3's runner-up gives the closest direct precedent: eight full-resolution
  256-channel SE blocks and approximately 10M parameters. See the
  [solution writeup](https://github.com/IsaiahPressman/kaggle-lux-2024/blob/main/write-up.md)
  and [model source](https://github.com/IsaiahPressman/kaggle-lux-2024/blob/main/python/rux_ai_s3/models/actor_critic/base.py).
- Lux S3's winner kept 24 full-resolution 128-channel SE blocks before a
  ConvLSTM and four small attention blocks. Counting the released checkpoint is
  an inference from the published artifact: about 23.07M parameters total,
  about 85.5% in the full-resolution ResNet and 2.75% in attention. See the
  [published solution](https://github.com/tonykozlovsky/lux-ai3-pub),
  [final configuration](https://github.com/tonykozlovsky/lux-ai3-pub/blob/main/final_versions/08_03_tune_against_mask_cont/submission_model/config.yaml),
  and [released checkpoint](https://github.com/tonykozlovsky/lux-ai3-pub/blob/main/final_versions/08_03_tune_against_mask_cont/submission_model/200000000_weights.pt).
- Lux S2 FLG is the relevant efficiency counterexample: a 128-channel
  `DoubleCone(4,6,4)` moved six blocks to lower resolution while retaining wide
  full-resolution blocks before and after the cone. It did not use a
  skinny-to-huge terminal pyramid. See the
  [official FLG writeup](https://www.kaggle.com/competitions/lux-ai-season-2/writeups/flg-flg-s-approach-deep-reinforcement-learning-wit).

Therefore `(64,128,192,256)` is the frozen first 10M hypothesis. The cheaper
`(24,48,64,320)` alternative would concentrate almost all capacity in the last
8x8 stage; keep it only as a later throughput/quality Pareto ablation if the
broad model is runtime-limited.

The same review reinforces the teacher contract. Lux S1 trained larger students
from behaviorally proven smaller teachers on current rollout observations; the
S3 runner-up decayed policy and value imitation while evaluating the teacher on
PPO experience observations. See the
[S1 training implementation](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/lux_ai/torchbeast/monobeast.py),
[S3 PPO defaults](https://github.com/IsaiahPressman/kaggle-lux-2024/blob/main/python/rl/config/ppo_default.yaml),
and [S3 PPO training code](https://github.com/IsaiahPressman/kaggle-lux-2024/blob/main/python/rl/train_ppo.py).
This supports a smaller but demonstrably strong same-distribution teacher; it
does not support using a merely finite teacher from an older map bank.

## Minimal causal comparison

Run two arms only:

1. `G-V8-XATTN-REWARM-CONTROL`: exact teacher architecture, parameter-only
   restart from the qualified teacher.
2. `G-V8-10M-XATTN-WARM`: channel-grown target from the same teacher.

Both use:

- the same frozen teacher for KL and value targets;
- a fresh optimizer and update counter;
- KL coefficient `1.0 -> 0` over 1,500 updates;
- value coefficient `0.5 -> 0` over 500 updates;
- learning-rate warmup for 100 updates;
- the exact same V8 map stage at every paired update;
- dense reward only;
- the same seed, transitions, PPO settings, and fixed evaluations.

The control is necessary: comparing a newly trained 10M student only with its
older teacher would confound capacity with extra PPO transitions and optimizer
restart. Compare common update numbers; wall-clock throughput is reported
separately. If the 10M arm needs a smaller PPO minibatch to fit, apply the same
minibatch setting to the control or do not claim a capacity-only comparison.

## Gates and run budget

1. CPU parameter/growth probe.
2. On all 720 exact frozen full-V8 promotion resets, record finite logits/values,
   teacher-student KL, value RMSE, and deterministic action agreement before
   PPO. This diagnoses transplant damage; it is not a behavioral result. The
   executable receipt is produced by `scripts/v8_10m_initialization.py`.
3. Matched four-RTX-4090 update-1 smokes. Require CUDA convolution backward,
   NCCL all-reduce, finite rollout/teacher tensors, gradients, model, optimizer,
   and loadable checkpoints.
4. Stage A uses one seed per arm for 2,000 updates on the two capability
   controls, checkpoints every 500 updates. The latest two promotion and
   development evaluations must each reach at least 12/16 exact successes in
   both conditions before nearby maps are admitted.
5. Nearby and full are separate later runs from the promoted checkpoint, with
   fresh optimizer state and immutable dense reward. The existing stage budgets
   remain 4,000 and 8,000 updates before any long continuation decision.
6. Report main promotion/development and the separate all-free capability
   promotion/development panels in one aggregate, family, and per-condition
   leaderboard. Online return is diagnostic only.
7. Replicate or grant 120-hour compute only if the 10M treatment shows held-out
   progress without family, worst-cell, capability, or integrity regression.

The map curriculum is part of this named treatment. The reward curriculum is
declared now but remains a separate checkpoint-bounded stage change: Stage A,
nearby, and full-map learning all remain dense. `terminal_margin` can begin only
after the fixed full-bank reward gate passes; `terminal_objective` follows only
after the margin stage passes its own fixed gate.

## Preparation checklist

- [x] Freeze a concrete approximately-10M architecture and derive its exact
  parameter count.
- [x] Implement a same-distribution teacher receipt and development gate.
- [x] Add focused CPU tests for parameter count and teacher admission.
- [x] Incorporate the primary-source Lux architecture-scaling review.
- [x] Add and test the parameter-tree growth diagnostic: all 174 target leaves
  are covered, the grown tree is finite, and no leaf takes the generic fresh-
  initialization path.
- [x] Add the real-teacher 720-reset pre-PPO output diagnostic.
- [x] Add the two-arm Euler update-1 and true 24-hour launcher.
- [x] Add the dependent common-prefix evaluator and aggregate/family/condition
  leaderboard, including separate all-free capability controls.
- [x] Run both smoke and screen launch plans locally with `SUBMIT=0` from a
  committed clean revision; no SSH, scratch, W&B, or Slurm mutation occurred.
- [ ] Run paired Euler update-1 smokes from the provisional teacher.
- [ ] Start the paired Stage-A screen only after both smokes pass.
- [x] Record explicit launch authorization and the teacher-performance waiver.
- [ ] Require the formal fixed full-bank gate before changing reward stage.

## Preparation verification

The prepared implementation currently passes:

- the full Terra-baselines suite: `344 passed`, including 3 subtests;
- Black formatting and Python byte-compilation for the new Python entrypoints;
- Bash syntax and ShellCheck at warning severity for all new launch scripts;
- the real-model growth probe at exactly `10,257,209` parameters;
- an independent launch-blocker re-review of the hardened teacher gate, with no
  remaining blocker found.

No 10M Euler job has been submitted. `SUBMIT=0` is the only allowed launcher
mode until a qualifying teacher receipt exists and a later explicit launch
instruction is recorded.
