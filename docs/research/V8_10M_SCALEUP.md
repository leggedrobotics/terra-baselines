# V8 10M scale-up experiment

- Status: **STAGE-B 20K JOBS SUBMITTED AFTER PAIRED SMOKE ADMISSION**
- Updated: 2026-08-06
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
Both completed Stage-A runs used `DENSE` for every PPO update. The decaying
kickstart KL/value coefficients are teacher-distillation schedules, not reward
annealing; `reward_transition_launched=false` in both run contracts.

The primary performance metric is not `train/episode_success_rate`. That value
is success among completed episodes from the currently active training stage
and is retained only as a sampler/optimization diagnostic. Primary policy
performance is deterministic exact success on every source-disjoint full-V8
development episode at horizon 450, with condition-balanced macro completion,
family results, worst condition, and all per-condition results reported beside
it. Promotion uses the separate promotion panel.

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
- KL coefficient `1.0 -> 0` over 3,000 half-batch updates;
- value coefficient `0.5 -> 0` over 1,000 half-batch updates;
- learning-rate warmup for 200 half-batch updates;
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
2. On all 720 frozen full-V8 promotion map slots, record finite logits/values,
   teacher-student KL, value RMSE, and deterministic action agreement before
   PPO. This diagnoses transplant damage; it is not a behavioral result. It
   uses deterministic exact-slot keys rather than claiming the combined main
   panel's inherited episode seeds are valid frozen resets. The executable
   receipt is produced by `scripts/v8_10m_initialization.py`.
3. Matched four-GPU update-1 smokes. Require CUDA convolution backward,
   NCCL all-reduce, finite rollout/teacher tensors, gradients, model, optimizer,
   and loadable checkpoints.
4. Stage A uses one seed per arm for 4,000 half-batch updates on the two
   capability controls, checkpoints every 1,000 updates. The latest two
   promotion and development evaluations must each reach at least 12/16 exact
   successes in both conditions before nearby maps are admitted.
5. Nearby and full are separate later runs from the promoted checkpoint, with
   fresh optimizer state and immutable dense reward. The existing stage budgets
   remain 4,000 and 8,000 updates before any long continuation decision.
6. Report main promotion/development and the separate all-free capability
   promotion/development panels in one aggregate, family, and per-condition
   leaderboard. Online return is diagnostic only.
7. Stage A runs on the 120-hour 4090 queue so compilation and the full 4,000
   updates cannot be truncated by the old 24-hour limit. Replication remains
   gated on held-out progress without family, worst-cell, capability, or
   integrity regression.

The map curriculum is part of this named treatment. The reward curriculum is
declared now but remains a separate checkpoint-bounded stage change: Stage A,
nearby, and full-map learning all remain dense. `terminal_margin` can begin only
after the fixed full-bank reward gate passes; `terminal_objective` follows only
after the margin stage passes its own fixed gate.

### Batch-size repair after the real smokes

The first 10M update-1 smoke (`9678493`) failed before an optimizer update when
XLA required a 17.38 GB training temporary at 1,024 environments per device.
The dependent screen was canceled automatically. The compact smoke passed, but
its pending screen was canceled before allocation so the arms could remain
matched.

The first repair used 512 environments and 16 minibatches. That reduced resident
environment state but accidentally retained a 1,024-sample local PPO minibatch;
the second 10M smoke (`9680835`) therefore reproduced the failure with a
17.50 GB allocation. Its dependent screen was also canceled automatically, and
the newly passing compact screen was canceled before allocation.

The final treatment uses 512 environments per device and **32** minibatches for
both arms, producing a 512-sample local PPO minibatch. It doubles updates,
annealing periods, LR warmup, entropy schedule, and checkpoint intervals to
preserve 262,144,000 Stage-A transitions, and halves the learning rate to
`1.5e-4` for the doubled sequence of half-batch optimizer steps. This preserves
the total number of sample presentations across two PPO epochs, while making
the unavoidable optimizer-step change explicit and identical in both arms.

## Euler launch receipt

The final paired update-1 smokes passed on 2026-08-05 from source revision
`d6bbba5bf60999f56663afb27e2a0b2b9931f877`:

| Arm | Smoke job | Result | Update-1 checkpoint SHA-256 |
|---|---:|---|---|
| `G-V8-XATTN-REWARM-CONTROL` | `9683427` | `COMPLETED`, receipt passed | `81ead9ff5216e0f6c32a2f5ffd0d5f60f89f97e80e492e3f7d27f3e5265c16f8` |
| `G-V8-10M-XATTN-WARM` | `9683428` | `COMPLETED`, receipt passed | `63429599fa17d44896991ec780b63ac9825e21e31a2b51f2450d00650580de03` |

Both smokes validated a CUDA convolution backward pass, distributed update,
finite optimizer state, and loadable update-1 checkpoint. The frozen
initialization diagnostic covered all 720 map slots. The compact control is
exactly output preserving at initialization (`KL=0`, action agreement `1.0`,
value RMSE `0`). The widened 10M transplant is finite but **not** function
preserving (`KL=19.965`, p95 `27.151`, action agreement `0.00972`, value RMSE
`5.243`). It is therefore recorded as a parameter warm start followed by
explicit KL/value distillation, not as a Net2Wider identity transform.

After both admissions passed, the paired 4,000-update Stage-A screens were
submitted from immutable source revision
`2a195b6c7112e56684d6088f1c9a073f3a3ff047`:

| Arm | Screen job | Runtime | Final state |
|---|---:|---|---|
| `G-V8-XATTN-REWARM-CONTROL` | `9685873` | `5:19:05` | `COMPLETED`, gate passed |
| `G-V8-10M-XATTN-WARM` | `9685874` | `9:33:43` | `COMPLETED`, gate passed |

The jobs had no Slurm dependency and a `119:45:00` limit. Both revalidated their
matching completed smoke receipt, ran all 4,000 updates, completed fixed
capability evaluation, and wrote a passing Stage-A gate receipt.

The selected frozen teacher is update 7,500 of
`G-DEEP-XATTN-V8-DIRECT-FULL-TEACHER`, SHA-256
`a6bebfffcf4d390df19ade9652d3c96d833eb7d2587ddb1b95035b7ad6a807f6`.
The user-authorized performance-waiver remains explicit: this checkpoint is a
finite same-V8 teacher, not a claim of full-bank mastery.

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
- [x] Add the two-arm Euler update-1 and 120-hour screen launcher.
- [x] Add the dependent common-prefix evaluator and aggregate/family/condition
  leaderboard, including separate all-free capability controls.
- [x] Run both smoke and screen launch plans locally with `SUBMIT=0` from a
  committed clean revision; no SSH, scratch, W&B, or Slurm mutation occurred.
- [x] Run paired Euler update-1 smokes from the provisional teacher.
- [x] Submit the paired Stage-A screen only after both smokes pass.
- [x] Record explicit launch authorization and the teacher-performance waiver.
- [x] Require the formal fixed full-bank gate before changing reward stage;
  Stage-A does not pass it, so reward fading remains locked.
- [x] Repair combined main-panel evaluation by separating exact map-slot keys
  from frozen environment/pose seeds; verify all 720 promotion resets locally.
- [x] Evaluate every Stage-A checkpoint on full-V8 promotion/development and
  publish the standardized family/condition leaderboard.
- [x] Evaluate the output-identical update-0 compact reference (the selected
  update-7,500 full-V8 teacher) with the repaired whole-V8 evaluator and bind
  its source contract plus all four fixed-panel hashes into Stage-B admission.

## Preparation verification

The launched implementation passes:

- the full Terra-baselines suite: `350 passed`, `73 warnings`, including 3
  subtests;
- Black formatting and Python byte-compilation for the new Python entrypoints;
- Bash syntax and ShellCheck at warning severity for all new launch scripts;
- the real-model growth probe at exactly `10,257,209` parameters;
- an independent launch-blocker re-review of the hardened teacher gate, with no
  remaining blocker found;
- paired update-1 smoke receipts from jobs `9683427` and `9683428`; and
- completed Stage-A jobs `9685873` and `9685874`, both with passing fixed
  capability receipts; and
- focused reset/reporting tests plus a local exact 720-slot reset receipt for
  the combined promotion panel; and
- completed reference-teacher evaluation job `9845019`, including all 720 main
  and 32 capability episodes on both source-disjoint splits.

The next operational check is paired update-1 Stage-B admission, followed by
the nearby-stage 20,000-update jobs under `bounded_replay25_v1`. Reward fading
remains locked until the later fixed **full-V8** gate passes; no online
completion signal may change the reward.

### Bounded replay population contract

Stage A necessarily samples its two capability conditions 50/50. For later
stages, the original archive-declared 50% replay is replaced by the named
training-protocol treatment `bounded_replay25_v1`; the map archive itself is
unchanged:

- nearby: 25% mastered capability replay, 75% newly active nearby conditions;
- full: 25% replay of the mastered nearby-stage mixture and 75% newly active
  constraints. Expanding the replay mixture gives 6.25% capability, 18.75%
  nearby core, and 75% constraints;
- every slice is still foundation/trench balanced;
- replay probabilities remain fixed inside a checkpoint-bounded stage;
- fixed capability/core panels guard retention. A retention failure restores
  the last passing checkpoint and previous mixture; it does not silently spend
  more population on mastered maps or use per-environment promotion/demotion.

This is the simplest anti-forgetting rule that still directs most data toward
the current unsolved support. The sampler profile and exact ordered probability
vector are stored in checkpoints and gate receipts.

### Stage-A result (2026-08-05)

Both jobs completed all 4,000 updates and passed the latest-two-checkpoint
capability gate:

| Policy | Job | Promotion exact | Development exact | Development macro |
|---|---:|---:|---:|---:|
| compact deep+xattn | `9685873` | `30/32` | `31/32` | `0.969` |
| 10M deep+xattn | `9685874` | `31/32` | `29/32` | `0.955` |

These are two-condition capability results, not whole-V8 success. The 10M
policy has therefore passed Stage A but has not yet shown a capacity advantage.
The next measurement enumerates all 720 promotion and 720 development episodes
for every Stage-A checkpoint before Stage B consumes more PPO compute.

### Combined-panel reset repair

The combined V8 main manifests correctly froze each episode's map identity and
environment seed, but those inherited seeds selected only 10 of 720 ordered
slots after V6 and V7 panels were concatenated. Evaluation now uses two keys:
an exact-slot key solely to materialize each ordered map once, and the frozen
manifest seed solely for pose/environment initialization. A local reset receipt
verified all 720 promotion slots, layers, metadata, zero initial steps, and the
ordered episode-seed hash. The benchmark maps and episode IDs were not changed.

### Corrected whole-V8 result and Stage-B decision (2026-08-06)

Job `9839960` completed with exit `0:0` after enumerating every one of the 720
promotion and 720 source-disjoint development episodes for all four Stage-A
checkpoints of both policies. The corrected benchmark selects checkpoints by
promotion exact successes, then promotion macro, while requiring both
capability controls to remain at least `12/16` on promotion and development.
Development reports generalization and never selects a checkpoint.

| Policy | Selected update | Promotion exact / macro | Development exact / macro | Capability P / D |
|---|---:|---:|---:|---|
| compact deep+xattn | 1,000 | `530/720` / `0.855` | `516/720` / `0.840` | `16+16` / `15+16` |
| 10M deep+xattn | 3,000 | `75/720` / `0.340` | `68/720` / `0.344` | `16+16` / `14+15` |

The compact checkpoint SHA-256 is
`5050b5c2fc890a176d1acf64fd7ba8c79a4c4ddbdd8b9ee1a64f727ec7833c80`;
the 10M checkpoint SHA-256 is
`ea6d61f7d95dd398c29819b67b4b8806c25dc0c6f33d092b222dc77c21182dd1`.
The immutable selection receipt is
`/home/lorenzo/moleworks/.artifacts/terra_v8_stagea_whole_eval_20260806/stage_b_selection.json`,
SHA-256
`63452163df74e0b413c4a8a6bbd50169c13dddce8c18f15c838dd355e94f715b`.

The 10M result is a learning signal, not a capacity win. It rises from
`0/720` development exact at update 1,000 to `68/720` at update 3,000, then
falls to `4/720` at update 4,000. The compact policy also regresses after its
early peak. Therefore neither terminal checkpoint is a valid parent and dense
reward remains frozen.

Selected-checkpoint nearby weaknesses are condition-specific:

| Policy | Weak nearby cells on development |
|---|---|
| compact | slab `2/16` (`0.611` mean completion), irregular `7/16` (`0.686`), courtyard-pads `8/16` (`0.701`), bearing-walls `8/16` (`0.714`) |
| 10M | courtyard `0/16` (`0.211`), slab `0/16` (`0.218`), bearing-walls `0/16` (`0.302`), courtyard-pads `1/16` (`0.310`), irregular `2/16` (`0.453`) |

The complete standardized family/condition histories are in
`/home/lorenzo/moleworks/.artifacts/terra_v8_stagea_whole_eval_20260806/leaderboard/`.

The independent full-V8 compact teacher was then evaluated by job `9845019`.
It is materially stronger than either selected Stage-A parent and is therefore
the common frozen distillation teacher:

| Reference teacher | Promotion exact / macro | Development exact / macro | Capability P / D |
|---|---:|---:|---:|
| compact deep+xattn update 7,500 | `552/720` / `0.875` | `538/720` / `0.868` | `31/32` / `31/32` |

The teacher checkpoint SHA-256 is
`a6bebfffcf4d390df19ade9652d3c96d833eb7d2587ddb1b95035b7ad6a807f6`.
Its source run contract and four fixed-panel files are independently
hash-pinned in the launcher. Local copies are archived under
`/home/lorenzo/moleworks/.artifacts/terra_v8_reference_teacher_full_eval_20260806/`.

Stage B is a paired 20,000-update allocation over 15 conditions and 96 real
layouts per condition (`1,440` distinct training maps,
`1,310,720,000` transitions per arm). Each random full reset samples the fixed
`bounded_replay25_v1` population: 25% capability replay, 75% nearby core, and
50/50 foundation/trench mass inside each slice. There is no per-environment
ratchet. Promotion remains global and checkpoint-bounded; every 1,000-update
checkpoint is retained for the fixed panels.

Both arms use the independent full-V8 compact update-7,500 checkpoint as one
frozen teacher on current Stage-B rollout observations. Policy KL decays from
`1.0` over 3,000 updates and value imitation from `0.5` over 1,000 updates,
with a 200-update learning-rate warmup. The compact arm begins from its
selected update-1,000 weights; the 10M arm begins from its selected
update-3,000 weights. This matched distillation treatment protects broad
competence while both policies learn the 1,440-map population. It is not
reward annealing.

An update-1 engineering smoke for the compact arm (`9846405`) passed against
the superseded compact-parent teacher wiring. The paired 10M job (`9846416`)
was cancelled while still pending, with zero elapsed compute, when the stronger
reference-teacher evidence arrived. Neither receipt qualifies the final
teacher-bound launch; both final smokes are rerun from one immutable revision.

The first teacher-bound attempt exposed a pre-PPO validator mismatch. Compact
job `9849664` failed after 41 seconds because the source teacher's original
`bank_v4` full-sampler vector was compared to the later Stage-B
`bounded_replay25_v1` digest. The pending 10M peer `9849666` was cancelled at
zero elapsed compute. Admission now reconstructs and validates the teacher's
hash-pinned legacy sampler separately; Stage B remains unchanged on the new
25/75 bounded-replay population.

The corrected submission then stopped before Slurm because the Euler scratch
user inode count (`1,105,009`) exceeded its `1,000,000` soft quota despite only
528.5 GiB of 2.273 TiB byte quota being used. No run was submitted. Rather than
delete unrelated historical artifacts, Stage-B outputs are rooted at
`/cluster/work/rsl/lterenzi/terra_v8_10m_nearby_long_v1`, whose 16 TiB quota
has ample byte and inode headroom. Inputs and model identities remain at their
original hash-pinned paths; this is storage routing, not an experiment change.

### Final Stage-B admission and allocation

Revision `f682f37d6a856c779b2c52e9e2d02a56cb04c15c` passed the complete local
suite (`351 passed`, 73 warnings, 3 subtests), Bash/ShellCheck validation, the
real-bank legacy sampler probe, and both Euler update-1 gates:

| Arm | Smoke | Runtime | Parameters | Receipt |
|---|---:|---:|---:|---|
| compact deep+xattn | `9854547` | `00:14:14` | `2,856,685` | `COMPLETED 0:0`, PASS |
| 10M deep+xattn | `9854549` | `00:14:36` | `10,257,209` | `COMPLETED 0:0`, PASS |

Both receipts bind the selected parent SHA, teacher checkpoint and source
contract, all four teacher fixed-evaluation hashes, 1,440 distinct maps,
`bounded_replay25_v1`, 25% capability replay, 75% nearby core, 50/50 family
mass within each slice, dense reward, full resets, and finite periodic/final
checkpoints.

The paired long jobs were then submitted independently to `gpuhe.120h`, each
requesting four RTX 4090s, 20,000 updates, and `4-23:45:00`:

| Arm | Slurm job | Initial scheduler state |
|---|---:|---|
| compact deep+xattn | `9858450` | `PENDING (Priority)` |
| 10M deep+xattn | `9858451` | `PENDING (Priority)` |

They have no dependency on each other. Each long job revalidates its matching
completed smoke before training and retains every 1,000-update checkpoint for
the fixed 720-map main and 32-map capability panels.

Both allocations subsequently started independently:

| Arm | Start (2026-08-06 CEST) | Node | Early steady throughput | State |
|---|---:|---|---:|---|
| compact deep+xattn | `14:13:44` | `eu-g6-065` | approximately 15.1k transitions/s | `RUNNING` |
| 10M deep+xattn | `14:19:48` | `eu-g6-023` | approximately 8.3k transitions/s | `RUNNING` |

The emitted contracts match the launch receipt, including each distinct
parent SHA, common admitted teacher, 1,440-map bounded-replay population,
dense reward, full resets, 20,000 updates, and checkpoint interval 1,000. Both
completed multiple real PPO updates and passed the update-10 finite-check
boundary. The 10M XLA/cudNN autotuner reported BF16 convolution-algorithm
comparison mismatches, rejected those algorithms, and continued at stable
throughput; there is no NaN, OOM, NCCL, or process failure. The projected
training times are about 24 hours for compact and 44 hours for 10M, before the
post-training fixed-panel sweep. These projections are operational only and
do not qualify either policy.

An early checkpoint observer was added in commit `335bf3c`. It reuses the
already validated one-checkpoint whole-V8 evaluator from immutable launch
revision `f682f37` and hash-pins each retained Stage-B checkpoint before
enumerating main promotion/development plus both all-free capability panels.
Its results are stored outside the training run under `checkpoint_eval/` and
cannot change optimizer or sampler state. One checkpoint is diagnostic only:
each capability condition must retain at least `12/16`, and only two
consecutive retained-checkpoint failures trigger rollback. Full Stage-B
promotion still requires the complete latest-two-checkpoint core, family,
capability, integrity, and development gates.

At the initial live observation, W&B reported the declared population within
sampling noise (approximately 25% capability, 75% nearby core, and exactly
50/50 foundation/trench). Active-stage completion was high for both arms, but
this is completed-episode behavior on the sampled training population. It is
not source-disjoint whole-V8 performance and cannot promote a checkpoint or
unlock reward progression.
