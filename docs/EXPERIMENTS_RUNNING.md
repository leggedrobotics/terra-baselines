# Experiments — current state (updated 2026-08-13 CEST)

## Active: v6.1 material-stall observation continuation

Submitted job `10616190` at 2026-08-13 17:25 CEST from terra-baselines
`6ad2eb157ef2724f88db74fd44d940d05260689d`.  Its authoritative scheduler
state at submission was `PENDING (Priority)` with no assigned node or start
estimate.  Slurm records partition `gpuhe.24h`, effective account
`gpuhe/es_hutter`, a 23:45 wall limit, one node, 8 CPUs, 64 GiB, and exactly
8 RTX 4090 GPUs.  This is queued compute, not runtime evidence; the first
completed PPO update and finite update-14,500 checkpoint remain the go/no-go
gate.

The next run continues the selected v6.1 reward-v2 checkpoint at update 14,000
and adds one observation only:

```text
stall_age = min(consecutive transitions without material-state change, 32) / 32
```

Material state is the raw soil/action map plus every active excavator's load
and carry-relocation credit.  Soil, load, or carry changes reset the counter;
pose and cabin motion do not.  Two zero-initialized 704-wide embeddings inject
the scalar before the existing actor and critic heads, leaving the existing
head matrix shapes and the update-14,000 outputs unchanged when the scalar is
zero.  The prepared checkpoint preserves model parameters, Adam moments and
clock, absolute update and entropy schedules, and the complete
`continuous_banded_v2` sampler state.

The production segment requests 8 RTX 4090 GPUs in `gpuhe.24h` for 23:45 and
uses 256 environments per device.  This reshapes the old 4x512 execution into
8x256 while retaining 2,048 total environments, 65,536 transitions per PPO
update, 32 minibatches, and two epochs.  The allocation itself runs the CUDA
convolution-backward and NCCL preflight before its first finite production
update; there is no separate 8-GPU smoke allocation.  Checkpoints are written
every 500 absolute updates toward update 40,000, so a wall-time stop is a
continuable segment.

This treatment does **not** change reward-v2, its timing coefficients, the
v6.1 spatial architecture, action masking, sampler rule, training bank, seed,
learning rate, PPO shape, or horizon.  It also deliberately excludes
time-to-go: remaining time is a separate finite-horizon/pacing observation and
is not the proposed mechanism for breaking the measured repeated-input loops.

Readout order:

1. verify the first completed update and finite rolling checkpoint;
2. compare fixed promotion checkpoints against the existing v6.1 curve;
3. inspect `train/stall_age_mean` and
   `train/stall_age_saturated_fraction`, plus the prior failure strata;
4. if recurrence persists, run a separate actor-only GRU-64 pilot with the
   v6.1 encoder, reward, and curriculum fixed.  That pilot must use contiguous
   32-step PPO sequences, carried/reset hidden state, and a matched
   sequence-batched feed-forward control; a GRU is not bundled into this run.

The superseded 8-GPU resume smoke `10572344` was cancelled while still pending
and never allocated, so it produced no checkpoint or training evidence.

Pinned implementation at launch preparation:

- Terra: `c2d2a94a124759e9f21c2b37930f717e299f0c46`
- terra-baselines core: `ae4252c` plus finite-step check `aaa1fdd`
- direct launcher: `2387f27`
- submitted source revision: `6ad2eb157ef2724f88db74fd44d940d05260689d`
- source u14k checkpoint SHA-256:
  `79312602176e88b696c8c006b3b9af71a4cf121907c7aa8c4865722bd4830609`
- prepared checkpoint SHA-256:
  `96600430af3fb0135e0fc94e8f9dd754476067fbfb8635a3db70d6c3519b6971`

Because this is not paired with a no-scalar continuation from u14k, its result
is a practical continuation screen, not by itself a causal stall-age ablation.
Any improvement must be interpreted together with the scalar telemetry and the
frozen recurrence/failure panel.

Current design and decision authority:

- [`research/V8_IMPROVEMENT_SET_20260810.md`](research/V8_IMPROVEMENT_SET_20260810.md)
- [`research/V8_REWARD_TERMINATION_AUDIT.md`](research/V8_REWARD_TERMINATION_AUDIT.md)
- [`research/V8_10M_SCALEUP.md`](research/V8_10M_SCALEUP.md), current-successor
  amendment; older v1/reward-fade sections are historical

Supporting historical evidence:

- [`research/V8_DEEP_XATTN_CURRICULUM.md`](research/V8_DEEP_XATTN_CURRICULUM.md)
  (superseded hard-stage campaign)
- [`research/P5_FOLLOWUP_GOAL.md`](research/P5_FOLLOWUP_GOAL.md)
- [`research/P5_RESULTS_ANALYSIS.md`](research/P5_RESULTS_ANALYSIS.md)
- [`research/P5_ACCEPTED_BANK_EXPERIMENTS.md`](research/P5_ACCEPTED_BANK_EXPERIMENTS.md)

The older E1--E10 spatial-encoder run history is retained in
[`EXPERIMENTS_SPATIAL_V3_RUNS.md`](EXPERIMENTS_SPATIAL_V3_RUNS.md). It is not
current scheduler state.

## Current map-curriculum decision

The completed compact/Atari comparison used one uninterrupted random-start
process per arm over all 47 V8 conditions. It had no capability-only,
nearby-only, or constraint-only training jobs. Its `continuous_banded_v1`
sampler used immutable depth 0/1/2 labels and live family-specific bands:

```text
0.10 * Uniform(all family conditions)
+ 0.75 * Uniform(the entire active depth)
+ 0.15 * Uniform(the next depth)
```

Foundation and trench retain 50% target probability each and advance their
bands independently from exact completed training episodes. The permanent
10% all-condition term gives every map type support from update 0. Fixed
source-disjoint panels select and audit checkpoints but never update sampler
state. The original dense-only implementation base is commit
`0982b7803777fee81a227ce26f30bd85004a9aaa`. The active architecture pair uses
terra-baselines `dcc4f955347182e57e6f16e9df81a3f170564d97` and runtime Terra
`eb3835c1d17af81e970b973ed5abf687ca6f3a26`; the bank protocol remains frozen
at Terra `a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4`.

The active reward-v2 system uses `continuous_banded_v2` from update 0: within
each family, 10% remains uniform support and 90% is spread over every
unmastered condition with depth weights `4:2:1`; eligible conditions graduate
independently. Foundation/trench mass is `0.5/0.5`. On the 47-condition V8
graph, fresh-v2 aggregate depth mass is `0.113464 / 0.383608 / 0.502928` for
d0/d1/d2, versus approximately `0.754 / 0.178 / 0.068` at the historical v1
start.

The completed scientific comparison is a constant-dense architecture pair. Both
arms retain that map curriculum, data, transition budget, PPO shape, seed,
horizon, and fixed evaluations:

| Arm | Architecture | Parameters | Smoke | Long job | Current state |
|---|---|---:|---:|---:|---|
| `compact_xattn` | compact deep SE plus cross-attention | 2,856,685 | `10128202` PASS | `10128518` | COMPLETED `0:0`; all fixed panels complete |
| `atari_base` | original Atari CNN plus base heads | 480,137 | `10128203` PASS | `10128519` | COMPLETED `0:0`; all fixed panels complete |

Both arms use seed `20260807`, 47 x 96 maps, `continuous_banded_v1`, full
450-step resets, dense reward, checkpoints every 500 updates, and fixed-panel
evaluation every 1,000 retained updates. Compact used the 120-hour queue
because the previous compact 20,000-update run took about 25h53. Atari is a
small-system replication control, not a pure encoder ablation, because its
policy, value, and local-map heads are also smaller.

The next experiment is one practical compact reward-v2 system trained from
random initialization, not another dense arm. It uses all 47 conditions,
fresh `continuous_banded_v2`, the nine-feature carry observation, canonical
global distance ledger, reward-v2, seed `20260807`, and the prior
compact PPO recipe. Entropy decays `0.15 -> 0.02` over the first 20,000 updates
and remains `0.02` through the 40,000-update target. A conservative first phase
runs to update 14,000 in `gpuhe.24h`; promotion main and capability panels
evaluate every 1,000-update checkpoint and development evaluates only the
promotion-selected checkpoint. A statistically continuous resume toward the
absolute 40,000-update target is launched only if that held-out result is
promising. The completed dense run
above is a descriptive reference. This is one bundled system change and does
not isolate its reward, sampler, distance ledger, carry input, runtime,
or extra-compute contributions. The active goal and exact contract are in
[`research/V8_R2_IMPLEMENTATION_GOAL.md`](research/V8_R2_IMPLEMENTATION_GOAL.md).
No revised phase-1 scratch job is submitted yet.

The superseded prepared-fork smokes `10292301` and `10292324` both failed
before checkpointing on the same Optax `FrozenDict`/plain-dict tree mismatch.
They are archived implementation failures; the scratch path does not repair
or use that loader.

Compact's promotion-selected update-20,000 checkpoint scores 580/752 exact and
0.859 macro on the combined promotion panels. On development it scores 546/720
and 0.861 on the main panel plus 31/32 and 0.977 on all-free capability. The
failure-mechanism and reward-semantics audit is in
[`research/V8_REWARD_TERMINATION_AUDIT.md`](research/V8_REWARD_TERMINATION_AUDIT.md).

The original dense-only job is retained in the ledger until both replacement
architecture smokes pass:

| Prior arm | Update-1 smoke | Long job | Current state |
|---|---:|---:|---|
| `G-V8-XATTN-CONTINUOUS-BANDED` | `10012150` PASS | `10015084` | CANCELLED after replacement smokes passed; elapsed `0:00` |

Smoke `10012150` completed `0:0` in 13m55s. Both update-1 and final
checkpoints reloaded; the receipt records all 47 conditions, family counts
25/22, depth counts 2/13/32, positive minimum probability `0.002`, resumable
`terra_continuous_banded_sampler_state_v1`, and graph SHA-256
`f0ad2c9c138cbb7d7139ac8bf50cb8c9b897d06d49c625b82b78d0a9c3e42b2d`.
This is engineering admission evidence, not a learning result.

The same-binary random-start reward smokes `10022782` (dense) and `10022786`
(old whole-objective anneal) both passed individually, but the anneal trigger
had not fired and the terminal mix remained zero. No long reward pair was
submitted. The two independent GPU updates diverged numerically despite
matching sampler and transition receipts, so these jobs are historical
implementation evidence only and are not a reward result or an R2 admission.

The replacement smokes completed `0:0` in 13m53s (compact) and 12m16s
(Atari). Each passed the in-job CUDA convolution-backward and NCCL all-reduce
preflight, generic finite checkpoint reload, all-47 continuous-sampler receipt,
and exact architecture receipt. Both long jobs then ran on 4xRTX4090 and
completed training plus every fixed panel without runtime failure. Their W&B
IDs are
`v8_architecture_dcc4f95534_screen_compact_xattn_10128518` and
`v8_architecture_dcc4f95534_screen_atari_base_10128519`.

Atari completed all 20,000 updates and fixed evaluations. Its
promotion-selected update-19,000 checkpoint scores 16/752 exact and 0.457
macro on promotion, then 20/752 and 0.428 on development. The terminal
checkpoint scores 15/752 and 0.403 on development. It mastered zero conditions
and ended at depth 0 for both families, so it is a negative small-system
replication result and not a reward-fork parent. The detailed paper-facing
receipt and caveats are in
[`research/V8_10M_SCALEUP.md`](research/V8_10M_SCALEUP.md), section
"Completed result: Atari-base small-system control (2026-08-10)."

## Cancelled nearby-only reward diagnostic

The compact update-20,000 nearby-policy checkpoint had been selected for a
matched reward fine-tuning diagnostic. This did not define the replacement map
curriculum or authorize a map-depth transition. Both proposed arms kept the
historical 15-condition `bounded_replay25_v1` sampler, full 450-step resets,
architecture, PPO settings, seed, and parent parameters fixed; both would have
started fresh optimizers without a teacher.

| Arm | Reward | Update-1 smoke | 2,000-update screen | State |
|---|---|---:|---:|---|
| `R-U20-DENSE-CONTROL` | dense skill | `10007282` PASS | `10009405` | CANCELLED, elapsed `0:00`, no compute consumed |
| `R-U20-TERMINAL-OBJECTIVE` | terminal completion plus soft workspace/step efficiency | `10007283` PASS | `10009411` | CANCELLED, elapsed `0:00`, no compute consumed |

The source contract is terra-baselines
`3567419073e15a3ae8394ed279ea8e7f4839dc6c` plus runtime Terra
`85e67c3574a34f6238d1bd92caa382bd069d7755`. The common parent SHA-256 is
`9c92eacdef0b6a2402df0bb2a621b8bffe5c730d5c9ce4f163673569bb2d930e`.
Because that parent narrowly failed its historical development gate and never
trained the 32 constraint conditions, the two queued screens were cancelled
before allocation. Their smoke artifacts are retained only as implementation
history; they provide no learning comparison.

## Historical staged campaign evidence (superseded)

The following section records the earlier hard-stage campaign and the evidence
that motivated full-support continuous sampling. Its scheduler language is
historical and is not the current launch design.

The corrected whole-V8 Stage-A evaluation job `9839960` and reference-teacher
evaluation job `9845019` both completed cleanly. Teacher-bound update-1 smokes
`9854547` (compact) and `9854549` (10M) then completed `0:0` with passing
finite-checkpoint receipts. The paired 20,000-update historical Stage-B jobs
then completed on the 120-hour queue:

| Arm | Parent | Smoke | Long job | Current state |
|---|---:|---:|---:|---|
| compact deep+xattn | update 1,000 | `9854547` PASS | `9858450` | COMPLETED, 20,000 updates, 25:52:43 |
| 10M deep+xattn | update 3,000 | `9854549` PASS | `9858451` | COMPLETED, 20,000 updates, 46:32:38 |

Both jobs passed their immediate smoke, parent, teacher, sampler, architecture,
CUDA, cuDNN, and NCCL checks inside Slurm and completed without a NaN, OOM,
NCCL, traceback, or quota failure. Compact's best fixed development result was
`548/720` exact with macro `0.863` (best macro `0.870` at the adjacent retained
checkpoint), but it regressed to `315/720`, macro `0.529` at update 20,000.
The 10M policy peaked at `427/720`, macro `0.710` and finished at `418/720`,
macro `0.666`. This does not establish a 10M capacity advantage and motivates
the all-47 continuous dense trunk rather than more 10M compute.

The compact arm retained update 1,000 with SHA-256
`5130856886889f4dccd3efa3b60a843e5c3af666e04c12a6e688000e05598f2d`.
Early fixed observer job `9864040` completed `0:0`: promotion is `548/720`
exact and `0.865` macro; development is `546/720` and `0.865`; capability is
`31/32` on both splits; all integrity counts are zero. Nearby trenches pass
their Stage-B family/cell thresholds, but foundations reach only `53/96`
promotion and `60/96` development versus the required `78/96`. The checkpoint
improves on its selected parent but does not promote. Compact update-2,000
observer `9884423` and 10M update-1,000 observer `9884425` also completed
cleanly. Compact development remains `546/720` while macro rises to `0.870`;
capability remains `31/32`, but nearby foundations remain only `59/96`. The
10M policy rises sharply from its selected parent's `68/720`, `0.344` to
`406/720`, `0.745` on development, with `32/32` capability; nearby foundations
remain `49/96`. Both policies already clear the nearby-trench gate and fail the
nearby-foundation gate. Matched update-9,000 observers `9964699` (compact) and
`9964703` (10M) completed; the complete fixed-panel trajectories are summarized
above. Dense reward and Stage C remained locked in that historical campaign.

Stage B contains 15 conditions x 96 layouts = 1,440 distinct training maps.
Random full resets draw 25% from the two mastered capability anchors and 75%
from the 13 nearby conditions, with 50/50 foundation/trench mass within each
slice. There is no per-environment promotion or demotion. Both arms use the
independent full-V8 compact update-7,500 checkpoint as one frozen
current-rollout KL/value teacher; its development result is `538/720` exact,
`0.868` macro, and `31/32` on capability. The matched distillation treatment
fades over 3,000/1,000 updates.

The compact parent reached `516/720` exact and `0.840` macro on fixed
development; the 10M parent reached `68/720` and `0.344`. This is evidence that
the 10M policy is learning but has not inherited broad competence. Dense reward
therefore remains frozen. Reward fading may start only after a future dense
full-V8 checkpoint passes the documented qualification gate.

Primary evidence:

- `/home/lorenzo/moleworks/.artifacts/terra_v8_stagea_whole_eval_20260806/leaderboard/LEADERBOARD.md`
- `/home/lorenzo/moleworks/.artifacts/terra_v8_stagea_whole_eval_20260806/stage_b_selection.json`
- `/home/lorenzo/moleworks/.artifacts/terra_v8_reference_teacher_full_eval_20260806/`
- [`research/V8_10M_SCALEUP.md`](research/V8_10M_SCALEUP.md)

The immutable launch revision is
`f682f37d6a856c779b2c52e9e2d02a56cb04c15c`. Stage-B output is under
`/cluster/work/rsl/lterenzi/terra_v8_10m_nearby_long_v1/`; no historical
scratch data was deleted to resolve the unrelated scratch inode soft limit.

## Most recent production jobs (completed fixed-evaluation screens)

All five allocated P5c update-1 smokes completed `0:0`, wrote
`status=PASSED`, and passed the explicit finite-checkpoint verifier. The five
matched 4,000-update screens were then submitted from the exact smoke-tested
revision and completed on Euler. They took 6.2--8.0 hours, crossed all finite
and transition-integrity checks, and ended with online training success still
rising. They are early learning-curve screens, not convergence results.

| P5c arm | Support | Sampler | Architecture | Budget | Smoke | Screen | State |
|---|---|---|---|---:|---:|---:|---|
| `G-MEDIUM-ADAPTIVE-WARM` | all 32 | adaptive | medium | 4,000 updates | `9458568` PASS | `9461489` | COMPLETE; 6.27 h early screen |
| `G-MEDIUM-UNIFORM-WARM` | all 32 | uniform | medium | 4,000 updates | `9458581` PASS | `9461500` | COMPLETE; 6.22 h early screen |
| `G-DEEP-UNIFORM-WARM` | all 32 | uniform | deep | 4,000 updates | `9458585` PASS | `9461504` | COMPLETE; 7.98 h early screen |
| `F-MEDIUM-UNIFORM-WARM` | 18 foundations | uniform | medium | 4,000 updates | `9458616` PASS | `9461507` | COMPLETE; 6.32 h early screen |
| `T-MEDIUM-UNIFORM-WARM` | 14 trenches | uniform | medium | 4,000 updates | `9458619` PASS | `9461512` | COMPLETE; 6.21 h early screen |

All five use the same P5 parent/teacher and low entropy
`0.02 -> 0.005 / 10,000`. Fixed evaluations were run every 500 updates on
constrained promotion/development and the separate all-free
promotion/development diagnostics. The specialists are family dose ceilings;
the causal pairs are medium adaptive versus medium uniform, then medium
uniform versus deep uniform.

All 160 declared fixed evaluations are complete and integrity-clean. Latest
generalist endpoints are:

| Arm | Promotion macro | Development macro | Promotion exact | Development exact |
|---|---:|---:|---:|---:|
| `G-MEDIUM-ADAPTIVE-WARM` | 0.607 | 0.547 | 133/512 | 116/512 |
| `G-MEDIUM-UNIFORM-WARM` | 0.565 | 0.565 | 115/512 | 94/512 |
| `G-DEEP-UNIFORM-WARM` | 0.624 | 0.586 | 168/512 | 143/512 |

Deep uniform at update 4,000 is the strongest descriptive checkpoint, not a
selected checkpoint. The full condition leaderboard is
`/home/lorenzo/moleworks/.artifacts/terra_p5c_leaderboard_20260803_3478af8/LEADERBOARD.md`.
Every campaign checkpoint and receipt is preserved read-only at
`/cluster/work/rsl/lterenzi/terra_p5c_campaign_20260803_3478af87950d3d35059344b078209d00785c8481/`.

## Most recent completed campaign: P5b

| Arm | Slurm | Selected checkpoint | Promotion / development macro | Exact P / D | State |
|---|---:|---:|---:|---:|---|
| `G-MEDIUM-ADAPTIVE-WARM` | `9378174` | 2,000 | 0.652 / 0.625 | 1/512 / 2/512 | COMPLETED, PASSED |
| `G-DEEP-ADAPTIVE-WARM` | `9378175` | 1,000 | 0.653 / 0.628 | 2/512 / 2/512 | COMPLETED, PASSED |
| `G-MEDIUM-UNIFORM-WARM` | `9378176` | 1,000 | 0.647 / 0.664 | 2/512 / 6/512 | COMPLETED, PASSED |

All three jobs completed 2,000 updates; “selected checkpoint” is the retained
fixed-evaluation checkpoint, not the terminal job update. At matched update
1,000, deep/adaptive and medium/uniform pass their bounded treatment gates,
but neither retains the gain at update 2,000. P5b is complete, not running.

Result and leaderboard roots:

- `/home/lorenzo/moleworks/.artifacts/terra_p5b_results_20260802_6c56610e`
- `/home/lorenzo/moleworks/.artifacts/terra_p5b_leaderboard_20260802_6c56610e/LEADERBOARD.md`

## Capability-floor diagnostics

`fnd-slab-allfree` and `trn-straight-allfree` keep the source and excavation
geometry of accepted-bank parents but accept every legal non-dig cell for
dumping. They remain outside the constrained macro.

| Checkpoint | Promotion / development macro | Exact P / D | State |
|---|---:|---:|---|
| Historical E8 parameters, current-protocol transplant | 0.013 / 0.027 | 0/32 / 0/32 | COMPLETE; compatibility diagnostic only |
| P5 parent | 0.385 / 0.465 | 0/32 / 0/32 | COMPLETE |
| P5b medium/adaptive @2,000 | 0.629 / 0.613 | 0/32 / 0/32 | COMPLETE |
| P5b deep/adaptive @1,000 | 0.718 / 0.736 | 0/32 / 0/32 | COMPLETE |
| P5b medium/uniform @1,000 | 0.484 / 0.540 | 0/32 / 0/32 | COMPLETE |
| P5 foundation specialist @2,000 | foundation 0.668 / 0.640 | 0/16 / 0/16 | COMPLETE |
| P5 trench specialist @2,000 | trench 0.556 / 0.606 | 1/16 / 0/16 | COMPLETE |

The controls are physically permissive but target-mask OOD. E8's legacy
near-1 online `swhr` used a different protocol and is not a counterexample.
P5c evaluated them separately at every checkpoint; adding them to training
would require a separate, versioned 34-condition treatment.

Diagnostic roots:

- `/home/lorenzo/moleworks/.artifacts/terra_unconstrained_controls_20260802_0306c3cd`
- `/home/lorenzo/moleworks/.artifacts/terra_unconstrained_control_eval_20260802`

## Long-run boundary

Future behavioral screens receive at least one healthy 24-hour allocation.
Configure more updates than can fit in that allocation, checkpoint every
100--500 updates, and classify a wall-time exit with a valid checkpoint as
`CONTINUABLE`. Promising policies continue with true `--resume_from` state on
the 120-hour queue; they do not restart parameters, optimizer, schedules, or
adaptive sampler state. Stop only after fixed held-out exact, macro, and tail
metrics plateau across multiple checkpoints. The full contract is
[`research/P5_ACCEPTED_BANK_EXPERIMENTS.md`](research/P5_ACCEPTED_BANK_EXPERIMENTS.md)
section 12.
