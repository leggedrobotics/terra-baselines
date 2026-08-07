# Experiments — current state (updated 2026-08-07 CEST)

Current design and decision authority:

- [`research/V8_10M_SCALEUP.md`](research/V8_10M_SCALEUP.md), section
  "Accepted decision: continuous all-47 bands (2026-08-07)"

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

The accepted successor uses one uninterrupted random-start compact deep+xattn
training process per matched arm over all 47 V8 conditions. It has no
capability-only, nearby-only, or constraint-only training jobs. Immutable depth
0/1/2 labels describe map difficulty; live family-specific bands only
redistribute target-assignment probability inside the same run:

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
`0982b7803777fee81a227ce26f30bd85004a9aaa`; its launch revision is
`d0aea30e475ec97442a86fcd443f690976572fe3`. The matched pair will record its
new common terra-baselines and runtime Terra revisions after implementation,
tests, and update-1 smokes pass.

The next scientific launch is a same-binary reward pair. Both arms retain that
map curriculum and every non-reward setting:

| Arm | Reward schedule | Initialization | Target | State |
|---|---|---|---:|---|
| `constant_dense` | dense for the full run | random parameters, no teacher | 20,000 updates | implementation and common-binary smoke pending |
| `dense_to_terminal` | dense, then a one-way 5,000-update linear fade after both families reach active depth 2 | random parameters, no teacher | 20,000 updates | implementation and common-binary smoke pending |

Both arms use seed `20260807`, 47 x 96 maps, `continuous_banded_v1`, full
450-step resets, the 2,856,685-parameter compact deep+xattn policy, checkpoints
every 500 updates, and fixed-panel evaluation every 1,000 retained updates.
The 20,000-update pair uses the 120-hour queue because the previous compact
20,000-update run took about 25h53.

The original dense-only job is retained in the ledger until the replacement
binary proves it can run both reward modes:

| Prior arm | Update-1 smoke | Long job | Current state |
|---|---:|---:|---|
| `G-V8-XATTN-CONTINUOUS-BANDED` | `10012150` PASS | `10015084` | PENDING, held by user (`JobHeldUser`), zero runtime; supersede only after both new common-binary smokes pass |

Smoke `10012150` completed `0:0` in 13m55s. Both update-1 and final
checkpoints reloaded; the receipt records all 47 conditions, family counts
25/22, depth counts 2/13/32, positive minimum probability `0.002`, resumable
`terra_continuous_banded_sampler_state_v1`, and graph SHA-256
`f0ad2c9c138cbb7d7139ac8bf50cb8c9b897d06d49c625b82b78d0a9c3e42b2d`.
This is engineering admission evidence, not a learning result.

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
finite-checkpoint receipts. The paired 20,000-update Stage-B jobs are submitted
to the 120-hour queue:

| Arm | Parent | Smoke | Long job | Current state |
|---|---:|---:|---:|---|
| compact deep+xattn | update 1,000 | `9854547` PASS | `9858450` | `RUNNING` at approximately 18.0k/20k; 17 checkpoints |
| 10M deep+xattn | update 3,000 | `9854549` PASS | `9858451` | `RUNNING` at approximately 9.25k/20k; 9 checkpoints |

Both jobs passed their immediate smoke, parent, teacher, sampler, architecture,
CUDA, cuDNN, and NCCL checks inside Slurm. After approximately 20.5 hours,
throughput is about 17.9k transitions/s for compact and 8.8k for 10M. Compact
has roughly two training hours left; 10M roughly 22. Neither log contains a
NaN, OOM, NCCL, traceback, or quota failure. These are operational facts, not
held-out learning results.

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
`9964703` (10M) are pending to test whether the overnight online gains
generalized. Dense reward and Stage C remain locked.

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
