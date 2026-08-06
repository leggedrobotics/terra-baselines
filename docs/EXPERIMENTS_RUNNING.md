# Experiments — current state (updated 2026-08-06 CEST)

Canonical design and decision authority:

- [`research/V8_DEEP_XATTN_CURRICULUM.md`](research/V8_DEEP_XATTN_CURRICULUM.md)
- [`research/P5_FOLLOWUP_GOAL.md`](research/P5_FOLLOWUP_GOAL.md)
- [`research/P5_RESULTS_ANALYSIS.md`](research/P5_RESULTS_ANALYSIS.md)
- [`research/P5_ACCEPTED_BANK_EXPERIMENTS.md`](research/P5_ACCEPTED_BANK_EXPERIMENTS.md)

The older E1--E10 spatial-encoder run history is retained in
[`EXPERIMENTS_SPATIAL_V3_RUNS.md`](EXPERIMENTS_SPATIAL_V3_RUNS.md). It is not
current scheduler state.

## Current scheduler decision

The corrected whole-V8 Stage-A evaluation job `9839960` and reference-teacher
evaluation job `9845019` both completed cleanly. Teacher-bound update-1 smokes
`9854547` (compact) and `9854549` (10M) then completed `0:0` with passing
finite-checkpoint receipts. The paired 20,000-update Stage-B jobs are submitted
to the 120-hour queue:

| Arm | Parent | Smoke | Long job | Current state at submission |
|---|---:|---:|---:|---|
| compact deep+xattn | update 1,000 | `9854547` PASS | `9858450` | `PENDING (Priority)` |
| 10M deep+xattn | update 3,000 | `9854549` PASS | `9858451` | `PENDING (Priority)` |

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
