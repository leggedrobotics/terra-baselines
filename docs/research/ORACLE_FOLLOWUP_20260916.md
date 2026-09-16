# Completion-first follow-up to the September 16 research review

## Decision and scope

Implement the foundation-specific teacher release first, alongside a bounded
failure replay. Keep corrected physics, reward-v2, full broad resets and zero
added behavior costs. The current broad student improves but is not qualified
for efficiency training: u5000 completes 198/384 foundations and 188/224
trenches, including 23/32 road cases. Foundation completion gained 65 cases and
lost 40 since u2500. These are completed fixed evaluations, not new pilot results.

The review supplied by Lorenzo supports this priority. Persistent foundation
guidance is a hypothesis for restricted adaptation, not an established cause.
The current-runtime foundation teacher completes only 140/384 while the student
completes 198/384, but that comparison alone does not prove that its guidance is
harmful on student-visited states.

This bounded teacher-release control answers a new question. It does not repeat
the early-penalty comparison. Do not allocate all proposed experiments at once.
Recover existing delayed-cost outputs and the latest broad checkpoint when
CSCS storage returns before choosing further unchanged training. The earlier
50,000-update continuation plan remains a conditional review budget, not a
promise of completion or an automatic second allocation.

## Implemented teacher-release experiment

The direct recipe is [foundation_teacher_release](../../scripts/foundation_teacher_release/README.md).
Both arms start from the locally preserved native generalist u5000 checkpoint,
with Adam step 320,000. The environment is the existing frozen `46738cde`
runtime; this change modifies only baselines. The original generalist baselines
snapshot was `96fcd811`; this branch starts from its documented successor
`6702fa3`.

| Property | Control | Foundation release |
| --- | --- | --- |
| Foundation guidance | Original cosine to u20000 | Linear fade from 0.85355339 at u5000 to zero at u6250 |
| Trench guidance | Original cosine to u20000 | Same original cosine |
| Model and observations | Parent configuration | Same |
| PPO and reward | Parent configuration, zero added costs | Same |
| Training layout | 4 GPUs × 256 environments × 32 steps | Same |
| First decision | u6250, 40.96M new transitions | Same |
| Conditional hold | u7500, 81.92M total new transitions | Same |

Each update contains 32,768 global transitions and 64 Adam steps. Preserve the
native optimizer and absolute clock. Resumes restart live environments, RNG
and action history, as in the existing trainer; both arms use the same restart
procedure. They are not bit-exact continuation of the original trajectory.

The opt-in `foundation_teacher_release_updates=1250` starts the release once.
The checkpoint stores its origin, initial coefficient and duration. Ordinary
resume restores that state, including after the coefficient reaches zero.
Changing the requested duration on continuation is rejected. The default
no-release loss retains the original scalar-KL computation. The treatment
weights pre-action foundation and trench rows before taking the global mean;
it does not renormalize families or change task sampling.

Evaluate all 608 development episodes at both milestones with the original
greedy450 contract. Report 384 foundations, 224 trenches, their 32-road subset,
all **38 evaluation conditions** (24 foundation and 14 trench), paired gained
and lost successes, material metrics and common-success efficiency. The training
bank has 40 conditions; these counts must not be interchanged.

Proceed through the bounded hold only when foundation completion is at least
control, at most two control trench successes and one road success are lost,
foundation excavation/disposal each lose no more than 0.5 percentage points,
and no condition loses more than two net successes. Gains elsewhere cannot
hide lost trench or road cases. Eight additional foundation completions with
these checks passing is a useful engineering signal, not statistical proof.
No continuation past u7500 or efficiency promotion is automatic.

One six-hour CSCS allocation runs the arms sequentially using all four GH200s.
It performs CUDA convolution/NCCL checks, bounded training and full evaluations.
Failed or incomplete evaluation stops the script. Regular native checkpoints
remain recoverable if a segment times out. The account is `lterenzi`.

## Failure and observation audit

[terra_failure_audit.py](../../scripts/analysis/terra_failure_audit.py) selects
12 u5000 failures covering foundations, road no-work/stalls, junctions and final
cleanup. It retains original manifest slots and reset seeds, complete initial
Agent states, full replay traces and snapshots after the latest material change
at or before action 400. This is purposeful case coverage, not a frequency
estimate over the bank.

Replay is 5,400 environment actions. A separate width-eight search considers
at most 12 successor actions per case using actual Terra transitions, within
the original remaining horizon. Report concrete successful suffixes separately
from partial material progress and work/disposal/translation witnesses. Failed
bounded search is not evidence of infeasibility; one legal escape does not prove
future access to every target. A replay that does not reproduce the original
failure is reported separately and is not attributed to that fixed result.
The comparison establishes matching failure/excavation/disposal endpoints,
not identical original action trajectories; the prior report lacks raw traces.

The CPU observation diagnostic already reproduces time aliasing with the actual
Terra observation function and u5000 preprocessing configuration: identical
physical state and five-action history at ages 50, 440, 449 and 450 produce
identical policy inputs; only age 450 terminates. This used a synthetic four-cell
excavation/disposal state and no policy inference. It establishes the missing
information, not its measured effect on learning.

### Completed 12-case replay and qualitative review

All 12 failures reproduce their original failure/excavation/disposal endpoints.
The completed audit uses 5,400 replay actions and 8,280 real successor candidates.
It finds additional legal material progress in **8/12** cases and sequences of
fresh work, accepted disposal and a subsequent base translation in **5/12**.
It finds **zero complete-success suffixes** within width 8/depth 12. These are
selected-case witnesses, not a feasibility rate for the broad bank.

The independent excavation-behavior review identifies:

- Movement/orientation cycles, including both blocked actions and successful
  movement that returns to the same poses. None of these 12 rollouts uses WAIT
  during its final 100 actions. Historical WAIT-loop diagnoses do not describe
  these traces.
- Stationary relift/dump cycles in road slots 343/296 with no improvement in
  excavation or accepted disposal. Their nearest declared disposal cells at the
  terminal pose are 9.37 m/8.93 m away, beyond the 6.5m direct dump reach. Foundation
  cleanup 63 has the same direct-reach issue at 7.96 m. Cabin turns alone cannot
  provide a direct accepted drop there; longer staging/relocation feasibility
  remains unresolved.
- Tee 503's sole remaining cell is at the far tip of its second section, not
  at the junction. This case does not support a junction-cell admission bug.
- Foundation slots 4/564 had executable fresh work and legal work/disposal/move
  alternatives at their saved roots. Useful opportunities remained when the
  policy stopped advancing the task.

Material changes alone are not progress: the two soil-cycling cases can look
active while excavation and accepted disposal remain flat. Keep task-progress
stalls separate from material-change stalls. This evidence supports testing
teacher release before stronger initial movement costs; it does not establish
that the teacher caused the loops.

All 12 real-state time-aliasing checks also pass. Search roots precede terminal
poses and are at action 0 for three no-work cases. An alternative from those
roots does not prove escape from the later final pose. No complete geometric
contradiction or motion-rule violation was established. Path figures overlay
terminal terrain, so crossing a drawn hole does not show travel over an already
excavated cell. Evidence: `failure_audit/summary.json`, `qualitative_review.md`,
`behavior_counts.json`, per-case states/traces and `failure_endpoints_disposal.png`.

## Separate next changes

1. Remaining-time input remains a separate experiment after this release
   screen. Use normalized time for actor and critic, a zero-initialized
   function-preserving projection and a matched constant-input control. Verify
   initial logits, values and optimizer-slot migration before training. No
   recurrence, pile-height, history or auxiliary-loss changes belong in it.
2. Recover the existing delayed-cost pair, reproduce its u20000 parent and
   evaluate any saved u22500/u25000 outputs before paying for more training.
   Preserve its original ramp and 63/64 requirement at both milestones.
3. Broad efficiency entry is separate from teacher release. The review proposes
   repeated 365/384 foundation, 213/224 trench and 31/32 road completion, at least
   80% per condition, and a teacher-free retention hold. These are engineering
   targets, not current achievements. Keep failures visible while feasibility
   questions are unresolved.
4. A later efficiency objective should charge retained work setups and transfers,
   including dump/relift poses. Current retained straight-line distance is a
   lower bound; validate route ranking against deployment navigation before
   claiming route efficiency. Do not substitute raw Terra travel or force large
   individual fresh areas during final cleanup.
5. Teacher-output caching, teacher-free rollout specialization and bookkeeping
   optimizations remain separate performance changes with numerical/transition
   checks. Model width, PPO layout and task sampling stay fixed in this screen.

## Validation and execution record

Evidence is under `.artifacts/terra_oracle_followup_20260916/` in the multi-repo
workspace. Focused tests verify unchanged control gradients, family weighting,
four-device global reductions, native release clocks, checkpoint/evaluation
metadata and observation identity. The independent implementation review found
no remaining blocking issue in the release and comparison logic.

The first local native CUDA smoke reached its 300-second bound during initial
compilation, before a PPO update; it is not a passing training check. The second
attempt, bounded to 600 seconds per phase, **passes all three phases** on one
RTX4090 with 128 environments: control and treatment each reach u5002 with
Adam 320128, then the treatment resumes to u5003/Adam 320192. Model, optimizer and
optimization/teacher diagnostics remain finite. Existing initialization records
confirm identical starting model, Adam, environment, history and RNG in both
arms. The release origin stays at u5000 through resume. The treatment's last
foundation/trench coefficients are 0.852188/0.853443, confirming separate clocks.
Evidence: `release/attempt2/verification.json`. These are diagnostic batches;
production rejects their checkpoints as parents and retains 4x256 throughout.

The 12-case GPU replay/search completed normally within its 900-second limit;
the timed replay/search portion was 571 seconds. Its bounded conclusions are
recorded above. In-allocation four-GH200 execution remains outstanding.

At the last live refresh (September 16, 12:59 CEST), SSH as `lterenzi` works but
`/capstor` is unavailable during the 07:00–19:00 maintenance reservation. Neither
Terra job is running or queued. The prior generalist 4672272 timed out; delayed
pair 4675576 failed for a still-uninspected reason. No new job has been submitted.
The final readiness snapshot is `cluster_readiness.json`.
The next useful cluster check is when storage returns; no polling worker is
scheduled. In-allocation four-GH200 execution remains outstanding.
