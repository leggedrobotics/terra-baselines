# Test-time compute for the corrected generalist

September 21, 2026. The user asked whether more sampled plans and short
geometry-specific PPO can improve completion and deployment efficiency while
the broad continuation runs.

## September 21: completed latest-policy geometry comparison

Euler14791590 completed in2h33m46s. The one-update finite/integrity smoke and
full-panel retention checks pass. Both arms use the original u109250 model.
Straight32 improves**2/32→26/32** under previous→corrected geometry, with
mean excavation93.32%→99.14%, accepted material90.63%→98.83%, and mean duration
447.6→181.0 actions including timeouts. There are26 gains and2 losses; both
old successful seeds621004/621006 now fail, so the improvement is not monotonic.

Full608 scores foundations383→382/384, trenches208→211/224, roads29→29/32.
There are four gained trench cases, one lost trench and one lost foundation.
Slot434, the original straight failure, is solved. Lost slab slot521 finishes
88.57% dug; lost two-section trench389 finishes44.12% dug with425steps of no
task progress. Six straight failures remain: seeds621000,621002,621004,621006,
621020,621028, with92.5–97.5% excavation; four still carry soil. Action tapes
support follow-up replay, but terminal aggregates do not establish feasibility.

Common-success foundation retained inter-setup distance30.342→30.362m and
workspace yield7.025→7.038m²; trench distance35.031→35.187m and yield2.832→2.829m².
The correction improves targeted completion without a general efficiency gain.
These are Euler within-runtime comparisons, not rankings against GH200 scores.
Keep the fix. CSCS4729577 retains its frozen prior geometry; no duplicate
continuation was submitted. See `.artifacts/terra_latest_geometry_20260921/`
for `results/`, `RESULTS_SUMMARY.json` and independent result review.

## Latest-checkpoint geometry comparison: Euler 14791590

Submitted September 21 at 15:55 UTC as `lterenzi`: one RTX3090, three-hour cap.
Initial scheduler state is PENDING/Priority, so GPU startup is unverified.
The fixed u109250 parent was the latest complete checkpoint at preparation.
A disposable one-update native resume smoke runs first; all old/new geometry
panels still evaluate the original weights. Compare straight32 and full608
sequentially on the same GPU with identical reset and inference layouts.
Actual cluster-Python configuration checks, staged-input checks and independent
review pass. The local cuDNN failure is retained as runtime evidence, not a
negative policy result.

No new CSCS allocation is submitted. Reserve a whole CSCS node for a later
qualified four-GPU continuation with zero added costs and no imitation.
The held-out straight witness remains outside generalist training. See
`.artifacts/terra_latest_geometry_20260921/` and the running-experiment ledger.
Recommended startup/evaluation check: 16:55 UTC; no monitor is scheduled.

## September 21, 15:25 UTC: bounded continuation improves; next step is fixed-runtime evaluation

Live CSCS check: 4725717 remains RUNNING on nid005700 (four GPUs); checkpoint
u108500 is present. The completed u105000 retention panel is PASS: foundations
381/384 (six gains, three losses versus u100000), trenches 211/224 (four gains,
two losses), road subset 31/32 (one gain, no loss). The parent scored
378/384, 209/224 and 30/32. This job uses the prior frozen geometry, including
float16 metadata and the earlier map-edge bounds, not today's local correction.
Let the existing bounded continuation reach its u110000 decision; no duplicate
continuation was submitted. Next useful check is its final evaluation or the
16:43 UTC allocation limit, whichever comes first; no monitor is scheduled.

Euler 14766038 completed at 13:29:52 UTC. Straight performed zero PPO updates;
T-junction has finite saved milestones through u50, and road-network reached
u250 (2,048,000 training transitions) with zero material/target/obstacle
violations. Road evaluation was explicitly deferred for insufficient remaining
budget. These are saved training outputs, not measured adapted-policy gains.

Sampling retry 14776814 FAILED at 13:14:47 UTC after 30m49s. Its actual 256-lane
sampling layout changed greedy actions relative to the reused Euler baseline:
`ValueError: Sample batch changes greedy actions; refusing to mix execution contracts`.
The failed-layout receipt remains in `results_14776814/layout_parity.json`;
no stochastic improvement is established. Diagnose the first divergence and
use a consistent inference and simulator batch layout before another comparison.

Recommended next work: a bounded GPU smoke and actual-policy evaluation under
the local geometry fix, first on the straight 32-start panel, then the full
608 cases including foundations. Reuse the already saved adaptation outputs
for evaluation before allocating more map-specific PPO. Regenerate any reused
demonstration observations from fresh original metadata. Keep this known-map
diagnostic out of generalist training. Resume a selected generalist only after
fixed-runtime retention is checked; add efficiency costs gradually after
completion stabilizes. This section records recommendations, not submissions.

## September 21, 14:32 UTC: geometry corrected; completion and exit verified locally

The local Terra development checkout now stores trench and foundation-border
geometry in float32, accepts continuous chassis corners exactly on either map
edge, and uses a 1e-5 m tolerance at closed trench distance limits. This removes
the 5.23 mm axis displacement and the asymmetric rejection of nominal 2 m
poses. Native footprint and swept-path collision checks still reject soil,
holes, obstacles and true overhang. The map itself was not shortened.

Fresh native metadata loading and the complete original initial Agent were
verified before replay. The recorded greedy sequence now completes **80/80
excavation and accepted disposal at action 46**, versus 78/80 previously. A
manual plan completes at **action 62**, then takes **seven legal physical
steps** to clear ground 9.73 m from the nearest target, unloaded, with all four
base movements available. A second full replay passes all material, target,
obstacle and finite-value checks.

The exit uses `step_no_reset` after native task completion and is recorded
separately. These are fixed-action/manual-plan diagnostics on one evaluation
geometry, not new learned-policy success rates or 32-start results. The map
and witness remain outside generalist training. Fresh policy evaluation and
normal GPU training qualification remain pending; existing frozen jobs and
snapshots retain their previous runtime.

The patch was qualified on CPU in isolated branch
`terra-geometry-clearance-20260921`, then applied to the active local Terra
checkout. Selected geometry/movement/metadata tests and independent review
pass; `VALIDATION.md` records the bounded split test runs and two repaired
legacy test assumptions. Old serialized float16 states and stored demo
observations require reconstruction from original metadata and action replay.

Evidence: workspace `.artifacts/terra_geometry_clearance_20260921/`, including
`geometry_fix.patch`, `qualified_plan.json`, `REPLAY_AND_EGRESS.md`,
`completion_egress.gif` and `same_actions_boundary_fix.gif`.

## September 21 status at 12:50 UTC

Sampling job **14765736** failed its cross-runtime greedy check after 43m26s.
The completed Euler and GH200 panels both solve 587/608, but 109 episodes
differ and four success outcomes flip. The selected diagnostic subset is
45/64 on Euler versus 43/64 on GH200; these gains are not sampling gains.
The cause of this runtime difference is not isolated.

Replacement **14776814** is RUNNING on eu-g4-023, with one RTX3090 and a
75-minute wall limit. It reuses the completed Euler baseline and preserves
the failed GH200 comparison. Its actual sampling layout must reproduce the
Euler greedy actions and outcomes before stochastic sampling. All three
sampling allocations together are bounded to 118m52s, within the original
two-hour budget. The new entrypoint is `search_euler.py`; the original runner
and failed outputs remain available. No sampled gain is verified yet.

Geometry-specific **14766038** remains within its four-hour allocation:

- Straight: zero PPO updates. The evaluation-first schedule consumed its
  budget on baseline inference. Greedy and best-of-12 both solve 0/32; mean
  excavation improves from 93.5938% to 97.7344%. This is a sampling result.
- T-junction: baseline greedy and best-of-16 both solve 32/32. Sampling reduces
  retained travel by 1.54%, with little change in workspace yield. Training
  has now saved finite updates 1, 2, 10 and 50 with clean material checks;
  adapted-policy evaluation is not yet available.
- Road network: before this case started, the future process entrypoint was
  changed to train/save first, with optional parent/latest greedy evaluation
  only if time remains and no N16. The active T-junction process is unchanged.
  Physics, optimization settings and the allocation budget are unchanged.

The original baseline-first ordering was poorly budgeted. The revised road
runner and local-baseline sampling runner passed focused checks and independent
review. Receipts are under `status_20260921_1228/`,
`launch/euler_local_submission.json` and `adaptation/train_first_handoff.json`
in the workspace artifact bundle. A recorded-action CPU replay is inspecting
the straight-trench endpoint failure; no geometry change has been made.

### Straight-trench recorded-action inspection

Exact frozen-physics CPU replay reproduces two failures, without policy
inference or terrain changes. Both use the 80-cell target at rows 31–32,
columns 12–51, with 0.5714 m cells and a 3.6429 m inner working radius.

| Replay | Final base (row, column) | Excavated and accepted | Remaining cells |
| --- | --- | --- | --- |
| Greedy seed 621000 | (32, 57), heading 0 | 78/80 | (31, 51), (32, 51) |
| Selected sampled seed 621006 | (34, 57), heading 0 | 79/80 | (32, 51) |

The greedy last fresh dig occurs at action 43 and disposal at 46; the next
404 actions make no task progress. The sampled last fresh dig occurs at 62
and disposal at 64; subsequent relifts do not advance the task. In both final
states, all four base-motion effects and all twelve fresh-dig counts are zero.

Actual native footprint checks isolate the immediate blockers:

- Greedy backward by one cell intersects only deposited soil at (35, 51).
  Longer retreats also intersect excavated trench cells.
- Sampled backward by one cell intersects an already excavated cell (31, 51)
  and spoil at (37, 51).
- Both forward moves and base turns exceed the current corner bounds. A
  one-cell forward move has no soil/hole collision but ends a chassis corner
  at coordinate 64; the code requires corners strictly below 64. This is a
  boundary-contract issue to review against its cell-center footprint model.
- The remaining cells are too close for the arm: greedy distances 3.429 and
  3.476 m; sampled distance 3.614 m, only 2.9 cm inside the inner radius.

**An actual action sequence completes this geometry under the unchanged
frozen physics.** Keep sampled seed 621006's first 38 recorded actions, then
apply 16 manual actions: align at row 28, column 55; dig at cabin 7; dump at
cabin 10 away from the next fresh-work cone; return to cabin 7, advance to
column 57, dig the last three cells, and dump at cabin 10. Native completion
at action 54 reports 80/80 excavated, 80/80 accepted and zero load. The full
54-action plan was replayed again from the original initial state and passed
all transition material, target and obstacle checks. An earlier attempted
suffix dumped at cabin 8, contaminating the next dig cone and causing relift
instead of fresh excavation. Disposal direction is part of the failure.

This witness establishes simulator task feasibility, not a deployable exit:
all four base-motion effects remain false after its final dump. Future clean
demonstrations should explicitly test egress if deployment requires it.
Immediate immobility is not itself proof of irreversible trapping: another
native trial relifts the greedy episode's blocking pile, dumps elsewhere and
retreats one cell, although that suffix does not finish the remaining work.

**A separate geometry-precision defect is confirmed.** `MapsBuffer.new`
stores trench axes in float16 (`terra/maps_buffer.py:131`). This map's JSON
axis `[0, -39.282, 1237.383]` becomes `[0, -39.28125, 1237.0]`, shifting the
line by about 5.23 mm. The nominally symmetric 2 m offset poses therefore
measure 1.994772 m at row 28 and 2.005228 m at row 35. Native admission allows
the former and rejects the latter. This is coefficient quantization, not a
float32 comparison tolerance error. It makes the viable endpoint approach
asymmetric. A follow-up precision correction should preserve geometry in
float32 and check boundary tolerance and mirrored admission separately.

Across all 32 selected best-of-12 trajectories, 7 leave one cell, 24 leave
two and 1 leaves three. The two inspected rollouts explain an endpoint trap;
they do not establish that every failed start has the same detailed cause.
No broad success rate or learned-policy gain follows from the single manual
witness. It remains a diagnostic on an evaluation geometry and is excluded
from the generalist demonstration bank.

Replay states, action trace, blocker cells, `clean_prefix.json`, successful
full-plan replay, contact sheets and GIFs are under
`.artifacts/terra_test_time_compute_20260921/straight_failure_inspection/`.

## Evidence and decision

The recovered September 13 campaign supports a bounded new comparison:

| Old development geometry | Frozen search | Short adaptation plus search |
| --- | --- | --- |
| Straight trench | Greedy 30/32; best-of-16 32/32 | u250 plus N16: 32/32, 6 productive poses |
| T-junction | 32/32 throughout; productive poses 9.44 greedy, 9.22 at N64, 9.09 at N1024 | u250 plus N16: 32/32, 8 productive poses |
| Road network | 0/32 through available N256 panels | 0/32 through u21000, about 45 training hours per seed |

All 18 old search cells have validated saved panels, but only eight reached
their requested maximum. No unsaved milestone is inferred. Old N16/N64 cold
panel time was approximately 20–24/22–32 minutes on one RTX3090 for 32 starts;
exact u250 adaptation time is unavailable. These timings are not per-plan
deployment latency. Old source `46738cde` and the u85500 specialist differ from
the corrected time-aware u100000 generalist. The road result is not evidence
of infeasibility under current physics.

Start with nested sampling of the current policy. Use the result to identify
where brief adaptation is useful rather than repeating long map-PPO runs.

## Submitted screen

Euler **14763650** passed its RTX3090/CUDA preflight but failed after 26 seconds:
the staged bank preserved local symlinks whose targets did not exist on Euler.
The map payload was replaced with real files and the full 608-map exact dataset
contract passed under the cluster Python. Replacement **14765736** requested one
RTX3090 and 1h59m, retaining the original total allocation budget. It completed
the greedy rollout on eu-g4-016, then failed parity as recorded above.
Local synthetic integrity/selection checks and independent code review pass.

Use the same corrected frozen source as the u100000 reference: 378/384
foundations, 209/224 trenches and 30/32 road cases. Recreate the complete
608-case native reset, then keep all 21 failures and 43 successful controls
with fixed identities. The panel contains 33 foundation and 31 trench cases
across 38 conditions. This deliberately enriched panel is not a population
efficiency estimate.

Keep greedy in the candidate pool. Evaluate nested N4/N16/N64 stochastic
candidates per initial state, then optionally N256 for still-failing cases
inside the same budget. N excludes greedy. Two selectors rank exact success
first, followed by retained work-pose travel or retained work-pose count.
Report productive poses, workspace yield, continuity, lateral digging,
relifts and raw Terra travel separately. Retained travel includes initial
approach and is a straight-line lower bound on the navigation-stack route.

The runner first checks the full greedy reference and then greedy action
parity under the actual sampling layout. A mismatch stops the run and saves
diagnostics. Sampling preserves complete candidate actions and validates
material integrity. Report cold/warm time, actual per-start sample counts,
active/executed transitions and completed prefixes. The process budget is
6000 seconds, checked between full batches; Slurm supplies the hard two-hour
limit. No policy updates, physics changes or efficiency penalties are added.

## Follow-up criterion

Increase N selectively if N16 to N64 continues to rescue cases or reduce
retained work. Where sampling saturates, compare a copy of the current parent
adapted for 2/10/50/250 updates plus N16 against frozen sampling at equal actual
wall time. Keep the original greedy plan available for final selection.
Freeze selectors before testing fresh confirmation starts. Adaptation to an
evaluation geometry is known-map planning, not held-out policy generalization;
its trajectories and weights stay out of the generalist training bank.

## Geometry-specific PPO comparison

Euler **14766038** submitted September 21 as `lterenzi`: one RTX3090, four-hour
wall limit. CPU tests, actual cluster-Python config validation and independent
review pass. The initial live check showed RUNNING on eu-g4-011 during CUDA
preflight. Later verified updates and the road scheduling revision are above.

The user subsequently requested submitting this comparison without waiting
for sampling results. Three independent warm starts use current failure
geometries: straight trench slot 434, T-junction slot 495 and road-constrained
four-section network slot 346. Their canonical arrays and finite trench
metadata are copied unchanged. None of these sources occurs in the recorded
generalist training manifest. Adaptation remains excluded from generalist
training and evaluation claims.

Each case starts from the same u100000 model with fresh Adam, preserving the
current time observation and expanded actor. Clear retired teacher schedules,
demonstration settings and added behavior costs. Use 256 environments on one
GPU, rollout length 32, two epochs and 32 minibatches: 8192 transitions and
64 Adam steps per update. Preserve the parent's learning rate and constant
0.02 entropy coefficient. Save and verify update 1 before proceeding, then
updates 2, 10, 50 and 250. A completed 250-update case uses 2.048M transitions.

Parent and adapted policies share 32 fresh fixed starts on each geometry.
Measure greedy at the early milestones and best-of-16 for the parent and final
adapted policy. These are matched same-geometry starts, not replay of the
original full-608 failure's complete initial Agent state. Thus success here
does not by itself prove that the original failed episode was rescued.

The allocation is capped at four GPU-hours, with three sequential cases and
70-minute process budgets. Unfinished cases retain their last verified
checkpoint; no automatic continuation follows. Compilation, baseline search,
training and evaluation all count toward the case budget. The independent
corrected generalist continuation, CSCS 4725717, is not duplicated.

## Artifacts

Workspace-local bundle: `.artifacts/terra_test_time_compute_20260921/`.
It contains `search.py`, `panel.json`, CPU check and independent review
receipts, the recovered old results and `launch/submission.json`.

Euler outputs:
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_test_time_compute_20260921/results_14765736/`.
The failed first attempt is retained in `results_14763650/`. Adaptation scripts,
maps, launch files and outputs are isolated under the same root's `adaptation/`.
The source pair is the existing frozen
`codex_terra_edge_validation/terra_trench_recovery_20260920/{terra,terra-baselines}`;
the parent is a staged copy of the native u100000 checkpoint.
