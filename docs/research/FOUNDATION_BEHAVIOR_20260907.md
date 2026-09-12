# Foundation behavior: base placement, lateral digging, and efficiency

September 7, 2026. Implemented in the paired `foundation-behavior` worktrees at
`/home/lorenzo/moleworks/.worktrees/terra_foundation_behavior_20260907/`, based on
Terra `46b140f8` and terra-baselines `da904ff4`. This contains the benchmark
measurements plus optional reward and observation changes. Defaults preserve the
previous behavior. The running cluster jobs have not been migrated to this treatment.

## What the current objective misses

The reported problems are digging sideways relative to the chassis and choosing
base poses with too little useful work, leading to unnecessary relocation.

The active reward is reward-v2, timing variant 0:

```text
r = 6 * success - failure - 1/450 + 0.9984 * Phi(next) - Phi(current)
```

The material potential rewards excavation and reduces remaining soil transport
work. Material transport distance is not the machine's travelled distance. A move,
base turn, cabin turn, and idle action with the same material state have the same
immediate reward. At the ordinary empty reset this is approximately -0.00822 per
action. Legacy dense movement penalties and terminal workspace efficiency are
bypassed in this mode. Increasing their weights would have no effect on this run.

Terra's arm direction is chassis yaw plus relative cabin yaw. The active geometry
has 30-degree heading increments and a workspace sector of +/-30 degrees, with
radial reach approximately 3.64–6.5 m from the base centre. This reach is identical
sideways and along the chassis. The nominal translation primitive moves five
tiles, approximately 2.86 m cardinally. It is relatively coarse for adjusting a
foundation work pose.

The environment excludes the base footprint from digging and checks candidate
movement/turn footprints against terrain and obstacles. It does not model contact
forces, overturning moments, centre of mass, soil bearing, or an excavation-edge
support margin. Cabin yaw alone can therefore express a digging preference, not
establish physical stability.

The relevant Terra functions are `_get_cabin_angle_rad`, `_build_dig_dump_cone`,
`_get_reward_v2`, `_reward_v2_state_values`, and `_reward_v2_behavior_costs`.

## Implemented reward treatment

Exact completion, material progress, the existing horizon, and junction admission
remain unchanged. Optional costs inside the active reward-v2 path are:

```text
r_new = r_R2
        - c_travel * executed_base_distance_m
        - c_turn   * abs(wrapped_base_yaw_change_rad)
        - c_side   * (fresh_excavated_volume / required_volume)
                   * sin(relative_cabin_yaw_before_dig)^2
```

The lateral term is zero along either longitudinal chassis direction, 0.25 of
its maximum at 30 degrees, 0.75 at 60 degrees, and maximal at 90 degrees. It applies
to actual new excavation towards the target by the excavator. Dumping, picking up
loose/already excavated soil, unsuccessful DO actions, and cabin swings incur no
lateral cost, regardless of cabin orientation.
Volume weighting avoids making the charge depend on how the policy splits the
same excavation into many small DO actions. Fore/aft symmetry is an initial
geometric assumption; a machine-specific envelope can replace it later.

Travel and base yaw costs use the executed excavator transition, including curved
movement: base displacement in metres and wrapped heading change in radians.
Failed motion has zero physical travel/turn cost; the existing step cost still
applies. The soft lateral preference leaves constrained corner/endgame poses
feasible. A hard minimum
workspace size can strand finishing cells; a hard sideways ban can require extra
base movements. Neither is needed for the first treatment.

Calibrate coefficients from successful foundation rollouts under the new metric
contract. One initial scale to test is a total lateral cost of 0.25 for a task
excavated entirely sideways, and travel/turn coefficients whose costs on a typical
successful episode are each around 0.1–0.3, compared with the completion bonus 6.
These are proposed starting budgets, not validated constants. The bounded smoke
uses `c_side=0.25`, `c_travel=0.005/m`, and `c_turn=0.02/rad`. Costs and executed
motion/fresh volume have six separate rollout diagnostic fields and W&B scalars;
episode reward reconstruction includes the new costs. Inspect completion
regressions and idle episodes before selecting production coefficients.

If many low-yield stances remain after pricing distance and turns, add a fixed
setup cost once per work-to-relocation transition, with continuous travel charged
by distance. Cabin swing at a stationary base should not trigger another setup.
This requires a precise state variable and tests; charging every movement action
as another setup would conflate setup cost with travel granularity.

These additions intentionally change the objective. Any optional *learning-only*
pose guidance should use a potential difference `gamma * F(next) - F(current)`,
with correct terminal treatment, instead of repeatedly paying a positive bonus
for being at an attractive pose. Potential-based shaping is the established way
to preserve the optimal policy of the chosen objective under its assumptions;
the new physical costs themselves are not policy-invariant shaping.
[Ng, Harada and Russell, 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

## Implemented observation and checkpoint handling

`--executable_dig_observation` changes the existing 12-direction
`local_map_admissible_dig` vector to the fresh target volume an empty excavator's
DO can actually remove at each relative cabin direction. It requires
`--admissible_dig_observation`. Parameter shapes, initialization, and policy input
width remain unchanged; the new semantics are recorded explicitly.

The observation and actual DO share `_dig_eligibility`: cleaned cone, base
footprint, previous-dig exclusion, foundation edge alignment, per-cell trench
admission, positive-pile priority, static-obstacle overlap, and capacity. A loaded
excavator or a cone whose DO would pick up loose soil reports zero fresh volume.
This describes executable digging at the current base pose, not the quality of
unvisited candidate base poses. The batch constructor selects the observation
statically so default runs do not execute the expensive new branch under `vmap`.

Train with any combination of:

```text
--lateral_dig_cost 0.25 --base_travel_cost 0.005 --base_turn_cost 0.02
--executable_dig_observation
```

Costs default to zero and the observation defaults off. Positive costs require
reward-v2. The four values are saved in train/environment configuration and, when
enabled, in the existing R2 protocol receipt. Old receipts retain their original
form when all four settings are disabled.

To change an existing R2 checkpoint to this treatment, supply
`--resume_from <parent> --finetune_foundation_behavior` plus the desired settings.
This explicitly permits only these four changes while preserving model weights,
Adam state, and the update clock; architecture, distance, shaping, timing, and
other receipt checks remain strict. Subsequent ordinary continuations must use
the saved settings. Fixed-bank evaluation, MCTS evaluation, and playback loaders
recover these values, reject conflicting metadata, and forward the static
observation selector. Old checkpoints retain legacy semantics.

## Later environment changes

1. **Represent productive candidate base poses.** A later feature can describe
   useful reachable target volume at nearby feasible base poses, after chassis
   preference and legal dump access. Compute unions over admissible cabin sectors
   so overlapping sectors do not double-count the same cell. Use the real
   footprint/collision rules. Start with a bounded local stencil and measure its
   JAX cost before considering a map-wide position-by-heading search.
2. **Refine placement only if the coarse primitive is limiting it.** A smaller
   local adjustment action near work poses, alongside the current travel action,
   could avoid oscillating between poses roughly 2.86 m apart. This changes the
   action space and checkpoint head; it is a separate architecture treatment.
   The current 64x64 grid already has approximately 0.57 m cells and the ordinary
   move advances five cells. A one-cell adjustment would not require more grid
   cells, though extra decisions and longer episodes can make learning harder.
   Neither grid resolution nor the movement primitive changes in this patch.
3. **Calibrate physical reach and support constraints.** Compare the nominal
   annulus and usable angular sectors with the downstream machine workspace.
   Add an excavation-edge clearance/support proxy only with calibrated geometry
   and corner/endgame reachability checks. Enlarging reach merely to reduce the
   benchmark workspace count would change the planning problem.

The junction fix remains unchanged: a trench cell can be admitted through any
owning aligned section. Chassis preference is an additional excavation cost, not
a replacement for the world-frame trench-axis rule or foundation edge alignment.

An elapsed-time observation is not required to charge executed travel or action
duration. To optimize real completion time later, calibrate durations for travel,
base setup, swing, and excavation volume. A semi-Markov treatment also needs
discounting consistent with those durations; merely exposing elapsed steps does
not distinguish costly base relocation from a useful cabin swing.

## Implemented benchmark measurements

`utils/behavior_metrics.py` measures raw state transitions in `eval_mcts.py`,
which also backs `eval_mixed.py` and the fixed-bank evaluator. The fixed-bank
path preserves the first episode's terminal state and exports per-map measurements
through JSON, the benchmark dashboard, and CSV.

| Field | Definition |
| --- | --- |
| `base_travel_m` | Sum of executed base displacement for active machines, using each map's tile scale. Includes the first and final actions. |
| `base_travel_per_sqrt_target_area` | Travel divided by square root of required target area; a scale-normalized travel measure, not shortest-path optimality. |
| `base_heading_change_deg` | Sum of absolute wrapped base yaw changes. |
| `cabin_swing_deg` | Sum of relative cabin angle changes, including useful dump swings. |
| `base_reposition_count` | Transitions changing base position and/or heading; failed moves do not count. |
| `productive_dig_actions` | Actions making new excavation progress on target cells. Relifts and repeat excavation of previously credited soil do not count. |
| `productive_base_stances` | Contiguous fixed-position-and-heading stances with new target excavation. Cabin swings do not split a stance; leaving and returning does. |
| `unique_productive_base_poses` | Productive agent/position/heading combinations, with revisits counted once. |
| `newly_dug_area_m2` | Target cell area first excavated during the evaluated episode. Pre-existing partial excavation is excluded. |
| `mean_dig_area_m2` | Newly dug area per productive dig action. |
| `mean_workspace_dig_area_m2` | Mean newly dug area per productive base stance; larger means more useful area per setup. |
| `p10_workspace_dig_area_m2` | Tenth percentile of productive stance areas; exposes small-yield placements. |
| `dig_area_per_travel_m` | Newly dug target area divided by actual travel. |
| `lateral_dig_volume_fraction` | Fraction of new excavator target volume dug more than 45 degrees from the nearest chassis longitudinal axis. |
| `mean_dig_lateral_score` | New-excavation-volume-weighted mean of `sin(relative cabin yaw)^2`. |

The old `productive_workspace_cycles` remains separately available. It counts
empty-to-loaded transitions, including relifts, and is not a distinct stance
count. Likewise, the historical `workspaces_efficiency` field is an assumed-area
cycle proxy; use the new fields for base placement comparisons.

The accumulator maintains each target cell's greatest depth progress since reset
to avoid re-crediting refill/redig loops. Further excavation within a cell already
partly excavated at reset can contribute volume and productive actions but no
new footprint area. Compare area metrics within matched depth/reset strata.
The workspace area describes work achieved, not the fraction of all geometrically
reachable work that was exhausted; the latter needs the candidate-workspace
calculation described above.

No-work and zero-denominator ratios are unavailable, not perfect efficiency.
Old records without these measurements remain unavailable. Auto-reset terminal
rows are unavailable because the true final pose has been replaced; their reset
teleport is never counted as travel. Stable raw agent slots prevent a switch of
acting machine from looking like a large movement across the map.

Report overall, foundation/trench, and condition summaries with all-episode and
successful-episode cohorts separate. For policy comparisons, compare travel,
workspace area, and lateral digging on the **same maps solved by both policies**,
alongside all-map success and progress. Low travel on a failed or idle episode is
not an improvement. Do not rank a policy by a weighted efficiency scalar that can
hide completion regressions.

## Validation and next comparison

Focused tests cover endpoint accounting, changing active agents, heterogeneous
map scales, wrapped angles, cabin swings, stance revisits, partial starts,
refill/redigging, volume weighting, zero denominators, and auto-reset exclusion.
A scripted real Terra rollout checks a forward dig, dump swing, and a final
base movement through the evaluator. Reporting tests cover unavailable historical
metrics and comparisons restricted to shared successful maps.

Validation passed: 46 focused tests plus four subtests, rendered dashboard
JavaScript syntax, and the seven-transition real Terra probe. The probe measured
2.857143 m travel, 90 degrees of cabin swing, one productive stance, and
1.306123 m² of newly excavated target area. Logs and the probe are in
`.artifacts/terra_foundation_behavior_20260907/` under the Moleworks workspace.
Independent review found no material correctness issues. The NumPy accumulator
copies terrain maps to the CPU during evaluation; its full-panel overhead has
not been benchmarked. It does not run in the PPO training path.

The implemented environment treatment passes 23 new focused tests, including
actual DO versus all 12 observation headings, obstacle/pile/last-dig/footprint/
capacity/depth cases, foundation edges, both junction approaches, legacy DO
eligibility parity, bitwise zero-cost reward parity, dump/loose-pickup exemption,
and reward reconstruction. Existing R2/admissible tests and adjacent trench/dump/
material/checkpoint tests also pass: 61 Terra tests plus 28 subtests in total.
Loader tests pass 49 tests plus 26 subtests. Native
resume/receipt/reward/logging tests pass 25 tests. Their logs are in the same
artifact directory; these counts describe separate suites and may overlap with
the older benchmark checks above.

The local one-GPU CUDA preflight and u1000→1002 native fine-tuning smoke passed.
The smoke uses 32 environments, four rollout steps, two PPO epochs, and 32
minibatches. Adam advances from 64000 to 64128; all 2,311,701 model parameters and
optimizer leaves are finite, the parameter tree is unchanged, and weights were
updated. All three costs are nonzero in the saved rollout diagnostics. Mass,
target-map, and obstacle-map integrity checks are zero. This small-batch smoke
does not estimate learning quality or production throughput. Exact command and
checkpoint checks are under `gpu/` in the artifact directory.

An ordinary continuation of that saved treatment also passed, u1002→1003, with
Adam step/count 64192, updated finite parameters, all six diagnostics finite, and
zero transition integrity residuals. Both reset and PPO-update executables hit
the persistent compilation cache. A fresh process still traces and lowers the
JAX program: the resumed training section took 133.31 s, so this is evidence of
cached executable reuse, not elimination of all startup overhead. The first
two-update run compiled the PPO update once and reused it for the second update.

These validate metric behavior, not policy quality. The next behavioral comparison
should first measure the same parent checkpoint on untouched fixed foundation
maps, then compare the unchanged reward against the travel/turn/lateral treatment
with the same checkpoint, bank, PPO shape, action mode, and training budget.
Keep success, terminal material progress, and trench/junction results visible.
Compare the optional executable observation separately from the reward treatment
before combining them; movement-primitive and support-envelope changes remain
later treatments. A changed reward must be recorded and must not silently
overwrite the receipt of an ordinary native continuation.

## September 11: historical comparison and repaired-environment replay

The earlier foundation sweep was a native continuation, not a scratch test.
After 10,000 additional easy-bank updates (old u15000), control/2x solved
63/64 and 64/64 of the same validation panel; current scratch u10000 solves
0/64 each, with 66.64% and 14.76% mean target excavation. The old parent had
327.68M prior generalist transitions, verified back to its original scratch
u1000 checkpoint. Total exposure is therefore 491.52M versus 163.84M, and the
optimizer/entropy clocks also differ. No matched historical easy-bank scratch
comparison was found; older constrained-bank scratch runs also had zero exact
foundation successes around this sample budget.

Replaying the exact old u15000 weights under repaired Terra ba9cc214 changes
control from 63/64 to 62/64 (99.67% to 99.56% dug), and 2x from 64/64 to
51/64 (100% to 95.27% dug). Both use the same local GPU runtime, 64 map/reset
identities, greedy decisions and 450-action horizon as their archived local
replays. All 128 new episodes pass integrity checks. The 2x loss spans all
three shapes; ten of its thirteen failures contain at least 300 no-effect
actions. Its earlier completion result does not transfer unchanged to the
repaired dynamics, although the frozen policy still works far more than the
scratch 2x policy. This replay does not isolate the effect on scratch learning
or identify the responsible transition change in each failure.

Keep initialization, total transitions, environment epoch and completion visible
when judging reward efficiency. Delaying behavior costs until basic competence
is a next-test hypothesis; the old 2x failure tails also require diagnosis.
Neither stronger penalties nor policy promotion follows from these results.
The [historical comparison and raw replay evidence](../../../../../.artifacts/terra_excavation_scratch_cscs_20260910/historical_foundation_comparison_20260911/REPORT.md)
record the matched panels, lineage, limits and independent checks.

## September 11: movement regression isolated and corrected

Exact first-transition replay identifies a movement bug in Terra ba9cc214.
Seven of the thirteen lost frozen-2x successes first diverge because separately
rounded intermediate chassis centers leave the straight path and collide with
nearby soil or holes. Every nominal endpoint is valid, and every first blocker
is outside the straight swept polygon under Terra's cell-center convention.
The remaining six first divergences are intended global soil corrections:
three old rotations enter height-one soil and three old relaxation updates
place material under a clear chassis. Foundation maps have no trench axes;
the junction gate does not explain these differences.

Terra 7fb30402, isolated in `terra_straight_sweep_20260911/terra`, checks each
candidate's entire straight swept path and valid endpoint, choosing the longest
clear candidate. It keeps the soil-free chassis invariant and does not allow
clear endpoints to bypass intervening obstacles. Eighteen focused CPU tests
pass, independent clear-ground checks preserve 384/384 old endpoints, and exact
GPU probes restore all seven old moves while preserving all six soil fixes.
Those thirteen transitions conserve mass and preserve chassis clearance.

The full matched replay recovers twelve of thirteen lost successes: **51/64 ->
63/64**, with no new losses and all 64 episodes passing existing integrity
checks. Checkpoint bytes/update, map/reset identities, full treatment fingerprint,
reward protocol, greedy policy, seed 20260907, horizon 450 and chunk size 32
match. Mean excavation improves 95.274% -> 99.938%, no-effect actions
76.000 -> 4.016 and steps 134.625 -> 60.875. Slot 38 remains unsolved at
96.0317% excavation. Restoring twelve completions includes five cases whose
first divergence was an intended soil change: first-transition labels were
not exclusive whole-episode causes. The original fa8d5d13 replay was 64/64.

The main demonstrated gain is restored completion. On the 51 common successes,
workspace area is 8.167 -> 8.199 m² and productive poses 5.078 -> 5.059;
the correction does not establish a learned larger-workspace strategy.

All thirteen original stalled states have a material-changing cabin heading
in independent actual-DO checks. Eleven expose fresh work after at most one
base action plus cabin adjustment. These local witnesses do not prove episode
completion or within-horizon recovery. Keep remaining policy action loops
separate from the confirmed movement bug, and do not infer an effect on scratch
learning speed from the frozen-policy comparison.

The original CSCS cohort and automatic evaluator still use ba9cc214; no weights,
PPO settings, reward costs or observations changed. The
[diagnosis, correction tests and replay evidence](../../../../../.artifacts/terra_excavation_scratch_cscs_20260910/foundation_regression_diagnosis_20260911/REPORT.md)
include the recorded geometry and independent review.

## September 12: corrected scratch cost comparison at u10000

Both task families now have the four cost combinations on the corrected
movement environment: zero costs, lateral-only, travel/turn-only and combined
2x. Seed, observations, architecture, PPO and task-specific banks are shared;
each policy starts from scratch. At u10000, each has 163.84M fresh training
transitions. This separates the cost components from the previously confirmed
movement bug, which is fixed in every new arm.

Foundation control/lateral/relocation/combined excavation is respectively
71.03%/55.00%/12.47%/8.28%, with 1/1/0/0 exact successes out of 64. Trench
success is 20/2/0/0 out of 224, with 47.26%/43.36%/0%/0% excavation. All
travel-cost trench episodes perform zero fresh excavation at both u5000 and
u10000. The control continues improving while the travel-cost arms suppress
productive work. Lateral-only also trails control, especially in trench
completion.

There is no foundation common-success set between control and lateral-only,
and their trench intersection contains only one episode. Comparisons against
travel-cost policies have no common successes. Lower travel, fewer work poses
or different workspace averages on failed episodes cannot establish improved
excavation efficiency. The raw reports retain all deployment-relevant work-pose
travel, workspace-area and adjacency metrics.

These results argue against increasing behavior penalties during early scratch
training. Reducing or introducing travel/turn costs after reliable completion
is a next-test hypothesis; it is not yet a demonstrated training strategy.
Finish the existing 24-hour screen and evaluate later checkpoints before a
convergence or saturation claim. One training seed and the two-node grouping
limit variance and factorial-interaction conclusions. No policy is promoted.

Full-panel reset identities, treatment compatibility and recorded integrity
checks pass; all u5000/u10000 checkpoint checks include finite parameters,
optimizer/loss, actual Adam counts and hashes. See the
[matched comparison and evidence](../../../../../.artifacts/terra_movement_restart_cscs_20260911/REPORT_20260912.md).
