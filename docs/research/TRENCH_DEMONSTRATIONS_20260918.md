# Demonstrations and successful-policy rehearsal

## September21, 09:42 UTC: checksum repair retry submitted

Live refresh confirmed no new generalist production updates after4721607;
Euler100-source bank remains complete and both legacy finetuning lanes run.
Within the existing authorized100k→110k continuation, submitted **4725717**,
one4-GH200 node,d130/normal,seven-hour cap (prior failed job used44m14s of the
original eight-hour budget). Scheduler startup remains unverified at submission.
The artifact runner now uses streaming SHA256 compatible with cluster Python.
Actual-parent CPU dry-run including the corrected retention report/checksum
passes, shell/Python checks pass, and independent review found no blockers.

The retry reuses completed corrected-u100000 evaluation and finite native
100001/save-resume100002 qualifications from4721607. Fresh CUDA/NCCL and exact
cluster-Python dry-run run before production. It resumes the untouchedu100000
model and Adam, not diagnostic checkpoints, under the same corrected frozen
runtime. No imitation, physics, reward, or PPO changes. Fixed retention panels
101000/102500/105000/110000 and rolling250-update saves remain. Both previous
calibration and alreadycompleted dataset collection are not repeated.

Remote outputs use `terra-overnight-20260920/retry_20260921/`; local submission
receipt `.artifacts/terra_overnight_20260920/launch/retry_submission.json`.
Next recommended check10:45UTC for queue/startup and first finite production
checkpoint. Recommended only; no monitor is scheduled. Do not count the
retry as healthy training until actual completed updates are present.

## September 21, 08:04 UTC: demonstration target reached; production continuation did not start

Live Daint/Euler checks succeeded. CSCS4721607 FAILED after44m14s. Corrected
u100000 panel completed378/384 foundations,209/224 trenches,30/32 roads with
zero integrity failures. Mean dug/accepted: foundations99.405%/99.320%,
trenches98.672%/98.216%. Compared to corrected u75000 counts375/206/30,
completion is up3 foundations and3 trenches; these are checkpoints of the
original training run replayed under corrected physics, not new adaptation.
Native finite u100001 and save/resume u100002 qualifications PASS with
Adam6400064/6400128. Production failed before updates in parent checksum
validation: `AttributeError: module 'hashlib' has no attribute 'file_digest'`.
The compatibility bug is fixed locally in the artifact continuation runner and
source run.py using streaming SHA256; exact digest parity on the actual parent
and syntax checks pass. Remote repair/resubmission has not been performed.
No retention-stop result or new production checkpoint exists.

Imitation calibration still stops on unchanged-parent cross-graph discrepancies:
logp0.05428314/value0.01160336. Dynamic parameter routing did not resolve the
native failure. No imitation PPO was run; do not weaken guards without diagnosis.

Euler14730608 COMPLETED56m09s with native oriented replay PASS. Collection
14730609_0/1 COMPLETED in3h01m/3h33m, yielding38+37 new sources. The full combined
bank was validated this turn at `qualified_combined_20260921`:100distinct sources,
102episodes,11875total transitions,11020supervised,94full plans and8corrective
suffixes. Existing native acceptance is retained; this combine pass checked
arrays/history/time/splits but did not rerun native simulation. Held-out source
and scenario overlap are zero across9manifests. Only1new road-disposal source
was solved; coverage remains uneven. Data collection is not learned policy
improvement. The optional known-network diagnostic failed separately and has
no completed replay result. Old Euler finetuning lanes continue; no new jobs
were submitted during this status check.

Evidence: `.artifacts/terra_overnight_20260920/status_20260921/` and remote
Euler `terra_trench_recovery_20260920/qualified_combined_20260921/`.

## September 20, 21:47 UTC: bounded overnight work submitted and runtime checked

**CSCS 4721607 is RUNNING on nid006492**, lterenzi/d130, one node with four
GH200 GPUs, 64 task CPUs, eight-hour ceiling ending September21 05:46 UTC
(07:46 Paris). Four-GPU identity, cuDNN backward and NCCL all-reduce PASS.
It is evaluating the native u100000 checkpoint under the corrected runtime;
no new PPO update is verified yet. The original-runtime 380/384 foundation,
213/224 trench and 31/32 road counts are not the current-runtime baseline.

The submitted sequence is:
1. Required corrected-runtime 608-case u100000 evaluation and integrity checks.
2. At most 30 minutes of old-u75000 imitation calibration, diagnostic only.
   Its failure is recorded separately and cannot select a production model or
   block the independent no-imitation continuation.
3. Independent native four-GPU finite u100001 and save/resume u100002 checks,
   including exact Adam6400064/6400128 and absent demonstration state.
4. Discard those diagnostic steps; resume the original native u100000 model and
   Adam toward u110000 (maximum327.68M added transitions), with all added costs
   zero and no imitation. Save every250 updates. Required matched panels at
   u101000, u102500, u105000 and u110000 compare to the corrected baseline.
   Stop for net losses above4 foundations,2 trenches,1 road case, or2 in any
   condition. No automatic promotion or further allocation follows this job.

This replaces the provisional plan to retry the old demonstration comparison.
The parent solves24/25 instruction resets; all26 instruction variants for those
successful resets require more raw Terra actions. That does not prove worse
retained-pose deployment efficiency, which was not compared. It does mean this
bank cannot be assumed to contain generally better expert behavior. The one
new successful strategy and larger training-only search remain useful research
inputs, but no hard-label training is launched tonight.

Calibration now passes rollout weights dynamically, matching production.
A varying-input CPU check of the actual parent has zero scan-versus-forward
logp/value discrepancy in BF16 and FP32; an earlier hoistable repeated-input
probe still drifts. The correction is not proof of the GH200 failure cause.
All native drift guards remain unchanged and the GPU diagnostic remains pending.

**Euler 14730608 is RUNNING on eu-g4-007**, one RTX3090,8CPUs,32GB,75minutes.
CUDA/cuDNN checks PASS; current phase is native scalar-path compilation after
325seconds of runtime initialization. Native oriented replay remains unverified.
Dependent collection **14730609_[0-1]** is PENDING/Dependency: two one-GPU lanes,
eight CPUs and32GB each, capped at six hours. Search clocks now exclude cold
compilation; each has at most five hours of search. Both require the new native
qualification receipt and afterok dependency. Target remains75 additional hard
training sources; zero new examples are claimed at this handoff. The old Euler
finetuning lanes remain untouched.

The CSCS runtime reuses September19 frozen Terra/baselines snapshots. Only the
artifact continuation runner's authorized ceiling and source routing differ;
training/Adam/PPO/reward logic is unchanged. Its actual-u100000 CPU dry-run
passes with4 virtual CPU devices,1024global envs,32768transitions/update and no
inherited imitation. Syntax, shell, EDF, source-diff and independent reviews
pass. GPU optimizer qualification, first production update and new checkpoints
remain outstanding. Walltime with a saved finite checkpoint is continuable,
not scientific failure; inspect any retention stop before resuming.

Artifacts: `.artifacts/terra_overnight_20260920/` (recovery, launch, Euler checks).
CSCS root: `/ritom/scratch/cscs/lterenzi/terra-training/runs/terra-overnight-20260920`.
Euler root remains `codex_terra_edge_runs/terra_trench_recovery_20260920` in
lterenzi scratch. Submission receipts record exact IDs and resources.
Next recommended startup check: **September20 22:45 UTC**, to establish native
qualification and the first PPO checkpoint; morning result check **September21
06:00 UTC**. These are recommendations, not scheduled monitors. The jobs perform
the described checks automatically. Report actual phases and completed panels.

## September20, 21:33 UTC: CSCS access restored and complete results recovered

Daint SSH and Slurm verified as lterenzi, project d130. Broad segment4685249
COMPLETED at native u100000 with Adam6400000 and PASS finite-state result.
The original-runtime fixed greedy450 panel improves from u75000 to u100000:
foundations379/384→380/384, trenches208/224→213/224, included road30/32→31/32.
All608 cases have zero integrity failures. Foundations gain4 and lose3 maps;
trenches gain7 and lose2. On376 common foundation successes, retained inter-setup
straight-line travel34.671→32.403m, unique area/productive setup6.804→6.953m²,
and fresh workspace edge adjacency86.06%→87.47%. These are original-runtime
results, not yet corrected-runtime qualification or evidence for cost promotion.

CSCS imitation4712295 FAILED after14m39s in calibration before either PPO arm.
Parent instruction comparison and bank assembly completed. The parent succeeds
on24/25 selected instruction resets, so most existing labels teach a different
successful sequence rather than a task the parent cannot complete. Calibration
stopped at unchanged-parent logp drift0.0507202/value0.0107465; cached parent
probabilities match within5.96e-8. There is no new imitation learning result.
The rollout embeds frozen weights as compile constants, unlike the dynamic
production path; a diagnostic correction is being checked without relaxing
numerical guards. The overnight run will preserve the matched u75000 recipe.

Euler qualification14671834 FAILED/124 after17m45s during pre-search startup;
no START2983 or native successful plan occurred. Its dependent collection array
14671835 was canceled before running. CUDA passed, bank loading took3m36s and
model initialization20s. The search budget incorrectly started before cold
compilation; it now starts after explicit warmup, with phase timing and a longer
bounded qualification. This is startup evidence, not a planner failure.

Recovery artifacts: `.artifacts/terra_overnight_20260920/recovery/`; Euler timing
and retry checks: `.artifacts/terra_overnight_20260920/euler/`. Source runtime,
physics and efficiency costs are unchanged. No new overnight job is claimed
started by this recovery entry; submission and startup will be recorded below.

## September 20, 05:22 UTC: Euler compute reallocated to hard-trench demonstrations

The user approved using Euler while CSCS access remains unavailable. Preserved
six native scratch checkpoints (model and Adam, 646 finite numeric leaves each)
in `/cluster/project/rsl/lterenzi/terra_trench_recovery_20260920/preserved_scratch/`.
The two scratch lanes and their already queued successors are now confirmed
CANCELLED: 14606054, 14615001, 14670279 and 14671603. The two finetuning lanes
14606393/14609052 remain RUNNING; their successors 14670278/14670291 are untouched.
This is a compute reallocation, not evidence that all scratch runs failed:
scratch straight seed14 reached 32/32 at u16000 and scratch T seed13 reached
32/32 at u17000. The preserved checkpoints retain those recoveries.

All four known-network variants remain 0/32 in the latest complete panels at
u15000–u18000. Finetune seed14 excavates 95.35% but accepts only 65.12%, with
25.25% off-zone and 4.98% loaded material. It needs a disposal/sequence audit;
this does not prove infeasibility. These adaptation maps originate in held-out
banks and are diagnostic-only, never imitation inputs.

Submitted replacement qualification **14671834** (one RTX3090, 45 minutes) and dependent
collection array **14671835_[0-1]** (one RTX3090 per lane, six hours each), account
lterenzi, gpuhe/es_hutter. At submission qualification is PENDING and collection PENDING/Dependency.
Native replay remains UNVERIFIED. The initial qualification 14671793 passed
CUDA/convolution checks but failed after 33 seconds because the staged bank
lacked its required parent source registry. Dependent array14671794 was
automatically CANCELLED without starting. The original registry was added;
the complete 3,840-map dataset contract now passes on CPU. Both old logs and
submission receipts are preserved.
The collection starts only after the new oriented planner completes a known
training source and independently replays it in 16 native lanes. A bounded
32-reset corrected-runtime network audit shares the qualification allocation;
its failure is recorded separately and does not imply a negative policy result.

Target: add 75 distinct hard training sources to the existing 25, best effort.
743 eligible unused sources remain after excluding all nine held-out manifests
(5,440 rows); the two disjoint lanes attempt at most 110 sources each. The new
proposal uses actual segment orientations for branch-first retreat planning.
Each source is bounded by 300 seconds and 120,000 transition queries; each lane
by 5M queries and 19,800 seconds internally. Every accepted plan must pass native
full-reset history/time replay, exact completion, mass, target/obstacle and
chassis checks. Corrective prefix actions retain context but are unsupervised.
CPU validation passed for 27 existing episodes and 110 negative mutations.
Remote input planning and both Slurm resource dry runs pass. No PPO, reward,
physics, or efficiency-cost changes are introduced; maximum new allocation is
12.75 GPU-hours. There is no automatic continuation.

Sources reuse the September19 corrected snapshots (Terra 6a0d7bd plus dirty
changes; baselines f550c04 plus dirty changes). Local artifacts are
`.artifacts/terra_trench_recovery_20260920/{launch,expansion,diagnosis}`. Euler run
root is `/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_trench_recovery_20260920`.
No duplicate of CSCS imitation comparison 4712295 was launched; its latest
confirmed state remains September19 13:19 UTC, not a current result.

Next recommended check: **September20 06:30 UTC**, to inspect qualification and
collection startup, then 3–6 hours after collection starts for accepted source
counts. No monitor or scheduled check was installed. Before any later imitation
assembly, exclude the enlarged instruction sources from fresh retention too;
do not reuse overlapping September19 retention rows. Merge only verified
accepted examples and report actual counts; 100 is a target, not a result.

The purpose is to teach access-preserving work order, terminal-pose selection
and complete disposal while retaining foundation competence. Clean manual or
planner plans supply missing strategies; successful current-policy trajectories
can provide broader rehearsal. Collection, finite training checks and improved
held-out policy behavior are separate results.

## September 19: revised retention and corrective instruction

The user authorized implementing the completed review and proceeding with a
bounded continuation. Both ordinary foundation and trench rows now use cached
selected-parent KL. Verified corrective instruction uses action cross-entropy.
Hard instruction sampling chooses source before dumping condition, then episode
and eligible step; retention keeps condition-first balancing. No reward,
physics, PPO layout, action mask or efficiency coefficient changes are added.
Twenty-nine focused CPU tests pass, including both KL anchors, mixed hard/soft
losses, source balancing, inactive-loss skipping and evaluation checkpoint
identity. Native allocation qualification is still required.

The selected parent is native u75000 of broad run 4685248. Its original-runtime
608-map panel passes: foundations 379/384, trenches 208/224, included road
subset 30/32. Mean excavation is 99.380% and 98.052%, respectively; accepted
material is 99.380% and 97.842%. This historical panel is not the retention
reference for the corrected runtime. Preparation job **4711914** completed its
current-runtime evaluation and collection with PASS. The shared comparison
baseline is **375/384 foundations, 206/224 trenches and 30/32 road cases**, with
zero integrity failures. Its CUDA, convolution-backward and four-GPU NCCL checks
also passed. No PPO runs in this preparation job.

Collection obtained eight qualified successes in each of all 40 training
conditions: **200 foundations and 120 ordinary trenches, 27,240 transitions**.
It attempted 336 episodes, with 334 successes, within the 512-attempt budget.
Source/scenario exclusions cover all nine held-out manifests and all reserved
instruction sources. Both families cache the selected parent's probabilities.
Full native resets, five-action history, remaining time and material/chassis
checks pass.

The qualified instruction payload has **27 episodes across 25 hard sources**:
seven networks, three segmented bends and fifteen T junctions. Twenty-one
sources are newly solved; four retain earlier native acceptance. There are
25 full plans and two corrective alternatives on existing sources, totaling
2,832 transitions and 2,727 supervised actions. Their 105 prefix actions remain
masked while preserving state and history. All new plans pass independent
native replay with exact success and accepted disposal. No straight examples
were added to inflate diversity, and all four manually solved evaluation
failures remain excluded. The unresolved training network 2166 is excluded;
bounded search failure does not prove infeasibility. Before training, the
parent is replayed on these exact initial Agents to measure what instruction
adds; successful demonstrations alone do not establish improvement over it.

Comparison **4712295** was submitted at 13:15 UTC with a six-hour ceiling,
sequentially on one additional four-GH200 node. PPO-only and revised imitation
both start from the same native u75000 model and Adam state, with the same seed,
reset protocol and corrected runtime. Live episodes restart on native resume;
this is not bit-exact continuation of the original running environment.
Each arm is bounded to 1,000 updates / 32.768M live transitions. The imitation
coefficient fades linearly over 500 updates / 16.384M transitions, followed by
500 updates without imitation; the control receives PPO alone throughout.
The initial coefficient is selected from 0.001, 0.003 and 0.01 by native-Adam
shadow-update calibration; the job stops if none qualifies. The data mixture
remains 50% foundation retention, 20% ordinary
trench retention and 30% corrective instruction, with 16 sampled rows per
device per Adam step. Those are sampling fractions, not gradient fractions.
Matching-parent KL gradients start near zero by design. Shared encoder changes
can affect critic values despite no direct demonstration critic loss.

Both arms require exact-layout finite first-update/save-resume checks. The
guided arm also checks the active-to-inactive fade specialization. Full fixed
evaluations at u75500 and u76000 compare to the shared corrected-runtime parent
panel. Existing net-loss stops remain foundations >4, trenches >2, road cases
>1, or any condition >2. Gross gains/losses and continuous/retained-work metrics
remain visible. A stop is a bounded retention decision, not a saturation claim
or a rejection of imitation. No policy or efficiency stage is promoted
automatically. Source, data, launch and later calibration evidence live under
`.artifacts/terra_imitation_revision_20260919/`.

At 13:19 UTC the comparison is RUNNING on nid006338. Four-GH200 device,
cuDNN-backward and NCCL checks pass; instruction-parent comparison is active.
New PPO startup is unverified.
Data assembly, four-GPU shadow calibration and each arm's independent native
finite-update/save-resume checks precede production. A completed control
retention stop permits the independently qualified imitation arm to run from
the original parent; unrelated failures stop the wrapper. If only one arm
reaches u76000, compare the arms at their shared u75500 milestone. The original
broad continuation remains active and all added efficiency costs remain zero.

## September 19 failure and imitation review

The user requested manual solutions for the four trenches lost from u35000 to
u50000, and a new Oracle review of reward and clean-demonstration integration.
The completed review is `terra_trench_churn_20260919/oracle/answer.md` under
the workspace artifact root. It recommends diagnosing policy drift and
separating retention from corrective instruction before another reward change
or stronger imitation coefficient. The original zero-imitation PPO churn and
the corrected-runtime imitation regression remain separate comparisons.

The lost panel slots are 340, 356, 396 and 499. The two network slots share one
source geometry. Three of the four regained slots were already solved at
u20000, lost at u35000 and regained at u50000. This is unstable retention of
demonstrated competence, not four monotonically acquired strategies.

The actual reward receipt and all six checkpoint cost coefficients were
checked. All additional costs are zero. Accounting for the retained terminal
potential, each old successful trajectory earns at least 5.50–8.55 more
discounted reward than an optimistic upper bound on its later failure.
Profitable stalling therefore does not explain these four trajectories.
However, mean fixed-panel return rises from approximately 5.150 to 5.214
overall while trench mean return falls from 5.180 to 5.136. The aggregate can
hide family-specific deterioration. These are recipe-level greedy return
bounds, not the sampled PPO objective or a causal gradient-interference test.

Independent inspection found no active demo history/time, sampling-weight or
schedule bug. A 512-row condition/source-balanced CPU probe shows the u35000
parent agrees with 91.56% of ordinary trench actions and 99.47% of DO labels.
Disagreement is mostly navigation/reorientation, so older hard labels are a
possible conflicting objective, not evidence of broadly wrong dig labels.
On the identical probe states, u35000 versus u50000 changes its preferred
action on 13.45% of rows. That demonstrates drift without attributing the four
held-out failures to a particular changed action.

Oracle's optional action-mask indexing finding was reproduced and fixed in
rollout, PPO, demonstration loss and both evaluators. All five lookups now use
the appended mask after optional policy features. Eighteen focused CPU tests
pass, including configuration-accepted alignment-plus-mask and exact unmasked
update parity. Native-smoke preparation also found that the paired environment
does not supply a raw action-mask observation. The unit fix therefore does not
enable native masked training; that separate interface remains absent. The
active runs are unmasked, so these issues do not explain their churn.
A native local GPU smoke also passes: 32 live transitions and one Adam update
with finite model, optimizer, rollout and loss values and zero integrity
errors. This smoke uses alignment inputs and a medium spatial ResNet without
masking, time inputs or demonstrations, on JAX 0.4.26/cuDNN 8; manual replays
use JAX 0.4.33/cuDNN 9.5. It is implementation evidence, not production-layout
qualification or a learning result. Patch/test evidence is in
`terra_trench_churn_20260919/mask_fix/`, with the native result in
`mask_smoke/receipt.json` under the same artifact root.

The smallest proposed next integration is cached selected-parent KL for both
foundation and ordinary-trench retention; hard labels remain for verified new
expert/corrective choices. Prefer successful parent-visited retention states,
sample expert geometry before dumping condition, and target training-source
junction/cleanup decisions and verified learner-prefix corrections. The current
six-source expert bank heavily repeats straights. Group sampling fractions are
not gradient fractions, and a zero KL gradient at the parent is expected for a
retention anchor. Calibrate auxiliary effect with native optimizer state and
keep a matched PPO control if measuring causal benefit. No replacement recipe
or new training allocation has been launched.

Manual slot 396 now completes in 98 actions, with all 88 cells excavated and
accepted, empty load, zero ineffective actions, and independent native 16-lane
replay passing all transition checks. It uses the locally corrected boundary
runtime, unlike the historical u35000/u50000 panel. Work order and finishing
poses were manually selected; BFS only routed between them. Two corrective
one-cell soil lifts are part of this witness, not proven universally necessary.
This is a feasible plan, not a shortest-plan claim. Slot 499 also completes:
104 actions, all 56 cells excavated and accepted, empty load, no ineffective
actions and independent native 16-lane replay passing all checks. Discrete
state and terrain agree exactly; the maximum normalized carry-work rounding
difference is 9.31e-10. Two-sided network slot 356 also passes native replay:
226 actions, all 141 cells excavated and accepted, empty load and all integrity
counters zero. Its plan completes branches before cutting the connector,
keeps the cleanup approaches opposite the branches clear and includes three corrective
soil lifts. These are choices in one verified plan, not universally required
actions. The old policy solved it in 93 actions, so this longer manual witness
does not demonstrate improved efficiency. One-sided road slot 340 also passes
native replay: 297 actions, all 141 cells excavated and accepted, empty load,
zero ineffective actions and no integrity failures. It relocates accepted
spoil before approaching its near-reach blind area, then uses an outside
aligned pose for the final three junction cells. All four requested episodes
are now verified within 450 actions. They cover three source geometries;
these feasibility/recovery witnesses do not establish why the historical
policy chose its failing actions or validate deployment navigation/egress.

A frozen-policy probe on all 98 slot-396 manual states preserves native history
and remaining time. Both policies assign very low probability to some manual
approach, soil relocation and final orientation choices, while both assign
over 99.9% to the final dig once correctly positioned. These are candidate
instruction targets, not proof that their preferred alternatives fail; those
alternatives were not executed, and the older policy already had a shorter
successful plan. All inspected evaluation plans remain excluded from training;
instruction data must use independent training sources.

Evidence: `terra_trench_churn_20260919/{README.md,reward_audit.json,
cohort_return_bounds.json,reward_cost_check.json,imitation_audit.md,
label_diagnostic.json,manual_policy_probe.json,gallery.html}` and
`manual/evaluation_{340,356,396,499}/{metadata.json,manual.gif}`. The Oracle answer and
its local validation are in `oracle/{answer.md,validation.md}`.

## September 19 morning update

At 05:48 UTC, old broad continuation 4685248 is RUNNING on nid006516 with
u67000 saved/logs near u67068, about 17.1k end-to-end transitions/s; 4685249 is
dependency-pending. Its original-runtime u50000 evaluation PASS matches the
u35000 source/panel/reset/protocol/fingerprint: foundations 374 to 377/384
(8 gained/5 lost), trenches 209 to 209/224 (4/4), road subset 29 to 30/32 (2/1).
Foundation dug/accepted material rises 98.752%/98.636% to 99.529%/99.528%;
trench values fall 98.863%/98.649% to 98.527%/98.396%. On 369 common foundation
successes, retained travel improves 3.6%, area/setup 0.8% and adjacency rises
85.313% to 86.384%. All integrity counts are zero. This is continued learning
with trench churn, not evidence of saturation or a matched corrected-runtime
control. The 15 remaining trench failures comprise four net4 cases (~50% dug),
two seg2 cases (61.5%/68.2%), eight straights (90.8–97.5%) and one T (89.3%).

The delayed easy-foundation comparison is now recovered: evaluation-only job
4709744 completed, fresh parent 64/64, both arms 63/64 at u22500 and 62/64 at
u25000. Five validation checks pass. The final completion floor fails for the
penalty arm, while the matched control has the same count. On 60 shared final
successes, penalties improve retained travel 10.3%, area/setup 4.3%, setups
4.0%, lateral score 22.3% and adjacency 0.21 pp. See
[the delayed-cost results](FOUNDATION_DELAYED_COSTS_20260916.md).

Main costs remain zero, the imitation candidate remains stopped, and the next
main evaluation is u75000. No automatic penalty switch or promotion is made.
Evidence: `.artifacts/terra_training_status_20260919/generalist/paired_35000_50000.json`
and `delayed_penalty/analysis.md`. Earlier decision/status sections below retain
their historical timing; the u50000 evaluation and delayed-output recovery are
now complete.

## September 18 implementation and outcome status

The combined imitation run 4709420 passed exact four-GPU qualification and
completed its 750-update fade, then stopped on trench retention at u35750.
Foundations improved 369/384 to 383/384, while trenches fell 213/224 to 205/224
on the same-runtime CSCS panel. The hold did not start and the policy is not
promoted. Full measurements and the continuing old-run status appear below.

The sector-boundary observation/DO mismatch is fixed locally, and optional actor
imitation is implemented. Five GPU geometry tests pass, including 432 translated
base/cabin poses, scalar/batched/nested masks, radial boundaries and loose-soil
priority. On the saved failing state, the corrected observation reports zero
fresh cells and scalar/batched DO both relift the expected one soil unit.

The additional geometry audit reports `NO_DISCRETE_MISMATCH_IN_PROBES` for
300 translated fixture poses, 648 synthetic float16 yaw-boundary probes and
288 wheeled-only probes. Scalar and batched checks cover sizes 16, 64 and 1,024,
with a separate receipt for the exact 256-environment per-device shape, using
default and highest operation precision. No discrete mismatch or nonfinite
result was found in these probes. This tests native trench admission, chassis
footprints and the remaining relative-rotation path; it is not an exhaustive
geometry, reachability or policy-impact certification.

The ten existing training plans were independently replayed with the fixed
geometry: all complete in their original 449 total actions with no ineffective
actions or transition-integrity failures. They still cover only four source
IDs, six trench conditions and no foundations. A second replay now also passes
through native MapsBuffer resets with float16 trench and foundation metadata,
using all ten original action sequences and reset seeds unchanged. Every action
has physical effect, passes transition checks and leads to exact completion.

The native export is
`boundary_fix/native_manual_replay/manual_demonstrations_native.npz` and
supersedes the earlier helper's float32-metadata observation bank. Arrays are
not identical: the receipt reports changes in standoff-error observations on
139 rows, yaw-error observations on 19 rows, reset context on 70 rows and agent
states on four rows. Counts are per field and can overlap. The successful action
sequences remain valid; use the regenerated native observations for subsequent
imitation work.

A real one-GPU diagnostic continuation completed u25000 to u25002, preserving
Adam step 1,600,000 to 1,600,128 with finite model and optimizer state. It used the
ten-plan bank, coefficient 0.05, batch size 32 and the existing native parent. The
diagnostic layout is 1x128 environments, or 4,096 transitions per update; it is
not a production parent for the established 4x256 layout. The recorded
foundation-release state and time/actor migration history were preserved.
This smoke used the earlier `boundary_fix/training_demonstrations.npz`; the
native-metadata observation export was regenerated afterward.
This original smoke is implementation evidence only. The later 331-plan
production screen has now completed its first retention evaluation, reported
below; its outcome cannot be inferred from this smoke.

**The ten-plan bank is smoke-only.** The larger training-only archive contains 1,280 accepted
plans across all 40 conditions, with its combined NPZ exported. The completed
Oracle review supports a smaller, explicitly sampled active bank; its mixture,
soft foundation targets and recovery-label masks are now implemented and have
focused CPU checks. The selected bounded continuation is described below; its
active bank has 331 plans and passed four-GPU qualification before stopping on
the trench-retention gate. The
prototype 0.05/81.92M-transition fade is not the selected recipe.

CSCS access was restored on September 18; the renewed certificate expires
September 19 at 17:48 CEST. The earlier afternoon check found existing job 4685246
running on `nid006532`, with dependents 4685248 and 4685249 pending. The newest
complete checkpoint metadata fetched at 15:46 UTC was u42000. This is historical
training status, not a u42000 evaluation; the later 20:01 UTC snapshot below
supersedes it. No scheduler changes were made during that earlier check.

Evidence: `boundary_fix/gpu_geometry_tests.log`, `saved_state_fix_result.json`,
`replay_summary.json`, `native_manual_replay/replay_summary.json` and
`smoke/result.json`, plus `remaining_geometry_audit.json` and
`remaining_geometry_256.json`, under the artifact root listed
below. An earlier aggregate CPU invocation reported a missing-pytest import and
a stale positional-config assertion; separate chassis tests (4 passed) and the
corrected positional-config test pass. That initial log is not a passing full
suite receipt.

## Historical u20000/u35000 comparison and runtime boundary

The historical, validated u35000 full 608-case panel improves over
u20000. Both were evaluated **before the local sector-boundary correction**:

| Cohort | u20000 completion | u35000 completion | Newly solved | Earlier successes lost |
| --- | ---: | ---: | ---: | ---: |
| Foundations | 367/384 | 374/384 | 14 | 7 |
| Trenches | 195/224 | 209/224 | 17 | 3 |

Foundation mean excavation rises from 98.13% to 98.75%, and accepted material
from 97.89% to 98.64%. Trench excavation rises from 96.34% to 98.86%, and
accepted material from 96.03% to 98.65%. These are continued-learning results
from the existing run, not evidence for imitation or the local geometry fix.
The paired receipt is
`boundary_fix/paired_u20000_u35000_historical_runtime.json`.

The fresh corrected-runtime u35000 panel is complete locally: **369/384
foundations, 213/224 trenches and 30/32 road cases**, or 582/608 overall. Against
the historical CSCS u35000 panel, foundations gain six and lose eleven cases;
trenches gain four and lose none, including one additional road success. See
`boundary_fix/u35000_corrected_runtime.json` and
`paired_u35000_geometry_fix.json`. Hardware also differs between those two
evaluations, so this is not an isolated causal measurement of the boundary fix.
The new run must produce its own fresh CSCS parent panel; the local result is a
qualified reference, not a report to import into the new campaign directory.

## Verified pilot

Eight complete planner-generated demonstrations are available: two horizontal
straight-trench layouts, each with one-sided, two-sided, alternating-side, and
tight one-sided disposal. They contain 191 pre-action examples, finish in
20–28 actions, and have no ineffective actions. Every episode excavates all
44 required cells, deposits all material in accepted zones, and finishes empty.

All actions in the original pilot executed through the then-current environment
with transition integrity checks. An additional readback checked saved history/time alignment,
finite arrays, per-observation mass conservation, immutable maps, forward fresh
digging, and exclusion from all nine evaluation manifests. At initial collection
there was no second independent physics replay after serialization; the fixed-
geometry and subsequent native-metadata replays above now provide that check.

The local GPU collection reached its 300-second budget after saving eight
complete episodes. The unfinished ninth proposal was excluded. Only the saved
complete episodes were assembled into `demonstrations.npz`; the report records
this budget boundary. The pilot's small geometric variety does not establish
generalization; the two manual junction additions below still leave lengths,
orientations and cleanup strategies sparsely covered.

These episodes now support the finite imitation implementation check. They
remain a small diagnostic dataset, not evidence of broad imitation learning.

## Manually worked junctions

The original manual planning step added two complete training plans under the
same runtime, with no policy-generated prefixes:

| Training slot | Geometry | Actions | Excavated / accepted cells |
| --- | --- | ---: | ---: |
| 2885 | T-junction, two-sided disposal | 83 | 96 / 96 |
| 2175 | Four-section network with three T-junctions, two-sided disposal | 175 | 165 / 165 |

The assistant selected branch order, work poses, and dump directions. A
terrain-fixed breadth-first search routed between requested poses. Offline
planning included unsuccessful proposals and backtracking; only the selected
successful sequence is exported. Each sequence was then independently replayed
from its original complete reset through actual Terra transitions, matching the
planned terrain and agent states. Neither exported episode teleports, edits
terrain, extends the horizon, relifts soil, or includes ineffective actions.
Transition integrity and exclusion from all nine evaluation manifests pass.

The T-junction plan clears the outer stem first, leaves the crossing traversable,
works the eastern trunk, finishes the stem from a northern offset pose, and
retreats along the western trunk. A lower offset pose reaches the final two
cells without picking up accepted spoil. One fresh dig uses a 30-degree cabin
offset; the others face forward along the chassis.

The network plan clears the outer portions of all three side branches, uses
the intact main trench as a travel corridor, and excavates that trunk while
retreating north. It then finishes the three junction residuals from the east.
Those final three digs face rearward along the chassis: they are longitudinal,
not sideways. This is an intentional extension beyond the forward-only straight
pilot. Dumps remain unrestricted in cabin orientation.

Two failure mechanisms appeared during manual planning. Accepted spoil inside
the next fresh digging cone makes DO pick up loose material instead of digging.
A valid dump can also block a later chassis pose or retreat. Disposal must
therefore preserve both navigation and the next fresh workspace. Moving the
base slightly or changing dump direction resolved these cases without changing
the physics.

These are feasibility witnesses and training examples, not optimal plans or
evidence that the learned policy has improved. Raw navigation includes extra
rotations and detours, and final egress is not certified. The manual episodes
contain 258 pre-action examples and are stored separately from the straight
pilot under `manual_junctions/train_2885/` and `manual_junctions/train_2175/`.
Each folder contains `demonstration.npz`, a full action/pose receipt, actual
replay frames, and a GIF.

An independent artifact readback also checked every saved pre-action state,
history and remaining-time value, material conservation, chassis occupancy,
and training-bank identity. Source, scenario and parent identities do not
overlap any of the nine evaluation manifests. The two episodes are assembled
with an explicit training-slot allowlist into
`manual_junctions/junction_training_demonstrations.npz` (258 transitions).
`junction_training_summary.json` records the input and output SHA-256 hashes.
The original straight and junction files remain separate. Their corrected-
geometry replays are assembled in `boundary_fix/training_demonstrations.npz`:
ten episodes and 449 pre-action examples, used only for the diagnostic smoke.
The later native-metadata replay replaces this observation export with
`boundary_fix/native_manual_replay/manual_demonstrations_native.npz`.

## Held-out junction recoveries

The two previously unresolved junction failures also have complete manual
sequences from their original full reset states. Independent replay through
unchanged Terra confirms all excavation and accepted disposal, empty load, no
off-zone soil and no ineffective actions.

| Evaluation slot | Geometry | u25000 policy excavation at 450 actions | Manual completion | Spoil relifts |
| --- | --- | ---: | ---: | ---: |
| 346 | Four-section road network, one-sided disposal | 50.35% | 214 actions, 143 / 143 cells | 3: 15, 33 and 8 units |
| 393 | Diagonal corner, two-sided disposal | 61.54% | 123 actions, 78 / 78 cells | 2: one unit each |

The map, original agent state and reset match the corresponding policy replay
(reset seeds 719 and 132). Road case 346 clears branch ends before the main
trench, relocates spoil obstructing the working corridor, then retreats along
the main trench. Corner 393 clears the far tip first, adjusts diagonal workspace
overlap to avoid a one-cell gap, and completes the horizontal arm from a west
pose plus two north-offset poses. These plans include recovery and additional
navigation; they are feasibility witnesses, not optimized clean examples.

These recovery counts describe the original pre-boundary-fix replay. Both remain
evaluation-only, including their saved observation/action files.
They are excluded from the training assembly. These results establish that the
original maps have successful sequences under the corrected rules; they do not
establish that every policy-produced late state remains recoverable. No map
shortening or alignment relaxation was used. All four new plans, with GIFs and
work-pose CSVs, are in `manual_junctions/gallery.html`.

## Reproduced sector-boundary observation mismatch

Manual corner planning exposed a separate issue in runtime `6a0d7bddd7cb7aa989408157ed8f22080b27ffca`
on JAX 0.4.33 / CUDA. Immediately before action 45 in the exported case 393,
the base is at row 29, column 43, base bin 8, relative cabin bin 0. The actual
saved `local_map_admissible_dig[0]` reports **eight fresh cells**. Scalar dig
eligibility and the actual DO instead pick up **one accepted-soil unit**, with
no fresh excavation. Material changes from `[42, 42, 0, 0]` to `[42, 41, 0, 1]`
in excavation / accepted / off-zone / load order.

The disputed loose cell is `[40, 43]`, on the angular sector edge. A 64-row
batched call to the same eligibility/cone methods excludes it and predicts
eight fresh cells; a scalar call includes it and selects relifting. Reloading
the serialized pre-action state reproduces the observation/transition
disagreement. This is not evidence that the terrain changed during navigation:
the comparison uses the identical state immediately before DO.

The saved state, reproducer and original numerical receipts are under
`manual_junctions/sector_boundary_mismatch/`. Follow-up GPU diagnosis isolated
the batched homogeneous-coordinate matrix arithmetic: its reduced precision
changed the disputed cell's angle from 0.52359855 to 0.52471066 rad across scalar
and batched execution, around the 0.52359879-rad boundary. Highest matrix
precision removed that discrepancy.

The implementation now subtracts the origin before elementwise rotation and
gathers base coordinates directly instead of multiplying by a one-hot vector.
Sector boundaries are inclusive within `1e-5` metres/radians, avoiding numerical
edge flips without a meaningful extension of the workspace. The saved failure
and geometry tests pass as described above. The prevalence of this bug across
training and its effect on learned completion remain unmeasured. This fix does
not establish that previous policy failures were all geometry bugs.

## Evidence and scope

In the September 18 failure audit, a hand-designed plan completed held-out
straight case 434 in 34 actions: reach the far end over intact ground, excavate
forward while retreating, and dispose of each load. Its policy-produced late
state could no longer reach the last two target cells. The original task was
solvable; earlier choices had destroyed access. Other verified alternatives
finished cases 405, 406, and 466. These held-out cases remain diagnostics and
must not become training demonstrations.

The first demonstration collector implements the hand-designed retreat
strategy on independently selected cardinal straight trenches in the existing
training bank. Terrain-fixed navigation search proposes the initial approach;
all approach, cabin, dig, dump, and retreat actions execute through unmodified
Terra. It does not use policy-generated action prefixes. Failed search attempts
are recorded separately and supply no imitation examples.

The automatic straight-trench collector remains deliberately narrow. It is not
a solver for junctions, networks, staging trips, diagonal trenches, or all
disposal constraints. The manual witnesses above do not establish a general
junction-solving algorithm.

## Broader collection: archive exported

`boundary_fix/collect_policy_rehearsal.py` now implements greedy rollout
collection from the native u25000 policy on the existing training bank. Its
default target is **32 accepted successful episodes per condition**, across
25 foundation and 15 trench conditions: at most 1,280 retained episodes if every
quota can be filled. This is a configurable collection target, not a claim that
1,280 episodes are necessary or sufficient.

The final `boundary_fix/policy_rehearsal_v1/summary.json` now reports
`QUOTAS_COMPLETE`: 1,280 accepted episodes, all 40 condition quotas filled,
116,265 transitions, 784 source geometries and 1,280 distinct scenarios. The
combined `rehearsal_demonstrations.npz` exists at its reported 41,836,975
compressed bytes. Its raw transition arrays occupy 5,290,987,620 bytes, or about
4.93 GiB per bank copy before training overhead. This archive need not all enter
the active PPO auxiliary bank, and its export does not qualify production
training memory or learning benefit.

An independent CPU readback through the current `load_demonstrations` passes
for the entire archive: 800 foundation and 480 trench episodes, every saved
history and remaining-time row, and all 26 current preprocessing inputs.
All nine held-out manifest exclusions pass again. Measured peak CPU RSS is
10.18 GiB; no GPU allocation, policy forward pass or PPO update was part of
this readback. See `boundary_fix/policy_rehearsal_v1/loader_validation.json`.

The bank has 3,840 slots, 1,821 distinct source IDs and 3,837 scenarios. Collection
excludes sources/scenarios present in all nine held-out manifests and removes
the three duplicate scenarios before sampling. The full default candidate
budget is at most 3,837 episodes, or 1,726,650 decisions at a 450-action horizon;
conditions stop early when their quotas fill. Source counts remain separate from episode
counts, since disposal variants can share an excavation source. Missing coverage
and rejections are reported rather than filled with extra easy cases.

The collector uses native MapsBuffer resets, including float16 trench metadata
and foundation border axes, and verifies every executed transition. It accepts
exactly completed episodes with at most 10% ineffective actions, no consecutive
ineffective run above 20 actions and no material-or-load stall above 100 actions.
These are explicit data-quality thresholds, not evidence of optimal navigation.
Legitimate relifting is retained. Successful-policy rehearsal therefore has a
different acceptance rule from the clean no-relift manual plans; both retain
complete action sequences and correctly aligned pre-action history/time.

CPU checks cover candidate exclusion/deduplication, quality thresholds and
streamed NPZ assembly/readback. A bounded one-GPU batch now passes: **16/16
episodes complete exactly and are selected**, covering 16 source IDs and 16
conditions, with eight foundation and eight trench episodes. The 1,290 actions
pass transition-integrity checks. Collection took 251.6 seconds including
compilation and exported 58,705,320 raw bytes, compressed to 424,981 bytes.
That earlier bounded batch is training-data collection evidence, not a held-out
policy result. Its own summary reports `quota_complete=false`, before the
larger collection completed. The helper also supplies the native-metadata reset
path used by the completed ten-plan manual replay.

The proposed data sources are successful current-policy foundations/trenches,
clean planner/manual strategies, and selected older-policy alternatives for
uncovered training cases. Older recordings must be replayed under current
physics; held-out success traces remain evaluation-only. The latest complete
historical-runtime panel is u35000, reported above. The old broad FF qualifies
at 140/384 foundations under the corrected runtime preceding this boundary fix;
139 of those successes overlap the u20000 student's successes and only one is
old-only.
These pre-boundary-fix measurements support using the current policy as the
primary source, without assuming that its training-map trajectories are all
efficient or that an old specialist adds no useful alternatives.

Foundation retention must be measured directly. Shared encoder updates from
trench imitation can alter foundation actions and critic representations even
when PPO continues on the full bank. The implemented sampler now draws by
explicit group weight, then condition, source, episode and eligible step. The
selected foundation/ordinary-trench/expert weights are 50%/20%/30%, independent
of the archive's episode proportions and the broad live PPO reset mixture.
Coverage, family completion, accepted material and common-success retained-pose
metrics should determine later expansion.

With 1,280 rehearsal episodes plus the ten manual/planner episodes, naive uniform
episode sampling would give the clean plans only `10 / 1290 = 0.775%` of draws;
the two junction plans together would receive `2 / 1290 = 0.155%`. Larger ordinary
rehearsal coverage therefore does not by itself ensure meaningful exposure to
the new junction strategies. The new explicit mixture addresses this sampling
problem; it does not create additional expert source diversity.

The small bank would receive 8,192 sampled labels per production PPO update at
four devices, batch size 32, 32 minibatches and two epochs: 20.48M draws over a
2,500-update fade. Repetition is not new coverage, and the auxiliary mean loss
is not automatically diluted by the larger online rollout count. This is why
the ten-plan recipe is retained only as an implementation smoke.

Storage must also be measured before scaling. Current raw observations occupy
about 45,508 bytes per transition; 1,280 episodes averaging 200 actions would take
about 10.85 GiB per raw bank copy. The collector keeps only one rollout batch in
memory and streams accepted episode files into the combined NPZ. PPO still
loads the complete bank and replicates it once per device, with additional host,
model and rollout memory. Dynamic JAX arguments avoid embedding the bank in
compiled constants and the bank is released after the fade; this does not
establish memory headroom for the full target on either platform.

## Independent review

The user-requested Oracle review is complete in the live
[review conversation](https://chatgpt.com/c/6aad58c3-5e78-83eb-b101-0be9ceb23fa2),
session `terra-scalable-imitation-20260918`, using GPT6Pro. The substantive
[saved review](/home/lorenzo/moleworks/.artifacts/terra_trench_failures_20260918/boundary_fix/oracle_review_result.md)
ends with the verified `TERRA_SCALABLE_IMITATION_ORACLE_20260918_COMPLETE`
sentinel. Its 17-file source bundle supports an implementation review; Oracle
did not independently replay the plans or measure production performance.

Accept the proposed direction: retain broad live PPO, native optimizer state
and zero added efficiency costs, with fading actor guidance and an
auxiliary-free retention check. Use cached parent action distributions for
foundation retention and executed-action cross-entropy for new hard plans.
Add explicit group/condition/source sampling so ordinary successes cannot
bury the junction instruction. Verified recovery episodes should retain their
complete history and elapsed time, with imitation restricted to a corrective
suffix through an eligible-step mask. These extensions are now implemented
with focused CPU tests; actual-bank four-GPU qualification remains pending.
Foundation rehearsal cannot guarantee retention through the shared encoder.

Oracle proposes the following initial active bank, separate from the larger
collection archive:

| Group | Accepted plans | Auxiliary weight |
| --- | ---: | ---: |
| Foundation rehearsal | 200 | 50% |
| Ordinary trench rehearsal | 120 | 20% |
| Hard planner/manual instruction | 48 | 25% |
| Verified recovery instruction | 16 | 5% |
| Total | 384 | 100% |

Those dataset counts and four-way weights remain engineering proposals rather
than the selected bank. The 750-update fade and 750-update auxiliary-free hold
have been selected for one bounded screen; at 32,768 transitions per update
each phase is 24.576M transitions. All auxiliary terms, including cached
foundation KL, fade. The proposed coverage includes 24 hard source geometries;
the two original manual junctions do not meet it. More current-policy successes
alone cannot supply a missing branch-order strategy.

Native float16-metadata replay of the ten manual plans has since passed. The
additional geometry audit found no discrete mismatch in the specified native
trench, footprint and wheeled probes, including the exact 256-environment
per-device batch; its limited scope is recorded above. The local corrected
u35000 evaluation is complete. Production-layout memory/finite-update checks,
fresh CSCS baseline and completed learning evaluations remain necessary before
an imitation-benefit claim. Use fresh evaluation output directories and the existing
nine-manifest exclusions for this bounded work; the review's additional hash
and receipt frameworks are not adopted.

The review's matched PPO-only control is advisory and has not been submitted.
The user selected an economical combined continuation; no matched-control
allocation or automatic promotion is planned.
Held-out completion, paired lost successes and retained-pose behavior after
the fade must establish benefit. The review cites
[DAPG](https://arxiv.org/abs/1709.10087) and
[CLEAR](https://arxiv.org/abs/1811.11682) as relevant precedents, not validation
of these Terra-specific counts, weights or retention outcomes.

## Clean manual/planner acceptance

- Use original training maps, metadata, ordinary full resets, and recorded reset
  seeds. Exclude source and scenario identities in the development, promotion,
  and sealed evaluation manifests.
- Execute every action with the active chassis, finite-section trench gate,
  disposal, and soil rules. Do not teleport, modify terrain, reset episode age,
  or extend the horizon.
- Require exact task completion, fully accepted disposal, no off-zone residue,
  and empty load. Check transition mass conservation, immutable target and
  obstacles, and no nonzero material under the chassis at every transition.
- Reject ineffective actions and soil relifting from the clean training
  examples. The straight pilot uses forward fresh digs. The manual junction
  examples additionally use documented 30-degree and longitudinal rearward
  digs. Cabin orientation for dumping remains unrestricted.
- Save raw pre-action observations, pre-action history, and the selected
  action. History begins at zero and stores the latest action at index zero.
  Regenerate history and remaining time by executing the actual clean plan.

Exact Terra completion does not require final egress. These demonstrations do
not certify a deployment exit route or optimality.

## Implemented PPO integration

The optional path adds a separate demonstration minibatch with actor guidance:

```text
row_loss = KL(cached_parent || pi) if foundation else -log pi(executed_action)
loss = PPO_loss + beta(global_transitions) * mean(row_loss)
```

Sampling implements group -> condition -> source -> episode -> eligible step;
group weights are applied once, through sampling. Longer plans do not receive
more total weight within a source. Recovery masks select labels without
discarding original history or elapsed time. Live PPO rollouts, GAE, old
action probabilities and value targets remain on-policy. Demonstrations create
no critic targets. Both paths use `obs_to_model_input`, and a separate sampling
RNG leaves live rollout/permutation RNG unchanged. Inputs require a declared
training split, contiguous complete episodes and checked pre-action history.

The coefficient has a linear fade in global transitions. Checkpoints retain
its origin and duration, so a Slurm restart does not restart imitation. Native
optimizer state, broad task sampling, released teacher schedules and zero added
efficiency costs are preserved. The extra forward pass stops after the
coefficient reaches zero. Imitation loss and demonstrated-action accuracy are
logged separately from PPO statistics, with group exposure and loss summaries.
After release the permanent inactive path has no imitation-loss measurements;
logged zero coefficient/active values are not evidence of teacher agreement.
The launcher evaluates the selected parent, fade end and post-fade hold, with
the retention stop below. It does not promote a policy or activate costs.

The shared encoder can still change the critic indirectly. A good demonstration
fit is not evidence of improved task completion. Judge the learned policy on
untouched fixed-panel maps, with foundation retention and trench completion
reported separately. Fourteen focused CPU tests cover zero-coefficient PPO/Adam
parity, finite mixed soft/hard losses, source-balanced sampling, label masks,
schedule restoration and dynamic bank passing. Five logging tests pass after
the independent review caught missing whitelist entries for the new group
metrics. The earlier native GPU smoke covers the original hard-label path;
it does not qualify the new grouped bank or establish policy improvement.

## Selected bounded continuation: stopped on trench retention at u35750

At the September 18 **20:01 UTC** check, CSCS job **4709420** had run 1:41:01
and stopped at u35750. Slurm FAILED/exit 1 records the intended retention-stop
exception, not an infrastructure fault. All exact four-GPU cuDNN/NCCL,
finite-first-update, native save/resume and fade-boundary checks passed. Parent
and u35750 evaluation statuses are PASS, with zero integrity failures across
the 608-case panel. Production checkpoints u35250, u35500 and u35750 are saved.
The 750-update hold did not start; there is no promotion or automatic retry.

The job used account `lterenzi`, project `d130`, one node/four GH200s and 64 CPUs,
with a six-hour wall limit. Its fresh campaign root is
`/ritom/scratch/cscs/lterenzi/terra-training/runs/terra-imitation-20260918`.
No local report was reused: both compared panels ran on the same corrected
runtime and CSCS hardware. The original submission receipt remains at
`boundary_fix/launch/cscs_submission.json` (16:51 UTC).

| Fixed cohort | Fresh u35000 parent | u35750 | Newly solved / lost |
| --- | ---: | ---: | ---: |
| Foundations | 369/384 (96.1%) | 383/384 (99.7%) | 14 / 0 |
| Trenches | 213/224 (95.1%) | 205/224 (91.5%) | 1 / 9 |
| Road trenches, included above | 30/32 | 29/32 | 1 / 2 |
| Complete panel | 582/608 | 588/608 | 15 / 9 |

`u35750.retention.json` returns STOP solely because the **net trench loss is
eight, above the allowed two**. No individual condition exceeds its two-case
net-loss tolerance. The total completion gain therefore does not qualify this
as a retained generalist at this checkpoint. This is a conservative 750-update
screen under the selected criteria, not evidence of plateau, saturation or
rejection of imitation in general. More training could change the outcome;
this bounded branch did not measure that counterfactual.

On the **369 foundation maps solved by both checkpoints**, retained inter-setup
distance rises 37.8407 to 38.7566 m (+2.4%); unique area per productive setup
falls 6.6175 to 6.5857 m² (-0.5%); fresh-workspace edge adjacency falls 86.02% to
85.51%; and lateral-digging score rises 0.4955 to 0.5253. There is no demonstrated
efficiency improvement. Across all 224 trenches, excavation falls 99.12% to
97.91%, accepted disposal falls 98.99% to 97.64%, and mean longest task-progress
stall rises 36.42 to 51.86 actions. Apparent travel/setup gains over all trench
episodes would be confounded by the additional incomplete episodes.

All nine lost trenches time out at 450 actions. Excavated fractions are 69.9%,
90.2%, 78.0%, 63.6%, 94.4%, 92.5%, 71.0%, 77.5% and 0%; their longest task stalls
span 209–450 actions, mostly at least 378. The parent already supplies successful
trajectories for these same cases under the corrected physics, so their loss
cannot be attributed to an unreachable bank. The sole remaining foundation
failure is slot 253 (`fnd-slab-side1-obj`), 49.3% dug with 379 ineffective actions.
These are failure outcomes, not a diagnosis of which training component caused
them: imitation versus ordinary PPO drift is not isolated by this combined run.

At the latest logged u35741, entropy is 0.24482, PPO KL 0.01384, and end-to-end
throughput about 15.1k transitions/s. These are training diagnostics, not evidence
against the measured retention failure. Added efficiency costs stayed zero.
Retrieved paired reports, evaluation status files and concise failed-case data
are under `boundary_fix/cscs_4709420_status_2001/`.

The run used the native u35000 model and Adam state, broad PPO, already released
online teachers and zero added efficiency costs. Cached parent foundation
action distributions supply retention targets; successful trench actions and
clean expert plans supply hard labels. The selected mixture is 50% foundation, 20%
ordinary trench and 30% expert. Expert collection and the final distinct-plan
count are recorded in `boundary_fix/active_demonstrations.json`. The assembled
active bank currently has **331 plans: 200 foundations, 119 ordinary trenches
and 12 experts**, with 27,613 transitions, 27,612 eligible labels, 299 distinct
sources and all 40 conditions. One ordinary trench scenario was replaced by
its expert version. The masked label remains in the complete trajectory/history.
The twelve expert plans cover six geometries: eight original straight disposal
variants, the two manual junctions and two new straight sources. The larger
1,280-plan archive is not the active bank.
Recovery masking is implemented, but no separate recovery group is selected
for this first screen.

`active_demonstrations.npz` is present at 10,376,666 compressed bytes, with
1,257,523,633 raw transition bytes; `.records.json` records the selected plans.
All nine evaluation manifests are excluded. Cached foundation distributions
come from u35000. The local grouped-bank GPU smoke now **PASSES**:
`boundary_fix/grouped_gpu_smoke/result.json` records native u35000 to u35001,
Adam 2,240,000 to 2,240,064, finite state and the complete 331-plan bank readback.
It uses 1x128 environments, or 4,096 transitions per update. This is a diagnostic
checkpoint and must never seed production. It verifies the grouped soft/hard
execution path, not the four-GPU layout, learning benefit or retention.

The two added straight source slots, 3175 and 3196, pass exact native replay in
49 total actions. Slot 3174 remains `UNVERIFIED_TIMEOUT` and is excluded. Four
locally tried network geometries remain unresolved; no new hard-junction
success has been established. The local first-eight-source search is frozen
for this bank after 8,014 action queries; a failed bounded search does not
establish infeasibility.

The single CSCS run requested four GH200 GPUs for at most six hours (24 GPU-hours
maximum). It kept 256 environments per device, 32 rollout steps, two PPO epochs
and 32 minibatches, with initial beta 0.01 and 16 auxiliary examples per device
per minibatch. It completed the 750-update fade to u35750, or 24.576M new live
transitions. The conditional 750-update hold to u36500 did not start because
retention failed. This bounded screen does not establish saturation; no control,
retry or extension is automatic.

Before production, `scripts/oracle_followup/run.sh` runs the existing CUDA
library/cuDNN-backward/NCCL preflight, followed by exact-layout finite native
update and save/resume checks. A separate disposable copy shifts only the
auxiliary origin so two updates cross the real fade-to-zero path; its model,
Adam clock, PPO layout and fade duration stay intact. Diagnostic children never
become production parents. All these allocation checks passed in 4709420.
Shell syntax, Python compilation and CPU dry configuration checks also pass.

The fresh campaign evaluation directory contains the new CSCS u35000 parent
panel and u35750 results with paired gains/losses, family/road/condition counts,
continuous material metrics and retained-work metrics on common successes.
The rule stops before the hold if net losses exceed four foundation cases, two trenches,
one road case or two cases in any condition. The same limits are reported at
the hold endpoint. These are engineering stop limits, not significance tests
or automatic promotion criteria. Failed/incomplete evaluation also stops the
bounded run. Seven pure threshold/identity checks pass; their synthetic
outcomes are not policy evidence.

The existing broad CSCS chain is **retained because the candidate fails
retention**. Job 4685246 timed out after 24 hours with u48750 preserved. Native
successor 4685248 was RUNNING about 27 minutes at the latest check, with u49250
saved at 21:59 CEST and about 17.3k end-to-end transitions/s; 4685249 remained
PENDING (Dependency). No old-chain cancellation or replacement promotion
occurred. The continuing run's latest checkpoint and lost-case material are
being retrieved for the next comparison.

Euler expert-search job **14555235 COMPLETED in 1:53:21 on eu-g4-004** as
`lterenzi`, using one RTX 3090 in `gpuhe.4h`, a three-hour wall limit, at most
7,200 search seconds and 400,000 action queries. CUDA/cuDNN backward passed;
NCCL all-reduce was correctly skipped with one GPU. It searched source ordinals
8–55: 23 straight and 25 junction candidates, excluding the locally tried
0–7 range. Existing Euler training lanes are untouched. This is training-only
expert generation, not a second learning arm. The completed collection reports
25 new plans from 25 sources: 23 straight and two compact T junctions, totaling
837 actions from 48 attempts and 44,701 action queries. Readback is complete.
The two Ts do not establish new multi-junction network solutions, and these
outputs were not in the completed 331-plan training bank. They have not been
merged. Combined with the existing 12 expert plans, the possible bank would
contain 37 plans from 31 sources: 27 straight, three T junctions and one network.
Only four junction sources are available; the coverage gap remains.

The next recommended check is September 18 at **20:45 UTC** for the continuing
old run's u50000 fixed panel. It is not scheduled; no watcher is configured.

The remote root is
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra-imitation-20260918`.
Frozen launch archives under `boundary_fix/launch/` record Terra `6a0d7bddd`
plus the local geometry patch and baselines `f550c048` plus grouped-imitation
and launcher patches. These are snapshots with explicit uncommitted changes,
not claims that the patches are present in those commit objects.

## Decision refinement: allow continued learning, then gradual efficiency

Keep the already-running broad continuation through its u50000 fixed-panel
evaluation; its latest verified checkpoint is u49250. The historical old-runtime
trench results still improved between u20000 and u35000: 195/224 to 209/224
successes and 96.34% to 98.86% mean excavation. This is evidence of continued
learning in that run, not a matched current-runtime control for the stopped
imitation branch. Neither a 750-update retention stop nor reaching u50000
establishes a learning plateau.

The next learning-stage priority remains **gradual efficiency after completion
competence**, rather than another initial-cost or control sweep. A proposed
bounded recipe is a linear **40.96M-transition ramp**, followed by a
**40.96M-transition hold**. At the established four devices, 256 environments
per device and 32 rollout steps, each phase is 1,250 updates. Compare fixed-panel
completion, retained travel, unique area per productive setup and workspace
continuity at milestones. Stage entry depends on observed performance; the
parent and exact coefficients are not yet selected. No such job is configured
or launched, and this proposal adds no new entry threshold.

The deployment target is the sequence of effective work poses, including dump
and relift poses, because the navigation stack reconnects those poses. Keep
the lateral orientation penalty limited to **fresh digging**, with no lateral
charge for dumping or moving loose soil. Raw action travel remains a diagnostic,
not a substitute for retained work-pose efficiency.

There is an implementation gap before a retained-cost curriculum is ready.
`terra/state.py:_reward_v2_retained_work_costs` implements setup, inter-work
straight-line distance and heading costs; its ledger retains dump/relift events.
However, `utils/behavior_cost_ramp.py:COST_KEYS` currently contains only
`lateral_dig_cost`, `base_travel_cost` and `base_turn_cost`. Retained-work costs
are not ramped, and the preceding retained work pose is not exposed in policy
observations. The proposed stage therefore requires preparatory ramp and
observation work; implemented cost terms alone do not establish launch readiness.

Recover the existing delayed-cost comparison outputs now that storage is
accessible before allocating compute to repeat them. Their recovered results
are not yet recorded here. This preserves the completion-first plan while
separating an economical retention screen from conclusions about saturation.

## Local artifacts

The original failure gallery and collector are under
`/home/lorenzo/moleworks/.artifacts/terra_trench_failures_20260918/`:

- `gallery.html`: six failed policy rollouts and four verified alternatives.
- `collect_training_demos.py`: explicit cardinal retreat planner and exporter.
- `training_demonstrations_pilot/`: eight accepted observation/action episodes,
  `demonstrations.npz`, and `summary.json`.
- `assemble_training_demos.py`: readback checks and assembly of fully saved
  episodes from the bounded collection.
- `manual_junctions/`: manually selected junction plans, independent reset
  replays, pre-action demonstrations, and GIFs. Its `workbench.py` routes between
  assistant-selected work poses; it does not select a complete plan itself.
- `manual_junctions/gallery.html`: two clean training plans and two separately
  labeled held-out recovery witnesses, with complete GIFs and work-pose CSVs.
- `manual_junctions/sector_boundary_mismatch/`: serialized pre-DO state and a
  reproduced executable-observation / actual-DO mismatch.
- `boundary_fix/`: fixed-geometry tests, saved-state parity, ten-plan replay and
  native diagnostic GPU-training receipts.
- `boundary_fix/remaining_geometry_audit.json` and
  `remaining_geometry_256.json`: no discrete mismatch in the specified geometry
  probes, including the exact per-device production batch shape.
- `boundary_fix/paired_u20000_u35000_historical_runtime.json`: full-panel
  improvement and paired gains/losses before the local boundary fix.
- `boundary_fix/u35000_corrected_runtime.json` and
  `paired_u35000_geometry_fix.json`: completed local corrected-runtime panel
  and pairing against the historical CSCS panel; hardware differs too.
- `boundary_fix/native_manual_replay/`: completed native float16-metadata replay
  of all ten plans, updated observation differences and
  `manual_demonstrations_native.npz` (449 actions).
- `boundary_fix/collect_policy_rehearsal.py` and `POLICY_REHEARSAL.md`: broader
  training-only collector and its stated budget, filtering and memory limits.
- `boundary_fix/policy_rehearsal_smoke/`: completed bounded native collection;
  16 successful training episodes, 1,290 actions and the exported rehearsal NPZ.
- `boundary_fix/policy_rehearsal_v1/`: completed 1,280-episode collection across
  all 40 conditions, final summary and exported rehearsal NPZ.
- `boundary_fix/active_demonstrations.npz`, `.json` and `.records.json`: assembled
  331-plan selected bank, mixture/count summary and per-plan records.
- `boundary_fix/grouped_gpu_smoke/result.json`: passed local grouped-bank
  diagnostic u35000 to u35001, Adam 2,240,064; never a production parent.
- `boundary_fix/cscs_4709420_status_2001/`: completed same-runtime CSCS parent
  and u35750 panels, PASS evaluation receipts, STOP retention report and
  `lost_trench_details.json`; no post-fade hold or promoted generalist.
- `boundary_fix/expert_expansion/local_first8_v1/summary.json`: two additional
  qualified straight sources and explicit unresolved/excluded search outcomes.
- `boundary_fix/launch/`: frozen source archives, patch/source identities,
  Euler/CSCS submission receipts and launch scripts. CSCS job 4709420 submission
  does not establish allocation qualification or production startup.
- `boundary_fix/oracle_review/`: independent review prompt and submitted bundle.
- `boundary_fix/oracle_review_result.md`: completed substantive Oracle review
  and its proposed active-bank, retention and qualification changes.
- `straight_length_clearance.json`: endpoint reach diagnostic.

For case 434, shortening the affected end by one cell (0.5714 m) admits a
centered finishing pose under current reach and chassis bounds. That diagnostic
does not establish a complete shortened-map trajectory or final egress. Future
generation should check end working clearance; reducing maximum trench length
alone does not account for where the trench is placed.
