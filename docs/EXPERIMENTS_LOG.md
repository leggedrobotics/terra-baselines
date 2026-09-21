# Experiments — completed log

## September 21, 21:05 UTC: control reaches110k; final panel running

CSCS4729577 remains RUNNING on nid006413, fourGH200s, elapsed3h44m56s.
Control saved its nativeu110000 checkpoint after completing all5,000 updates;
the final fixed608 panel is in progress (150/450 steps observed). No completed
u110000 score is available yet. Update throughput was about17.5k transitions/s.
The penalty arm has passed native first-update and resume qualification through
u105002 and will start automatically after control evaluation. Both arms'
initial model/Adam/RNG/environment receipts match. No production penalty
result exists yet, and no new job was submitted.

Completed controlu107500 remains PASS:381/384 foundations (3gains/3losses),
212/224 trenches (3gains/2losses),31/32roads (no churn), relative tou105000.
On378 common foundation successes, area/productive setup improves1.68%,
productive setups fall2.10%, retained travel falls0.14%, adjacency rises0.72pp,
and lateral score falls4.63%. On209 common trench successes, retained travel
rises0.83% and area/setup falls0.42%. These are zero-added-cost control results,
not evidence for the still-pending penalty treatment.

Separate Euler geometry result remains verified from completed artifacts:
straight32 improves2→26 successes with the identicalu109250 policy; full608
changes383→382foundations,208→211trenches and29→29roads. The active CSCS
comparison keeps its original frozen geometry in both arms.

Evidence: `.artifacts/terra_efficiency_p25_20260921/status_20260921_2105/`.
Recommended next check23:15UTC for treatment progress/evaluation; allocation
endsSeptember22 at01:20UTC. No monitor is scheduled.

## September 21, 20:14 UTC: latest-policy geometry comparison completed

Euler14791590 completed in2h33m46s. Finite native resume and integrity pass.
With identical u109250 weights, straight32 improves2→26/32 after the float32
geometry/map-edge/tolerance correction (26 gains,2 losses). Full608 retention
passes: foundations383→382/384, trenches208→211/224, roads29→29/32.
Four trench gains, one trench loss and one foundation loss remain explicit;
six straight starts still fail. Full-panel efficiency changes are small.
See the test-time-compute note and
`.artifacts/terra_latest_geometry_20260921/results/`.

## September 21, 17:20 UTC: retained-work comparison submitted; queued for resources

CSCS **4729577** is **PENDING/Resources** under `lterenzi`, accountd130:
one node, four GH200 GPUs,64CPU cores, eight-hour ceiling. This is the
bounded control versus25% retained-work stage from native **u105000**;
2,500 updates of linear ramp plus2,500 of hold per arm, run sequentially.
No new training update is claimed yet. Slurm's start estimate is17:26UTC,
which can change.

Focused CPU checks, actual-parent migration parity and independent review
pass. The local1x32 GPU smoke completed u105001, saved/reloaded finite
model/Adam, and preserved the6,720,000→6,720,064 optimizer clock. It exposed
and then verified a fix for three missing retained-cost scalar names in the
W&B logging contract. This diagnostic checkpoint is not a production parent.

Inside the allocation, fresh four-GPU convolution/NCCL checks and real-parent
GPU output parity precede independent native4x256 first-update/resume checks
for **both** arms. Their initial model/Adam/RNG/environment receipts must match
before production. Production uses the qualified u105002 states; the ramp
origin remains105000. Checkpoint/evaluation milestones107500/110000 compare
to the frozen105k panel (381/384F,211/224T,31/32roads), with explicit retention
stops. Both teachers remain released, no imitation, raw navigation costs zero.
W&B runs offline; cluster first-update/checkpoint evidence remains pending.

The separate September21 metadata/map-edge geometry changes are excluded from
both arms. Source and rationale: [retained efficiency](research/RETAINED_EFFICIENCY_20260921.md).
Local evidence: `.artifacts/terra_efficiency_p25_20260921/`; remote root:
`/ritom/scratch/cscs/lterenzi/terra-training/runs/terra-efficiency-p25-20260921/`.

Recommended next check **18:15UTC (20:15CEST)**: verify allocation, both native
qualifications and the first production checkpoint. No monitor is scheduled.

## September 21, 17:10 UTC: u110000 reaches retention stop; bounded efficiency stage prepared

CSCS4725717 reached u110000 and exited after6h44m52s with the policy-retention
stop:381/384 foundations,206/224 trenches,29/32 roads; trench net loss3>2
against its u100000 parent. The stronger u105000 remains381/384,211/224,31/32.
This is a completed evaluation, not an infrastructure failure or convergence claim.

Prepare the authorized control versus25%-strength retained-work penalty stage
from native u105000, with a2500-update linear ramp then2500-update hold.
Both arms receive zero-initialized previous-work-pose context; Adam and native
clocks are preserved. Raw navigation costs stay zero. One four-GPU CSCS node
runs both arms sequentially after independent finite first-update/resume checks.
Runtime stays frozen to the u105000 panel; separate local metadata/map-edge
corrections are excluded. No new allocation is submitted at this preparation
entry. See [retained efficiency](research/RETAINED_EFFICIENCY_20260921.md).
Evidence: `.artifacts/terra_efficiency_p25_20260921/`.

## September 21, 15:56 UTC: geometry diagnostic qualification and submission

Latest-parent u109250 passed actual Euler-Python resume configuration checks;
all evaluation/training arrays and teacher files are staged without broken
links. Independent launch review passed after increasing the full608 timeout
to 3300 seconds and retaining Slurm's assigned CUDA device. The local full
network failed with `CUDNN_STATUS_EXECUTION_FAILED`, despite a successful small
CUDA probe and three GPU geometry tests. No local policy result was produced.

Euler job14791590 requests one RTX3090 for at most three hours and is initially
PENDING/Priority. It checks one disposable PPO update, then the unchanged
u109250 policy under previous/corrected geometry on straight32 and full608.
Startup, policy results and any subsequent CSCS continuation remain pending;
no new CSCS allocation was submitted. Artifacts:
`.artifacts/terra_latest_geometry_20260921/`.

## September21, 15:22 UTC: u105000 retention passes; training near108560

CSCS4725717 RUNNING5h38m; latest saved milestoneu108500. Live training around
u108559, about1.87s/update. Completed corrected-runtime u105000 panel PASS:
381/384foundations,211/224trenches,31/32roads versus parent378/209/30.
Foundations gain6/lose3 and trenches gain4/lose2; per-condition checks pass.
Foundation mean dug99.775%, accepted99.578%; trench dug98.610%, accepted98.536%.
On common successful foundations retained inter-setup distance falls1.55%,
area/productive setup falls0.60%, adjacency rises0.15percentage points. Trench
retained distance falls0.86%, workspace yield essentiallyunchanged. Thus modest
completion gains, no established major efficiency gain; no costs/imitation active.
Next/final panelu110000 is pending. At current throughput ~45minutes of updates
remain, plus evaluation. Recommend nextcheck17:00UTC, notscheduled; no newjobs.
Evidence: `.artifacts/terra_overnight_20260920/status_20260921_1228/u105000.retention.json`.

## September21, 12:28 UTC: corrected-runtime continuation training; two panels pass

CSCS4725717 RUNNING on nid005700 for2h44m. Production reached approximately
u103669; latest saved milestoneu103500 (35,345,010bytes). Observed end-to-end
throughput17.4ktransitions/s, about1.9s/update. Checksum configuration path and
four-GH200 CUDA/cuDNN/NCCL checks passed. No final checkpoint claim yet.

Corrected-runtime panels (foundations/trenches/road): parentu100000378/209/30,
u101000381/209/31, u102500381/208/30, denominators384/224/32. Both retention
checks PASS, including per-condition tolerance. At102500 foundations gain6/lost3;
trenches gain4/lost5. Thus the foundation gain is retained, trench exact completion
is flat/slightly lower, and individual solved maps still change.

Continuous mean dug/accepted at102500: foundations99.606%/99.498% versus
parent99.405%/99.320%; trenches98.749%/98.557% versus98.672%/98.216%.
Efficiency is not clearly improving: on375 common successful foundations,
retained inter-setup distance31.460→32.832m (+4.36%), unique area/productive
setup7.039→6.847m² (-2.72%), edge adjacency88.576%→86.660%. On204 common trench
successes retained distance35.419→35.262m and workspace yield2.817→2.820m².
No efficiency costs or imitation are active. Continue the existing bounded
run; next required panelu105000 should clarify whether completion/behavior
changes persist. Recommend next result check near14:00UTC, not scheduled.
Evidence: `.artifacts/terra_overnight_20260920/status_20260921_1228/`.

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

## September 20 evening: overnight submission and CPU/runtime checks

Actual-u100000 native continuation dry-run passes at4virtualCPU devices, preserving
Adam and PPO layout with zero costs and no imitation. Calibration dynamic-weight
routing and Euler post-warmup budget-clock corrections pass focused checks and
independent review; native GPU effects remain unverified. CSCS4721607 started
on fourGH200, passed cuDNN/NCCL, and entered corrected-runtime u100000 evaluation.
Euler14730608 passed its3090 runtime checks and is compiling; dependent collection
14730609 remains blocked until native replay succeeds. No new training result is
claimed. See the running ledger for the final bounded110k plan, which supersedes
the earlier provisional old-bank imitation retry.

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

## 2026-09-20: Euler scratch preservation and bounded collection submission

Six scratch checkpoints passed native load and finite model/Adam validation and
were copied to project storage. Both scratch lanes and their successors were
cancelled after preservation; both finetuning lanes continue. CPU demonstration
checks pass for 27 episodes and 110 rejected mutations; source exclusions yield
743 eligible new hard layouts and disjoint collection lanes. Qualification
14671834 and dependent array 14671835 are submitted, native replay unverified; no new
policy result or PPO run. See the running ledger and
`.artifacts/terra_trench_recovery_20260920/launch/submission_retry.json`.

## 2026-09-19, 13:17 UTC: revised imitation preparation complete

Preparation **4711914 COMPLETED**, exit 0, in 22m06s. Four-GH200 CUDA,
convolution-backward and NCCL checks pass. Fresh corrected-runtime u75000
evaluation completes 375/384 foundations, 206/224 trenches and 30/32 road
cases, with zero integrity failures. The same checkpoint's original-runtime
379/384 and 208/224 counts are not the new comparison baseline.

The current-parent retention collection obtained 320 selected successes from
336 attempts (334 successful), covering all 40 conditions at eight episodes
each, with 27,240 transitions. Both families cache soft parent targets. All
held-out and reserved instruction sources are excluded. Independent hard
instruction now covers 25 geometries: seven networks, three segmented bends
and fifteen T junctions. Its 27 episodes contain 2,832 transitions and 2,727
supervised actions; 105 prefix actions remain masked. Twenty-one source layouts
are new and four carry previous native replay acceptance. All new full and
corrective plans pass native exact-completion and integrity checks.

Twenty-nine focused CPU trainer/evaluation tests pass. Hard instruction now
samples source before condition; both ordinary families use parent KL. Native
Adam shadow calibration has a passing CPU gradient/clock fixture, not yet a
GPU calibration result. No new environment or reward rule is added here.

Matched comparison **4712295** is submitted and RUNNING on one additional
four-GH200 node, six-hour maximum. It includes mandatory actual-GPU calibration
and independent finite first-update/save-resume checks before two sequential
1,000-update arms from the same original parent. Guided fade/hold is 500/500;
PPO-only receives no imitation. Full fixed panels at u75500/u76000 use the
shared corrected-runtime parent and strict family/condition retention stops.
No new PPO update or learned-policy gain is yet verified. Costs remain zero.

Evidence: `.artifacts/terra_imitation_revision_20260919/`, including
`preparation/`, `instruction/qualified/`, `implementation/`, `calibration/`
and `launch/`. Live follow-up is in `EXPERIMENTS_RUNNING.md`.

## 2026-09-19, 05:48 UTC: u50000 panel complete; delayed results recovered

CSCS 4685248 is RUNNING on nid006516, elapsed 10h12m, u67000 saved/logs near
u67068 at about 17.1k end-to-end transitions/s; 4685249 is dependency-pending.
Original-runtime u35000/u50000 identities match and evaluation passes: foundations
374 to 377/384 (8 gains/5 losses), trenches 209 to 209/224 (4/4), included road
subset 29 to 30/32 (2/1). Foundation dug/accepted material improves to
99.529%/99.528%; trench values are 98.527%/98.396%. Common-success foundation
retained travel improves 3.6%, area/setup 0.8%, and adjacency 1.071 percentage
points. Integrity failures are zero. Costs stay zero; next panel is u75000.

Evaluation-only recovery **4709744 COMPLETED in 19 minutes, exit 0**, using the
original source, image, saved policies and 64-map protocol. The original job
4675576 had finished training; a 600-second evaluation timeout caused exit 124.
Recovery repeats evaluation only, not the training/control run. Fresh parent
64/64 and all five validation checks pass. Both arms score 63/64 at u22500 and
62/64 at u25000, so the completion gate fails. Penalty versus control on the
60 shared final successes: retained travel -10.3%, area/setup +4.3%, setups
-4.0%, lateral score -22.3%, adjacency +0.21 pp. This is measured efficiency
improvement without stage acceptance; equal completion counts do not identify
penalties as the cause of the parent-relative losses.

Evidence is under `.artifacts/terra_training_status_20260919/`, in
`generalist/paired_35000_50000.json` and `delayed_penalty/analysis.md`/`analysis.json`.
The imitation candidate remains stopped. No new job, automatic penalty switch
or policy promotion was made.

## 2026-09-18: retain learning horizon; prepare delayed efficiency

User steering clarified the next decision: the 750-update imitation retention
STOP is a conservative screen, not a saturation result or rejection of all
imitation. The existing broad run continues to its u50000 fixed-panel check
(last verified checkpoint u49250). Its historical old-runtime trench trend,
195/224 to 209/224 successes and 96.34% to 98.86% dug between u20000 and u35000,
shows continued improvement; it is not a matched current-runtime control.

Prioritize gradual efficiency after a mature checkpoint qualifies by measured
performance. Proposed budget: linear 40.96M-transition ramp, then 40.96M hold
(1,250 + 1,250 updates at 4x256x32), evaluating completion and retained travel,
workspace yield and continuity at milestones. Parent and exact coefficients
remain unselected; no new job is configured or launched. No new initial-cost
or control sweep is proposed, and u50000 is not a presumed plateau.

The deployment objective uses effective work poses, including dump/relift;
lateral orientation penalties apply to fresh excavation only. Code audit finds
retained-work costs implemented, but absent from the existing lateral/raw travel/
raw turn ramp, with previous retained pose not observed. Those gaps require
preparation before a retained-cost stage is ready. Existing delayed-comparison
outputs are being recovered rather than rerunning completed compute.

Euler's 25 new plans have completed readback. They have not been merged; adding
them to the 12 existing expert plans would yield 37 plans from 31 sources:
27 straight, three T junctions and one network. Four junction sources remain
limited coverage. No training or scheduler change accompanies this refinement.

## 2026-09-18, 20:01 UTC: imitation stopped on trench retention; Euler collection completed

CSCS **4709420** ran 1:41:01 and reached u35750. Slurm FAILED/exit 1 is the
intended retention-stop exception, not an infrastructure fault. Exact four-GPU
cuDNN/NCCL, finite update, native save/resume and fade-boundary qualification
passed. Fresh parent and u35750 fixed panels both pass validation, with zero
integrity failures across all 608 cases.

On the same corrected runtime and CSCS hardware, foundations improve
369/384 to **383/384** (14 gained, none lost), while trenches decline
213/224 to **205/224** (one gained, nine lost). The road subset declines
30/32 to 29/32 (one gained, two lost). Overall 582/608 to 588/608 does not pass
the family gate: eight net lost trenches exceeds the allowed two. No hold
started, no policy is promoted, and no retry or extra job is automatic.
Production checkpoints u35250, u35500 and u35750 remain available; costs stayed
zero. At u35741, entropy is 0.24482, PPO KL 0.01384 and end-to-end throughput
about 15.1k transitions/s.

For the 369 common foundation successes, retained distance is 37.8407 to
38.7566 m (+2.4%), area/setup 6.6175 to 6.5857 m² (-0.5%), adjacency 86.02% to
85.51% and lateral score 0.4955 to 0.5253: no efficiency gain. Across all
trenches, excavation declines 99.12% to 97.91%, accepted disposal 98.99% to
97.64%, and mean longest task stall increases 36.42 to 51.86 actions. All nine
lost trenches time out at 450 actions; their parent successes under identical
physics provide executable witnesses. This does not isolate imitation from
ordinary PPO drift. The remaining foundation failure is slot 253,
`fnd-slab-side1-obj`, 49.3% dug and 379 ineffective actions.

The original broad continuation is retained because the candidate fails
retention. Job 4685246 timed out at 24 hours with u48750 preserved; 4685248 is
RUNNING from that native parent, with u49250 saved at 21:59 CEST and about
17.3k end-to-end transitions/s. Job 4685249 remains dependency-pending.

Euler **14555235 COMPLETED in 1:53:21**, reporting 25 plans from 25 new sources,
23 straight and two compact T junctions, 837 actions across 48 attempts and
44,701 action queries. Retrieval and readback subsequently completed. This is added
training data, not performance evidence or a verified multi-junction-network
solution. The new plans were not used by the completed imitation run.

Evidence: `.artifacts/terra_trench_failures_20260918/boundary_fix/`
`cscs_4709420_status_2001/`, including `u35750.retention.json`, both validated
fixed panels and `lost_trench_details.json`. Recommended next old-run check:
September 18 at 20:45 UTC for the u50000 fixed panel. No scheduled watcher exists.

## 2026-09-18: Euler preflight passed; bounded imitation run submitted

CSCS job **4709420** was submitted at 16:51 UTC, requesting one node/four GH200s
for six hours under account `lterenzi`, project `d130`. The latest Slurm snapshot
is **PENDING (Priority)**, no dependency, one node/four GPUs/64 CPUs. The estimated
start is September 18 at 20:25 CEST (18:25 UTC), subject to scheduler changes.
Exact allocation qualification and production startup remain unverified. Its fresh campaign is
`/ritom/scratch/cscs/lterenzi/terra-training/runs/terra-imitation-20260918`, with
no local evaluation report reused. The completed submission receipt is
`boundary_fix/launch/cscs_submission.json` under the artifact root below.

Euler job **14555235** was submitted at 16:40 UTC and subsequently verified
RUNNING on `eu-g4-004` as `lterenzi`: one RTX 3090, `gpuhe.4h`, three-hour wall
limit, 7,200-second search budget and 400,000 action-query cap. CUDA/cuDNN
backward passed; NCCL was skipped because this is a one-GPU job. Search covers
the 48 deterministic source ordinals 8–55 (23 straight, 25 junction), distinct
from the local first eight. Existing Euler training lanes were not disturbed.
Expert generation is **STARTUP_UNVERIFIED**, distinct from the passed preflight.
At 12m54s elapsed, model initialization was complete, but no WARMUP, SOURCE,
BEST or RESULT record or accepted expert plan was present. Remote root:
`/cluster/scratch/lterenzi/codex_terra_edge_runs/terra-imitation-20260918`.

The local first-eight-source search accepted two additional straight sources,
slots 3175 and 3196, with exact native replay in 49 total actions. Slot 3174
remains `UNVERIFIED_TIMEOUT` and is excluded. Four network attempts remain
unresolved, with no additional hard-junction success. This is bounded search
evidence, not a feasibility proof.

The active bank now contains **331 plans: 200 foundations, 119 ordinary
trenches and 12 experts from six expert geometries**. Its 27,613 transitions
include 27,612 eligible labels and cover 299 source IDs and all 40 conditions.
One ordinary trench scenario is
replaced by its expert plan. The explicit mixture is 50%/20%/30%; foundation
targets are cached u35000 distributions, and trench/expert targets are executed
actions. Nine evaluation manifests are excluded. The complete active NPZ is
10,376,666 bytes compressed, with 1,257,523,633 raw transition bytes; it is a
selection from the broader archive plus clean expert plans, not the entire
1,280-plan archive.

The grouped-bank local GPU smoke subsequently **PASSES** one native update at
1x128: u35000 to u35001, Adam 2,240,000 to 2,240,064, finite state and final
331-plan bank readback. This diagnostic checkpoint must never seed production.
`grouped_gpu_smoke/result.json` is the completed receipt. Four-GPU qualification
remains outstanding. The submitted single six-hour/four-GH200 run is bounded at u36500,
with a 750-update auxiliary fade from u35000 and a 750-update hold. Beta is
0.01, auxiliary batch is 16/device/minibatch, all added costs remain zero and
online teachers remain released. Exact four-GPU qualification and a fresh
same-hardware parent panel remain outstanding. The retention stop uses net
losses greater than four foundations, two trenches, one road case or two in
any condition; it does not promote policies or extend the run automatically.

The local corrected u35000 panel completes 369/384 foundations, 213/224
trenches and 30/32 road cases. The historical CSCS u35000 panel completes
374/384, 209/224 and 29/32. Hardware differs alongside the runtime correction;
the new CSCS campaign will establish its own baseline. The existing broad
CSCS chain 4685246/4685248/4685249 is untouched. Retirement is pending, not
automatic, and requires verified new production progress and checkpoint saving,
preserving its latest complete checkpoint.

Recommended checks on September 18: Euler's first native expert output at
17:40 UTC; CSCS allocation qualification and first production checkpoint at
18:55 UTC. These are recommendations only; no scheduled check or watcher exists.

Evidence root: `.artifacts/terra_trench_failures_20260918/boundary_fix/`.
`launch/euler_submission.json` records the submission; live allocation and
preflight are verified, while generation startup remains unverified.
`launch/source_revisions.json`, source archives
and patches record Terra `6a0d7bddd` and baselines `f550c048` with the declared
geometry/imitation/launcher modifications. Bank details are in
`active_demonstrations.json` and `.records.json`; native extra-plan evidence is
in `expert_expansion/local_first8_v1/summary.json`.

## 2026-09-18 08:26 CEST: long continuation healthy; broad foundations reach 95.6%

CSCS4685246 is RUNNING on nid006532 with four GH200s, from September17
21:23:31 CEST to its scheduled September18 21:23:31 wall limit. Training has
advanced from the native u7500 parent to approximately u25158, with u25000 /
Adam1600000 downloaded and validated finite. Transition integrity records zero
mass residual, obstacle mutation and target mutation. CUDA/NCCL pass, warmed
throughput is about17k transitions/s, and both successors remain dependency-held.

The fixed608panels at u10000 and u20000 both pass. Foundations rise from
301/384 to367/384 (95.6%), after the parent u7500 had322/384. Trenches remain
195/224 (87.1%), compared with193/224 at the parent. Road trenches rise25/32
to27/32. Total completion is496/608 then562/608, versus515/608 at the parent.
The10k-to20k interval gains73/loses7 foundation successes and gains11/loses11
trench successes. Accepted disposal improves91.01% to97.89% for foundations,
but slips96.16% to96.03% for trenches. This is broad foundation improvement
with unfinished trench progress, not a completed efficiency qualification.

Independent comparison verifies matching608episode identities, reset receipts,
treatment fingerprint and evaluation contracts, with zero integrity/nonfinite/
mass errors. Relative to u7500, foundation excavation rises93.70% to98.13% and
off-zone residual falls0.68% to0.155%. On313 common successful foundations,
retained inter-setup travel improves10.6%, workspace yield2.6%, edge adjacency
82.09% to84.82%, and lateral score13.7%. Trench retained travel rises1.0% and
workspace yield falls1.1% on its common-success cohort, so behavior gains are
primarily foundational. Full paired detail is in the local analysis artifact.

Both teacher coefficients are zero at the native u25000 checkpoint; the log
confirms teacher-free execution from u20000. All six added costs remain zero
in trainer and environment. Entropy0.237, approximate PPO KL0.0124, clip0.0648;
all saved rollout/gradient/model diagnostics are finite. Fixed evaluation runs
successfully beside the paused trainer and training resumes afterward.

The repeated efficiency-readiness result remains false. Continue the authorized
run unchanged toward u100000. Next meaningful report is u35000, estimated early
afternoon; recommended manual check14:15 CEST/12:15 UTC, not scheduled. Evidence:
`.artifacts/terra_oracle_long_20260917/status_20260918_morning/` and timestamped
scheduler status in the campaign artifact root.

## 2026-09-17 18:41 CEST: long continuation still waiting for priority

Live scheduler refresh confirms4685246 remains PENDING(Priority), with4685248
and4685249 PENDING(Dependency). No node, training process, new checkpoint or
evaluation exists yet. Slurm's provisional start estimate moved to September18
00:24 CEST. No failure or configuration change is reported. Next recommended
manual startup check: September18 00:45 CEST (September17 22:45 UTC), not scheduled.
Evidence: `.artifacts/terra_oracle_long_20260917/latest_status.json`.

## 2026-09-17 16:55 CEST: long native continuation submitted

Lorenzo approved continuing u7500 toward u100000 on one node/four GH200s.
Submitted sequential CSCS jobs4685246,4685248,4685249, each with a24-hour cap,
using native resume and one absolute final target. At16:55 the first is
PENDING(Priority), successors PENDING(Dependency), with first start estimated
20:13 CEST. Startup remains unverified; no production update has occurred.

Removed the2,500-update pilot cap and9,000-second timeout. Added fixed-panel
checkpoint callbacks at10k/20k/35k/50k/75k/100k without restarting the PPO process.
Atomic saves every250updates preserve continuation across Slurm limits; failed
or cancelled predecessors stop the chain. An evaluation failure is logged and
does not discard continued training. Final-checkpoint evaluation recovery and
training-only throughput accounting were corrected during independent review.

All six added costs remain zero. Native Adam480000, time input and actor branch
are preserved; no migration/fade is repeated. Foundation KL remains0, trench
KL reaches0 at the original u20000, LR3e-4 and entropy0.02 remain constant.
Source is Terra6a0d7bdd / baselines407e85e. Thirteen focused CPU tests and ten
subtests, native dry run, shell validation and a finite local CUDA continuation
to u7502/Adam480128 pass. The local smoke does not replace the four-GPU parent.

Artifacts: `.artifacts/terra_oracle_long_20260917/`. Next recommended check:
20:30 CEST/18:30 UTC for startup and first scheduled checkpoint; no automatic
status monitor exists. GPU coexistence during the first fixed panel remains
to be measured. Old Capstor campaign-output recovery is separate and unchanged.

## 2026-09-17 15:56 CEST: combined run completes; broad foundations recover

CSCS 4682135 COMPLETED, exit 0:0, from 09:43:44 to 12:02:28 CEST on nid006024.
One node/four GH200s used about 9.25 GPU-hours including two fixed evaluations.
The native u5000 parent reaches u7500 / Adam480000 after 81.92M new transitions.
Both downloaded u6250/u7500 checkpoints pass finite model/Adam and native-clock
validation. Runtime sources are Terra 6a0d7bdd / baselines b61ce031.

Complete fixed panels show foundations **198 to 316 to 322/384**, trenches
**188 to 195 to 193/224**, and the road subset **23 to 22 to 25/32** at
u5000/u6250/u7500. Foundation mean excavation rises 73.39% to 93.70%, accepted
disposal 70.91% to 92.88%. Independent analysis verifies matching recorded
manifest/reset/evaluation contracts and episode identities; all panels have
zero integrity, nonfinite and mass errors. Complete initial Agent-state hashes
are absent from these reports, limiting exact state-parity claims.

On 181 common successful foundations, retained inter-setup distance falls
20.5%, raw travel 31.9%, and adjacency rises 73.99% to 82.77%. Area per
productive setup falls 2.5% and setup count rises 4.4%, so this is better travel
and continuity without demonstrated larger workspaces. Relative to u5000,
u7500 gains 141 foundation successes and loses 17; trenches gain 20 and lose15.
The last u6250-to-u7500 segment has substantial churn: foundation gains42/losses36,
trench gains13/losses15, with slightly worse mean trench material completion.

Foundation guidance is zero from u6250, trench guidance retains the original
cosine, and all six added costs stay zero. The numerical efficiency-readiness
result is false. Recommend another bounded completion phase and review of
regressions before any efficiency stage. This combined continuation cannot
identify causal effects of time, capacity, release or extra training separately.
Warmed loop median is 15,533 transitions/s; last-100-update means are entropy
0.468, approximate KL0.0191 and clip fraction0.111.

At 15:56 CEST the account queue is empty, no reservations are listed, and the
old Capstor root is accessible again. Its earlier campaign outputs remain
unrecovered. No new allocation or automatic extension was started. Evidence:
`.artifacts/terra_oracle_combined_20260916/status_20260917_completed/`.

## 2026-09-17 09:44 CEST: combined job submitted on restored Ritom inputs

CSCS 4682135 starts on nid006024 at 09:43:44, with one node/four GPUs and a
four-hour limit. The known evaluated native u5000 parent branches to u7500
with remaining time, the larger actor head, foundation KL release and cached
teachers; added costs remain zero. The user chose this one combined run to
save compute, so no unchanged-policy control or second node was submitted.

Daint now exposes `/ritom/scratch/cscs/lterenzi` as `$SCRATCH`; `/capstor` is
unavailable. Local inputs and parent were restored; the JAX24.10 runtime was
rebuilt from the established Dockerfile/requirements. Current source is Terra
6a0d7bdd / baselines b61ce031. A frozen-bank fingerprint failure caused by new
zero-default cost fields was fixed with exact default checks; seventeen tests
and independent validation pass. The staged accepted panel preserves all608
ordered episode identities. By 09:51 CEST, CUDA convolution backward, four-GPU
NCCL and finite native initialization pass (2,940,829 model parameters,
u5000 / Adam320000). At 09:53:42, training advances to u5043, with finite
model/Adam/loss checks passed through u5041 and warmed loop throughput around
16,040 transitions/s. Offline history subsequently verifies u5081 with PPO
KL0.01718, clip fraction0.1553 and entropy0.7229. The first u5250 save remains outstanding. Old later
checkpoints and delayed-cost outputs remain unrecovered. Next recommended
manual check is 10:10 CEST (08:10 UTC), not automatically scheduled.

## 2026-09-16 23:02 CEST: authenticated access cannot yet be restored

Lorenzo reports CSCS online and authorizes the needed experiments. A dedicated
agent followed `cscs-auth` using the existing key and pending signing request.
The certificate expired at 20:45:34 CEST. The CLI exits on token-poll timeout,
and independent HTTPS attempts to `auth.cscs.ch` time out; saved browser login
and authenticator recovery are available. Ela rejects the expired certificate,
so current Daint storage, queue and old outputs remain unverified. No new
job or duplicate authentication request was created. Task-created sensitive
logs/screenshots were removed.

Independent launch review finds no blocker in the prepared combined recipe.
Proceed with one four-GPU continuation and a baseline evaluation if choosing
a newer recovered parent. Recover delayed-cost outputs before allocating its
remaining training; never restart completed work or reset the u20000 ramp.
The old delayed wrapper has an expired deadline/start cutoff. Local readiness
record: `.artifacts/terra_oracle_combined_20260916/cscs_readiness_2302.json`.

## 2026-09-16 18:52 CEST: CSCS storage still unavailable

SSH works as `lterenzi`; the account queue is empty. `/capstor` itself returns
ENOENT on `daint-ln003`, while `/iopsstor` and the user home are available.
Generalist 4672272 remains TIMEOUT and delayed pair 4675576 remains FAILED;
their final outputs cannot be recovered yet. The combined run is still local,
not staged or submitted. No new policy evaluation or training result exists.
The prior all-node maintenance reservation is no longer listed; the next one
is September 17 14:00 through September 18 14:00 CEST. A manual storage check
at 19:30 CEST is recommended, not scheduled. Evidence:
`.artifacts/terra_oracle_combined_20260916/cscs_readiness_1852.json`.

## 2026-09-16: combined time, actor-capacity and teacher-throughput implementation

Lorenzo selected one combined continuation because separate learning controls
would exceed the compute budget. Added remaining time to actor/critic, a
zero-output residual actor head, explicit migrations preserving native Adam,
frozen-teacher caching and a teacher-free execution specialization. Optional
retained-work costs and numerical readiness diagnostics are implemented but
remain disabled for the broad policy. Corrected physics and reward-v2 remain.

CPU suites, four-device cache/gradient parity and independent source review
pass. Actual native-parent CUDA smoke reaches u5002 / Adam320128 and native
resume reaches u5003 / Adam320192 with finite states and original release
origin 5000. Model size is 2,940,829 versus 2,311,701. Initial GPU logits/values
and live/cached teachers match exactly. Warmed RTX 4090 batch 256 forward is
5.695→5.677 ms, forward-plus-gradient 22.196→22.533 ms. This is local runtime
and kernel evidence, not learning/throughput promotion on GH200.

Prepared one four-GH200 run with 81.92M new-transition ceiling and midpoint/end
608-case evaluations. No job submitted: lterenzi auth succeeds but campaign
storage is absent at 15:44 CEST during maintenance. The earlier two-arm release
launcher is superseded. Source and the item-by-item status are in
[the Oracle follow-up](research/ORACLE_FOLLOWUP_20260916.md).

## 2026-09-16: controlled teacher-release implementation and failure diagnostics

Accepted the independent review's completion-first priority. Added an opt-in
foundation-only linear KL fade, persistent native checkpoint clock, independent
family coefficient logging and a bounded u5000→u6250→u7500 matched recipe.
Trench guidance retains its original cosine; broad behavior costs remain zero.
The hold requires complete fixed evaluations and explicitly counts lost trench
and road successes, so new gains cannot hide those losses.

CPU schedule, actual-gradient, multi-device family weighting and checkpoint
metadata checks pass. The CPU observation test reproduces identical physical
state/history inputs at ages50/440/449/450 with different terminal semantics;
this is time aliasing evidence, not a measured benefit of time-aware training.
Finite CUDA control/treatment continuations both reach u5002/Adam320128 from
identical initial state; treatment resume reaches u5003/Adam320192 without
restarting its fade. These use 1x128 diagnostic batches, not production data.
All 12 failure replays match their previous endpoints. The bounded search finds
eight progress witnesses and five work/disposal/translation witnesses, but no
complete suffix. The independent qualitative review identifies movement loops
and stationary soil cycling, not WAIT. Tee503's last cell is at a far branch
tip rather than a junction. Full findings and limits are recorded in the
[follow-up note](research/ORACLE_FOLLOWUP_20260916.md).

CSCS scratch remains unavailable during maintenance; no pilot, duplicate
generalist, or additional delayed-cost job was submitted. Existing remote final
checkpoints and the delayed-job failure reason remain uninspected. Artifacts:
`.artifacts/terra_oracle_followup_20260916/`.

## 2026-09-16 09:19 CEST: generalist improves; overnight allocations ended

Complete local greedy fixed evaluations at u2500/u5000 pass native, source,
reset, exact-termination and integrity validation. Morning rechecks bind the
actual checkpoint, report and summary hashes. Easy foundations improve 49→55/64,
broad foundations 173→198/384, trenches 175→188/224, and the road subset 19→23/32.
All-map broad-foundation excavation improves 69.52→73.39%; trench excavation
improves 92.40→95.26%. On 133 common successful broad foundations, workspace
yield improves 3.89% and retained work-pose distance falls 4.78%. Easy-map and
road-trench mean excavation decline despite more exact completions; this is
improvement with remaining regressions, not uniform mastery.

Slurm reports generalist 4672272 TIMEOUT at 05:52:10 after nine hours and
delayed-cost pair 4675576 FAILED at 03:37:28 after 2:54:49. `/capstor` is absent
on the reachable Daint login node during maintenance, blocking final checkpoint
and failure-log inspection. Do not infer final update counts, failed training
rather than failed evaluation, or a negative penalty result from accounting.
The latest locally validated generalist checkpoint is u5000.

Lorenzo reiterated the agreed order: learn completion first, introduce stronger
penalties gradually later. The pair already follows this order from the 64/64
u20000 parent; its control is a bounded continuation for the new late-stage
intervention, not another early-penalty experiment. The generalist keeps added
costs zero. No new job, control extension or automatic continuation was added.
Evidence: `.artifacts/terra_training_status_20260916_morning/`.

## 2026-09-15 21:50 CEST: post-teacher pair complete; recommend ending easy screen

Evaluated existing u30000 checkpoints after 10,000 coefficient-zero updates
(163.84M transitions per arm) on the unchanged 64-map fixed panel. Both score
63/64 versus 64/64 at u20000. Scratch slot1, easy-foundation-l-00258, times out
at 76% excavated/75.33% accepted; pretrained slot55, easy-foundation-l-00312,
times out at 19.16% excavated/accepted. Both previously completed those maps.
Hash/native/Adam1,920,000/source/reset/material integrity checks pass. The two
evaluations exited normally, with no remaining local GPU worker.

Use the 62 four-way common successes for efficiency: workspace area increases
2.4%/1.3%, retained inter-setup distance falls 2.4%/1.0%, and lateral score falls
3.6%/6.5% (scratch/pretrained). Adjacency changes 96.93→96.55% and 96.09→97.99%.
Retain both u20000 policies for completion and u30000 for diagnosis. Recommend
ending this unchanged easy-map comparison and keeping the separate broad
generalist and its fading KL schedule. This does not establish an asymptotic
initialization winner, causal KL benefit or the cause of the two timeouts.
No scheduler changes, cancellation, new allocation or monitor were made.

The one-shot evaluation helper was reviewed. Its timeout cleanup was improved
for future invocation; original executed bytes and explicit root supervision
are recorded. Both actual evaluations finished normally with no orphan process.
Evidence: the foundation campaign's `status_20260915_evening/` decision,
comparison, execution and independent-review receipts, plus `evaluation/paired/30000.json`.

## 2026-09-15 21:30 CEST: generalist starts; both foundation students solve 64/64

Reduced the existing pending generalist 4672272 from 16 hours to 9 hours at
20:51:49, with owner/account/job/state guards and unchanged requested resources,
source, teachers, bank and training recipe. Slurm admitted it at 20:51:55 on
nid005475; the segment ends September 16 at 05:51:55 before 07:00 maintenance.
Full four-GPU CUDA/conv-backward/NCCL and finite 4x256 u1/u2/FINAL gates pass.
Production restored the exact u2 model/Adam (step 128), then exceeded u500 at
approximately 15–16k transitions/s. Actual downloaded u500 passes SHA, finite
model/Adam/loss, Adam 32,000, current physics/teacher/protocol and zero integrity
checks. There is no generalist held-out result yet. No new allocation or
monitor was created.

Foundation pair 4665916 remains healthy at approximately 31.4k updates each,
with latest observed u31000 saves. Actual downloaded u30000 checkpoints pass
hash/native/finite checks, Adam 1,920,000 and zero integrity failures. Teacher
coefficient is zero; added behavior costs remain zero. Training entropy is
0.1220 scratch versus 0.1192 pretrained, a single-batch comparison.

The eight planned fixed evaluations completed normally at 15:46. Paired exact
successes at u2500/u5000/u10000/u20000 are 62/62, 63/63, 64/63 and 64/64 out of
64 (scratch/pretrained). Frozen reference: 62/64. At u20000, area per productive
base pose is 6.207/6.349 m², retained inter-setup distance 16.349/15.895 m and
workspace edge adjacency 96.81/96.21%. Independent review verifies all 576
student/reference rows, actual checkpoint hashes, native clocks and fixed
identities. No overall initialization winner or broad-foundation claim.
The latest replay is at the teacher fade boundary, so sustained completion
after guidance removal is still untested despite healthy subsequent PPO.

Evidence: both campaigns' `status_20260915_evening/` folders. Suggested next
manual generalist check: 22:45 CEST. Maintenance additionally covers September
17 14:00–September 18 14:00; refresh scheduling before further continuations.

## 2026-09-15 14:09 CEST: broad-teacher replacement submitted after full qualification

Recovered mature V8 broad policies, confirming FF670/720 and GRU677/720
historical promotion completion. Their bank included2400foundations among4512
maps. Completed FFu86000 replays on the same608development panel give legacy
332/384foundations+218/224trenches, current140/384+1/224, and current physics with
historical traversability113/384+4/224. All integrity checks pass. On the360
foundations with unchanged initial Agents, old/current successes are312/132;
all24 foundation conditions regress. The runtime bundles differ, so no single
physical rule is established as the cause. Observation rollback is rejected.

Select the broad FF only as a fading foundation prior, alongside the updated
trench specialist192/224. Full3840-map training bank passes current metadata/R2
loading and disjointness against all9 broad held-out manifests. Native teacher
adapter/routing tests28+10subtests and actual CUDAu1/u2/nativeu3 pass, with finite
model/teacher/Adam/loss, Adam64/128/192 and zero transition integrity failures.

Canceled held narrow4670716 before starting at14:07, then submitted one four-GPU
replacement4672272 at14:08:09, after final independent pinned-payload review.
At14:09:04 it is PENDING/Priority,16hours, lterenzi/d130, no node/start estimate.
Existing foundation comparison4665916 remains RUNNING. Fresh student/Adam,
KL1 cosine to0 at20k, all current maps and zero added behavior costs are retained.
Full4x256 CUDA/conv/NCCL and finiteu1/u2 remain in-allocation gates.

The original serial remote hash scan timed out at300seconds. All original
payload bytes were subsequently verified; an isolated8-thread verification
helper preserves every hash and the final source/launch/input/container scan
passes in36.2seconds. This is not a controlled throughput comparison. Training
source/inputs/recipe did not change. Pinned source: baselines96fcd811 and Terra
46738cde; repair/source/smoke/bank/review/submission receipts are in
`.artifacts/terra_generalist_broad_teachers_20260915/`. No automatic continuation,
new monitoring worker or penalty stage was scheduled; next check14:30CEST.

## 2026-09-15 12:40 CEST: hold narrow campaign; recover broad teachers

Held pending job4670716 after the user corrected the teacher and dataset scope.
Verified scheduler receipt: PENDING(JobHeldUser), no allocation or training.
The actual local V8FFu86000 checkpoint matchesSHA2fe5d23c86cc7702b188d33ca1ca9a42066a9a2515150e8795f8c640bbbeb4af.
Historical completed promotion results:341/384foundations,329/336trenches,
670/720total. GRUu40k reached343/384foundations and677/720total. Both trained
on4512maps, including2400foundations. The previous0/384 result applies only
to the easy-map control and does not characterize these broad policies.
Current-runtime qualification and corrected full3840-map bank preparation are
in `.artifacts/terra_generalist_broad_teachers_20260915/`. Existing4665916 unchanged.


## 2026-09-15 10:57 CEST: additional mixed generalist submitted on CSCS

Job **4670716**, one node/four GPUs, `lterenzi`/`d130`. Submission and all staged
source/input checks succeeded. At 11:00 CEST its wall time was reduced from
24 to 18 hours so it can fit before the September 16, 07:00–19:00 CEST maintenance.
At 11:03:26 it remained PENDING with reason Priority; startup is unverified.
The change affects the first allocation length, not PPO, the bank, teachers,
reward costs or the absolute 500,000-update target. No continuation is queued.

The fresh student uses task-specific KL teachers on the qualified easy
foundation plus 15-condition trench bank. Frozen teacher influence fades to
zero by update 20,000. Existing foundation comparison 4665916 is unchanged.
Baselines source `a80fe8bfe14ac4b26cdcd8f306e54cfd97e5fca9` and Terra `46738cde`
are immutable; local/source/runtime/evidence hashes passed independent review.
Local CUDA u1/u2 and native u3 pass, with model/Adam/teacher/clock/soil-integrity
checks. Full four-GH200 runtime evidence remains pending in the allocation.

Receipts: `.artifacts/terra_generalist_teachers_20260915/{job.json,submission_review.json,maintenance_walltime_receipt.json}`.
See the [design](research/GENERALIST_TASK_TEACHERS_20260915.md) for the
stage-one distribution, teacher qualification and continuous behavior metrics.
Next recommended queue/startup check: September 15 at 10:00 UTC; no automatic
monitoring or additional allocation was scheduled.

## 2026-09-15 task-specific teacher generalist preparation

Implemented two frozen policy teachers routed by the student's pre-action map
family, preserving each teacher's native digging observation semantics. Added
per-task KL/exposure diagnostics and native continuation binding to teacher
bytes and family roles. Evaluation clears both teacher flags.

The current-rules easy-foundation control screen completed all608development
cases:0/384 harder foundations and0/224 trenches, zero integrity failures.
Together with its62/64 easy-map result and the recovered trench teacher's
192/224 result, this motivates a first mixed run on easy foundations plus the
fifteen broad trench training conditions. The qualified2976-slot bank is
source/scenario/map disjoint from all eleven checked held-out manifests.

43CPU tests and15subtests pass, with two existing skips. Local RTX 4090 startup
u1/u2 and native u3 pass actual teacher, fresh/native initialization,
model/optimizer/loss, checkpoint-clock and transition-integrity gates.
The staged one-node CSCS campaign has not yet been submitted at this entry.
See [design and evidence](research/GENERALIST_TASK_TEACHERS_20260915.md).

## 2026-09-07 local pipeline correctness smoke

Validated the junction observation, native checkpoint replay, and JAX cache
fixes on starship's single RTX 4090 with W&B disabled. Paired Terra is
`46b140f8373e098ad832e4968d8136a5ba861bf6`; the trainer changes are on
`fix-training-continuation-cache`, based on `d410062b`. No Slurm job was launched.

The current v2 encoder (2,311,701 parameters) completed four updates at 8
environments × 4 steps, 2 epochs, and 2 minibatches. A separate process then
resumed checkpoint 2 with receipts 3/4 already present and completed absolute
update 5. All 431 optimizer leaves were restored exactly at step 8; final Adam
count and train-state step were 20. Model, optimizer, and stored losses were
finite. Replayed receipts 3/4 were archived unchanged and canonical receipts
1–5 remained available.

Both processes had one stable update signature and an explicit persistent-cache
hit for `pmap__update_step`. The original reset counter's weak-to-strong int32
change had caused a second full compile; its two reset initializers are now
explicit int32. Periodic cache eviction is removed. Process startup still traces
and lowers the graph; this small-batch smoke is not a production throughput,
multi-GPU, or learning-quality benchmark.

CPU validation passed junction/DO parity, reward and counter regressions, and
39 trainer/launcher tests. Four edited launchers passed shellcheck. Full local
commands, logs, checkpoints, and independent review are under
`/home/lorenzo/moleworks/.artifacts/terra_pipeline_fixes_20260907/`.

The first table is the historical spatial-encoder line. Metrics:
eval/positive_terminations /
eval/rewards (final), swhr = eval/success_within_horizon_rate, ep_len =
eval/avg_positive_episode_length. Full incident trail:
`docs/EXPERIMENTS_SPATIAL_V3_RUNS.md`.

| Date | Run | W&B | Config | Final | Verdict |
|---|---|---|---|---|---|
| 07-19/20 | spatial A/B | pqtmfmqy | resnet_spatial_8x8 base, 10k updates, mb32 | 2.810 / 0.131 | spatial beats atari +39% per-sample at 3.9× wall-clock; became the E3 teacher |
| 07-20 | atari control | nnsksyva | atari base, 10k, mb32 | 2.020 / 0.095 | control; 113k steps/s |
| 07-21 | E2 | nr032qs7 | se+bf16+critic512 FROM SCRATCH, 20k | 2.329 / 0.108, swhr 0.976 | below teacher — from-scratch drag of heavier bundles (bf16 confirmed +50% throughput) |
| 07-22 | E1 | 3buorfp3 | spatial_8x8 + algo fixes (no value clip, flat shuffle, 19k ent), 20k | 2.833 / 0.132, swhr 0.997, ep_len 59.2 | beats teacher on its own encoder → PPO fixes net-positive |
| 07-22 | E3 | j0bs2fkl | medium se, grown init + kickstart from pqtmfmqy, 20k | **3.054 / 0.142, swhr 0.997, ep_len 55.2** | **best policy to date (+8.7%); kickstart playbook validated — surpassed teacher at 30% budget** |
| 07-22 | E4 | k8vnwp5u | se_xattn FROM SCRATCH, 20k | 2.586 / 0.121, swhr 0.993, ep_len 63.0 | xattn beats SE from scratch +11% (2.59 vs 2.33) — real architecture win; still below warm-started runs |
| 07-22 | E5 | gud7cbwg | dumpzone transfer (E3 warm-start + teacher), CANCELLED @850/20k | 0.000 / −0.005 | naive cross-task kickstart does NOT bootstrap a new task family (no reward stream found); E5b = 2-stage curriculum + higher entropy when dumpzone becomes priority |
| 07-22 | E9 | — | 128×128 pilot, 5-stage medium SE, teacher_obs_downsample=2 | FAILED in smoke | teacher module was built with the student 128 env, causing 128-row position embeddings for a 64-row teacher checkpoint; fixed locally with regression + real-checkpoint CPU smoke before relaunch |
| 07-22 | E9b | — | fixed 128×128 pilot relaunch, same 512 env/GPU memory shape | FAILED in smoke | teacher-env fix worked, but 128×128 PPO update OOMed (`RESOURCE_EXHAUSTED`, temp 10.70 GiB plus 11.49 GB allocation attempt); next gate should try `num_minibatches=64` |
| 07-23 | E9c | 0ixsswn4 | fixed 128×128 pilot relaunch, same 512 env/GPU with `num_minibatches=64` | CANCELLED @~1.4k; all-NaN from smoke update 0 | memory-fit shape worked, but smoke gate missed NaN loss/params: smoke FINAL and production checkpoint both had all model params NaN; forward-only behavior was a symptom, not ordinary learning failure |
| 07-23 | E9d | — | E9c plus embedding clamp, local-map `IntMap`, local-map area scale 4, loaded/downsample fix, finite guard | SUBMITTED 8323457 | exact local full-shape 1-update gate passed finite before Slurm submit; job pending priority |

Cross-run findings:
- Warm-start (grow + kickstart) >> from-scratch for introducing architecture/capacity
  changes (E3 vs E2; E4' line continues this).
- bf16 encoder compute: +50% production throughput (43k vs 28.8k steps/s), numerics clean.
- Finite-loss/param checks are now mandatory for smoke gates: E9c completed update 0 and
  entered production despite all-NaN model params/loss scalars. The E9d script keeps
  `--fail_on_nonfinite` through smoke and production.
- Attention follow-up launched as E10 (pending at submission): `--attention_compute_dtype
  float32` isolates attention-softmax precision inside a bf16 trunk;
  `--token_mixer_residual_init_scale 0.001` wakes v5 mixer gradients without changing the
  default identity-at-init contract.
- pmap scaling ~95% (single-GPU probe 11.4k vs 10.8k/GPU ×4).
- Entropy-schedule stretch makes mid-run comparisons vs 9.5k-schedule runs invalid —
  compare finals or matched entropy phase only.
- Episode length: E3 55.2 vs E1 59.2 steps — bigger warm-started net is also faster per
  episode; 300-step horizon (E6) exerts no pressure since episodes are ~55 steps.

## Completed training awaiting fixed selection

The 2026-08-21 paired movement-feedback GRU pilot completed 50,000 updates in
both arms. Control job `11364188` and six-bit feedback job `11364189` exited
`0:0`; their final checkpoint SHA-256 values are respectively
`5459bd5347dbdf64431cd78df5f61f22b75ee56bc2b15662d9751fb2959a7f84`
and `8cde5ccd4fd4ef5b1ed716a9c5c3a4c4b43f69d44db66d29ed7db86f2ad7d7df`.

The final 1,000-update online aggregate has effectively tied success
(0.99019 control, 0.99037 feedback), while feedback reduces no-effect rate
from 0.03152 to 0.01450. This supports retaining the optional observation path
but is not a promotion verdict. The accepted development-720 evaluation is
still missing, so neither final policy is selected and feedback stays off by
default. See
[`research/V8_MOVEMENT_FEEDBACK_PILOT_20260821.md`](research/V8_MOVEMENT_FEEDBACK_PILOT_20260821.md)
for the frozen decision gate and full provenance.

## Accepted-bank campaigns

The metrics below are deterministic fixed-bank terminal absolute completion,
not the legacy online spatial-run summaries above. Promotion and development
remain separate. Exact is solved maps / evaluated maps.

| Date | Campaign / arm | Slurm | Selected update | Macro P / D | Exact P / D | Verdict |
|---|---|---:|---:|---:|---:|---|
| 08-02 | P5 six-arm accepted-bank screen | — | 2,000 | generalists 0.574--0.588 / 0.574--0.577 | at most 1/512 | all six completed and passed; `G-ADAPTIVE` selected only by the predeclared retention gate, not as a general scheduler claim |
| 08-02 | P5b `G-MEDIUM-ADAPTIVE-WARM` | `9378174` | 2,000 | 0.652 / 0.625 | 1/512 / 2/512 | completed 2,000, `PASSED`; strongest selected constrained distance axis |
| 08-02 | P5b `G-DEEP-ADAPTIVE-WARM` | `9378175` | 1,000 | 0.653 / 0.628 | 2/512 / 2/512 | completed 2,000, `PASSED`; transient matched gain, not retained at 2,000 |
| 08-02 | P5b `G-MEDIUM-UNIFORM-WARM` | `9378176` | 1,000 | 0.647 / 0.664 | 2/512 / 6/512 | completed 2,000, `PASSED`; best selected development family floor, transient at 2,000 |
| 08-03 | all-free capability-floor evaluation | local fixed eval | selected P5/P5b checkpoints | 0.385--0.718 / 0.465--0.736 | generalists 0/32; trench specialist 1/32 promotion only | integrity-clean diagnostic; physically easier but target-mask OOD, excluded from constrained macro |
| 08-03 | P5c five-arm low-entropy screen | `9461489`, `9461500`, `9461504`, `9461507`, `9461512` | none | deep latest 0.624 / 0.586 | deep latest 168/512 / 143/512 | all fixed evaluations clean; no arm passed the long-run gate at two consecutive checkpoints |
| 08-10 | V8 Atari-base small-system control | `10128519` | 19,000 (descriptive promotion selection) | 0.457 / 0.428 | 16/752 / 20/752 | completed 20k; zero mastered conditions and depth 0/0; negative 480k-system result, not an encoder-only ablation |
| 08-14 | V8 v6.1 reward-v2 + stall age + final-v3 continuation | `10625259` | 40,000 | 0.956 / 0.959 | 657/720 / 663/720 | completed u14->u40; promotion exact 407->657 with 254 conversions and 4 regressions; u39->u40 was high-churn 38/32 for only +6 net, so this is a strong combined-treatment capability result but not stall-age or sampler attribution |
| 08-23 | trench-aligned 37-condition generalist, initial production attempt | `11529891` | 0 | — | — | exact update-1 smoke `11529665` passed, but production failed before update 1 on `eu-g6-065` with four-replica `CUDNN_STATUS_EXECUTION_FAILED`; no checkpoint and zero W&B training updates, so this is a runtime incident only |

P5b result root:
`/home/lorenzo/moleworks/.artifacts/terra_p5b_results_20260802_6c56610e`.
Standard leaderboard:
`/home/lorenzo/moleworks/.artifacts/terra_p5b_leaderboard_20260802_6c56610e/LEADERBOARD.md`.
Capability-floor results:
`/home/lorenzo/moleworks/.artifacts/terra_unconstrained_control_eval_20260802`.
The parameters-only current-protocol E8 compatibility replay scores
`0.013/0.027` macro and `0/32` exact; its historical near-1 online `swhr` is a
different evaluation contract and is not numerically comparable.

P5b deep used function-preserving growth (`2,441,223 -> 2,699,117`), a fresh
optimizer, and the frozen parent as KL/value teacher. E8 was not a larger E3:
both had `2,441,223` parameters. The likely recipe mismatch is P5b entropy
`0.15 -> 0.005 / 7,600`, still about `0.137` at the synchronized update-1,500
decline when KL reached zero. This is the P5c hypothesis, not a post-hoc claim
that entropy caused the decline.

## Completed fixed-evaluation screen: P5c

P5c freezes five 4,000-update arms with evaluation every 500 updates: medium
adaptive, medium uniform, deep uniform, foundation medium-uniform, and trench
medium-uniform. Allocated update-1 smoke jobs `9458568`, `9458581`, `9458585`,
`9458616`, and `9458619` all completed `0:0` and passed. Screen jobs `9461489`,
`9461500`, `9461504`, `9461507`, and `9461512` were then submitted from
revision `3478af87950d3d35059344b078209d00785c8481` and crossed finite update
1 with transition-integrity checks enabled. All five subsequently completed
4,000 updates in 6.2--8.0 hours. All share entropy
`0.02 -> 0.005 / 10,000` and the common P5 parent/teacher. The specialists are
family dose ceilings; they do not enter the causal sampler/depth comparison.
All 40 numbered checkpoints were evaluated on constrained
promotion/development and diagnostic all-free promotion/development: 160
integrity-clean evaluations and 43,520 episodes with no integrity failure.
Deep uniform at update 4,000 is the strongest descriptive endpoint:
promotion/development macro `0.624/0.586`, foundation `0.556/0.533`, trench
`0.711/0.654`, and exact `168/512` / `143/512`. Its checkpoint SHA-256 is
`4d178c39443009cb4e57d83713421553689f6e3989da0be674184237c14d86cc`.

This endpoint is not a formal selection. Foundation specialist had one clean
interval at update 3,000, deep uniform one at 4,000, medium adaptive one at
2,500, trench specialist one at 3,500, and medium uniform none. No arm passed
the predeclared improvement/retention gate at two consecutive checkpoints.
No 120-hour continuation or P6 training was launched. Online training success
was still rising, so the result is also not saturation evidence; it says the
current learning curve is unstable on fixed held-out panels.

The condition-balanced leaderboard is frozen at
`/home/lorenzo/moleworks/.artifacts/terra_p5c_leaderboard_20260803_3478af8/LEADERBOARD.md`
with input digest
`ac665b7088942b66159a52f7170c1484dc6e36175f2ec7decbd8c4383094c5ac`.
The complete read-only campaign archive is
`/cluster/work/rsl/lterenzi/terra_p5c_campaign_20260803_3478af87950d3d35059344b078209d00785c8481/`;
its payload-manifest SHA-256 is
`605922d0965206f82e7fe54a10fac202e028b548de24454febcd2691709ff42f`.
Future behavioral screens receive at least 24 healthy hours with an oversized
absolute update target; an admitted checkpoint continues with true resume
state on the 120-hour queue. See
[`research/P5_ACCEPTED_BANK_EXPERIMENTS.md`](research/P5_ACCEPTED_BANK_EXPERIMENTS.md)
section 12.


## September 9, 2026: excavation reliability submission preparation

Status: local gates passed; Euler inputs staged; no job submitted. Terra
`ba9cc214` supplies strict occupied footprints, eligible-soil selection and
short tracked maneuvers. Baselines `9354b89` adds retained work-pose metrics and
a one-RTX 4090, 24-hour full-bank 2x recipe. Four local native updates including
an ordinary process restart pass finite and transition-integrity checks; the
PPO executable cache is reused. The specialist regression screen is 175/224
versus 188/224. The mixed-bank generalist u5000 remains the initializer: it
completes 32/608 full-panel cases under the repaired environment, versus 0/608
for the easy-foundation 2x u15000 checkpoint. Both remain weak on foundations.
Independent review and Slurm test-only validation pass. Ready for one bounded
training allocation; see the [current readiness report](../../../../.artifacts/terra_excavation_reliability_20260909/SUBMISSION_READINESS.md)
for the selected checkpoint, exact source/runtime/bank identities and launch.
Canonical checkouts and unrelated running/queued jobs are preserved.

## September 9, 2026: scratch foundation comparison on CSCS

The user replaced the proposed Euler continuation with scratch initialization
on the four-GPU CSCS node. The prepared comparison is control versus 2x costs,
each at seeds 20260909 and 20260910, using the same repaired Terra `ba9cc214`
and easy-foundation bank. One 24-hour node is at most 96 GPU-hours. No generalist
or Euler duplicate is included. No job has been submitted: the CSCS certificate
expired at 15:10:46 CEST and SSH currently rejects authentication.

The scratch launcher passes 43 focused recipe tests and shellcheck. A local
512x32 two-update scratch smoke passes: next update 2, Adam step 128, finite
model/optimizer/loss and zero transition-integrity counters. Independent code
review found no actionable issues; actual Daint binding/runtime still needs
the in-allocation checks. See the [current plan and evidence](../../../../.artifacts/terra_foundation_scratch_cscs_20260909/PLAN.md).


## September 10, 2026: paired foundation and trench scratch comparison

The user approved foundation control/2x and trench control/2x as four independent
one-GPU policies on one 24-hour CSCS node (at most 96 GPU-hours). All use seed
20260909 and repaired Terra `ba9cc214`; no historical initializer is imported.
This replaces the two-seed foundation-only proposal above. Foundation uses the
existing 256-map easy bank, trench the existing 1,440-map/15-condition pooled
bank including junctions. Comparisons are within task, with one paired seed.
Job **4634548** was submitted September 10 at 00:29:56 CEST and is
**PENDING (Priority)** at 00:31:34. The provisional Slurm start estimate is
22:55 CEST September10 and can change. ReqTRES: 256 CPUs, one node, four GPUs;
normal/d130, 24 hours. Staged source is baselines `30f5f97` with Terra `ba9cc214`.
The 44 focused tests, independent review, both local family scratch smokes and
Slurm test-only passed. The new trench smoke completed u1/u2 with finite
model/optimizer/loss and zero integrity counters. Its local log retains a cuDNN
bf16 autotuning mismatch warning. Daint has not allocated a node or run its
numerical gates yet. See the [current plan and execution evidence](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/PLAN.md)
for current submission/runtime status. No generalist or Euler duplicate is
included. Evaluate matched checkpoints separately on the 64 easy-foundation
validation cases and 224 trench rows of the unchanged 608 development panel.


### September 10, 09:46 CEST: all four scratch gates passed

Job 4634548 is RUNNING on nid005895, started 09:32:30 with four distinct GH200s;
end time is September 11 09:32:30. Node-level conv backward/NCCL and each arm's
own conv backward/u1/u2 gates passed. All four checkpoints have finite model,
optimizer and loss, Adam 128 and zero transition-integrity counters. Each arm
restored its own new u2 optimizer state for production; first-update compilation
was still running at 09:46:18. No production checkpoint or matched behavioral
evaluation exists yet. This is a numerical-startup result only.

The fixed environment's random-transition and recorded-state probes support
mass/occupancy correctness and restored short moves. Frozen specialist replay
regressed 188→175/224; generalist trench replay improved 30→32/224 with foundations
still 0/384. Thus there is no demonstrated overall learned-policy improvement.
The scratch comparison holds the fixes constant and tests zero versus 2x costs
within each task, with one paired seed. See the [status and interpretation](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260910_MORNING.md)
and its per-arm live evidence. No jobs or training settings changed during the
status check.

At 09:48:15, both foundation arms had production receipts through u31 and steady
updates at roughly 7,150 transitions/s (2.3 s/update); both trench arms were still
compiling their first resumed production update. The foundation u31 batch had
512 timeouts and no successes in each arm; this is too early to rank costs or
claim saturation. The linked morning status retains the exact observations.


### September 10, 13:35 CEST: all arms past u5000; cost concern

CSCS 4634548 remains RUNNING on four GH200s, with recorded updates F0=5500,
F2=5921, T0=5381, T2=5861. Latest saved checkpoints are 5500/5500/5000/5500.
All four retrieved u5000 checkpoints have finite model/optimizer/loss, zero
transition integrity and Adam 320000; native continuation is working. No
execution/cuDNN/nonfinite error was found. Rates are around 6100-7200
transitions/s/GPU. No training settings or jobs changed.

The matched online update 4000-5000 window raises concern about early cost
suppression: foundation mean excavation 68.9% control vs 15.4% 2x; trench 48.3%
vs 0.32%. These are sampled training episodes, not a held-out efficiency result.
The authorized 24-hour screen continues. Four matched u5000 evaluations are
running locally; the completed foundation control is 0/64 success and 48.9%
excavated, with zero integrity failures. Other results are pending; comparison
is scheduled after all four finish. See the [afternoon evidence report](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260910_AFTERNOON.md).


### September 10, 15:31 CEST: completed u5000 comparison

Job 4634548 remains healthy at about six hours: updates F0=8171, F2=8901,
T0=8041, T2=8891; no execution/cuDNN/nonfinite errors. All four u5000 fixed
results are complete. Foundation control/2x: 0/64 successes each and 48.91% versus
15.49% excavation. Trench control/2x: 3/224 versus 0/224 and 46.71% versus 0.178%
excavation. Trench 2x digs no fresh soil on 220/224 maps. There are no common
successes within either pair; reduced travel cannot be promoted as efficiency.
All 1344 evaluated episodes have zero integrity failures/unavailable counters.
Independent review checked matching identities, reset receipts, hashes and
settings; all result counts reproduce. The newer sampled online u7000-8000
window still shows large 2x progress suppression. These are early one-seed
cost-treatment results, not convergence or an environment-fix ablation.
Continue the authorized 24-hour screen; the next planned evaluation is u10000.
See the [completed comparison report](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260910_1530.md).

At 15:41 CEST, a reviewed local helper started waiting for all four u10000
checkpoints and u10001 receipts (managed session 88723, PID 3052299). It has a
three-hour readiness deadline, checks the actual Adam count and local GPU
availability, then runs the unchanged matched evaluations and summary inline.
No u10000 result is available yet; no Slurm job or training setting changed.


### September 11, 00:30 CEST: component arms submitted; evaluation recovery

Original CSCS 4634548 remains RUNNING (last live check 00:18), about 14h46m into
its 24-hour allocation. Recorded updates are F0=20201, F2=22291, T0=19911, T2=22501.
The sampled online u18000-19000 window gives exact success 4.84%/0% foundations
and 25.69%/0% trenches (control/combined 2x). Excavation is 86.66%/5.48% and
70.36%/0.135%. No runtime failure or physical/per-step integrity violation was
found; one small informational accumulated reward-drift count is documented.
These online results are separate from the fixed held-out evaluation.

The old automatic u10000 evaluation had stopped before downloading checkpoints
on SSH exit 255. All four checkpoints were retrieved and passed finite checks
and actual Adam 640000 today; recovery evaluation began 00:08:59 locally.
Foundation results are complete: control 0/64 exact, 66.64% dug; combined 2x 0/64,
14.76% dug. Both integrity checks pass. The trench evaluations are still running.

Lorenzo authorized four additional overnight experiments. Submitted CSCS 4642631
at 00:18:13 for one additional 24-hour four-GPU node (at most 96 GPU-hours), with
foundation/trench lateral-only (0.5,0,0) and relocation-only (0,0.01,0.04), same
scratch seed 20260909, data, model, PPO and repaired environment. Original jobs
are unchanged. New immutable source: Terra ba9cc214 and baselines 275571b;
changes are launcher/verifier wiring and docs only. The 38 existing launch tests,
CLI comparison, shellcheck, known finite checkpoint verifier and independent
review passed. Each new arm still requires its actual per-GPU finite-u2 gate.

New job state at 00:18:41 was PENDING(Resources), with an estimated 16:35 start
that may change. The SSH certificate expired 00:19:13 after submission. Two
read-only shorter-segment queries failed authentication; no walltime change or
extra submission occurred. Root requested renewal if Lorenzo was still awake.
Submitted/running jobs do not require continued SSH access.

At 00:29:41 the reviewed serial evaluation driver started in local tmux
terra-components-eval-20260911 (worker 3821255), waiting for current u10000 before
original u15000/u25000 and new component u5000/u10000. Retrieval requires renewed
authentication; each helper has a 12-hour bounded readiness/download window.
No new generalist, recipe promotion, or saturation claim is made.
See the [current status](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260911_0025.md)
and [component plan](../../../../.artifacts/terra_excavation_cost_components_cscs_20260911/PLAN.md).


### September 11, 09:03 CEST: completed u10000 comparison; live status blocked

The four original u10000 evaluations completed at 00:46:14. Foundation
control/2x have 0/64 exact each and 66.64%/14.76% excavation. Trench control
reaches 14/224 exact and 52.05% excavation; trench 2x has 0/224 and no fresh
digging on any of the 224 cases. Both pairs still have no common successes, so
no comparative efficiency result is available. All 1344 episodes pass integrity
checks. Independent review confirmed hashes, update/Adam 640000, fixed settings,
per-map/reset identities and the result counts.

Against u5000, control foundation excavation improves 48.91%→66.64%; trench
success 3→14/224 comprises one retained, thirteen new and two lost successes.
All 14 current trench successes are straight-trench conditions; segmented,
network and T-junction conditions remain unsolved. This is an early checkpoint
trend, not the end-of-allocation or convergence result.

SSH still fails authentication at 09:03. The last confirmed scheduler states
remain original 4634548 RUNNING and additional 4642631 PENDING(Resources) at
00:18:41; do not report these as current. New-arm startup and later checkpoints
are unverified. The persistent local evaluation driver is alive, waiting at
original u15000 retrieval within a 12-hour deadline until about 12:46:56. No
u15000/u25000 or component u5000/u10000 results exist yet. Lorenzo was asked to
renew authentication. No jobs, training settings or runtime source changed.
See the [morning report](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/STATUS_20260911_MORNING.md).

### September 11: frozen foundation regression traced to intermediate rounding

Corrected Terra 7fb30402 restores 12/13 lost frozen-2x completions on the same
64-map easy validation panel: 51/64 -> 63/64, zero new losses, 95.274% ->
99.938% mean excavation. The earlier original-environment result was 64/64.
Intermediate grid rounding introduced sideways chassis excursions and false
collisions; the correction checks the entire straight sweep to each candidate
endpoint and retains all soil protections. Eighteen focused CPU tests, 384
independent endpoint comparisons and thirteen exact GPU probes pass. Independent
review confirms matched metadata and result counts. This is frozen-policy
replay evidence, not a scratch-learning or reward-cost result. No cluster job
or training setting changed; the original ba9cc214 cohort remains separate.
See the [diagnosis, corrected replay and tests](../../../../.artifacts/terra_excavation_scratch_cscs_20260910/foundation_regression_diagnosis_20260911/REPORT.md).

### September 11: corrected eight-arm scratch comparison submitted

Submitted CSCS **4645193** (foundation/trench × control/combined 2x) and
**4645194** (foundation/trench × lateral/relocation), both pending for priority
at 11:48 CEST. Each is one four-GH200, 24-hour allocation with four independent
one-GPU policies. The shared clean source snapshot is Terra main 46738cde and
terra-baselines main 2fb7863. Each policy starts from zero; the production
phase continues only its own two fresh smoke updates.

Local corrected-environment foundation and trench smokes passed u1/u2,
finite model/Adam/loss and recorded integrity checks. Independent launch/EDF
review, shell checks and Slurm dry submissions passed. On-node runtime,
binding and per-arm smoke checks remain unverified while pending. This is a
submission result, not evidence that scratch learning or behavior improved.

Original 4634548 timed out at 24 hours. Its saved checkpoints F0=u33000,
F2=u36500, T0=u32500 and T2=u37000 are also preserved locally. Never-started
component job 4642631 was cancelled before replacement; the old local driver
waiting for it was stopped. No other jobs changed. The 192-GPU-hour replacement
budget, fixed PPO/cost matrix and equal-update evaluation contracts are in the
[restart plan](../../../../.artifacts/terra_movement_restart_cscs_20260911/PLAN.md).

### September 12: corrected u10000 comparison favors zero behavior costs

All eight policies have retained u5000/u10000 held-out evaluations; the local
driver completed at 01:51:17 CEST. At u10000, foundation control/lateral/
relocation/combined exact success is 1/1/0/0 out of 64, with mean excavation
71.03%/55.00%/12.47%/8.28%. Trench success is 20/2/0/0 out of 224, with
47.26%/43.36%/0%/0% excavation. Control trench successes increased from 2 at
u5000 to 20 at u10000; both travel-cost policies still perform no fresh trench
excavation. There are too few common successes to establish an efficiency gain.

The full 5,376 episode rows across both milestones pass recorded integrity and
reset checks. A missing optional trench geometry field had stopped summary
generation; fixing the presence/value comparison recovered saved rollouts
without repeating them. Checkpoint update/Adam/hash and matched treatment
contracts remain verified. This is an early scratch-cost result, not a
convergence or frozen-policy environment-fix result.

CSCS authentication expired September 12 at 11:32 CEST; live scheduler state
and later checkpoints could not be refreshed in the morning. No training
settings or jobs changed. See the [complete comparison](../../../../.artifacts/terra_movement_restart_cscs_20260911/REPORT_20260912.md).

### September 12: prepare completion-first behavior-cost continuation

The user approved introducing penalties after the policies complete most maps.
Continue the corrected zero-cost controls separately for foundations and
trenches; preserve native Adam/update/entropy clocks, banks, PPO settings and
the fixed environment. The first cost stage requires two successive greedy
450-step fixed-panel evaluations at least 2,500 updates apart with at least
58/64 foundation or 202/224 trench exact successes. Saved u10000 results of
1/64 and 20/224 do not qualify.

The prepared offline launcher advances through 25%, 50% and 100% of the old
combined 2x costs in 5,000-update stages. Each subsequent pair of evaluations
must retain 90% completion and remain within three percentage points of the
original accepted zero-cost reference. It validates the native parent and
bank, uses the existing trainer and records each stage for later continuation.
Retain a zero-cost sibling and compare efficiency on common successes at equal
additional updates. The
[recipe](../scripts/foundation_reward_sweep/README.md) supersedes applying the
costs from scratch; it is not evidence of improved behavior yet.

Validation: 72 focused CPU tests and 22 subtests pass, together with shellcheck
and syntax checks. Both actual u10000 control checkpoints pass native finite
model/Adam/loss and bank checks at Adam 640000. Each rejects deliberately wrong
update/Adam metadata, partial-reset state and nonfinite model values; the real
completion gates remain false. Independent review has no remaining findings.
Evidence is in `.artifacts/terra_delayed_penalties_20260912/` at the workspace root.
No new training, submission or cancellation occurred. CSCS authentication still
fails; latest control checkpoints, live scheduler state and the required next
allocation's runtime/native continuation smoke remain pending.

## 2026-09-12 native smooth-cost ramp runtime check

Implemented an optional checkpointed linear behavior-cost ramp. TrainConfig
and the R2 receipt retain declared target costs; saved EnvConfig contains
effective last-rollout costs. Checkpoint loading verifies these against the
absolute schedule, and evaluation uses effective costs. Ordinary native resume
restores the ramp and optimizer clock, while parameters-only warm starts
discard the schedule. Values change only between PPO rollouts with stable
array shapes and dtypes. Fixed-cost training and Terra legality are unchanged.

Validation: 122 focused CPU tests plus 22 subtests, shell checks, independent
review, and a real local RTX 4090 convolution/native-resume smoke passed. Seven
checkpoints spanning u10001..u10005 preserve finite model/Adam/loss values,
zero integrity counters and Adam640064..640320. The u10002 checkpoint resumes
from half weight, reaches full weight at u10004, and holds through u10005.
The resumed PPO executable was a persistent-cache hit, with one signature
per process. The 32-environment run is a functional check; multi-GPU scaling
remains pending in Euler diagnostic13935300 and the prepared CSCS diagnostic.
Full evidence: `.artifacts/terra_delayed_penalties_20260912/smooth_ramp/`.

## 2026-09-12 latest zero-cost controls and bounded automatic continuation

Corrected CSCS allocations 4645193/4645194 ended after 24 hours with `TIMEOUT`,
exit 0:0. Under restored lterenzi access, foundation u33000 and trench u32000
parents passed native finite-state, Adam-clock and zero-cost checks. Fixed
greedy 450-step evaluation, using baselines 866e8e and Terra 46738cde, gives
foundation **8/64 exact**, **93.2860% dug**, **90.9650% disposed**; trench
**36/224 exact**, **65.9106% dug**, **62.8468% disposed**. All 672 full-panel rows
(64 foundation plus the original 608-row development panel) have zero recorded
integrity/nonfinite/mass-residual failures. Neither family reaches 58/64 or
202/224, so no penalties are introduced.

CSCS diagnostic **4652857** started at 16:35:10 CEST on nid005780. At 16:37,
four-GH200 binding, CPU parent/bank checks and cuDNN/NCCL preflight passed.
At 16:40 it remains RUNNING, tracing its first graph with no native checkpoint.
Its hook submits one 24-hour
zero-cost four-GPU trench continuation after all diagnostic checks pass and matched
foundation speedup reaches >=1.5x. Its parent is the verified diagnostic
trench u32016 FINAL; production repeats runtime and two native-update checks.
No child is submitted yet.

Euler replacement **13939497** is submitted under lterenzi with four RTX 4090s,
a 45-minute limit and `AUTO_CONTINUE_FOUNDATION=1`. At 16:40 it is PENDING
(nodes down, drained or reserved), with no allocated GPU or reliable start
estimate. Old 13935300 was cancelled
only while pending under lterenzi. The hook chooses four GPUs at speedup
>=1.5x, otherwise one, and continues the actual u33000 parent for one 24-hour
allocation after runtime acceptance. The older diagnostic parent is not used
for production. Both recipes preserve 512 global environments and 64 Adam
steps per update. No further allocation chain or automatic penalty promotion
is configured; a new layout needs its own qualifying zero-cost evaluation pair.

Training source b6d1597, published to baselines main, includes smooth ramp
b6754540, account cleanup
c2df04a and the portable Python 3.10 hash fix; Terra remains 46738cde. Checks:
122 CPU tests plus 22 subtests, 65 post-compatibility tests, independent review,
shell/submission-stub checks and seven finite real local RTX 4090/32-env ramp
checkpoints with persistent-cache reuse. Four-GPU native training acceptance
remains pending. See the [current status and evidence](../../../../.artifacts/terra_delayed_penalties_20260912/STATUS.md).

At 16:44 CEST, CSCS4652857 passed the foundation four-GPU segment from
u33000 to u33016: 17 finite periodic/FINAL checkpoints, Adam2113024 at the
end, zero integrity counters and no added costs. Median global throughput
after the first two updates is 15,935.585 transitions/s; elapsed process time
including startup/compilation is 455 seconds. The one-GPU comparison is
running. Scaling and trench resume checks still precede any production child.

## September 12, 23:03 CEST: accepted CSCS scaling and live trench continuation

Diagnostic **4652857** completed 16:35:10–17:07:07 CEST with exit 0:0. All four
phases passed. Matched foundation median throughput is 15,935.585 versus
6,961.24 transitions/s on four versus one GPU (**2.2892x**); four-GPU trench
resume reaches 15,778.96 transitions/s. This supports the allocation choice;
it is not a measured learning improvement.

The resumed trench diagnostic reports a persistent cache hit for the same
`pmap__update_step` executable. Its XLA compilation stage took 4.153 seconds,
versus 65.402 seconds in the first process. Python tracing and lowering still
run on restart; this does not eliminate all startup overhead.

Its hook submitted the authorized zero-cost trench continuation **4652918**,
which started at 17:10:25 on nid005935 with four GH200s. CPU parent checks,
CUDA/NCCL preflight and native u32016→u32018 startup checks passed. At 23:03
the job is RUNNING around u48919, with recent throughput around 13.5–13.9k
transitions/s. It keeps 512 global environments, 64 Adam steps per update and
500-update checkpoints; the single 24-hour allocation ends around September
13 at 17:10 CEST. W&B remains offline as
`terra-trench-control-4gpu-after-4652857-4652918`.

Retrieved u48500 passes finite model/optimizer/loss and zero-cost/integrity
checks at Adam 3,104,000. Its 608-row greedy 450-step evaluation completed at
23:40. The 224 trench rows improve from **36/224 to 147/224 exact**, 65.9106%
to **91.3255% dug**, and 62.8468% to **87.0979% disposed**. All full-panel
integrity checks pass. Reset/bank/R2 and normalized treatment checks match,
allowing only the training GPU layout and linked run name. T-junctions improve
1/32 to 25/32, two-sided networks 0/64 to 51/64, and multi-segment trenches
0/32 to 23/32. Road-constrained networks remain 0/32. There are 115 gained and
four lost successes; efficiency on 32 common successes is almost unchanged.
No penalty stage is active: 147/224 is below 202/224, and a second qualifying
checkpoint would still be required. Foundation remains 8/64 at u33000.

At 23:41, CSCS4652918 remains RUNNING around u50831, with u50500 saved and
recent median throughput 13,681.645 transitions/s. Euler **13939497** is still
PENDING/Priority, with no reliable start estimate or foundation production job.
Device-local advantage
normalization changes after layout migration, so completion qualification
must use the selected layout. No job or training-setting mutations occurred in
this status check. Evidence: [status_2302](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_2302/).

## September 13, 08:43 CEST: recover foundation continuation after ramp timeout

CSCS trench **4652918** remains RUNNING near u77361, with u77000 saved and
13,618.59 transitions/s. The allocation ends today at 17:10 CEST. Local u77000
inspection passes finite model/Adam/loss at Adam 4,928,000, zero costs and zero
integrity counters. Its complete full 608-row greedy 450-step evaluation
finished at 08:59 CEST: **161/224 exact trench completions**, up from 147/224
at u48500; mean dug 91.3255% → 94.4698%, disposed 87.0979% → 89.8239%.
There are 24 gained and 10 lost successes; all 608 integrity checks are zero.
Straight improves 48/64 → 52/64, T-junction 25/32 → 27/32, two-sided networks
51/64 → 60/64, multi-segment falls 23/32 → 22/32, and road networks stay 0/32.
Matched resets, bank, source, treatment and four-GPU layout pass comparison.
On 137 common successes, productive poses (9.358 → 9.314), unique area/setup
(2.712 → 2.724 m2), retained-work travel (40.754 → 40.705 m) and edge adjacency
(89.58% → 89.45%) are essentially unchanged. Added penalties stay zero.

Euler diagnostic **13939497** ended `FAILED`, exit 124:0, after
00:38:02–01:13:23 CEST. Its ramp startup exceeded 650 seconds before producing
a checkpoint. The two completed zero-cost controls were recovered and checked:
all 34 periodic/FINAL checkpoints pass finite/native/Adam/global-batch and
zero-cost/integrity checks. Matched throughput is 13,503.175 versus 4,585.225
transitions/s on four versus one GPU (**2.9449x**). That evidence accepts
zero-cost continuation only; the separate ramp remains unqualified.

The unused ramp gate prevented the foundation continuation from submitting.
The already-authorized single 24-hour foundation run is therefore being moved
to CSCS, whose current resource estimate is September 13 at 12:13 CEST versus
September 14 at 11:45 on Euler. No Euler production job exists and no extra
trial is added. Actual replacement **4655350** started at 09:00:11 CEST on
nid005799 with four GH200s, with no dependency or requeue and a 24-hour limit
ending September 14 at 09:00. Parent/bank checks and cuDNN-backward/NCCL
preflight passed. Native startup updates are in progress; production is not yet
verified. The submitted job resumes the accepted CSCS diagnostic
foundation u33016 FINAL and must repeat runtime/parent/bank checks plus two
finite native updates before production. It preserves b6d1597/Terra46738cde,
the 256-map bank, zero costs, 512 global environments and 64 Adam steps per
update; it is bounded to one 24-hour allocation without another job chain.
Foundation's last evaluated result remains 8/64 at u33000. Evidence:
[status_20260913_0843](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_0843/).

At 09:04 CEST, the trench worker has reached approximately u78310. An
independent review reproduced all u77000 evaluation counts, grouped results,
matched identities and common-success metrics from the raw reports, with no
remaining findings. Foundation runtime checks have passed; its first native
training updates are still compiling at this observation.

## September 13, 11:44 CEST: both continuations training; native checkpoints pass

CSCS foundation **4655350** is RUNNING on nid005799 after 2h44m, around
u41113 (8095 production updates beyond u33018). Its startup smoke passed
all three finite native checkpoints through u33018 in 425 seconds, and
production is now sustained at roughly 14.7–15.3k global transitions/s.
The saved u41000 checkpoint passes finite model, optimizer and loss checks
at Adam 2,624,000. Allocation ends September 14 at 09:00 CEST.

CSCS trench **4652918** is RUNNING on nid005935 after 18h33m, near u85992.
Recent log samples report 13.1–13.3k transitions/s. Saved u85500 passes the
same native checks at Adam 5,472,000. Allocation ends today at 17:10 CEST.
Both checkpoints preserve four devices × 128 environments, training-bank
identity, native parent/clock, zero added behavior costs and zero recorded
transition-integrity counters. No jobs or training settings changed.

No new held-out evaluation was run during this status check. The latest
complete results remain trench u77000 at 161/224 exact (71.9%), 94.4698% dug,
and foundation u33000 at 8/64. These are not evaluations of the newer
checkpoints. Workspace efficiency and road-network completion have no new
measurement; the completion gate remains unmet and penalties stay off.
Evidence: [status_20260913_1144](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_1144/native_validation.json).

## September 13 evening: CSCS auth expired; retained foundation u41000 regresses

At 22:22 CEST, live CSCS status could not be refreshed: lterenzi's SSH
certificate expired at 15:52:28 and retry returned `Permission denied
(publickey)`. Last verified scheduler/runtime observations remain those at
11:44. Trench4652918 was scheduled to end September13 17:10; foundation4655350
was scheduled to end September14 09:00. These scheduled times do not establish
current job state. No further jobs or reward changes were made.

The already-downloaded, native-validated foundation u41000 and trench u85500
checkpoints were evaluated locally in persistent tmux using the frozen source
and unchanged respective panels. These checkpoints predate training later in
the day. Foundation's complete64-row greedy450 evaluation gives **3/64 exact**
versus **8/64 at u33000**, **79.4091% versus93.2860% dug**, and **77.1420%
versus90.9650% disposed**. All64 reset identities and integrity checks pass;
checkpoint hashes and effective zero-cost treatment match. Independent review
reproduced the counts from raw rows.

All8 earlier successes are lost and3 new successes appear. Square drops8/21
to1/21, rectangle rises0/22 to2/22, L remains0/21. Average excavation decreases
in all three shape groups. On each checkpoint's failing rows, longest material
stall rises387.18 to 408.87steps; failure sets differ. No common-success
intersection exists, so no success-conditioned workspace-efficiency comparison
is available. Raw travel across all64 rows rises54.54 to 104.74m while average
excavation falls. Fewer productive poses here cannot establish efficiency.

This establishes regression at the retained u41000 checkpoint, not a cause
or the latest remote model's behavior. Continued learning and layout migration
from1x512 to4x128 are confounded; per-device advantage normalization changes
while the global batch is preserved. The added penalties remainzero. Retrieve
and evaluate the latest native checkpoint after authentication renewal before
deciding how to continue or change the foundation recipe.


## September 13, late evening: historical trench replay and manual regression audit

Fresh matched trench u85500 evaluation gives **162/224 exact** versus 161/224
at u77000, with mean excavation 93.53% versus 94.47%. The 151 common successes
show essentially unchanged productive-pose, area-per-setup, retained-work travel
and workspace-adjacency metrics. Road-constrained completion remains 0/32.

Historical fixed reports confirm 196/224 at u51500 and 188/224 at u83500.
The latter completed 25/32 road maps; non-road completion is almost unchanged
at 163/192 historically versus 162/192 now. A fresh replay of old u83500 weights
under the current repaired environment completes **179/224**, including
**20/32 roads**, with matched resets and zero integrity failures. Thus the
current environment still permits successful road behavior. This comparison
retains each policy's trained observation interface and different training
histories; it does not isolate the cause of the new policy's deficit.

Manually inspected trajectories show the current policy repeating blocked
base movements, ineffective cabin/dig cycles and WAIT despite legal productive
alternatives. The old policy completes the two selected road cases in 95 and
231 actions under current physics. For foundations, old u33000 completes a
selected square in 63 actions while u41000 waits 448 actions after two setup
actions. Same-input inference confirms a changed learned action preference.
Sampled u41000 diagnostic completes 6/64 versus greedy 3/64; this is not a
qualified deployment improvement.

Native clocks, optimizer state and effective environment settings pass audit;
63 focused CPU environment tests pass. No hidden added penalties or positive
WAIT reward were found. Relevant differences still requiring controlled tests
are per-device PPO advantage normalization (512 to 128 local samples) and the
historical trench run's four-times-larger transition budget per update and
entropy-decay experience budget. Current loose-soil observations remain present.

Foundation greedy replay endpoints match all 64 rows. The current trench trace
replay differs on eight unselected episodes; all five selected clips match
checked endpoints. The full-panel parity failure is retained and not presented
as a pass. No production source, reward or job changes were made. CSCS state
remains unverified after certificate expiry; these are morning checkpoints.

See [completion regression audit](research/COMPLETION_REGRESSION_20260913.md)
and the [manual rollout viewer](../../../../.artifacts/terra_delayed_penalties_20260912/smooth_ramp/status_20260913_2222/qualitative/index.html).


## September 13, 23:49 CEST: two authorized overnight recovery screens submitted

The user explicitly authorized new overnight training. Two distinct 24-hour
Euler allocations were submitted as lterenzi, each requesting four RTX 4090 GPUs:

| Arm | Job | Native parent | Batch | Treatment | Fixed evaluation |
| --- | ---: | ---: | --- | --- | --- |
| Foundation screen |14055215|u33000|4×128env|Global minibatch advantage normalization|u41000,64-map greedy validation|
| Old trench recovery |14055463|u83500|4×512env|Original local normalization and legacy observation, current physics|u86000,224 trench rows of608-map greedy development|

Foundation was confirmed PENDING (Priority); neither job has passed its remote
startup gate at this submission observation. Trench's sbatch receipt is valid;
queue queries are intermittently timing out. Do not call submitted jobs healthy
training. Existing CSCS state is still unverified after certificate expiry.
No duplicate local-normalization foundation control or CSCS run was submitted.

Both local actual-parent GPU smokes passed two updates and three saved native
checkpoints at 1×32 environments. Model/Adam/loss are finite, actual clocks advance
128 Adam steps, and restored environment/bank/zero-cost contracts pass. This is
functional evidence only; the allocation requires CUDA convolution backward,
NCCL and two validated updates at the exact production shape before proceeding.
Startup failure ends the allocation without silently shrinking the batch.

The normalization change is opt-in, preserves the default arithmetic, records
its mode in checkpoints/W&B/evaluation fingerprints, and rejects native resumes
that would silently drop it. 26 CPU tests pass, including four-device versus
merged-minibatch equivalence through 64 real PPO/Adam steps. Independent review
found no remaining source, native-validator or watcher findings. Training source
is baselines 46adaec50ed4dd225117b2fb8b721ee85649efa7 paired with
Terra 46738cde28e455da7c466fc0a2cb64f677d86401; remote files pass SHA verification.

All added behavior penalties remain zero. Both runs target absolute u500000,
save every 500 updates and are bounded by 24 hours, with no automatic next job.
The old trench arm preserves seed 20260901, 65,536 transitions/update, 64 Adam steps,
entropy floor 0.02 and executable_dig_observation=False. Foundation preserves
seed 20260909, 16,384 transitions/update and 64 Adam steps, opting into global rather
than per-device normalization. Its existing CSCS control differs in hardware,
RNG and restart boundaries; this is a screen, not a causal A/B claim.

A 26-hour local tmux watcher, terra_recovery_overnight_20260913, monitors both
jobs, downloads only the selected milestones, verifies native checkpoint SHA/
mode and serializes fixed greedy evaluation on an idle local GPU. It reuses
completed reports after restart, kills its evaluation process group on timeout,
and neither submits jobs nor enables penalties. The evaluator stays frozen at
baselines 866e8e2/Terra 46738cde; a separate native SHA/mode receipt accounts for
its historical treatment fingerprint, which predates the normalization flag.

[Campaign plan and evidence](../../../../.artifacts/terra_regression_recovery_20260913/PLAN.md) ·
[Submission manifest](../../../../.artifacts/terra_regression_recovery_20260913/manifest.json) ·
[Overnight status](../../../../.artifacts/terra_regression_recovery_20260913/overnight_status.json).


## September 14, 12:54 CEST: recovery evaluations complete; both Euler jobs training

Both Euler jobs are RUNNING and have passed their full production-workload
startup gates: four RTX 4090 GPUs, CUDA convolution backward, NCCL, two finite
native updates and checkpoint/environment/bank/integrity validation.

| Run | Approximate live update | Saved checkpoint | Recent global transitions/s | Scheduled end (CEST) |
| --- | ---: | ---: | ---: | --- |
| Foundation global normalization 14055215 |57,686|57,500|12,594|September 15 03:37|
| Old trench recovery 14055463 |87,591|87,500|16,923|September 15 08:05|

Latest saved checkpoints above are located, not evaluated or independently
loaded during this update. The completed milestone evaluations below use
validated earlier checkpoints, fixed greedy 450-step panels and zero added costs.

| Evaluation | Exact completion | Mean dug | Accepted disposal |
| --- | ---: | ---: | ---: |
| Foundation global u41000 |11/64 (17.2%)|91.09%|90.61%|
| Foundation prior local u41000 |3/64 (4.7%)|79.41%|77.14%|
| Foundation common parent u33000 |8/64 (12.5%)|93.29%|90.97%|
| Trench recovery u86000 |192/224 (85.7%)|96.78%|95.97%|
| Old trench parent u83500 under current physics |179/224 (79.9%)|94.71%|93.66%|

Trench road completion rises 20/32→24/32; non-road 159/192→168/192.
There are 22 gained and 9 lost successes. On 170 common successes, productive
poses 9.524→9.553, unique area/setup 2.808→2.803m², retained-work travel 44.03→
44.48m and workspace adjacency 88.97%→88.49% show no efficiency improvement.
The main gain is completing more maps and disposing more spoil correctly.

Foundation completes 8/21 squares,3/22 rectangles and 0/21 L maps. It improves over
the regressed local-normalization run but remains weak and does not exceed the
parent's average excavated fraction. Hardware/RNG/restart differences mean the
normalization comparison is a screen, not proof that normalization alone caused
the improvement.

Both reports finished with their EVAL_DONE markers. All 672 rows have zero
recorded integrity failures; reset identities and actual checkpoint hashes pass.
The bounded overnight watcher completed both planned evaluations and exited
normally at 11:27. It is no longer a live monitor. Training continues within the
existing 24-hour allocations; no new jobs or reward changes were made today.
Penalties remain off: neither 58/64 foundation nor 202/224 trench qualifying
completion threshold has been met. Let the existing allocations continue.

CSCS SSH access was checked again at 12:54 and still returns Permission denied(publickey);
its older jobs' final states and checkpoints remain unverified.

[Live runtime receipt](../../../../.artifacts/terra_regression_recovery_20260913/status_20260914_1253/live_runtime.json) ·
[Foundation evaluation](../../../../.artifacts/terra_regression_recovery_20260913/evaluation/foundation_global/fixed.json) ·
[Trench evaluation](../../../../.artifacts/terra_regression_recovery_20260913/evaluation/trench_recovery/fixed.json) ·
[Watcher completion](../../../../.artifacts/terra_regression_recovery_20260913/watcher_finished.json)


## September 14, 13:49 CEST: old foundation policies still solve 62–63/64 under current rules

Fresh complete frozen-weight replays under Terra 46738cde and evaluator 866e8e2
solve **62/64 for old zero-cost control u15000** and **63/64 for old 2× u15000**,
versus **11/64 for new global-normalization u41000**. The same old weights had
solved 63/64 and 64/64 in their original reports. The current comparison uses
identical 64-map resets, greedy inference, seed 20260907 and horizon 450. Both
replays have EVAL_DONE markers; independent review validates all 448 rows across
seven historical/current reports and the five actual retained checkpoint hashes.

Old control excavates 99.56% and correctly disposes 99.55%; old 2× achieves
99.94% for both. Each completes every map solved by new u41000, plus 51 and 52
additional maps. The old zero-cost control is the primary foundation reference.

The foundation bank has no trench metadata and receives no trench-yaw restriction.
All 70 native environment fields agree except live episode counters, and all
1,280 local training reset arrays match their manifest. Global soil/chassis and
movement repairs do affect foundations; the known intermediate-rounding collision
bug is fixed. Current Terra has the exact corrected 7fb30402 runtime tree.

The old runs adapted a pretrained generalist with native Adam/clock state; new
foundation learning started from random initialization. Old u15000 had 491.52M
total transitions versus 671.744M for new u41000. The new scratch u33000 already
scored 8/64 at the old 1×512 layout, before four-GPU normalization changed.
Neither fewer total samples nor later GPU scaling alone explains the large gap.
Initialization, optimizer state, experience distribution, entropy schedule history
and physical-rule training history remain confounded. Successful old-policy
execution does not prove unchanged difficulty of scratch learning.

Keep soil-free chassis protections and zero added costs. The old control is a
stronger recovery-parent candidate; no training jobs, source code or rewards were
changed in this audit. This entry adds historical comparison, not a new live
scheduler check or an evaluation of checkpoints newer than u41000.

[Foundation regression analysis](research/FOUNDATION_LEARNING_REGRESSION_20260914.md) ·
[Fresh report comparison](../../../../.artifacts/terra_foundation_lineage_audit_20260914/current_environment_comparison.json) ·
[Independent review](../../../../.artifacts/terra_foundation_lineage_audit_20260914/independent_review.md).


## September 14, 14:21 CEST: foundation teacher-KL recovery ready locally; CSCS auth blocks submission

The user requested a foundation-only CSCS run initialized from a strong policy
with KL guidance from a frozen teacher. Prepared one 24-hour, four-GH200 Daint
allocation using lterenzi. No new job has been submitted. CSCS SSH again rejects
the certificate with Permission denied(publickey); Euler jobs are unchanged by
this preparation.

Student and teacher use old zero-cost foundation control u15000, SHA
`d1a6c07d9d8a40b7b7b60bd0b54313aa46a9b50fb09c789ed4ebffddb2b488b3`,
which freshly solves 62/64 under current Terra 46738cde. Keep the same easy bank,
seed20260907, current soil/chassis physics, native Adam and entropy clock. The
4×128 layout keeps global512 environments and64 Adam steps/update; global
advantage moments match merged minibatch mathematics. Teacher KL is1.0 at
u15000, cosine-annealing to0 at u35000; value distillation and LR warmup are0.
All added behavior penalties and their ramp remain disabled. Evaluate u17500,
u20000 and u25000 on the same64-map greedy450-step panel, including workspace
and retained-work travel metrics. This is retention/adaptation, not a causal
KL A/B or proof of full-foundation generalization.

Teacher continuation needed fixes for its absolute coefficient origin and
constant-LR optimizer structure. The actual-parent GPU smoke also caught a
missing executable-dig observation selector in the teacher model stub; that
interface is now preserved and validated. Inference clears the teacher origin
when disabling distillation. Seven focused native-teacher and47 existing
training-utility tests pass; independent source/launch review has no findings.

The corrected local1×32 run completes two finite updates and a further native
resume: u15001/u15002/u15003, Adam960064/960128/960192. Teacher KL measures
0.0264/0.0444/0.0516 with a coefficient near1, immutable teacher SHA, finite
student/teacher/Adam/loss and unchanged environment/bank/R2/zero-cost contracts.
The original failed smoke is preserved. The second process still spent about
206 seconds before its first update; cache reuse is not claimed. Full4×128
CUDA/conv/NCCL plus two-update qualification remains an in-allocation gate.

Source and input staging, duplicate-safe submission, and the bounded milestone
watcher are prepared. They are not running remotely. Authentication renewal is
the outstanding submission dependency.

[Teacher-KL run design](research/FOUNDATION_TEACHER_KL_RECOVERY_20260914.md) ·
[Native resume smoke](../../../../.artifacts/terra_foundation_strong_recovery_20260914/local/resume_checks.json) ·
[Manifest](../../../../.artifacts/terra_foundation_strong_recovery_20260914/manifest.json).


## September 14, 14:54 CEST: matched scratch versus pretrained teacher-KL pair ready locally

The user reports better past results from random student initialization with
teacher KL and requested checking that against direct strong-policy initialization.
The unsubmitted single native-recovery proposal is superseded. Prepare one
24-hour CSCS d130/lterenzi node with four GH200s total: two independent2-GPU
arms, scratch_kl(primary) and pretrained_kl(full-model parameters only).
Both start with fresh Adam and clocks0, the same old zero-cost u15000 teacher,
seed20260907, easy foundation bank, current Terra46738cde physics and zero
added behavior penalties. Each arm keeps global512envs, 16,384transitions and
64Adamsteps/update. Constant LR3e-4/entropy0.02 and teacherKL1 cosine-to0 over
20,000newupdates are identical; value distillation and LR warmup are0.

The optional initialization receipt verifies actual first-rollout model, Adam,
reset and RNG states. All57CPUtests pass. Both local1x32 arms complete finite
u1/u2 plus nativeu3, with Adam64/128/192 and zero integrity failures. Initial
reset/RNG/history/teacher and fresh-optimizer hashes match; pretrained model
matches the teacher, scratch differs; both resumes preserve their own u2 state.
Independent review covers arm/parent binding, immutable source/launch/input
hashes and paired metrics. Actual weak-policy fixtures exposed legitimate
missing workspace metrics; aggregation now retains missingness and matched
per-metric sample counts. These are runtime gates, not learning results.

Fixed evaluations will compare both arms atu2500/5000/10000/20000 on the same
64-map greedy450-step panel. Report all-map coverage, common-success workspace
and retained-pose continuity metrics, and secondary raw Terra travel. No
additional allocation or penalty stage is automatic. The full2x256-per-arm
CUDA/conv/NCCL/startup and paired-state gates are mandatory in the allocation.
CSCS still rejects SSH authentication; no new job was submitted. Existing
Euler jobs are unchanged. The old native-only proposal is explicitly disabled.

[Current design](research/FOUNDATION_TEACHER_KL_RECOVERY_20260914.md) ·
[Campaign manifest](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/manifest.json) ·
[Paired local qualification](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/local/paired_checks.json).


## September14,16:32CEST: CSCS foundation teacher-KL initialization pair submitted

Authentication now succeeds as lterenzi on daint-ln004. Submitted one reviewed
24-hour d130/normal job, **4665916**, at16:28:16CEST. It requests one node with
two tasks, each two GH200GPUs and32CPU cores; Slurm confirms totalfourGPUs and
per-task binding. Scratch+KL is the primary arm; pretrained full-model weights
with the same teacherKL are the comparison. Both retain fresh Adam and clocks0,
global512 environments each, identical schedules/maps and zero behavior costs.

Pinned training sources are baselinesa3a2119af014f9afe018ed69b023a2e9c1acc4dc
and Terra46738cde28e455da7c466fc0a2cb64f677d86401. Remote source, launch, bank,
teacher and container hashes match the reviewed staging receipts. There are
no other Terra jobs in the lterenzi CSCS queue at the authentication check;
unrelated running Newton job4663069 was left unchanged.

At16:31CEST job4665916 is PENDING/Priority. The provisional start estimate is
September15 at06:03CEST, not a reservation or proof of startup. There are no
new training updates/checkpoints or evaluations. Both full2x256 startup gates
must pass before production. The queue-aware runtime coordinator is running in tmux
terra_foundation_kl_init_runtime_20260914. Its live pending-state probe passed;
it starts the reviewed26-hour evaluation window when the allocation begins
and captures both full-size startup receipts. Pending time does not reduce
evaluation coverage. No second allocation or penalty phase is automatic.

[Submission receipt](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/submission_receipt.json) ·
[Scheduler receipt](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/scheduler_after_submission.txt) ·
[Live manifest](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/manifest.json).


## September 15, 08:44 CEST: both arms recover strong easy-foundation completion

CSCS job **4665916** is RUNNING on nid005694, started at04:10:32CEST,
with walltime ending September16 at04:10:32CEST. Both full2x256 startup and
paired initialization gates passed. Live production logs have reached about
7,700 updates per arm; both have u7500 checkpoint files. The latest checkpoints
independently downloaded and native-validated for evaluation are u5000.
Training currently takes about2seconds/update per arm (about8,200 transitions/s).

| Arm | u2500 exact completion | u5000 exact completion | u5000 mean dug |
| --- | --- | --- | --- |
| Random student + teacher KL | 62/64 | 63/64 | 99.740% |
| Pretrained student + teacher KL | 62/64 | 63/64 | 99.933% |

All four fixed evaluations are complete:64 identical validation resets each,
greedy450steps, verified checkpoint hashes, Adam counters and pinned sources.
An independent agent checked all256 rows and transition integrity. On the62
common successful maps at u5000, scratch/pretrained productive poses are
6.839/6.823, unique area per productive setup6.136/6.132m², retained straight-line
travel26.574/26.579m, and fresh-workspace adjacency97.480% for both. There are
no exact-pose revisits or ABA returns in this cohort. Retained straight-line
travel is a geometric lower bound, not a Nav2 path measurement.

Scratch fails slot43/mapL00300 at83.33%dug; pretrained fails slot33/mapL00290
at95.72%dug. There is no clear completion or efficiency winner. Both match or
slightly exceed the frozen control's62/64 on this easy panel. This is one paired
seed with a shared teacher, not a no-KL ablation or full-dataset result.
Teacher KL is still active (coefficient about0.854 at u5000); the bounded watcher
is alive and will evaluate both arms at u10000 and u20000. Both numerical
completion gates now pass, but promotion is separate: all added behavior costs
remain zero and no new allocation or penalty phase has been started.

[Live evidence](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/live_status_20260915.json) · [Paired u5000 report](../../../../.artifacts/terra_foundation_kl_init_comparison_20260914/evaluation/paired/5000.json).


## September 15, 23:57 CEST: approved foundation comparison stopped

Canceled CSCS 4665916 as lterenzi after the user approved ending the unchanged
easy-foundation scratch-versus-pretrained comparison. The scheduler no longer
lists it, and accounting reports cancellation at 23:55:43 with elapsed time
19:45:11. Both arms' u20000 and u30000 checkpoints were verified and preserved
before cancellation. The separate broad generalist 4672272 remains running;
no other allocation was changed. Stop and preservation receipts are under
`.artifacts/terra_foundation_kl_init_comparison_20260914/stop_20260915/`.

The practical decision follows complete same-panel results: both arms solved
64/64 at u20000 and 63/64 at u30000 after 10k teacher-free updates. Common-success
workspace and retained-travel gains were small. This does not establish an
asymptotic plateau or a causal effect of removing KL; it ends this unchanged
comparison and preserves the stronger completion checkpoint for the next test.


## September 16, 00:43 CEST: one delayed-cost comparison queued

Submitted CSCS **4675576** at 00:35 as one four-GPU node with two matched 2-GPU
arms. Both resume the successful scratch u20000 parent and native Adam. Control
keeps costs zero; treatment ramps over 2,500 updates to lateral 0.125, travel
0.0025 and turn 0.01, then holds for 2,500. The u25000 target is 81.92M new
transitions per arm. Teacher coefficient remains zero. The source, bank, PPO,
architecture, physical rules and observation contract are unchanged.

Both local CUDA startup/native-resume gates pass, with exact matching initial
parameters and Adam/reset/RNG/history. The actual penalty ramp continues from
u20002 instead of restarting. The current evaluator reproduces 64/64 frozen
parent completions. Independent review passes and all staged source, launch,
input, checkpoint and container hashes match. Full 2x256 GH200 startup remains
mandatory inside the allocation; no cluster learning or quality result is claimed.

The first scheduler estimate was 02:07, too late for a five-hour request to
finish before the explicit 06:55 deadline. The same unstarted job was held at
00:37, changed to a four-hour ceiling with a matching 02:55 start cutoff,
independently reviewed, and released at 00:42. Original and adjusted payload
receipts are archived under admission_adjustment/. Slurm TimeLimit is the
authority for the cached submitted batch. At release the job remains PENDING.
No second allocation was submitted and the update budget is unchanged.

The allocation will replay the parent and both u22500/u25000 milestones on the
same 64-map panel, then write paired coverage/efficiency and stage-gate receipts.
At least 63/64 at both treatment milestones is necessary for considering the
behavior gains. No automatic next stage or continuation is permitted by this
campaign. Four hours is a ceiling; incomplete work is reported as incomplete.

The separate broad generalist **4672272** continues unchanged. Its u2500/u5000
checkpoints are downloaded with matching hashes and pass CPU native validation.
A finite local GPU batch started at 00:35 for those exact two checkpoints, each
on the easy 64 and full 608 panels. Source/launcher/checkpoint/panel receipts
are captured before inference; complete final validation is required. No
quality comparison is available at this snapshot.


## September 16, 00:56 CEST: both delayed-cost arms pass full cluster startup

Job 4675576 started at 00:42:39 on nid005363 and ends at 04:42:39 CEST. Both
2x256 arms completed u20001/u20002 and saved finite native checkpoints after
CUDA/conv/NCCL preflight. Paired initialization passes and all four physical
GPUs are disjoint across the two processes. Downloaded FINAL checkpoints match
remote SHA256 and pass local CPU validation. Production initialization restores
each exact model and Adam at step 1,280,128. The original ramp is preserved.
The second startup update measured roughly 9.1k transitions/s per arm; production
was still compiling in the latest captured log. The first u20500 save remains
outstanding, with a recommended unscheduled check at 01:20 CEST.

Separately, the generalist u2500 replay completed and passed final source,
native-state, panel/reset and integrity validation: 49/64 easy foundations,
173/384 broad foundations, 175/224 trenches and 19/32 road trenches. The u5000
replay is running, so the paired milestone comparison is incomplete. No new
penalty-model quality or broad-generalist saturation conclusion is drawn.
