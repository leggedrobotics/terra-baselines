# Terra Baselines Research Context

## Objective

Train and evaluate a global excavation-planning policy for Terra that handles
trench and foundation tasks, including bulk excavation, precise edge finishing,
navigation/reorientation, and legal dumping. The goal is a reproducible
generalist planner whose exported actions or plans can later be consumed by the
Moleworks ROS execution stack.

The sibling `terra` repository owns environment dynamics and map generation.
This repository owns optimization, models, experiment configuration,
evaluation, inference, checkpoints, and cluster execution.

## Canonical entry points

- [Delayed retained-work efficiency](docs/research/RETAINED_EFFICIENCY_20260921.md): CSCS4729577 runs on one four-GPU node; both arms pass native update/resume qualification. Control reached and savedu110000; final evaluation is running. Its completedu107500 panel passes381F/212T/31roads. The25% retained-work arm is qualified throughu105002 and awaits its sequential turn. Ramp2500 updates, hold2500, native Adam preserved; selected parentu105000, previous frozen geometry retained.
- [Current-policy test-time compute](docs/research/TEST_TIME_COMPUTE_20260921.md): Euler14791590 completedPASS. Same u109250 policy improves straight32 from2/32 to26/32 after geometry correction; full608 changes383→382foundations,208→211trenches,29→29roads. Six straight failures and two lost full-panel cases remain. The correction is excluded from the active matched efficiency comparison.
- [Demonstrations and successful-policy rehearsal](docs/research/TRENCH_DEMONSTRATIONS_20260918.md): September21 combined bank passes with100sources/102episodes/11020supervised actions. CSCS4721607 corrected u100000 scores378/384F,209/224T,30/32roads; native qualifications pass, but production never starts due Python checksum API incompatibility. Retry4725717 finished u110000 at381/384F,206/224T,29/32roads and triggered the trench-retention STOP. Selected u105000 scores381/384F,211/224T,31/32roads for the next bounded efficiency comparison. Imitation drift remains unresolved; the completed continuation had zero added costs.
- [Oracle implementation and combined broad continuation](docs/research/ORACLE_FOLLOWUP_20260916.md): remaining time, native actor growth, earlier foundation release and cached teachers; zero added costs.
- [Network capacity and representation diagnostics](docs/research/NETWORK_SCALING_20260916.md)

- [Foundation efficiency with delayed behavior costs](docs/research/FOUNDATION_DELAYED_COSTS_20260916.md): evaluation-only recovery 4709744 completed. Both arms score 63/64 at u22500 and 62/64 at u25000, below the final 63/64 floor. The penalty arm improves retained travel by 10.3% and workspace yield by 4.3% on 60 common successes, but the stage is not accepted.

- [Full generalist using recovered broad teachers](docs/research/GENERALIST_BROAD_TEACHERS_20260915.md)

- [Fresh mixed generalist with separate foundation and trench teachers](docs/research/GENERALIST_TASK_TEACHERS_20260915.md)

- [Foundation scratch versus pretrained initialization with teacher KL](docs/research/FOUNDATION_TEACHER_KL_RECOVERY_20260914.md)

- [Foundation historical competence and scratch-learning regression](docs/research/FOUNDATION_LEARNING_REGRESSION_20260914.md)

- [September 13 overnight recovery screens](docs/EXPERIMENTS_RUNNING.md#september-13-2349-cest-two-authorized-overnight-recovery-screens-submitted)

- [September 13 completion regression and manual behavior audit](docs/research/COMPLETION_REGRESSION_20260913.md)

- [Audited training, PPO, curriculum, and evaluation protocol](docs/TRAINING_PROTOCOL.md)
- [Dataset categories, splits, and terrain figures](https://github.com/leggedrobotics/terra/blob/main/docs/DATASET.md)
- [Environment, reward equations, and exact termination](https://github.com/leggedrobotics/terra/blob/main/docs/ENVIRONMENT.md)
- [Scratch foundation and trench comparison on CSCS](scripts/foundation_reward_sweep/README.md)
- [Foundation behavior and efficiency metrics](docs/research/FOUNDATION_BEHAVIOR_20260907.md)
- [V8 movement-feedback pilot](docs/research/V8_MOVEMENT_FEEDBACK_PILOT_20260821.md)
- [V8 paper-experiment handover](docs/research/V8_PAPER_EXPERIMENT_HANDOVER_20260818.md)
- [V8 fixed-panel benchmark inspector](docs/research/V8_BENCHMARK_INSPECTOR_20260820.md)
- [V8 recurrent actor pilot](docs/research/V8_RECURRENT_ACTOR_GRU_20260817.md)
- [V8 GRU v1 plateau diagnosis](docs/research/V8_RECURRENT_GRU_PLATEAU_DIAGNOSIS_20260817.md)
- [Training and evaluation overview](README.md)
- [`configs/training_configs.yaml`](configs/training_configs.yaml): named
  agent/map/curriculum presets
- `train_mixed.py`: primary mixed/generalist PPO training path
- `eval_fixed_bank.py`: source-identified fixed-panel evaluation
- `eval_mixed.py` and `eval_mcts.py`: online/legacy and optional search paths
- [Single-map inference](inference/README.md)
- [Cluster workflow](cluster/README.md)

## Implemented training contract

### Configuration and environment batching

`train_mixed.py` is the current production PPO entry point. Named YAML presets
select agent/action types, map families, curriculum thresholds, capacities,
border-alignment behavior, and reward multipliers; explicit CLI values can
override the preset. Record the resolved values rather than citing only the
preset name.

The trainer uses `num_devices * num_envs_per_device` parallel Terra
environments. One PPO update contains `num_steps` transitions from every
environment, so global environment steps per update are:

```text
num_steps * num_envs_per_device * num_devices
```

`total_timesteps` and all learning-curve comparisons use this global count.
Changing devices, environments, rollout length, minibatches, or update epochs
changes either the data distribution or optimizer workload and must be reported.

The current full-reset path requires `env_steps == 0`, checked by
`assert_initial_env_steps_zero`; it does not randomize the initial elapsed step.
Legacy policies do not observe elapsed or remaining time. The combined Oracle
continuation enables normalized remaining time for both actor and critic,
using explicit checkpoint migration. Partial-completion resets, where enabled
by the resolved recipe, change the material state rather than shorten the
initial horizon. Their exact schedule is documented in the
[training protocol](docs/TRAINING_PROTOCOL.md).

### PPO optimization and checkpointing

Each update samples categorical policy actions, steps the auto-resetting Terra
batch, computes GAE, shuffles complete environment trajectories into
minibatches, and performs clipped PPO actor/critic updates. Multi-device
training uses `pmap`; terminal counts and success metrics must be explicitly
reduced across devices before host logging.

The entropy coefficient normally follows a cosine schedule. Checkpoints with
the current format store model parameters, optimizer state, train-state step,
environment configuration, and the next update index. Resume restores those
items when present, but environment state, RNG, and previous-action history
restart, so a resumed run is not bit-exact. Older parameter-only checkpoints
are warm starts with a fresh optimizer unless their continuation semantics are
recorded explicitly.

Optional actor imitation now uses separate demonstration observations/actions,
the normal preprocessing path, and a coefficient fading in global transitions.
PPO rollouts, advantages and value targets remain on-policy. Its checkpointed
origin survives native resume; the extra demonstration forward pass stops after
release. A diagnostic one-GPU continuation from u25000 to u25002 preserved Adam
and produced finite state. This verifies implementation, not foundation
retention or improved trench completion. The original ten-plan bank has only
four sources and no foundations, so it is not the selected production dataset.
All ten plans now pass native float16-metadata replay with unchanged actions
and reset seeds; regenerated observations supersede the earlier helper export.
The broader archive now exports 1,280 successful plans across all 40 foundation
and trench conditions: 116,265 transitions from 784 sources, about 4.93 GiB of
raw arrays per bank copy. This does not qualify production training memory.
Cached soft foundation targets, explicit group/condition/source sampling and
recovery masks are now implemented and CPU-tested. The selected single-run
recipe uses 50%/20%/30% foundation/trench/expert guidance, beta 0.01 and 16
auxiliary examples per device/minibatch, fading over 750 updates with a
conditional 750-update hold. The selected bank has 331 plans:
200 foundations, 119 ordinary trenches and 12 experts from six expert sources.
The local grouped-bank GPU smoke passes u35000 to u35001 at 1x128, preserving
Adam through step 2,240,064 and reading 27,613 transitions/27,612 eligible labels.
It is diagnostic only. CSCS job 4709420 subsequently passed exact four-GPU
cuDNN/NCCL, finite-update, native save/resume and fade-boundary checks. It ran
1:41:01 and stopped as designed at u35750: Slurm FAILED/exit 1 records the
retention exception, not an infrastructure failure. Same-runtime, same-CSCS
panels pass validation and improve foundations from 369/384 to 383/384
(14 gained, none lost), but regress trenches from 213/224 to 205/224
(one gained, nine lost). The net eight-case trench loss exceeds the allowed two.
Road trenches decline from 30/32 to 29/32; overall completion rises 582/608 to
588/608, which does not override the family retention gate. The hold did not
start; there is no promotion or automatic retry. All added costs remain zero.

The 369 common foundation successes show no efficiency gain: retained distance
37.84 to 38.76 m, unique area/setup 6.618 to 6.586 m², adjacency 86.02% to 85.51%,
and lateral score 0.4955 to 0.5253. All nine lost trenches time out at 450 actions;
the parent supplies successful witnesses under the same physics. Their losses
cannot be assigned to bank infeasibility, and imitation versus ordinary PPO
drift is not isolated. Euler 14555235 completed in 1:53:21 with 25 new plans from
25 sources (23 straight, two compact T junctions), totaling 837 actions.
Readback of these new collection outputs is complete; they were not part of the
completed 331-plan training bank and have not been merged. Combining them with
the existing 12 expert plans would yield 37 plans from 31 sources: 27 straight,
three T junctions and one network. Only four sources are junctions, leaving a
substantial coverage gap.

On September 18, CSCS access was restored and existing job 4685246 was running
on `nid006532`; dependents 4685248/4685249 remained pending, with no scheduler
changes. Complete u42000 checkpoint metadata was fetched at 15:46 UTC. The
historical u35000 full panel is 374/384 foundations and 209/224 trenches,
with paired gains/losses of 14/7 and 17/3 relative to u20000. Both panels use the
runtime before the local boundary fix. The local corrected-runtime u35000 panel
completes 369/384 foundations, 213/224 trenches and 30/32 road cases. The fresh
CSCS parent panel reproduces those counts and supplies the controlled baseline
for the stopped imitation screen. At the 20:01 UTC update, old job 4685246 had
timed out with u48750 preserved; native successor 4685248 was running with
u49250 saved and about 17.3k end-to-end transitions/s, while 4685249 remained
dependency-pending. Keep this chain because the candidate failed retention.
The 750-update imitation stop is a conservative retention screen, not a plateau
or rejection of all imitation. Keep the already-running broad policy through
its u50000 evaluation. Historical old-runtime trenches improved u20000 to
u35000 from 195/224 to 209/224 and 96.34% to 98.86% dug; this supports allowing
continued learning, but is not a matched current-runtime control. The next
recommended u50000-panel check is 20:45 UTC; no watcher is scheduled.
See the demonstration research note and `boundary_fix/cscs_4709420_status_2001/`
for paired reports, failure details and the bounded geometry-audit scope.

The September 21 efficiency stage starts from qualified u105000, with a linear
81.92M-transition ramp (2,500 updates at 4x256x32) and an equal hold. Control
and treatment both observe the previous effective work pose through initially
zero actor/critic projections; native Adam leaves and clocks are preserved.
The ramp now supports all six costs and restores historical three-cost stages
unchanged. Effective work includes excavation, dumping and relifting; lateral
cost applies only to fresh excavation, and raw navigation costs remain zero.
The one-node comparison and its completion/behavior criteria are recorded in
[retained-work efficiency](docs/research/RETAINED_EFFICIENCY_20260921.md).
Its frozen environment excludes the separately validated September 21 geometry
correction. The earlier easy-map delayed-cost stage is complete and failed its
final completion floor; it is not evidence for promoting the new treatment.

### Policy inputs and architecture

The shared actor-critic combines:

- embeddings/MLPs for up to four active-agent states;
- the acting agent's nine local terrain, target, dumpability, obstacle, and
  foundation-edge maps;
- a finite previous-action history; and
- seven global maps: traversability, optional reachability, action state,
  target, padding/obstacle, dumpability, and current interaction workspace.

Those seven are source layers, not the final channel count for every encoder.
The current attention encoder adds remaining-dig, dump-deficit, and two spatial
coordinate channels. The September observation variants add a relocation
distance channel and a tenth local admissible-dig feature. Use the resolved
encoder and observation table in the [training protocol](docs/TRAINING_PROTOCOL.md)
when specifying policy input dimensions.

The default `atari` encoder is a compact CNN. `resnet_global_pool` preserves the
older residual topology and checkpoint preprocessing. `resnet_spatial_8x8`
keeps an 8 x 8 spatial readout before its dense projection. Encoder names and
preprocessing are part of the checkpoint API; compatibility aliases must not be
silently repurposed.

`model_size=medium` and `large` widen the configurable Atari and spatial
encoder/head paths. The current `resnet_global_pool` topology has fixed channel
sizes, so model-size language must not imply that every encoder is widened.

The normal MLP core concatenates encoded features before separate categorical
policy and scalar-value heads. A lightweight transformer core exists as an
experimental alternative, not as the paper baseline unless a pinned experiment
uses it.

Two observation details matter for scientific claims:

- the normal PPO path samples unmasked logits even though Terra can compute
  action feasibility; and
- `clip_action_maps=True` clips the global action map to `[-1, 1]`, aliasing
  positive pile heights before the model sees them.

Treat masking and height-preserving observations as controlled research
variables, not undocumented implementation details.

## Evaluation contract

Inline evaluation uses an environment-step budget. Because environments
auto-reset, legacy counts such as `eval/positive_terminations` and
`eval/total_terminations` are episodes per initial environment and may exceed
one. They are not probabilities.

The current `eval_episodes` CLI/config field does not control this evaluation.
Inline evaluation instead runs `num_rollouts_eval=200` environment steps over
the full training-shaped cohort. Many configured task horizons are 450-800
steps, so the default inline metric censors the initial cohort by construction.
Treat it as a frequent training diagnostic, not a paper evaluation.

Use:

- `online_eval/success_within_horizon_rate` as the bounded online metric: the
  fraction of the initial reset cohort that succeeds before the fixed horizon;
- `online_eval/termination_within_horizon_rate` to expose censoring at that horizon;
- `online_eval/completed_episode_success_rate` only as a secondary success-among-ended
  measure; and
- `train/episode_success_rate` as a bounded online diagnostic, reported as NaN
  when no episode ends in the rollout window.

Older logs use `eval/*` names. These online fields remain training-bank
diagnostics; the current paper-oriented path is `eval_fixed_bank.py` on a
declared source-disjoint panel, with exact success evaluated at 450 actions.

`progress/episode_completion_rate` is a legacy final-step termination fraction
that includes timeouts. It must not be reported as task success.

Paper-level checkpoint selection and promotion require a separate, pinned,
source-disjoint fixed bank. Report at least task success within horizon, raw
productive workspace count, raw steps, failure/timeout strata, map-family
breakdown, action mode, checkpoint hash, Terra revision, and evaluation code
revision. Online return or pooled online success is diagnostic only.

The current standalone `eval_mcts.py` is not yet that promotion evaluator:

- it enables `shuffle_maps=True`, which flattens and reshapes curriculum levels
  and therefore destroys per-family identity;
- it records no source map IDs, manifests, or dataset hash;
- it treats `task_done` as success without explicitly intersecting it with the
  terminal `done` event, unlike inline evaluation;
- its terminal/horizon branch can retain the pre-action observation for final
  coverage and workspace-change calculations; and
- its reported workspace efficiency is a successful-episode proxy derived from
  action-map change events, not a raw count of distinct executed workspaces.

Fix and regression-test those contracts before using standalone output for
paper tables or checkpoint promotion.

MCTS in `eval_mcts.py` is an optional inference-time policy improvement that
uses learned PPO policy priors and value estimates while stepping the exact
Terra simulator in its recurrent function; it is not a separately learned world
model. Report its simulation budget, discount, JIT warmup exclusion,
changed-decision count, and raw workspace/step outcome separately from plain
PPO. It is not part of the trained policy unless explicitly stated.

## Plan extraction and ROS handoff

`inference/inference_single_map.py` produces a rendered rollout and scalar
summary; it is not the deployment-plan path. `isaac_sim/extract_map.py`
separately records terrain-modifying load/unload events, and its plain-PPO path
currently samples stochastically rather than offering deterministic argmax.
`isaac_sim/serialize_plan.py` converts matched load/unload pairs into schema-v2
JSON:

- each pair becomes a dig and dump waypoint with masks and agent state;
- `pos_base` remains a floating-point plan-grid coordinate, discrete headings
  become radians, and alignment metadata supplies metric scale, origin, and
  plan-to-map yaw;
- unpaired load or unload events are retained as metadata instead of silently
  becoming executable waypoints.

This serializer establishes a file-format boundary, not execution success.
The downstream `moleworks_ros` loader independently validates schema, digest,
alignment, masks, and waypoint pairs before the BehaviorTree can execute them.
Record both the raw rollout and serialized plan hash for a deployment result.
The current schema does not carry the policy checkpoint/config hash, seed,
planning mode, source map hash, or final task-success provenance; preserve
those in the surrounding experiment manifest.

Fresh extraction and standalone re-serialization are not currently equivalent
when runtime map resolution differs from metadata/defaults. The current
`5c985fe` extraction path injects the resolved Terra tile size into fresh
schema-v2 output, but standalone `serialize_plan.py` can still fall back to
`0.1 m/tile`. Keep plan resolution provenance explicit and test both paths
before relying on re-serialized plans.

## Checkpoint and logging limitations

Current version-2 checkpoints store parameters, native Adam state and clocks,
resolved training/environment configuration, losses and transition-integrity
counts. When enabled, they also store the R2 protocol, adaptive sampler,
reward-annealing and partial-reset records. The audited experiment recipes
retain numbered checkpoints every 500 updates. The generic rolling-save
default is a different setting.

Resume still restarts RNG, live environment state, action history and recurrent
carry. Source versions, exact bank/evaluation identities, checkpoint checksum
and completed segment budgets must remain available in the surrounding
experiment records. See the [training protocol](docs/TRAINING_PROTOCOL.md) for
the precise resume and changed-batch accounting rules.

Current online success/completion statistics are reduced across devices, and
inline `online_eval/*` logging includes `train/update`. Training also writes
episode aggregate records. These remain diagnostics under the live sampling
distribution; fixed source-disjoint evaluation supplies the comparative
endpoint.

## Current research threads

### Generalist global planner

Train across both foundations and trenches, then evaluate by map family and
completion phase rather than relying on a pooled scalar. Preserve the exact
Terra revision, map dataset, action/agent types, encoder, reward configuration,
seed, and checkpoint hash with every comparison.

### Foundation edge finishing

Hard edge constraints create state-dependent action feasibility and a distinct
endgame phase. The current diagnosis and literature-backed intervention order
are documented in:

- [Edge-digging RL research brief](docs/edges_trainings/brief.md)
- [Recovered literature follow-up](docs/edges_trainings/deep_research_recovered_followup.md)
- [Manual source-backed notes](docs/edges_trainings/manual_literature_notes.md)

Treat action masking, explicit edge affordances, phase-aware value prediction,
and targeted edge curricula as hypotheses to test through controlled ablations.
Re-check the current code before assuming the older notes' branches or line
numbers still match.

## Research hygiene

- Keep live jobs and checkpoint decisions in a dated experiment ledger when a
  training campaign starts; do not put volatile scheduler state in `AGENTS.md`.
- Compare checkpoints only under the same environment, dataset, and evaluation
  contract, or label the comparison as a new epoch.
- Export enough map, frame, mask, and policy metadata for downstream ROS plan
  execution to reproduce the result.
- Do not describe a finite first update, active scheduler job, online return,
  or successful serialization as generalization or deployment evidence.
- Keep environment semantics in `terra`, training/evaluation evidence here, and
  real/sim execution evidence in `moleworks_ros`.

## Mixed V8 curriculum

This section describes the adaptive mixed V8 lineage. The September v2
generalist/trench pools and foundation screen instead use one uniformly sampled
level, with no adaptive condition sampler or partial resets. Their resolved
settings are in the [training protocol](docs/TRAINING_PROTOCOL.md).

The current method is
[Continuous Banded v3](docs/research/CONTINUOUS_BANDED_V3_DESIGN_20260812.md).
It is one uninterrupted process over all 47 V8 conditions:

- 80% of assignment mass is distributed globally over open conditions with
  immutable depth weights `4:2:1`;
- 20% is uniform replay over mastered conditions;
- no foundation/trench quota affects sampling;
- a 15% per-condition water-fill cap prevents a single-condition monopoly;
- exact-success EMA with minimum exposure controls mastery and demotion; and
- source-disjoint fixed panels audit checkpoints but never update the sampler.

Earlier family-balanced samplers are historical experiment provenance, not
selectable training modes. The selected v6.1 update-14,000 source is converted
offline into a native v3 checkpoint; runtime training has no compatibility or
sampler-migration mode.

Here, depth is immutable map-difficulty metadata and band is a changing sampler
role. Online success is weighted by the live sampler distribution; it is not a
whole-V8 benchmark result. Map allocation and reward design remain separate
causal variables.

The completed primary experiment held reward dense and trained two random-start
all-47 controls: the 2.856M compact deep+xattn policy and the original 480k
Atari-base policy. They shared the map sampler, transition budget, PPO shape,
seed, horizon, and fixed evaluations. The Atari policy is a deliberately small
system control, not a pure encoder ablation, because its actor, critic, and
local-map heads are also smaller.

The completed capability run continued the selected v6.1 reward-v2 checkpoint
with two declared changes: Continuous Banded v3 and one normalized material
stall-age observation. Reward-v2, the v6.1 spatial architecture, action-mask
setting, PPO shape, learning rate, horizon, bank, and seed remain fixed. This
is a practical combined treatment, not a causal sampler or observation
ablation. Fixed source-disjoint panels—not reward return or online success—
provide its behavioral evidence: the final checkpoint reaches 657/720 exact
versus 407/720 for the matched source, while five selected recurrent failures
and high late-checkpoint churn remain.

The reward and termination audit preserves exact success, separates strict
completion from continuous material progress, and records the diagnostics and
analytic admission gates that preceded R2. Its fixed-checkpoint experiment
proposal superseded the older sampler-depth reward trigger and whole-objective
fade.

The implementation, fixed-panel evidence, failure-mechanism readout, and
bounded next actions are tracked in
[V8 failure-remediation execution](docs/research/V8_FAILURE_REMEDIATION_EXECUTION_20260814.md).

The mechanism is informed by, but does not copy constants from Prioritized
Level Replay, Self-Paced Deep Reinforcement Learning, Replay-Guided Adversarial
Environment Design, ACCEL, and C-Procgen. These sources motivate the mechanisms
only; fixed source-disjoint Terra evaluations decide whether the chosen masses
and thresholds work.
