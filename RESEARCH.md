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

- [V8 recurrent actor pilot](docs/research/V8_RECURRENT_ACTOR_GRU_20260817.md)
- [Training and evaluation overview](README.md)
- [`configs/training_configs.yaml`](configs/training_configs.yaml): named
  agent/map/curriculum presets
- `train_mixed.py`: primary mixed/generalist PPO training path
- `eval_mixed.py` and `eval_mcts.py`: evaluation paths
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

At the first reset, the trainer randomizes each environment's elapsed step
within its configured timeout, but the policy input does not expose elapsed or
remaining episode time. This creates hidden horizon state in the first training
episode and is a known experimental limitation.

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

### Policy inputs and architecture

The shared actor-critic combines:

- embeddings/MLPs for up to four active-agent states;
- the acting agent's nine local terrain, target, dumpability, obstacle, and
  foundation-edge maps;
- a finite previous-action history; and
- seven global maps: traversability, optional reachability, action state,
  target, padding/obstacle, dumpability, and current interaction workspace.

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

- `eval/success_within_horizon_rate` as the primary bounded metric: the
  fraction of the initial reset cohort that succeeds before the fixed horizon;
- `eval/initial_episode_completion_rate` to expose censoring at that horizon;
- `eval/completed_episode_success_rate` only as a secondary success-among-ended
  measure; and
- `train/episode_success_rate` as a bounded online diagnostic, reported as NaN
  when no episode ends in the rollout window.

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

Current checkpoints are sufficient for a warm optimizer continuation, but they
do not store repository revisions, dataset/map identities or hashes, evaluation
bank, RNG, live environment/curriculum state, previous-action history, or their
own checksum. Saves overwrite pickle targets directly rather than through an
atomic/best-model retention protocol.

Global online success/completion counts are correctly reduced across devices.
Some reward-component and terminal-completion diagnostics are still taken from
device-local or final-timestep values, and inline evaluation logging does not
share the explicit PPO-update step used by training logs. Use these fields for
debugging only until population reductions and episode receipts are audited.

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

## Current V8 curriculum

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
