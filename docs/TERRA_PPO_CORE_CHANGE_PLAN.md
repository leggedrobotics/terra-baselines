# Terra PPO Core Change Plan

Date: 2026-05-19

Sources:

- `/home/lorenzo/terra-ppo-oracle-recommendations.md`
- `/home/lorenzo/terra-ppo-oracle-no-deepresearch-recommendations.md`
- `docs/LARGER_NETWORK_DEEP_RESEARCH.md`

Scope: improve the current Terra PPO stack for the `medium_deep` and
`large_deep` delayed-ResNet runs. Keep PPO, the primitive action interface, the
unmasked actor default, terrain-derived map channels, critic-only affordances,
timeout bootstrap, terminal reward gating, GPU preflight, and early checkpoint
snapshots.

Action masking is not a planned feature in this change set. It remains a
diagnostic/future exact-semantic-mask ablation only.

## Current Diagnosis

The current `medium_deep` versus `large_deep` comparison is not a clean
architecture comparison. It changes model size and PPO optimization geometry at
the same time:

| run kind | envs/GPU | steps | minibatches | global samples/update | global samples/minibatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `medium_deep` | 1024 | 32 | 64 | 131072 | 2048 |
| `large_deep` | 1024 | 32 | 128 | 131072 | 1024 |

The large model therefore gets a smaller, noisier PPO minibatch while also
using the same high post-imitation entropy schedule. Because advantages are
currently normalized inside each PPO minibatch, the two runs also get different
advantage normalization statistics. A poor `large_deep` result would not prove
the 10M architecture is worse.

The strongest shared recommendation from the two Oracle notes and two follow-up
code reviews is:

1. first make PPO diagnostics and config truthful;
2. then fix the handoff and minibatch geometry;
3. only then test architecture variants.

First implementation batch:

- PPO diagnostics.
- Global advantage normalization.
- Comparable effective SGD batch for `medium_deep` and `large_deep`.
- CLI-exposed PPO/entropy knobs.
- Lower entropy handoff.
- Deterministic eval.
- `--load_env_from_checkpoint` default fix.
- One compute-constrained YOLO architecture run with a bottlenecked spatial
  encoder and critic-heavy trunk.

Explicitly postpone until after that batch:

- teacher-KL regularization during PPO;
- flattened env-time minibatches;
- full architecture ablation ladders such as `large_deep_thin` versus many
  critic-heavy widths, and full actor/critic encoder separation;
- reward changes.

Because compute is limited, the first launch does not need to exhaustively
ablate `medium_deep`, `large_deep`, `large_deep_thin`, and critic-heavy
variants. The practical plan is to keep the PPO cleanup needed for a meaningful
run, then spend the next expensive slot on the best single architecture bet.

## P0: Truthful Config, Resume Semantics, And Diagnostics

These changes should land before any new 120h relaunch.

### 1. Fix `--load_env_from_checkpoint` default

Target:

- `train_mixed.py` CLI parser.

Problem:

- `MixedAgentTrainConfig.load_env_from_checkpoint` defaults to `True`.
- The CLI comment also says default true.
- But `argparse` `store_true` defaults to `False` when the flag is absent, so
  the post-parse `if args.load_env_from_checkpoint is None` branch does not
  fire.

Change:

- Set the parser default explicitly:

```python
parser.set_defaults(load_env_from_checkpoint=True)
```

Acceptance:

- A parser smoke proves no flag gives `True`, `--no-load-env-from-checkpoint`
  gives `False`, and `--load_env_from_checkpoint` gives `True`.

### 2. Rename or implement true resume

Target:

- `train_mixed.py` checkpoint save/load path.

Problem:

- `--resume_from` restores model params and optionally env config, but not PPO
  optimizer state, RNG, current update, env state, or entropy schedule position.
- For the current workflow this is really `init_from`, not a true resume.

Plan:

- Short term: rename docs/help text to "initialize from checkpoint" and avoid
  calling this a resume in run ledgers.
- Medium term: add a true resume checkpoint with `params`, `opt_state`, `rng`,
  update index, train config, env config, and enough runner state to restart the
  entropy schedule and optimizer without replaying update zero.

Acceptance:

- Restarting from a true resume checkpoint logs the restored update and entropy
  coefficient, not a fresh update-zero schedule.

### 3. Add PPO trust-region diagnostics

Targets:

- `train.py::ppo_update_networks`
- `train_mixed.py::make_mixed_agent_train()._update_step`

Log at minimum:

- `train/approx_kl`
- `train/clip_fraction`
- `train/value_clip_fraction`
- `train/ratio_mean`
- `train/ratio_std`
- `train/adv_mean`
- `train/adv_std`
- `train/target_mean`
- `train/target_std`
- `train/value_mean`
- `train/value_std`
- `train/global_grad_norm`

Interpretation:

- Low KL plus low clip fraction: under-updated policy.
- High KL or high clip fraction: PPO step too aggressive or minibatch too
  noisy.
- Grad norm always clipped: reduce LR or increase effective minibatch.
- Entropy remains high after imitation: handoff is too exploratory.

Acceptance:

- CPU validation covers metric shapes and finite values.
- W&B rows for update 1 include the diagnostics for both medium and large smoke
  runs.

### 4. Make PPO knobs CLI-controllable and captured

Targets:

- `train_mixed.py::MixedAgentTrainConfig`
- `train_mixed.py` CLI parser and config construction.
- Slurm launchers.

Expose:

- `--clip_eps`
- `--gamma`
- `--gae_lambda`
- `--vf_coef`
- `--max_grad_norm`
- `--ent_schedule_start`
- `--ent_schedule_end`
- `--ent_schedule_steps`

Also make untyped config values dataclass fields so they are captured in
`asdict`/W&B:

- `num_prev_actions`
- `clip_action_maps`
- `local_map_normalization_bounds`
- `maps_net_normalization_bounds`
- `loaded_max`
- `cache_clear_interval`

Acceptance:

- W&B config and checkpoint `train_config` include all PPO and model-shape
  fields needed to reproduce a run.

### 5. Fix misleading curriculum/agent logging for solo runs

Targets:

- `train.py::get_curriculum_levels`
- `train_mixed.py` call site.

Problem:

- `train_mixed.py` currently calls `get_curriculum_levels(..., timestep=None)`.
- The helper falls back to a fake mixed-agent default, so solo runs can log
  misleading tracked/skidsteer agent counts.

Change:

- Pass the current `timestep` when logging curriculum stats, or remove the
  fallback agent-type counts from solo run logs.

Acceptance:

- Solo-excavator W&B rows no longer report fake second-agent counts.

## P1: PPO Geometry And Handoff Fixes

These are the first behavioral changes. They should be implemented behind flags
and tested with short local/Euler smokes before 120h jobs.

### 6. Normalize advantages globally before minibatching

Targets:

- `train.py::calculate_gae` caller path.
- `train_mixed.py::make_mixed_agent_train()._update_step`.
- `train.py::ppo_update_networks`.

Problem:

- Advantages are currently normalized inside `ppo_update_networks`, after
  minibatching.
- Since large uses more minibatches than medium, large gets noisier advantage
  statistics.

Change:

- Add `global_advantage_norm=True`.
- Normalize advantages once after GAE and before minibatching.
- Use cross-device statistics with `jax.lax.pmean`.
- Remove minibatch-local normalization when global normalization is active.

Acceptance:

- Diagnostic logs show global `adv_mean ~= 0`, `adv_std ~= 1`.
- Medium and large use comparable advantage statistics for the same rollout
  shape.

### 7. Match effective SGD batch between medium and large

Target:

- `scripts/euler/terra_train_deep_resnet_4gpu_120h.sbatch`.

Primary ablation:

```bash
--num_envs_per_device 512
--num_steps 64
--num_minibatches 64
```

This keeps total samples/update at `131072`, doubles temporal depth, and gives
both medium and large `2048` global samples/minibatch.

Fallback if large OOMs:

```bash
--num_envs_per_device 1024
--num_steps 64
--num_minibatches 128
```

This doubles samples/update but still gives large `2048` global
samples/minibatch.

Acceptance:

- W&B-disabled full-shape smoke passes on Euler before online training.
- Run docs include samples/update and samples/minibatch, not only env count.

### 8. Lower entropy after imitation

Targets:

- `train_mixed.py` entropy schedule CLI.
- Deep ResNet Slurm launcher.

Initial settings to test:

| run kind | entropy start | entropy end | entropy steps |
| --- | ---: | ---: | ---: |
| `medium_deep` | 0.02-0.03 | 0.002-0.005 | 2000-5000 |
| `large_deep` | 0.01-0.02 | 0.001-0.003 | 3000-5000 |

Rationale:

- Post-distill students already have structured teacher behavior.
- Restarting PPO with entropy `0.15` can erase teacher structure before sparse
  task returns stabilize.

Acceptance:

- Post-PPO teacher drift and eval behavior are logged.
- If deterministic eval improves while stochastic eval fails, entropy remains
  too high.

### 9. Add post-distill handoff gates

Targets:

- `train_mixed.py::_distill_update_step`
- checkpoint/eval scripts.
- docs/run tables.

Add imitation diagnostics:

- `imitation/policy_kl`
- `imitation/value_loss`
- teacher/student argmax agreement
- teacher/student entropy
- held-out teacher KL if cheap enough
- post-distill stochastic and deterministic rollout summary

Default gates before PPO:

- `imitation/policy_kl <= 0.10` preferred, `<= 0.15` acceptable.
- value loss decreasing or `<= 0.03-0.05`.
- post-distill rollout has productive `DO` and terrain-change/action near the
  teacher.
- no high `DO_NOTHING` drift.

Initial schedule:

| run kind | imitation updates | imitation LR | imitation minibatches |
| --- | ---: | ---: | ---: |
| `medium_deep` | 100-300 | 1e-4 | 64 |
| `large_deep` | 300-500 | 1e-4 | 64-128 |

Acceptance:

- PPO launch scripts can run imitation-only ladders and select a
  `_POST_DISTILL_update_XXXX.pkl` based on the gates.

## P2: Evaluation And Clean Comparison Tools

These improve decision quality and should land before interpreting a failed
large run as an architecture result.

### 10. Add deterministic eval alongside stochastic eval

Targets:

- `utils/utils_ppo.py`
- `eval_ppo.py::_rollout_impl`
- `train.py::eval_log_metrics`
- `train_mixed.py` eval call site.

Change:

- Keep stochastic eval.
- Add deterministic argmax eval metrics with a prefix such as `eval_argmax/*`.
- Clarify `eval_episodes`: current eval effectively uses all parallel envs and
  a fixed horizon, not exactly the requested episode count.

Acceptance:

- W&B shows whether a high-entropy policy has a usable argmax policy even when
  stochastic success is poor.

### 11. Make value clipping switchable

Target:

- `train.py::ppo_update_networks`.

Rationale:

- Current PPO uses value clipping.
- The Deep Research brief and PPO ablation evidence suggest value clipping can
  hurt critic learning.

Change:

- Add `value_clip=True` config flag.
- Log `value_clip_fraction`.
- Run a focused ablation with `value_clip=False` only after diagnostics are in.

## P3: Later PPO Stabilizers

Do not include these in the first clean medium-vs-large relaunch. They are
useful only after P0/P1/P2 make the current comparison diagnosable.

### 12. Add optional frozen-teacher KL during early PPO

Targets:

- `train_mixed.py` PPO loss path.
- teacher model load path.

Add config:

```python
teacher_kl_coef_start: float = 0.0
teacher_kl_coef_end: float = 0.0
teacher_kl_anneal_updates: int = 0
teacher_kl_temperature: float = 1.0
teacher_value_coef_ppo: float = 0.0
```

Initial test values:

| run kind | teacher KL start | anneal updates | teacher value coef |
| --- | ---: | ---: | ---: |
| `medium_deep` | 0.03 | 1000 | 0.02 |
| `large_deep` | 0.05 | 3000 | 0.02 |

Acceptance:

- W&B logs `train/teacher_kl` and `train/teacher_kl_coef`.
- If teacher KL jumps while success stays zero, the handoff is forgetting.

### 13. Flatten env-time samples before PPO minibatching

Targets:

- `train.py::make_train()._update_step()._update_epoch`
- `train_mixed.py::make_mixed_agent_train()._update_step()._update_epoch`

Problem:

- Current minibatches shuffle envs and keep complete rollout fragments
  together.
- The model is feed-forward and previous actions are already in the
  observation, so full trajectory grouping is not required.

Change:

- Add `flatten_time_minibatches=True` flag.
- Flatten `[time, env]` into samples, shuffle individual samples, and reshape
  into minibatches.

Acceptance:

- Shape tests cover both old trajectory minibatches and flattened minibatches.
- Compare only after global advantage normalization is in place.

## P4: Architecture Follow-Ups

Do not make the immediate next experiment "even larger." The compute-constrained
path is to add one bottlenecked/critic-heavy architecture now, then leave the
remaining variants as later follow-ups if the YOLO run is inconclusive.

### 14. Add `bottleneck_critic` as the first architecture bet

Target:

- `utils/models.py`
- `train_mixed.py` model-size choices.
- `scripts/euler/terra_train_deep_resnet_4gpu_120h.sbatch` or a dedicated
  launcher.

Hypothesis:

- Terra needs more spatial reasoning and value-function capacity, not just a
  wider final feature vector.
- A bottlenecked encoder increases effective receptive field by doing many
  residual blocks at lower spatial resolution.
- A critic-heavy trunk gives the value function more capacity without making
  the deployable actor larger or adding actor-visible affordances.

Shape:

```text
model_size = bottleneck_critic
map_encoder = resnet_bottleneck
map_feature_dim = 384

encoder:
  high-res stem: 2-3 residual blocks
  downsample to mid resolution
  bottleneck residual stack: 6-10 blocks
  optional second downsample if map is still large
  global avg/max pool
  dense to 384

actor:
  trunk = (384, 192)
  pi head = (192, 64)

critic:
  trunk = (768, 384, 192)
  value head = (512, 256, 1)
  receives critic affordances + episode_progress
```

Keep for this first architecture run:

- shared spatial encoder;
- separate actor/critic trunks after the shared representation;
- critic-only edge/progress affordances;
- unmasked actor;
- no full actor/critic encoder duplication.

Why not full actor/critic encoder separation yet:

- Lux-style PPO agents generally use separate policy/value outputs and often
  separate heads, but public evidence does not require fully duplicated visual
  encoders as the first move.
- Fully separate encoders double the expensive conv path and add a large memory
  and optimization confound.
- Shared bottleneck encoder plus critic-heavy trunk tests the main hypothesis
  with less risk.

### 15. Add `large_deep_thin`

Target:

- `utils/models.py::_model_size_kwargs`.

Purpose:

- Test deeper spatial capacity with less optimization burden than full
  `large_deep`.

Suggested shape:

```text
model_size = large_deep_thin
map_feature_dim = 384
resnet_channels = (32, 64, 128, 192)
resnet_blocks_per_stage = 4
resnet_pool_dense_dim = 384
intermediate_mlp_layers = (512, 256)
intermediate_mlp_dim = 256
actor_trunk_layers = (512, 256)
critic_trunk_layers = (512, 256)
hidden_dim_pi = (256, 128)
hidden_dim_v = (256, 128, 1)
local_map_hidden_dim_layers_mlp = (512, 128)
```

### 16. Add a critic-heavy preset

Target:

- `utils/models.py`.

Purpose:

- Test the Deep Research hypothesis that Terra needs more critic/value
  capacity without symmetrically scaling the actor.

Preferred direction:

- actor no larger than medium/large baseline;
- critic trunk 2x actor width/depth;
- optionally branch before the shared intermediate MLP in a later ablation;
- keep actor deployment inputs unchanged.

### 17. Optional MLP LayerNorm for large models

Target:

- `utils/models.py` MLP/trunk blocks.

Rationale:

- ResNet blocks already use normalization, but large MLP/trunk stacks do not.
- Treat this as a secondary stability ablation, not the first fix.

## Slurm Launcher Changes

Target:

- `scripts/euler/terra_train_deep_resnet_4gpu_120h.sbatch`.

Changes:

- Require `RUN_KIND`; do not default to `medium_deep` for a 120h job.
- Add environment overrides for PPO and entropy knobs:

```bash
LR
CLIP_EPS
GAE_LAMBDA
VF_COEF
MAX_GRAD_NORM
ENT_START
ENT_END
ENT_STEPS
NUM_ENVS_PER_DEVICE
NUM_STEPS
NUM_MINIBATCHES
UPDATE_EPOCHS
IMITATION_UPDATES
IMITATION_NUM_MINIBATCHES
```

- Keep `--xla_gpu_autotune_level=0` as a smoke/debug or explicit override, not
  an unconditional long-run default, unless full-shape large runs still fail
  without it.
- Print a run-geometry table before both smoke and online training.

## First Relaunch Recipes

Run these only after P0/P1 diagnostics are in.

### Medium

```bash
RUN_KIND=medium_deep
NUM_ENVS_PER_DEVICE=512
NUM_STEPS=64
NUM_MINIBATCHES=64
UPDATE_EPOCHS=2
LR=0.0002
CLIP_EPS=0.15
GAE_LAMBDA=0.97
ENT_START=0.02
ENT_END=0.002
ENT_STEPS=2000
IMITATION_UPDATES=300
```

### Large

```bash
RUN_KIND=large_deep
NUM_ENVS_PER_DEVICE=512
NUM_STEPS=64
NUM_MINIBATCHES=64
UPDATE_EPOCHS=2
LR=0.0001
CLIP_EPS=0.10
GAE_LAMBDA=0.97
ENT_START=0.01
ENT_END=0.001
ENT_STEPS=3000
IMITATION_UPDATES=500
```

If the large `NUM_MINIBATCHES=64` smoke OOMs, rerun with
`NUM_MINIBATCHES=128` and keep `NUM_STEPS=64`.

### YOLO Architecture Run

Use the same PPO cleanup settings, but swap the architecture:

```bash
RUN_KIND=bottleneck_critic
NUM_ENVS_PER_DEVICE=512
NUM_STEPS=64
NUM_MINIBATCHES=64
UPDATE_EPOCHS=2
LR=0.0001
CLIP_EPS=0.10
GAE_LAMBDA=0.97
ENT_START=0.01
ENT_END=0.001
ENT_STEPS=3000
IMITATION_UPDATES=500
```

If the bottleneck architecture OOMs, first increase `NUM_MINIBATCHES` to `128`
before shrinking the model.

## Decision Rules

At post-distill:

- Do not start PPO if teacher KL is still high or post-distill rollout has
  much worse productive `DO`/terrain-change than the teacher.

At PPO update `500`:

- Kill/relaunch if completion is below about `0.45-0.50`, final dig coverage is
  below about `0.50`, `eval/DO < 0.05`, or `eval/DO_NOTHING > 0.25`.
- Continue if completion/dig coverage improve even without success.

At PPO updates `1000-2000`:

- Kill/relaunch if success is zero and final dig coverage/terrain-change are
  not improving.
- Reduce LR/increase effective minibatch if `approx_kl > 0.03` or
  `clip_fraction > 0.3`.
- Increase LR/epochs only if `approx_kl < 0.005` and `clip_fraction < 0.05`.
- If deterministic eval improves but stochastic eval fails, lower entropy
  before changing architecture.

## Validation Gates

For every implementation phase:

1. `python3 -m py_compile train.py train_mixed.py eval_ppo.py utils/models.py utils/utils_ppo.py scripts/validation/validate_edge_mask_changes.py`
2. focused CPU gates for changed behavior;
3. full CPU validation:

```bash
validate_edge_mask_changes.py --case all --jax-platforms cpu \
  --dataset-path /home/lorenzo/moleworks/terra_data/train --dataset-size 1
```

4. local RTX 4090 smoke for any memory-sensitive shape;
5. Euler W&B-disabled full-shape smoke before online W&B;
6. update `docs/EXPERIMENTS_RUNNING.md` and `docs/EXPERIMENTS_LOG.md` with run
   geometry, W&B id, post-distill KL, samples/minibatch, and decision status.
