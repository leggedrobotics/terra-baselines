# Terra training and evaluation protocol

Source audit: 8 September 2026. This is the local methods reference for PPO,
policy inputs, curricula, evaluation and continuation. The sibling
[environment reference](https://github.com/leggedrobotics/terra/blob/main/docs/ENVIRONMENT.md) owns dynamics, rewards
and termination; the [dataset reference](https://github.com/leggedrobotics/terra/blob/main/docs/DATASET.md) defines
terrain families, constraints and splits. Results and completed training
budgets belong to their experiment records.

Source links identify public code where available. References marked **local**
are paths relative to the original `/home/lorenzo/moleworks` workspace. Input
banks and unpublished experiment snapshots are separate from this documentation.
The September optional behaviors describe those recorded snapshots and may not
be implemented in the published main branch.

## Which experiment this describes

There is no single recipe called “the current Terra experiment.” The mixed V8
paper pilots, corrected trench-gate training and foundation behavior screen use
different banks, observations and training histories. Their shared optimizer
does not make their outcomes interchangeable.

| Experiment | Training distribution | Policy and initialization | Source of resolved settings |
| --- | --- | --- | --- |
| August V8 feed-forward relay pilot | 47 conditions, 4,512 ordinary map slots; additional partial states for 96 `fnd-slab-apron-d16` maps | Feed-forward actor, fresh initialization, seed 20260815 | [FF launcher](../scripts/run_v8_relay_partial_v1.sh), [paper handover](research/V8_PAPER_EXPERIMENT_HANDOVER_20260818.md) |
| August V8 recurrent relay pilot, concat-skip v2 | Same full-start and partial banks | Actor-only GRU64, fresh initialization, seed 20260817 | [GRU launcher](../scripts/run_v8_relay_gru_v2.sh), [recurrent implementation record](research/V8_RECURRENT_ACTOR_GRU_20260817.md) |
| September v2 generalist and trench specialist | Generalist: 3,840 maps, 25 foundation + 15 trench conditions; specialist: 1,440 maps, the same 15 trench conditions | Feed-forward actor; seed 20260901; native continuations of their own earlier checkpoints | generalist resolved parent config (local: `.artifacts/terra_foundation_sweep_20260907/parent_recipe_comparison.json`), specialist restart command (local: `.artifacts/terra_training_restart_20260907/launches/cscs.sh`) |
| September foundation behavior screen | New, separate bank: 256 train / 64 validation / 64 test foundations; squares, rectangles and L shapes; no obstacles and broad legal dumping | Feed-forward actor; all arms transfer the same generalist u5000 parameters and Adam state; seed 20260907, plus E repeat 20260908 | resolved parent/recipe comparison (local: `.artifacts/terra_foundation_sweep_20260907/parent_recipe_comparison.json`), frozen training wrapper (local: `.worktrees/terra_foundation_sweep_20260907/terra-baselines/scripts/foundation_reward_sweep/train.sh`) |

The original training bank's 4,512 slots contain 4,509 unique scenario IDs:
three duplicate pairs are confined to its all-free trench control. The dataset
audit finds no cross-split scenario overlap. Uniform slot sampling gives those
repeated scenarios additional exposure within that condition; retain both slot
counts and unique scenario counts in a data table.

The original V8 promotion panel contains 45 conditions × 16 maps = 720 maps.
It excludes the two additional all-free training conditions. The September
generalist is a pooled slice of the finite-enriched V8 R2 release; it is not the
original 47-condition sampler. The easy-foundation split is a separate
adaptation and efficiency study, not a replacement benchmark for constrained
trenches and foundations.

### Source and runtime identity

The local baselines checkout used during the audit had HEAD
`7989a9bea7c86b3b969cb32e5aee3fcfef44a9ad`, with existing local changes. The
training snapshots below are separate source pairs; that local HEAD alone does
not identify their environment.

| Experiment lineage | Baselines revision | Terra revision and location |
| --- | --- | --- |
| August FF relay | `2778766683fb8a0a53a761385fae05cf9396dda9` | Base `25f855db3d913fd638c4e56b1740437a2b7122ca` plus effective pre-start patch `ebdc3ad7b0e7ef505bb6d442a97d18d986cced44`; [handover provenance](research/V8_PAPER_EXPERIMENT_HANDOVER_20260818.md) |
| August GRU concat-skip v2 | `33d26213327d66921b66753a5a6018a37d6f2e81` | `25f855db3d913fd638c4e56b1740437a2b7122ca`; [handover provenance](research/V8_PAPER_EXPERIMENT_HANDOVER_20260818.md) |
| September 7 generalist/specialist continuation | `da904ff4be9ab9d5f71e89e8820233af9facd6a8` | `46b140f8373e098ad832e4968d8136a5ba861bf6`; paired snapshot (local: `.worktrees/terra_training_restart_20260907`), restart manifest (local: `.artifacts/terra_training_restart_20260907/restart_manifest.json`) |
| September 7–8 foundation screens | `c797ea3ab8ae780d7515abe028553d9ace4353a2` | `fa8d5d133a2491d5b7d58f265aa07151afd829e0`; paired snapshot (local: `.worktrees/terra_foundation_sweep_20260907`), source/runtime record (local: `.artifacts/terra_foundation_sweep_20260907/RUNS.md`) |

The September Euler snapshot uses JAX 0.4.33 / CUDA 12.6 / cuDNN 9.5. CSCS
uses the recorded NVIDIA JAX 24.10 container image on GH200 hardware. The
foundation screen has independent one-GPU policies on both venues, including
a venue-matched observation control. Four CSCS processes in that screen are
four policies, not four distributed workers of one policy. In the specialist,
four GH200 GPUs instead optimize one shared policy. Hardware, precision and
runtime are part of the comparison; the paired snapshot and saved evaluation
record are needed to interpret exact trajectories.

## PPO objective and optimizer

The primary entry point is [train_mixed.py](../train_mixed.py). The shared
implementation in [train.py](../train.py), functions `calculate_gae` and
`ppo_update_networks`, computes generalized advantage estimates backwards over
each rollout:

```text
delta_t = r_t + gamma * (1 - done_t) * V_old(o_{t+1}) - V_old(o_t)
A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
R_t     = A_t + V_old(o_t)
```

The final nonterminal transition bootstraps from the critic's next-state value.
`done` masks both this bootstrap and the GAE recursion at task completion and
at timeout. There is no separate time-limit bootstrap. Thus a 450-step timeout
is treated as the end of the finite task, rather than as a censored transition
of an infinite-horizon task.

Advantages are standardized **inside each device-local minibatch**, using
`(A - mean(A)) / (std(A) + 1e-8)`. They are not standardized once over the full
distributed rollout. The targets `R_t` retain the environment reward scale;
the PPO loss adds no running return normalization.

All experiment recipes in the table use the following minimized loss:

```text
rho_t   = pi_theta(a_t | o_t, history) / pi_old(a_t | o_t, history)
L_actor = -mean(min(rho_t * Ahat_t,
                    clip(rho_t, 1 - epsilon, 1 + epsilon) * Ahat_t))
L_value = 0.5 * mean((V_theta(o_t) - R_t)^2)
L       = L_actor + c_v * L_value - c_H(u) * mean(entropy(pi_theta))
```

Value clipping is disabled in these recipes. The generic trainer can instead
clip the change from the old value estimate to ±`clip_eps`, taking the maximum
of unclipped and clipped squared error. Auxiliary map decoding and teacher
distillation are disabled in the listed runs. Approximate KL and clipping
fraction are diagnostics; the listed optimizer does not stop an epoch at a KL
threshold or adapt the learning rate to KL.

| Setting | Listed experiment recipes | Generic `MixedAgentTrainConfig` default when different |
| --- | --- | --- |
| Optimizer | Adam, constant learning rate `3e-4` | Same |
| Adam moment decays / epsilon | Optax defaults `beta1=0.9`, `beta2=0.999`; explicit epsilon `1e-5` | Same |
| Discount `gamma` | `0.9984` | Same |
| GAE `lambda` | `0.95` | Same |
| Policy clipping `epsilon` | `0.2` | Same |
| Critic coefficient `c_v` | `2.0` | Same |
| Gradient global-norm cap | `0.5` | Same |
| PPO epochs per rollout | `2` | Same |
| Minibatches per epoch | `32` | `16` |
| Value clipping | Off | On |
| Entropy coefficient | Cosine `0.15 → 0.02` over 20,000 absolute PPO updates | Cosine `0.15 → 0.005` over 9,500 updates |
| Teacher / auxiliary decoder | None / coefficient `0` | Same |
| Policy action masking | Off | Off |
| Previous-action history, single excavator | 5 actions | Runtime derives `5 * number_of_agents` |
| Reward objective | `reward_v2`, timing variant `0` | `dense_skill` |
| Checkpoint retention | Every 500 updates, keep history | Every 100 updates, overwrite rolling target |

Adam uses Optax's default moment decays; the trainer explicitly overrides its
learning rate and epsilon. Gradients and losses are averaged across devices
with `pmean`; the optimizer then clips the averaged gradient and updates the
replicated parameters. There is one optimizer update per synchronized
minibatch, not one additional update per GPU.

For zero-based absolute rollout-update index `u`, the actual entropy schedule is:

```text
f      = min(1, u / 20000)
c_H(u) = 0.02 + 0.5 * (0.15 - 0.02) * (1 + cos(pi * f))
```

The stored `ent_coef=0.06` is a fallback field; `train_mixed.py` supplies the
scheduled coefficient to the loss. A continuation from u5000 resumes the
schedule at u5000. It does not restart entropy at 0.15. The September specialist
continuation from u51500 is already on the 0.02 floor.

## Rollouts, minibatches and training budget

With `D` devices, `E` environments per device, `T` rollout steps and `M`
minibatches per epoch:

```text
parallel environments                  = D * E
new transitions per PPO update         = D * E * T
transitions per local minibatch        = E * T / M
transitions per synchronized minibatch = D * E * T / M
optimizer steps per PPO update         = epochs * M
```

Each new rollout is reused for two epochs. Counting its second optimization
pass again as new environment experience would double the reported training
budget incorrectly.

| Recipe | `D × E × T` | Parallel environments | New transitions/update | Local / synchronized minibatch transitions | Adam steps/update |
| --- | --- | ---: | ---: | ---: | ---: |
| V8 FF relay | `8 × 256 × 32` | 2,048 | 65,536 | 256 / 2,048 | 64 |
| V8 GRU relay | `4 × 512 × 32` | 2,048 | 65,536 | 512 / 2,048 | 64 |
| September generalist or specialist | `4 × 512 × 32` | 2,048 | 65,536 | 512 / 2,048 | 64 |
| Each foundation screen arm | `1 × 512 × 32` | 512 | 16,384 | 512 / 512 | 64 |

The FF relay launcher enables `flat_minibatch_shuffle`: it shuffles individual
environment-time transitions after computing GAE. The GRU shuffles intact
32-step environment sequences with their saved pre-rollout hidden state. A
GRU local minibatch contains 16 sequences × 32 steps. Hidden-state gradients
stop at that pre-rollout state; memory persists between rollout boundaries
until an episode ends. The September feed-forward recipes use the default
environment-axis shuffle: 16 complete 32-step trajectories per local
minibatch, flattened only for the network forward pass. These are different
sampling/normalization layouts despite matching some global batch counts.

`num_updates = floor(total_timesteps / (D * E * T))`. `total_timesteps` is the
absolute target count for the configured batch shape, not an additional segment
budget. The September generalist and specialist target 100,000 updates
(`6,553,600,000` transitions at their unchanged four-GPU shape). This is a
configured ceiling, not evidence that either completed that budget.

Every foundation screen starts from the same u5000 parent, which already
contains `5,000 × 65,536 = 327,680,000` transitions of generalist training and
Adam step 320,000. Its subsequent batch is four times smaller. For a saved
foundation checkpoint whose `next_update` is `U`:

```text
additional foundation transitions = (U - 5000) * 16384
cumulative lineage transitions    = 327680000 + (U - 5000) * 16384
Adam step                        = 320000 + (U - 5000) * 64
```

The original screen's target u505000 lets the allocation wall time bound the
run. The upper-cost screen's u15000 target is a short diagnostic ceiling.
Its u7000/u10000/u15000 endpoints add 32,768,000 / 81,920,000 / 163,840,000
foundation transitions. Report completed checkpoint budgets and allocated
GPU-hours separately from requested targets. The upper-screen plan (local: `.artifacts/terra_foundation_sweep_20260908_upper/PLAN.md`)
preserves the concrete limits.

## Policy architecture and observations

The listed recipes use `model_size=medium`, `model_core=mlp`, and
`map_encoder=resnet_spatial_8x8_se_sa_xattn`. The name describes the encoder;
`model_core=mlp` does not mean that its spatial encoder lacks attention.
Architecture and preprocessing are implemented in [utils/models.py](../utils/models.py).

| Component | Resolved architecture |
| --- | --- |
| Residual map trunk | Channels `(24,48,64,96)`; blocks `(2,2,3,3)`; squeeze/excitation; 64×64 input reduces to an 8×8 grid |
| Spatial token mixing | Two self-attention blocks over 64 spatial tokens; four heads, Q/K/V width 96; learned positional table; residual initialization scale 0.1 |
| Flatten readout | 1×1 reduction to 32 channels, flatten `8×8×32`, Dense(192) |
| Cross-attention readout | One active-agent query plus eight learned latent queries; four heads; 96-wide Q/K/V; output projection to 160 |
| Map embedding | Concatenated flatten and attention branches projected to 160 |
| Local workspace features | MLP widths `(320,64)` for concatenated local summaries |
| Feed-forward actor | Fused features → Dense(160) → Dense(48) → eight categorical logits |
| Critic | Fused features → Dense(512) → Dense(256) → scalar value |
| Recurrent actor alternative | Dense(160) → GRU(64); concatenate GRU output and current Dense(160) features → Dense(48) → eight logits; critic remains feed-forward |
| Precision | bfloat16 map convolution/readout computation; float32 attention, parameters and recurrent state |

The pure-GRU v1 without the current-observation skip is a different historical
architecture. The default `atari`, `resnet_global_pool`, and plain
`resnet_spatial_8x8` encoders are also separate models. In particular, the
generic medium preset defaults to fewer residual blocks than the explicit
`(2,2,3,3)` override above. A model-size label alone is insufficient to
reconstruct these runs.

[utils/utils_ppo.py](../utils/utils_ppo.py) defines the observation order. Inputs
include agent state and active-agent identity, five previous actions, local
workspace summaries, and global maps. Although the network supports up to four
agent slots, these recipes use one tracked excavator (`agent_types=[0]`,
`action_types=[0]`).

The global encoder takes seven source maps: traversability, reachability,
current terrain/action state, target, padding/obstacles, dumpability, and current
interaction workspace. The SE encoder adds four derived channels: remaining
dig cells, empty positive-target cells, and normalized x/y coordinates. Thus
the original relay policy uses **11 global channels**, rather than seven.
The September relocation-distance observation adds a twelfth channel. It uses
the static obstacle-geodesic distance map from the reward protocol, scaled by
the protocol bound; it is not an independently learned distance predictor.
Reachability is zero when its environment option is disabled, as in the
preserved September configuration.

There are nine original local workspace summaries: negative/positive current
soil, negative/positive targets, dumpability, obstacles, foundation-border
workspace, alignment error, and diggable border. September adds a tenth
summary of admissible fresh digging for each of 12 cabin orientations. These
are workspace summaries indexed by orientation, not a second set of global
image channels.

| Observation option | August FF / GRU relay | September generalist / specialist | Foundation screen |
| --- | --- | --- | --- |
| Carried-work state | On | On | On |
| Reset context `[Q_reset, H_reset/V0]` | On | Off | Off |
| Trench alignment vector | Off | On | On; inapplicable for ordinary foundation cells |
| Geodesic relocation map | Off | On | On |
| Admissible-dig local summary | Off | On | On |
| Executable-dig observation semantics | Not this treatment | Legacy semantics | A: legacy; B–F and upper doses: enabled |
| Stall age / movement feasibility / previous-outcome features | Off | Off | Off |
| Action-logit masking | Off | Off | Off |

The executable-dig option changes the meaning of existing digging observations
without changing their tensor widths. The preserved September model has
2,311,701 parameters in the parent (local: `.artifacts/terra_foundation_sweep_20260907/checkpoints/parent_check.json`),
every foundation arm, and the specialist continuation (local: `.artifacts/terra_training_restart_20260907/checkpoints/cscs_smoke_gate.json`).
It is not the 2,307,645-parameter earlier specialist that lacked the two later
observations. The resolved comparison (local: `.artifacts/terra_foundation_sweep_20260907/parent_recipe_comparison.json`)
records the observation flags and unchanged model settings.

`clip_action_maps=True` clips the global current height map to `[-1,1]` before
the encoder. Positive pile heights above one are therefore aliased in that
channel. The spatial encoder linearly normalizes current/target map heights
with bounds `(-10,10)`, and the first four local summaries use bounds
`(-16,16)`. These are fixed affine transforms, not running statistics or a
promise that every transformed value lies inside `[-1,1]`. The listed policies
receive carried-work state and local summaries, so global-height clipping
should not be described as erasing every source of volume information.

Training samples categorical actions from all eight logits. No action mask is
applied in any listed recipe. The environment can reject an ineffective or
inadmissible action while the policy remains unmasked; an environment digging
gate and a policy action mask are different mechanisms.

## Curriculum, resets and reward selection

### Historical mixed V8 curriculum

The August relay recipes use [Continuous Banded v3](research/CONTINUOUS_BANDED_V3_DESIGN_20260812.md),
implemented in [utils/pooled_sampler.py](../utils/pooled_sampler.py). Every 150
PPO updates, 80% assignment mass is distributed over open conditions with
curriculum-depth weights `4:2:1` for depths 0/1/2. The other 20% is uniform
over mastered conditions. If a pool is empty, its mass goes to the other pool.
A water-filling cap limits each condition to 15%. The cap can change the final
80/20 allocation when it binds.

An eligible refresh window requires at least 32 completed episodes for that
condition; mastery then requires exact-success EMA ≥0.80. The EMA coefficient
is 0.30, applied to each eligible window's success fraction. A mastered
condition reopens below 0.65. Insufficient-data windows leave its estimate and
mastery unchanged.
Foundation/trench labels are reporting strata, not fixed sampling quotas.
Curriculum depth denotes a difficulty band, not vertical excavation depth.
Only full-start training episodes update this mastery estimate. Evaluation
outcomes and synthetic partial-start outcomes do not update it.

The partial-start schedule in `train_mixed.py::partial_reset_schedule` is:

| Zero-based PPO update index | Supported completion tiers | Target fraction of partial-reset lanes |
| --- | --- | ---: |
| 0–2,499 | 90% | 25% |
| 2,500–4,999 | 75%, 90% | 25% |
| 5,000–7,499 | 50%, 75%, 90% | 25% |
| 7,500–9,999 | 50%, 75%, 90% | Linear decay from 25% to 0% |
| ≥10,000 | None | 0% |

The same schedule is present in the preserved August FF and GRU worktrees.
Lane assignment is not the fraction of completed episodes or transitions:
episode lengths and delayed resets affect realized exposure. The active
historical sidecar covers only the 96 d16 foundation maps, so these pilots do
not establish broad relay-reset coverage across all 47 conditions. Partial
states conserve material and are geometrically screened; they are not
demonstrated-success trajectories. The reset-context observation exposes the
episode's fixed reward baseline.

### September pooled training and foundation screens

The September generalist and specialist presets use one pooled map level with
450-step episodes and the adaptive condition sampler disabled. They do not
use Continuous Banded v3 or partial resets. With 96 training maps per source
condition, uniform map sampling in their complete pools induces equal
condition exposure in expectation; the generalist then has 25/40 foundation
and 15/40 trench mass. The one-level increase/decrease thresholds in the preset
do not create a multistage curriculum.

The foundation screen likewise samples uniformly from its 256 training maps
in one level and has no partial resets. The initial transfer changes bank,
the R2 distance sidecar and the selected behavior settings while retaining
native Adam and the absolute update counter. Ordinary continuations then load
the saved environment settings and disable the transfer overrides.

All listed recipes start ordinary episodes with `env_steps == 0`, zero action
history and, for GRU, zero recurrent carry. The current trainer explicitly
asserts the initial zero elapsed counter. Earlier prose describing random
elapsed-step initialization does not describe these audited source versions.
Elapsed and remaining episode time are still absent from the policy input.
Training auto-resets completed episodes; history and GRU state are cleared at
episode boundaries.

`reward_stage=reward_v2` with `reward_v2_timing_variant=0` selects the global
material/transport reward objective. A map-level `rewards_type: DENSE` is not
evidence that the old dense dig/dump reward formula is used. The frozen R2
distance protocol is `obstacle_geodesic_8_physical_global_v1`. Its sidecar
identity is bank-specific: the September pooled banks use
`f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980`, whereas
the easy-foundation bank uses
`6b2675998403ed2d6125d955fca446404fbdf260e0a0c2cf7b9864cbdd1fb2bf`.
Use the environment reference for the reward equation and exact completion
predicate, and the saved `r2_protocol_receipt` for the executed coefficients.

The preserved specialist receipt (local: `.artifacts/terra_training_restart_20260907/checkpoints/cscs_smoke_gate.json`)
fixes `alpha=1`, `beta=1.5`, shaping weight `1`, shaping discount `0.9984`,
success bonus `6`, horizon failure penalty `1`, total horizon step cost `1`,
distance reference `16 m`, and distance bound `2.5`. Legacy reward fields
retained inside the saved environment are not all active in reward-v2.

The foundation screen changes three additional costs, preserving the base
reward-v2 objective. The lateral term applies only to newly excavated target
soil relative to chassis orientation; travel and turn costs measure actual
base translation and rotation. Dumping and loose-soil pickup are exempt from
the lateral term.

| Foundation arm | Executable digging observations | Lateral coefficient | Translation coefficient /m | Base rotation coefficient /rad |
| --- | --- | ---: | ---: | ---: |
| A legacy | Off | 0 | 0 | 0 |
| B observation control | On | 0 | 0 | 0 |
| C lateral only | On | 0.25 | 0 | 0 |
| D movement only | On | 0 | 0.005 | 0.02 |
| E combined; repeat uses seed 20260908 | On | 0.25 | 0.005 | 0.02 |
| F doubled combined | On | 0.5 | 0.01 | 0.04 |
| Upper E×4 / E×8 / E×16 / E×32 | On | 1 / 2 / 4 / 8 | 0.02 / 0.04 / 0.08 / 0.16 | 0.08 / 0.16 / 0.32 / 0.64 |

Exact arm/venue assignment is retained in the original launcher (local: `.artifacts/terra_foundation_sweep_20260907/run_arm.sh`)
and upper-dose launcher (local: `.artifacts/terra_foundation_sweep_20260908_upper/run_arm.sh`).
The upper doses change three coefficients together and use one seed per dose;
they cannot attribute an outcome to one individual cost term.

## Evaluation and publication endpoints

The primary policy endpoint is exact completion within a fixed 450-decision
horizon from an ordinary full start on identified held-out maps. Fixed
evaluation uses deterministic argmax, explicit map slots and reset seeds,
first-episode outcomes, and preserved terminal states. A sampled policy or
MCTS evaluation is a separate treatment.

| Measurement | Implemented meaning | Appropriate use |
| --- | --- | --- |
| `train/episode_success_rate` | Exact successful training terminations divided by completed training episodes; NaN when none completed | Online learning diagnostic; affected by training sampler and reset mix |
| `online_eval/success_within_horizon_rate` | Fraction of a freshly reset initial training-bank cohort that succeeds within the inline budget | Bounded, but in-bank and horizon-limited diagnostic |
| `online_eval/termination_within_horizon_rate` | Fraction of that initial cohort that terminates within the budget | Exposes unfinished/censored episodes |
| `online_eval/completed_episode_success_rate` | Success among all completed episodes in the auto-reset inline stream | Secondary diagnostic; excludes unfinished episodes |
| Fixed-bank exact success | Fraction of named first episodes that satisfy exact completion within 450 decisions | Main comparative endpoint under the recorded bank/runtime contract |
| Partial completion, no-effect rate and behavior costs | Separate progress and behavior diagnostics | Explain failures and efficiency; do not replace exact success |

Older logs use `eval/*` names, including unbounded episodes-per-environment
counts. Those fields are not automatically comparable to current
`online_eval/*`. The legacy `progress/episode_completion_rate` also counts
timeouts and is not a success rate.

Inline evaluation in [eval_ppo.py](../eval_ppo.py) samples policy actions and
runs `num_rollouts_eval=200` steps across the training-shaped cohort. The
`eval_episodes=100` field does not determine the number of those episodes.
`train_mixed.py` resets these inline cohorts to full starts, even when training
uses partial resets. A 200-step diagnostic cannot be reported as 450-step task
success. It is enabled every 100 updates in the September generalist and
specialist, and disabled in the August relay wrappers and foundation screen.

[eval_fixed_bank.py](../eval_fixed_bank.py) evaluates named maps, uses the
checkpoint's matching model contract, and retains the first terminal state.
The August [benchmark inspector](research/V8_BENCHMARK_INSPECTOR_20260820.md)
records the 720-map FF/GRU comparison. The policies differ in seed, device
layout, minibatch shuffling and effective Terra runtime; that result is a
capability comparison, not an isolated causal estimate of recurrence.

The foundation fixed-evaluation wrapper (local: `.worktrees/terra_foundation_sweep_20260907/terra-baselines/scripts/foundation_reward_sweep/eval.sh`)
selects the same 64 validation maps, seed 20260907, horizon 450,
`exact_visible_dump_v1`, no MCTS, and forward chunks of 32. The original
screen evaluates at absolute u7000/u10000/u15000/u25000; the upper screen uses
u7000/u10000/u15000. The checkpoint hook (local: `.artifacts/terra_foundation_sweep_20260907/checkpoint_eval.py`)
pauses training for a child evaluator and then resumes the retained live
training state. Its separate 64-map test split is reserved for final testing.
Repeated validation episodes do not increase the number of independent maps.

Report overall and family/condition exact success, completion and failure
strata. For efficiency, report raw actions, travel, productive base setups,
fresh area per setup and soil rehandling, with paired comparisons on maps both
policies solve. Keep failed episodes visible separately; reduced work by an
unsuccessful policy is not an efficiency gain. In the intended ROS deployment,
navigation plans motion between retained work poses. Raw Terra movement is
therefore a diagnostic/learning surrogate, not measured deployed Nav2 travel
or execution time; see foundation behavior priorities (local: `terra-baselines/docs/research/FOUNDATION_BEHAVIOR_20260907.md`).

## Checkpoints, continuation and evidence limits

Version-2 checkpoints contain model parameters, Adam state, optimizer step,
resolved training/environment configuration, `update`, `next_update`, loss
diagnostics and transition-integrity counts. When active, they also contain
the R2 reward/distance receipt, adaptive sampler state, reward-annealing state
and partial-reset curriculum receipt. It is incorrect to describe all current
checkpoints as parameter-only files with no protocol information.

`--resume_from` restores the native optimizer and absolute schedule position.
`--warm_start_from` loads only model parameters and starts fresh optimization;
it does not describe the September foundation transfer. A native resume still
restarts the JAX RNG stream, live environment states, action histories and
recurrent carry. Adaptive sampler state is restored separately when present.
Continuation is therefore not bit-exact replay of an uninterrupted process.

For each paper result, retain the exact checkpoint and its checksum, actual
completed update and Adam clocks, paired source versions and effective patches,
resolved flags, bank/split/source IDs, distance sidecar, seed, evaluation mode,
hardware/runtime, precision, and per-map evaluation records. Those surrounding
records remain necessary even when a checkpoint contains an R2 receipt.
For a changed batch shape, compute experience segment by segment using the
formula above. The preserved foundation parent is
`generalist_parent_u5000.pkl`, SHA-256
`f8c6dc7fc34bb5b9917206b8feb488604f1f1f2c9f191b995a9e42dfe8b8cc8a`.

The current documentation is sufficient to explain the implemented method and
separate these experiments. It does not turn the historical pilots into a
matched, replicated ablation, establish held-out generalization from inline
curves, prove full finite-budget training, or establish robot execution. Those
claims require their own completed, matched experiment evidence.
