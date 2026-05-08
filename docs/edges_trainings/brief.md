# Terra Edge-Digging RL Research Brief

## User Observation

We are training a discrete-state, discrete-action RL agent for foundation digging in a gridworld-like setting. Core foundation tiles can be dug from the excavator workspace cone/radial sector. Edge tiles now have stricter physical constraints: to dig a clean foundation edge, the robot base must be on/near the edge tile and the cabin/arm must be aligned with the edge direction. If those conditions are not met, the simulator does not allow the edge tiles to be dug. Since adding edge constraints, agents finish much less often and the value function shows frequent small spikes, suggesting the critic is not learning the new return structure well. This may be a multi-task or hierarchical setup: core excavation and edge finishing have different affordances/preconditions.

## Code Context

Repos:

- `/home/lorenzo/moleworks/terra`: JAX Terra environment on branch `multi-agent`.
- `/home/lorenzo/moleworks/terra-baselines`: PPO training code on branch `multi-agent`.

Relevant environment path:

- `terra/terra/state.py`
- `terra/terra/env.py`
- `terra/terra/config.py`
- `terra/terra/maps_buffer.py`
- `terra/terra/env_generation/generate_foundations_real_ring.py`

Relevant training path:

- `terra-baselines/train.py`
- `terra-baselines/train_mixed.py`
- `terra-baselines/utils/models.py`
- `terra-baselines/utils/utils_ppo.py`
- `terra-baselines/configs/training_configs.yaml`

## Current Mechanics Observed In Code

1. Border metadata is generated from the final rasterized foundation mask.
   - `generate_foundations_real_ring.py` builds `foundation_border_axes_ABC` from contours.
   - `maps_buffer.py` loads that metadata into fixed-size arrays.

2. Edge/border digging is hard-gated in the simulator.
   - `state.py:_get_foundation_border_mask` creates an inside border band from target dig tiles.
   - `state.py:_get_foundation_border_alignment_mask` requires the current arm angle to align with the touched border segment and the base to be close to the border line.
   - `state.py:_mask_out_wrong_dig_tiles` multiplies the dig workspace mask by this border alignment mask when `enforce_foundation_border_alignment=True`.
   - If the dig cone only touches edge tiles and alignment/proximity fail, the `DO` action becomes a no-op or partial no-op rather than explicitly invalid.

3. The current policy samples from an unmasked categorical action distribution.
   - `utils_ppo.py:policy` returns `tfp.distributions.Categorical(logits=logits_pi)`.
   - `utils_ppo.py:select_action_ppo` samples directly from that distribution.
   - `state.py:_get_infos` documents that `action_mask` is informational only and currently returns zeros.
   - The PPO update recomputes log-probs from the same unmasked categorical.

4. The observation contains some relevant spatial context, but not an explicit edge-precondition/affordance signal.
   - `env.py:_state_to_obs_dict` includes agent pose, local target/action/dumpability/obstacle maps, global target/action maps, and a global `interaction_mask`.
   - It does not expose a per-action valid-action mask, a border alignment mask, the nearest border segment direction, or "edge vs core" phase/progress.

5. The model is a shared actor-critic with a common feature trunk.
   - `utils/models.py:SimplifiedCoupledCategoricalNet` encodes all agent states, local maps, global maps, and previous actions into one feature vector.
   - It has separate MLP heads for value and policy, but the same trunk and no task-conditioned or phase-conditioned value head.
   - Agent type is embedded, but there is no explicit task/phase embedding for core excavation vs edge finishing.

6. Rewards and returns are likely challenging after edge constraints.
   - Dense reward includes movement/turn penalties, dig/dump rewards, existence penalty, and a large terminal reward.
   - `state.py:_calculate_terminal_reward` uses thresholded/exponential completion reward.
   - `state.py:_get_reward` adds terminal reward whenever `done` is true, not only when `done_task` is true; below the completion threshold this can be zero, but the value target can still change abruptly near episode end.
   - `train_mixed.py` additionally backfills terminal reward over the last `num_agents - 1` alternating steps.
   - PPO uses `gamma=0.9984`, `gae_lambda=0.95`, `clip_eps=0.5`, `vf_coef=5.0`, value clipping with the same `clip_eps`, and advantage normalization.

## Working Hypotheses

1. The edge rule creates a state-dependent action-validity problem. The agent must discover a low-probability precondition sequence before `DO` has an effect on border tiles. Without action masks or affordance labels, the critic sees many visually similar states where `DO` has sharply different consequences.

2. The task is now phase/mode structured:
   - Core digging: any aligned radial sector that covers target tiles can make progress.
   - Edge finishing: progress requires a precise base position and arm/cabin orientation relative to the foundation boundary.
   This is closer to multi-task, option, or hierarchical RL than a homogeneous gridworld excavation problem.

3. Value spikes may come from heterogeneous returns mixed in one scalar value head: core-progress states, edge-setup states, near-finish states, and timeout states may alias under the current observation/features. Sparse successful edge completions plus large terminal reward can create high-variance GAE targets.

4. The environment currently hides a useful structured signal. The simulator can compute edge legality, touched border segments, alignment error, proximity error, and valid action masks, but the policy/critic do not receive them directly.

5. The first likely fixes to investigate are:
   - action masking for invalid/no-op `DO` and impossible movement/cabin actions;
   - exposing edge affordance features: border mask, nearest border tangent/normal, alignment error, base-to-edge distance, edge/core remaining counts;
   - curriculum over edge constraints: train core-only, then soft edge shaping, then hard edge gating; oversample edge states;
   - separate or conditioned critics for phases/tasks, possibly with auxiliary heads predicting edge-validity/progress;
   - hierarchical policy/options: navigate-align-to-edge option followed by dig-edge option, plus core dig option;
   - reward shaping that is potential-based or at least aligned with the edge preconditions, not only with final terminal success.

## Deep Research Prompt

Act like an expert reinforcement learning researcher and robotics ML engineer. I need a literature-grounded set of recommendations for a discrete gridworld-like excavation RL task.

Problem: an agent digs a foundation in a discrete state/action space. A normal `dig` action removes multiple tiles in the robot workspace, a radial sector/cone. Core foundation tiles can be dug whenever the workspace covers them. Edge/foundation-boundary tiles are different: in the real robot, clean/smooth edge digging requires the base to be on/near the edge tile and the cabin/arm aligned with the edge direction. The simulator now hard-enforces this: if an edge tile is not dug from the right base pose and cabin/arm alignment, the edge tile is not removed. After adding this edge constraint, success rate dropped sharply and the value function shows frequent small spikes, suggesting critic/return-learning instability.

Current implementation summary from code:

- Environment is JAX Terra. Edge legality is a hard simulator mask in `terra/terra/state.py`: it builds a foundation border band, computes the touched border segment direction from metadata or gradients, checks arm/cabin alignment tolerance and base-to-edge proximity, then masks out edge tiles in the dig workspace if the preconditions fail.
- PPO policy in `terra-baselines` samples from an unmasked categorical distribution over discrete actions. There is an `action_mask` field in env info, but it currently returns zeros and is informational only. The policy and PPO update do not use valid-action masking.
- Observations include agent pose, local action/target/dumpability/obstacle summaries, global target/action maps, and interaction masks. They do not explicitly include an edge-affordance map, nearest border direction, alignment error, proximity error, or phase label such as core-dig vs edge-finish.
- Actor and critic share one feature trunk with separate MLP heads. There is agent-type conditioning, but no task/phase-conditioned critic or multi-head critic.
- Rewards include dense move/turn/dig/dump/existence terms, a thresholded/exponential terminal completion reward, and for multi-agent runs terminal reward is backfilled across recent alternating turns. PPO uses GAE, clipped value loss, advantage normalization, and relatively large value coefficient.

Research questions:

1. What literature is most relevant to this failure mode: hard state-dependent action preconditions, invalid/no-op actions, sparse edge-subtask completion, long-horizon gridworld manipulation, and critic instability/value spikes?
2. What does the literature say about invalid action masking in policy-gradient/PPO methods, especially when invalidity is state-dependent and deterministic? When is action masking preferable to penalizing invalid/no-op actions?
3. This looks like a multi-task or hierarchical problem: core excavation vs edge finishing require different behaviors and affordances. What architectures are supported by literature: shared trunk with task-conditioned heads, multi-head critics, UVFA/goal-conditioned critics, option/hierarchical policies, auxiliary losses, or successor features?
4. What critic-specific changes might reduce value spikes here: value normalization/PopArt, distributional value functions, separate critics by phase/task, lower value coefficient, Huber value loss, return/terminal reward scaling, GAE/lambda/gamma changes, bootstrapping changes at timeout, or more explicit phase features?
5. What curriculum/replay/data-generation strategies are supported for rare edge-finishing success: staged constraint hardening, edge-state oversampling, reset-to-edge curriculum, demonstrations/behavior cloning, hindsight relabeling, or self-imitation?
6. For this exact Terra-style setup, produce a prioritized intervention plan: quick low-risk changes, medium architecture changes, and research-heavy options. Include expected failure modes and what metrics should be logged to tell whether each intervention helped.

Please return:

- A concise taxonomy of the problem in RL terms.
- A list of relevant papers/books/blogs with one-sentence relevance each.
- Concrete recommendations for actor, critic, observation, reward, and curriculum changes.
- A short experimental plan with ablations and diagnostics.
- Explicit caveats where the literature may not transfer to deterministic gridworld excavation.
