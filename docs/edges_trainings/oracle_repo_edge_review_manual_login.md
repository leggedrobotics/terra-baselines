## Main diagnosis

The edge rule changed `DO` from “usually useful when the cone covers target” into a **state-dependent feasible action**: for border tiles it only works when the base is near the touched border segment and the arm/cabin angle is aligned. In the current training stack, that feasibility is enforced inside the simulator, but PPO still samples and updates an **unmasked categorical policy** and the critic does not see a clear edge-affordance signal.

The highest-probability failure mode is therefore:

`raw cone says target is reachable` → `edge mask silently removes the target tiles` → `DO becomes a no-op or partial no-op` → PPO still assigns probability to it → critic sees similar observations with very different returns → value loss spikes near edge/terminal states.

## Ranked recommendations

### 1. Make action masking real end-to-end in PPO

**Expected impact: highest.**

The simulator already computes legality, but the trainer ignores it. In `terra/terra/state.py:2022-2080`, `_mask_out_wrong_dig_tiles()` applies the border alignment mask to digging. But `terra-baselines/utils/utils_ppo.py:43-66` builds `tfp.distributions.Categorical(logits=logits_pi)` without any mask, and `terra/terra/state.py:3536-3547` currently returns an all-zero `action_mask`.

Fix this before reward or architecture tuning.

Concrete changes:

1. Fix the env mask first. `_get_action_mask_tracked()` and `_get_action_mask_wheeled()` in `state.py:3371-3487` return only 7 entries, but both action classes define 8 actions including `DO_NOTHING=7` in `terra/terra/actions.py:13-26` and `actions.py:99-112`. Append `DO_NOTHING=True`.

2. Fix the multi-agent comparison bug before using the mask. In `_get_action_mask_tracked()` and `_get_action_mask_wheeled()`, the code compares `new_state._get_prev_agent_state()` against `self._get_current_agent_state()`. Since these handlers are called directly and do **not** call `_swap()`, this is wrong for multi-agent runs. Compare `new_state._get_current_agent_state()` to the old current agent instead.

   Use the same pattern for movement, base angle, cabin angle, wheel angle, and `DO`.

3. Make `DO` validity detect real state changes, not only `loaded` changes. Current `bool_do` only checks `loaded`, which misses skidsteer shovel toggles and can miss terrain-only changes. A safer first pass:

   ```python
   old_cur = self._get_current_agent_state()
   new_cur = new_state._get_current_agent_state()

   bool_do = (
       jnp.any(new_cur.loaded != old_cur.loaded)
       | jnp.any(new_cur.shovel_lifted != old_cur.shovel_lifted)
       | jnp.any(new_state.world.action_map.map != self.world.action_map.map)
   )
   ```

4. Ensure the mask never becomes all-false:

   ```python
   action_mask = jnp.concatenate(
       [action_mask, jnp.array([True], dtype=jnp.bool_)], axis=0
   )
   action_mask = action_mask[:num_actions]
   action_mask = jax.lax.cond(
       jnp.any(action_mask),
       lambda: action_mask,
       lambda: action_mask.at[7].set(True),
   )
   ```

5. Put the mask in the **observation**, not only in `info`. `select_action_ppo()` samples from `prev_timestep.observation` in `train.py:407-427` and `train_mixed.py:837-861`; it does not use `info`. Add `"action_mask"` in `TerraEnv._state_to_obs_dict()` at `terra/terra/env.py:301-362`.

6. Use the same masked distribution for sampling, log-probs, entropy, and PPO update:

   ```python
   def apply_action_mask(logits, mask):
       mask = mask.astype(jnp.bool_)
       mask = mask.at[..., 7].set(True)  # no-op fallback
       return jnp.where(mask, logits, jnp.full_like(logits, -1e9))
   ```

   Then in `utils_ppo.policy()`:

   ```python
   value, logits_pi = apply_fn(params, model_obs)
   logits_pi = apply_action_mask(logits_pi, action_mask)
   pi = tfp.distributions.Categorical(logits=logits_pi)
   ```

7. Store the mask in `Transition`. PPO ratios must use the same action support that was used during sampling. Do not recompute a next-state mask during update. Add `action_mask` to `terra-baselines/train.py:111-120`, store it in `_env_step`, reshape it in `ppo_update_networks()`, and pass it to `policy()` at `train.py:184-192`.

8. Update eval/inference too. Several files directly create unmasked categoricals: `eval.py`, `eval_mcts.py`, `eval_mixed.py`, `visualize_mixed.py`, `visualize_paths.py`, and `inference/inference_single_map.py`. A masked training policy evaluated unmasked will look worse than it is.

Important nuance: action masking can only mask primitive actions, not individual dig tiles. It will prevent pure no-op `DO`, impossible movement, and disabled cabin actions. It will not by itself explain *which* edge tiles are legal. That requires observation changes below.

---

### 2. Disable or fix the terminal reward backfill in `train_mixed.py`

**Expected impact: very high for value-loss spikes, especially with more than one active agent.**

`terra-baselines/train_mixed.py:875-907` tries to distribute terminal reward to the last `num_agents - 1` alternating steps. The current shift direction appears backwards.

Current code:

```python
def shift_back(arr, k):
    zeros = jnp.zeros_like(arr[:k])
    return jnp.concatenate([zeros, arr[:-k]], axis=0) if k > 0 else arr
```

For a terminal reward at time `t`, this places the bonus at `t + k`, not `t - k`. Because `TerraEnv.step()` auto-resets on `done` in `terra/terra/env.py:257-283`, this can leak terminal reward into the next episode’s reset states. That is a direct recipe for critic spikes.

First experiment: turn this backfill off completely and compare value loss, explained variance, and completion.

If keeping it, fix three things:

```python
def shift_to_previous(arr, k):
    zeros = jnp.zeros_like(arr[:k])
    return jnp.concatenate([arr[k:], zeros], axis=0) if k > 0 else arr
```

Also do not use the whole terminal-step `reward_seq` as the bonus. It includes dense action reward and existence reward. Store and backfill only `reward_components["terminal"]`.

Also gate by `task_done`, not `done`. In `state.py:3364-3369`, `done` means `task_done OR max_steps`. Timeout and success should not share the same terminal-credit logic.

---

### 3. Add explicit edge-affordance observations

**Expected impact: high.**

Right now the model sees raw geometry but not the new feasibility rule. `env.py:337-362` exposes agent states, local target/action maps, global maps, and `interaction_mask`. But `interaction_mask` is not a legality mask: `terra/terra/wrappers.py:82-128` builds it from raw workspace cones, and in multi-agent it is the union of all active agents’ cones.

Similarly, `LocalMapWrapper._build_local_cartesian_masks()` at `wrappers.py:177-234` uses `_get_dig_dump_mask_cyl()` only. It does not apply `_mask_out_wrong_dig_tiles()` and therefore does not include the border alignment/proximity rule.

Add signals that match the rule in `state.py:1204-1419`:

Recommended minimum observation additions:

```text
foundation_border_mask          [H, W] bool
remaining_core_dig_mask         [H, W] bool
remaining_edge_dig_mask         [H, W] bool
current_raw_cone_mask           [H, W] bool
current_legal_dig_mask          [H, W] bool
current_blocked_edge_mask       [H, W] bool
valid_do                        scalar/bool
edge_remaining_count            scalar
core_remaining_count            scalar
legal_edge_tiles_in_cone        scalar
blocked_edge_tiles_in_cone      scalar
```

The cheapest high-value feature is:

```python
raw_cone = state._build_dig_dump_cone()
legal_dig = state._mask_out_wrong_dig_tiles(raw_cone)
```

Expose `legal_dig.reshape(H, W)` or at least counts from it. This directly tells the actor and critic whether `DO` can currently make dig progress.

For better edge behavior, expose nearest active border-segment features:

```text
nearest/touched border angle
arm-to-edge angle error
base-to-edge distance
proximity margin = threshold - distance
alignment margin = tolerance - abs_angle_error
```

These are already almost computed in `_get_foundation_border_alignment_mask()` at `state.py:1229-1419`, but currently discarded.

Implementation locations:

* Add fields in `terra/terra/env.py:_state_to_obs_dict()`.
* Add corresponding entries in `terra-baselines/utils/utils_ppo.py:obs_to_model_input()`.
* Update dummy obs in `terra-baselines/utils/models.py:get_model_ready()`.
* Add channels in `MapsNet` or low-dimensional features in `SimplifiedCoupledCategoricalNet`.

Do not overload the existing `interaction_mask` name. It is a cone mask, not a legal-action or legal-dig mask.

---

### 4. Add critic phase features before making the critic more complex

**Expected impact: high for value stability.**

The model has one shared trunk and one scalar value head in `terra-baselines/utils/models.py:405-530`. That value head must fit several different return regimes:

```text
core digging
edge setup / repositioning
edge-aligned digging
loaded transport/dump
near-success terminal states
timeout states
```

The new edge constraint makes these regimes more distinct.

Start with phase features, not a large architecture change:

```text
core_remaining_count
edge_remaining_count
edge_fraction_remaining
loaded
valid_do
valid_edge_dig_available
core_done_edge_remaining
distance/alignment margins
task_done_possible / completion_percentage
```

Feed these to both actor and critic, but they matter most for the critic.

Then compare:

1. **Single value head + phase features**
   Lowest risk. Likely enough if the main issue is aliasing.

2. **Gated value heads**
   Example heads: `v_core`, `v_edge`, `v_transport`, `v_terminal_setup`. Select or softly gate using known phase features.

3. **Value normalization / PopArt**
   Useful if return scales differ strongly between curriculum levels or between normal and near-terminal states.

Also tune value loss only after fixing mask/observations/backfill:

* `terra-baselines/train.py:41-45` uses `clip_eps=0.5` and `vf_coef=5.0`, which is aggressive.
* `train_mixed.py` uses lower `clip_eps=0.2`, `vf_coef=2.0`, but the value coefficient is still high if targets are noisy.

Recommended first settings:

```text
vf_coef: 0.5 or 1.0
policy clip_eps: keep 0.2
separate vf_clip_eps: 0.2–0.5
value loss: Huber instead of squared loss
```

Also split timeout from task termination. `calculate_gae()` in `train.py:123-147` uses `transition.done` to stop bootstrapping. But `done` includes max-step timeout. If timeout is a truncation rather than a true terminal MDP state, this creates artificial value discontinuities. At minimum, log timeout separately; ideally store `terminated=task_done` and `truncated=done & ~task_done`.

Because `env.step()` auto-resets before returning the next observation, correct timeout bootstrapping would require returning the final pre-reset observation or final value. Otherwise, keep timeout terminal but avoid adding/backfilling terminal rewards on timeout.

---

### 5. Add an edge-specific reset/curriculum path

**Expected impact: high after masks/observations are fixed.**

The edge rule creates a rare endgame subtask. Full-episode PPO now has to discover:

```text
dig core → move near border → align arm/cabin to edge tangent → dig edge → repeat sides/corners
```

The current default curriculum in `terra/terra/config.py:227-233` uses `foundations_real_ring` with `max_steps_in_episode=200`. The mixed YAML `solo_excavator` preset uses `max_steps=550` in `terra-baselines/configs/training_configs.yaml:18-29`, which is more realistic. Make sure the actual run is using the YAML override, not the 200-step default.

Add reset distributions for edge finishing:

1. **Core already dug, one edge side remains.**
2. **Core already dug, one corner remains.**
3. **Only border band remains.**
4. **Agent starts near the selected border, with aligned and misaligned buckets.**
5. **Full task.**

Mix them rather than replacing full episodes. A good initial mixture:

```text
40% full task
30% edge-only side/corner
20% core-dug border remaining
10% random hard starts
```

Then anneal toward full tasks.

The repo already loads optional `actions/` maps in `terra/terra/maps_buffer.py:236-242`, so partial excavation states can be represented with initial `action_map`s. For pose-specific edge starts, extend `Agent.new()` or add a reset override that samples base pose and angle from border metadata.

Also stage the constraint itself:

```text
wide proximity + wide tolerance
current proximity + wide tolerance
current proximity + current tolerance
full rule with corners/diagonals
```

Current config is strict: `foundation_border_proximity_tiles=3.5` and both tolerances are `0.436 rad` at `terra/terra/config.py:203-208`.

---

### 6. Add deterministic edge-geometry tests before changing the math

**Expected impact: medium to high; very high if diagonal/corner maps dominate.**

The border math is subtle because Terra uses an unconventional coordinate convention. `generate_foundations_real_ring.py:52-59` creates metadata as `A*x + B*y + C = 0` using image-style `x=col`, `y=row`. Terra agent poses are effectively `[row, col]`. So the swapped distance in `state.py:1324-1326` is likely intentional, and `line_dist_direct` at `state.py:1327-1329` should not be blindly substituted.

Instead, add regression tests that lock the convention:

```text
horizontal edge: arm angle 0 should align
vertical edge: arm angle +/- pi/2 should align
positive diagonal edge
negative diagonal edge
base just inside/outside proximity threshold
workspace touching two segments at a corner
metadata missing fallback path
```

Test `_get_foundation_border_alignment_mask()` directly on small synthetic maps.

Also check these edge-specific code risks:

* `state.py:1410-1417`: metadata fallback enforces alignment but not proximity. If a map bypasses metadata validation, the rule semantics change.
* `maps_buffer.py:63-65`: `foundation_border_axes` are cast to `float16`. For 64×64 maps this may be okay; for 96/128 maps and diagonal lines near a 3.5-tile threshold, use `float32` to avoid avoidable boundary errors.
* `wrappers.py:207-213`: local-map wrapper uses `EnvConfig().agent.angles_cabin` instead of `state.env_cfg.agent.angles_cabin`. This will silently break if the configured cabin discretization changes.

---

### 7. Improve diagnostics before interpreting learning curves

**Expected impact: high for debugging; low implementation cost.**

Current training logs can mislead. In `train_mixed.py:1050-1053`, `progress/episode_completion_rate` is `mean(timestep.done)`, but `done` includes timeouts. Rename that to termination rate and log `task_done` separately from `timestep.info["task_done"]`.

Add these metrics:

```text
success/task_done rate
timeout rate
core_remaining_tiles
edge_remaining_tiles
core_done_edge_remaining rate
valid action count
all-masked fallback count
raw policy probability mass on invalid actions before masking
masked entropy
DO action rate
DO no-op rate
DO legal-dig rate
legal_edge_tiles_in_cone
blocked_edge_tiles_in_cone
alignment error / proximity margin histograms
value loss by phase: core, edge, loaded, near-terminal, timeout
explained variance by phase
terminal reward mean/std/max
```

For the edge mask specifically, log:

```text
border tiles touched by cone
segments touched
segments allowed
base distance to touched segment
arm-edge angle diff
```

This will tell whether agents fail because they cannot reach border poses, cannot align, sample invalid `DO`, or suffer critic instability after reaching edge states.

---

## Likely code-level issues and missed signals

Highest-priority issues:

1. **`action_mask` is all zeros and unused**
   `state.py:3536-3547`, `utils_ppo.py:43-66`.

2. **Existing `_get_action_mask_*` omits `DO_NOTHING`**
   Mask length is 7, action space is 8.

3. **Existing `_get_action_mask_*` compares against previous agent in multi-agent mode**
   Use `new_state._get_current_agent_state()`, not `_get_prev_agent_state()`.

4. **`train_mixed.py` terminal backfill shifts reward forward, not backward**
   `train_mixed.py:891-906`. This can contaminate post-reset states.

5. **Backfill uses whole reward and all `done`, not terminal component and `task_done`**
   This amplifies value target noise.

6. **Observation exposes raw cone, not legal cone**
   `wrappers.py:82-128`, `wrappers.py:177-234`, `env.py:337-362`.

7. **Completion logging uses `done`, not `task_done`**
   `train_mixed.py:1050-1053`.

8. **Eval/inference are unmasked in several files**
   `eval.py`, `eval_mcts.py`, `eval_mixed.py`, `visualize_mixed.py`, `visualize_paths.py`, `inference/inference_single_map.py`.

Lower-priority but worth fixing:

* `maps_buffer.py:63-65` casts border axes to `float16`; prefer `float32`.
* `wrappers.py:207-213` uses a fresh default `EnvConfig()` instead of `state.env_cfg`.
* `state.py:1410-1417` fallback path lacks proximity enforcement.
* `interaction_mask` name is misleading because it is not an action or legality mask.

## Suggested experiment order

1. **No-learning geometry sanity test**
   Verify edge mask truth tables on horizontal, vertical, diagonal, and corner synthetic maps.

2. **PPO action mask only**
   Fix mask bugs, add masked categorical, store masks in transitions. No observation or reward changes yet.

3. **Disable mixed terminal backfill**
   Especially for multi-agent runs. Compare value loss and explained variance.

4. **Add current legal-dig mask and edge/core remaining features**
   Keep architecture otherwise unchanged.

5. **Add phase-conditioned critic input**
   Then try lower `vf_coef`, Huber value loss, and separate `vf_clip_eps`.

6. **Edge reset curriculum**
   Mix edge-only and full-task starts. Track edge completion, not only total success.

7. **Optional heavier changes**
   Multi-head critic, PopArt, auxiliary edge-legality prediction, or hierarchical options for “reposition to edge → align → dig edge.”

The first three items are the most likely to explain the observed “finish less often + critic spikes” without invoking broader RL instability.
