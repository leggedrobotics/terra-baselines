# D3 action-mask wiring: where the zero mask comes from (2026-08-10)

Investigated while implementing `spatial_v6_3m`. Conclusion first: **this is not
a terra-baselines wiring bug and cannot be fixed in this repository.** The mask
never reaches terra-baselines at all — it is zeroed inside the Terra env, and it
lives in `info`, not in the observation. Deferred with the exact patch points
below.

## What actually computes a mask

`terra/state.py::State._get_action_mask(dummy_action)` (line 3654) returns a
bool vector over the action enum, `1 = allowed`. It routes by action type to
`_get_action_mask_tracked` (3534) / `_get_action_mask_wheeled` (3593), then
disables the two cabin actions for trucks (agent_type 1) and skid steers (2),
then truncates to `num_actions`.

The per-action test is *effect-based, by simulation*: it applies each handler
(`_handle_move_forward`, `_handle_clock`, `_handle_do`, ...) to a copy of the
state and asks whether the relevant field changed (pos_base / angle_base /
angle_cabin / loaded). So `do` is marked illegal exactly when a full dig/dump
handler would be a no-op — which is the obstacle-veto case the audit's severe
obstacle loops sit in — but it costs one shadow application of every handler
per step.

## Where it becomes zeros

1. `terra/state.py::State._get_infos` (line 3699):

   ```python
   # Keep infos cheap: target_tiles is already materialized by the wrapper as
   # interaction_mask, and action_mask is currently informational only.
   ...
   "action_mask": jnp.zeros((dummy_action.get_num_actions(),), dtype=jnp.bool_),
   ```

   `_get_action_mask` is never called here. The zeroing is deliberate and
   documented as a cost decision, not an oversight.

2. `terra/env.py` reset (line ~123) builds `dummy_info` with the same zero
   `action_mask`, to keep the reset info pytree aligned with `step()`.

## Why terra-baselines cannot fix it

- The mask is an **info** field. `terra/env.py::_state_to_obs_dict` (~line 545)
  emits no `action_mask` key, so no observation carries it.
- `utils/utils_ppo.py::obs_to_model_input` (lines 60-84) assembles the 22-entry
  model input from the observation dict only; `utils/models.py::MapsNet`
  consumes seven global maps (traversability, reachability, action, target,
  padding, dumpability, interaction). There is no mask input to route.
- Nothing in terra-baselines reads `info["action_mask"]`. The only consumer of
  the helper is `scripts/analysis/diagnose_f0_trajectories.py:460`, which calls
  `state._get_action_mask(dummy_action)` directly on replayed states for
  offline analysis — that path already gets the true mask.

## Exact patch points for later (Terra repo)

| # | File | Change |
|---|---|---|
| 1 | `terra/state.py::_get_infos` (3699) | return `self._get_action_mask(dummy_action)` instead of zeros |
| 2 | `terra/env.py` reset `dummy_info` (~123) | compute the same mask at reset, or keep zeros and document that step 0 is unmasked |
| 3 | `terra/env.py::_state_to_obs_dict` (~545) | only if the POLICY is to see it: add `"action_mask"` to the observation dict |
| 4 | `terra-baselines/utils/utils_ppo.py::obs_to_model_input` | append the mask as obs[22] |
| 5 | `terra-baselines/utils/models.py` | consume it (concatenate to the fused feature vector, or add logits masking in `policy`) |

Steps 1-2 alone make the mask *observable to the training loop* (diagnostics,
D3's oracle upper bound); steps 3-5 change the observation space and the
parameter tree, so they are a new encoder version and a checkpoint break.

## Decisions this note does not make

- Whether the mask should be an observation channel or a logits mask. Logits
  masking is the cheap version but changes the policy's action distribution
  semantics (entropy, KL, and the entropy schedule all move), so it is not a
  drop-in for a matched screen.
- The per-step cost of making the mask live (7 shadow handler applications per
  agent per step). Measure before adopting; the existing comment says it was
  zeroed precisely to keep infos cheap.
- V8_REWARD_TERMINATION_AUDIT.md D3 is an oracle immediate-effect masking
  *probe* on the seven severe obstacle-loop cases — an upper bound, explicitly
  "not a deployable mask result". Do the probe with patch 1 before spending
  anything on 3-5.
