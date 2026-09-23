"""Joint team policy: shared trunk, per-agent readout, MAT-style decoding."""

from types import SimpleNamespace as NS
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from terra.config import BatchConfig, MapsDimsConfig
from terra.env import AGENT_VIEW_OBS_KEYS

from train import ppo_update_networks
from utils.models import get_model_ready
from utils.team_migration import team_optimizer_state, team_params_from_single_agent
from utils.utils_ppo import (
    joint_obs_to_model_input,
    obs_to_model_input,
    random_agent_order,
    update_prev_actions,
)

A, B, MAP = 3, 4, 64


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class TeamTransition(NamedTuple):
    obs: Any
    prev_actions: Any
    done: Any
    action: Any
    log_prob: Any
    agent_order: Any
    agent_values: Any
    value: Any
    reward: Any


def _config(**changes):
    return Config(
        clip_action_maps=True, loaded_max=100,
        local_map_normalization_bounds=(-16, 16), maps_net_normalization_bounds=(-10, 10),
        model_core="mlp", model_size="base", num_prev_actions=5,
        map_encoder="resnet_spatial_8x8_se_sa_xattn", resnet_stage_channels=(4, 4, 8, 8),
        resnet_blocks_per_stage=(1, 1, 1, 1), actor_residual_head=True,
        clip_eps=0.2, vf_coef=2.0, ent_coef=0.01,
        **changes,
    )


ENV = NS(batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=MAP)))


def _team_obs(rng):
    keys = iter(jax.random.split(rng, 32))
    active = (jnp.arange(4) < A).astype(jnp.int8)
    obs = {
        "agent_states": jax.random.randint(next(keys), (B, A, 4, 9), 0, 12).astype(jnp.float32),
        "agent_active": jnp.broadcast_to(active, (B, A, 4)),
        "num_agents": jnp.full((B,), A, jnp.int32),
        "agent_width": jnp.full((B,), 7, jnp.int32),
        "agent_height": jnp.full((B,), 11, jnp.int32),
    }
    for key in (
        "local_map_action_neg", "local_map_action_pos", "local_map_target_neg",
        "local_map_target_pos", "local_map_dumpability", "local_map_obstacles",
        "local_map_border_workspace", "local_map_edge_alignment_error",
        "local_map_border_diggable",
    ):
        obs[key] = jax.random.normal(next(keys), (B, A, 12))
    for key in ("reachability_mask", "action_map", "target_map", "padding_mask", "dumpability_mask"):
        obs[key] = jax.random.randint(next(keys), (B, MAP, MAP), -1, 2).astype(jnp.int8)
    # Per-view maps: own chassis / own workspace differ between agents.
    for key in ("traversability_mask", "interaction_mask"):
        obs[key] = jax.random.randint(next(keys), (B, A, MAP, MAP), -1, 2).astype(jnp.int8)
    return obs


def _models():
    single_cfg, team_cfg = _config(), _config(agent_types_override=(0,) * A)
    single_model, single_params = get_model_ready(jax.random.PRNGKey(0), single_cfg, ENV)
    team_model, team_init = get_model_ready(jax.random.PRNGKey(1), team_cfg, ENV)
    team_params = team_params_from_single_agent(single_params, team_init)
    return single_cfg, team_cfg, single_model, team_model, team_params


def _activate_decoder(params, rng):
    params = jax.tree_util.tree_map(lambda x: x, params)
    decoder = params["params"]["intent_decoder"]
    out = max((n for n in decoder if n.startswith("Dense_")), key=lambda n: int(n[6:]))
    decoder[out]["kernel"] = 0.5 * jax.random.normal(rng, decoder[out]["kernel"].shape)
    return params


def test_migrated_team_is_the_single_agent_policy_on_every_view():
    single_cfg, team_cfg, single_model, team_model, params = _models()
    single_params = {"params": {k: v for k, v in params["params"].items() if k != "intent_decoder"}}
    obs = _team_obs(jax.random.PRNGKey(2))
    prev = jax.random.randint(jax.random.PRNGKey(3), (B, A, 5), 0, 8)
    order = random_agent_order(jax.random.PRNGKey(4), B, A)
    actions = jax.random.randint(jax.random.PRNGKey(5), (B, A), 0, 8)
    values, logits = team_model.apply(
        params, joint_obs_to_model_input(obs, prev, team_cfg, A), actions, order,
        method="joint_policy",
    )
    for slot in range(A):
        view = {k: (v[:, slot] if k in AGENT_VIEW_OBS_KEYS else v) for k, v in obs.items()}
        value, logit = single_model.apply(
            single_params, obs_to_model_input(view, prev[:, slot], single_cfg)
        )
        np.testing.assert_allclose(logits[:, slot], logit, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(values[:, slot], value[:, 0], rtol=1e-5, atol=1e-5)


def test_sequential_sampling_matches_teacher_forcing_and_is_causal():
    _, team_cfg, _, team_model, params = _models()
    params = _activate_decoder(params, jax.random.PRNGKey(6))
    obs = joint_obs_to_model_input(
        _team_obs(jax.random.PRNGKey(7)), jnp.zeros((B, A, 5), jnp.int32), team_cfg, A
    )
    order = random_agent_order(jax.random.PRNGKey(8), B, A)
    actions, log_probs, _, _ = team_model.apply(
        params, obs, order, jax.random.PRNGKey(9), False, method="joint_act"
    )
    _, logits = team_model.apply(params, obs, actions, order, method="joint_policy")
    forced = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], -1)[..., 0]
    np.testing.assert_allclose(log_probs, forced, rtol=1e-5, atol=1e-5)

    rows = jnp.arange(B)
    first, last = order[:, 0], order[:, -1]
    # The first agent decides alone: later agents' actions cannot move it.
    changed_last = actions.at[rows, last].set((actions[rows, last] + 3) % 8)
    _, logits_last = team_model.apply(params, obs, changed_last, order, method="joint_policy")
    np.testing.assert_allclose(logits_last[rows, first], logits[rows, first], atol=1e-6)
    # The last agent conditions on the first agent's chosen action.
    changed_first = actions.at[rows, first].set((actions[rows, first] + 3) % 8)
    _, logits_first = team_model.apply(params, obs, changed_first, order, method="joint_policy")
    assert float(jnp.abs(logits_first[rows, last] - logits[rows, last]).max()) > 1e-4


def test_team_ppo_update_trains_the_intent_decoder():
    _, team_cfg, _, team_model, params = _models()
    team_cfg = Config(team_cfg, use_value_clip=True)
    state = TrainState.create(apply_fn=team_model.apply, params=params, tx=optax.adam(1e-3))
    steps = 2
    obs = jax.tree_util.tree_map(
        lambda *x: jnp.stack(x, axis=1),
        *[_team_obs(jax.random.PRNGKey(10 + t)) for t in range(steps)],
    )
    prev = jnp.zeros((B, steps, A, 5), jnp.int32)
    order = random_agent_order(jax.random.PRNGKey(20), B * steps, A)
    flat = jax.tree_util.tree_map(lambda x: x.reshape((B * steps,) + x.shape[2:]), obs)
    model_obs = joint_obs_to_model_input(flat, prev.reshape((B * steps, A, 5)), team_cfg, A)
    actions, log_probs, agent_values, _ = team_model.apply(
        params, model_obs, order, jax.random.PRNGKey(21), False, method="joint_act"
    )
    transitions = TeamTransition(
        obs=obs, prev_actions=prev, done=jnp.zeros((B, steps), bool),
        action=actions.reshape(B, steps, A), log_prob=log_probs.reshape(B, steps, A),
        agent_order=order.reshape(B, steps, A),
        agent_values=agent_values.reshape(B, steps, A),
        value=agent_values.mean(-1).reshape(B, steps),
        reward=jnp.zeros((B, steps)),
    )
    advantages = jax.random.normal(jax.random.PRNGKey(22), (B, steps))
    targets = transitions.value + advantages

    def update(s, tr, adv, tgt):
        return ppo_update_networks(s, tr, adv, tgt, team_cfg)

    batched = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x)[None], (state, transitions, advantages, targets)
    )
    new_state, info = jax.vmap(update, axis_name="devices")(*batched)
    assert np.isfinite(float(info["total_loss"][0]))
    assert abs(float(info["approx_kl"][0])) < 1e-5  # fresh rollout: ratio is one
    decoder = new_state.params["params"]["intent_decoder"]
    out = max((n for n in decoder if n.startswith("Dense_")), key=lambda n: int(n[6:]))
    assert float(jnp.abs(decoder[out]["kernel"][0]).max()) > 0


def test_prev_actions_keep_one_history_per_agent():
    prev = jnp.arange(2 * A * 5).reshape(2, A, 5)
    action = jnp.array([[7, 8, 9], [1, 2, 3]])
    updated = update_prev_actions(prev, action, jnp.array([False, True]))
    np.testing.assert_array_equal(updated[0, :, 0], [7, 8, 9])
    np.testing.assert_array_equal(updated[0, :, 1:], prev[0, :, :4])
    np.testing.assert_array_equal(updated[1], 0)


def test_team_optimizer_keeps_the_parents_adam_moments():
    _, _, _, _, params = _models()
    single_params = {"params": {k: v for k, v in params["params"].items() if k != "intent_decoder"}}
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4, eps=1e-5))
    grads = jax.tree_util.tree_map(jnp.ones_like, single_params)
    _, parent = tx.update(grads, tx.init(single_params), single_params)
    team = team_optimizer_state(parent, tx.init(params))
    adam, parent_adam = team[1][0], parent[1][0]
    assert int(adam.count) == int(parent_adam.count) == 1
    for tree in ("mu", "nu"):
        mine, theirs = getattr(adam, tree)["params"], getattr(parent_adam, tree)["params"]
        np.testing.assert_array_equal(
            mine["mlp_pi"]["layers_0"]["kernel"], theirs["mlp_pi"]["layers_0"]["kernel"]
        )
        assert all(float(jnp.abs(x).max()) == 0 for x in jax.tree_util.tree_leaves(mine["intent_decoder"]))
    updates, _ = tx.update(jax.tree_util.tree_map(jnp.ones_like, params), team, params)
    assert jax.tree_util.tree_structure(updates) == jax.tree_util.tree_structure(params)
