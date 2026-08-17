from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax.training.train_state import TrainState
from terra.config import BatchConfig, MapsDimsConfig

from train import Transition, ppo_update_networks
from utils.models import get_model_ready
from utils.utils_ppo import obs_to_model_input, recurrent_policy_sequence


class Config(dict):
    __getattr__ = dict.__getitem__


def _config(actor_core="gru"):
    return Config(
        clip_action_maps=True,
        loaded_max=127,
        local_map_normalization_bounds=(-16, 16),
        maps_net_normalization_bounds=(-10, 10),
        local_map_area_scale=1.0,
        model_core="mlp",
        actor_core=actor_core,
        actor_gru_hidden_dim=64,
        model_size="medium",
        map_encoder="atari",
        num_prev_actions=5,
        clip_eps=0.2,
        vf_coef=2.0,
        ent_coef=0.01,
        use_value_clip=True,
        flat_minibatch_shuffle=False,
        teacher_obs_downsample=1,
        aux_coef=0.0,
        action_logit_masking=False,
    )


def _env(edge=32):
    return SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=edge))
    )


def _observation(shape_prefix, env):
    edge = env.batch_cfg.maps_dims.maps_edge_length
    angles = env.batch_cfg.agent.angles_cabin
    state_width = env.batch_cfg.agent.num_state_obs

    def zeros(*tail, dtype=jnp.float32):
        return jnp.zeros(shape_prefix + tail, dtype=dtype)

    agent_active = zeros(4, dtype=jnp.int8).at[..., 0].set(1)
    grid = lambda: zeros(edge, edge)
    local = lambda: zeros(angles)
    return {
        "agent_states": zeros(4, state_width),
        "agent_active": agent_active,
        "num_agents": jnp.ones(shape_prefix, dtype=jnp.int32),
        "local_map_action_neg": local(),
        "local_map_action_pos": local(),
        "local_map_target_neg": local(),
        "local_map_target_pos": local(),
        "local_map_dumpability": local(),
        "local_map_obstacles": local(),
        "local_map_border_workspace": local(),
        "local_map_edge_alignment_error": local(),
        "local_map_border_diggable": local(),
        "traversability_mask": grid(),
        "reachability_mask": grid(),
        "action_map": grid(),
        "target_map": grid(),
        "agent_width": zeros(dtype=jnp.int32),
        "agent_height": zeros(dtype=jnp.int32),
        "padding_mask": grid(),
        "dumpability_mask": grid(),
        "interaction_mask": grid(),
    }


def test_gru_step_matches_sequence_with_terminal_reset():
    env = _env()
    config = _config()
    model, params = get_model_ready(jax.random.PRNGKey(0), config, env)
    obs = _observation((2, 3), env)
    prev_actions = jnp.zeros((2, 3, 5), dtype=jnp.int32)
    model_obs = obs_to_model_input(obs, prev_actions, config)
    dones = jnp.array([[False, True, False], [False, False, False]])
    h0 = jnp.zeros((2, 64), dtype=jnp.float32)

    values, logits, final_hidden = model.apply(
        params, model_obs, h0, dones, method="actor_sequence"
    )
    carry = h0
    step_values = []
    step_logits = []
    for step in range(3):
        step_obs = [leaf[:, step] for leaf in model_obs]
        value, step_logit, carry = model.apply(
            params, step_obs, carry, method="actor_step"
        )
        carry = jnp.where(dones[:, step, None], 0, carry)
        step_values.append(value)
        step_logits.append(step_logit)

    # Encoder batch shapes differ in this tiny test; bf16 convolution rounding
    # is therefore close rather than bit-identical. Production uses 512 rows
    # for both rollout and PPO replay.
    assert jnp.allclose(values, jnp.stack(step_values, axis=1), atol=2e-3)
    assert jnp.allclose(logits, jnp.stack(step_logits, axis=1), atol=2e-3)
    assert jnp.allclose(final_hidden, carry, atol=2e-3)

    _, feedforward_params = get_model_ready(
        jax.random.PRNGKey(0), _config(actor_core="mlp"), env
    )
    # GRU64 cell (43,392) + concat-skip head: post_gru consumes
    # [gru_output 64, actor_input 160] instead of the feed-forward 160.
    assert (
        sum(x.size for x in jax.tree.leaves(params))
        - sum(x.size for x in jax.tree.leaves(feedforward_params))
        == 46_336
    )


def test_recurrent_ppo_update_is_finite():
    env = _env()
    config = _config()
    model, params = get_model_ready(jax.random.PRNGKey(1), config, env)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(1e-3),
    )
    obs = _observation((2, 2), env)
    prev_actions = jnp.zeros((2, 2, 5), dtype=jnp.int32)
    done = jnp.array([[False, True], [False, False]])
    h0 = jnp.zeros((2, 64), dtype=jnp.float32)
    model_obs = obs_to_model_input(obs, prev_actions, config)
    value, dist, _ = recurrent_policy_sequence(
        model.apply, params, model_obs, h0, done
    )
    action = jnp.argmax(dist.logits_parameter(), axis=-1)
    transition = Transition(
        done=done,
        task_done=jnp.zeros_like(done),
        action=action,
        value=value[..., 0],
        reward=jnp.zeros((2, 2), dtype=jnp.float32),
        log_prob=dist.log_prob(action),
        obs=obs,
        prev_actions=prev_actions,
        prev_reward=jnp.zeros((2, 2), dtype=jnp.float32),
    )
    advantages = jnp.ones((2, 2), dtype=jnp.float32)
    targets = value[..., 0] + 0.25

    def update(state, batch, adv, target, hidden):
        return ppo_update_networks(
            state,
            batch,
            adv,
            target,
            config,
            actor_hidden_init=hidden,
        )

    add_device = lambda leaf: jnp.asarray(leaf)[None]
    new_state, info = jax.vmap(update, axis_name="devices")(
        jtu.tree_map(add_device, train_state),
        jtu.tree_map(add_device, transition),
        advantages[None],
        targets[None],
        h0[None],
    )
    assert jnp.isfinite(info["total_loss"]).all()
    assert info["diagnostics/grads_all_finite"].item() == 1.0
    assert info["diagnostics/params_all_finite"].item() == 1.0
    assert all(jnp.isfinite(x).all() for x in jax.tree.leaves(new_state.params))
