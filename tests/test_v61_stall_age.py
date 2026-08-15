import copy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from terra.config import BatchConfig, MapsDimsConfig

from scripts.prepare_v61_stall_age_continuation import (
    FUSED_WIDTH,
    PARAMETER_NAMES,
    add_zero_adam_moments,
    add_zero_embeddings,
    migrate_sampler_state,
)
from utils.models import get_model_ready
from utils.utils_ppo import obs_to_model_input
from train_mixed import _attach_stall_age_receipt


class Config(dict):
    __getattr__ = dict.__getitem__


def config(stall_age_observation: bool):
    return Config(
        clip_action_maps=True,
        loaded_max=100,
        local_map_normalization_bounds=(-16, 16),
        maps_net_normalization_bounds=(-10, 10),
        model_core="mlp",
        model_size="medium",
        num_prev_actions=5,
        critic_hidden_dims=(512, 256),
        encoder_compute_dtype="bfloat16",
        attention_compute_dtype="float32",
        map_encoder="resnet_spatial_8x8_se_sa_xattn",
        resnet_stage_channels=(24, 48, 64, 96),
        resnet_blocks_per_stage=(2, 2, 3, 3),
        token_mixer_residual_init_scale=0.1,
        flatten_reduce_channels=32,
        attn_latent_queries=8,
        aux_coef=0.0,
        carry_work_observation=True,
        stall_age_observation=stall_age_observation,
        action_logit_masking=False,
        local_map_area_scale=1.0,
    )


def raw_obs(batch=3):
    maps = {
        key: jnp.zeros((batch, 64, 64), dtype=jnp.float32)
        for key in (
            "traversability_mask",
            "reachability_mask",
            "action_map",
            "target_map",
            "padding_mask",
            "dumpability_mask",
            "interaction_mask",
        )
    }
    return {
        "agent_states": jnp.zeros((batch, 4, 9), dtype=jnp.float32),
        "agent_active": jnp.ones((batch, 4), dtype=jnp.int8),
        "num_agents": jnp.ones((batch,), dtype=jnp.int32),
        **{
            key: jnp.zeros((batch, 12), dtype=jnp.float32)
            for key in (
                "local_map_action_neg",
                "local_map_action_pos",
                "local_map_target_neg",
                "local_map_target_pos",
                "local_map_dumpability",
                "local_map_obstacles",
                "local_map_border_workspace",
                "local_map_edge_alignment_error",
                "local_map_border_diggable",
            )
        },
        "agent_width": jnp.zeros((batch,), dtype=jnp.int32),
        "agent_height": jnp.zeros((batch,), dtype=jnp.int32),
        "stall_age": jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32),
        **maps,
    }


def test_zero_stall_embeddings_preserve_outputs_and_optimizer_tree():
    env = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
    )
    source_model, source_params = get_model_ready(
        jax.random.PRNGKey(1), config(False), env
    )
    target_model, target_params = get_model_ready(
        jax.random.PRNGKey(2), config(True), env
    )
    grown_params = add_zero_embeddings(source_params)
    assert jax.tree.structure(grown_params) == jax.tree.structure(target_params)

    observation = raw_obs()
    previous_actions = jnp.zeros((3, 5), dtype=jnp.int32)
    source_input = obs_to_model_input(observation, previous_actions, config(False))
    target_input = obs_to_model_input(observation, previous_actions, config(True))
    source_value, source_logits = source_model.apply(source_params, source_input)
    target_value, target_logits = target_model.apply(grown_params, target_input)
    np.testing.assert_array_equal(np.asarray(target_value), np.asarray(source_value))
    np.testing.assert_array_equal(np.asarray(target_logits), np.asarray(source_logits))

    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(3e-4, eps=1e-5),
    )
    source_state = TrainState.create(
        apply_fn=source_model.apply, params=source_params, tx=tx
    )
    grown_opt_state = add_zero_adam_moments(source_state.opt_state)
    assert jax.tree.structure(grown_params) == jax.tree.structure(
        grown_opt_state[1][0].mu
    )
    assert jax.tree.structure(grown_params) == jax.tree.structure(
        grown_opt_state[1][0].nu
    )
    target_state = TrainState.create(
        apply_fn=target_model.apply, params=grown_params, tx=tx
    ).replace(opt_state=grown_opt_state)
    updated = target_state.apply_gradients(
        grads=jax.tree.map(jnp.zeros_like, grown_params)
    )
    assert jax.tree.structure(updated.params) == jax.tree.structure(
        updated.opt_state[1][0].mu
    )
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree.leaves((updated.params, updated.opt_state))
    )
    for name in PARAMETER_NAMES:
        np.testing.assert_array_equal(
            np.asarray(updated.params["params"][name]),
            np.zeros((FUSED_WIDTH,), dtype=np.float32),
        )


def test_stall_age_input_is_required_and_appended_after_existing_features():
    observation = raw_obs()
    previous_actions = jnp.zeros((3, 5), dtype=jnp.int32)
    model_input = obs_to_model_input(observation, previous_actions, config(True))
    assert len(model_input) == 23
    np.testing.assert_array_equal(
        np.asarray(model_input[22]), np.asarray(observation["stall_age"][:, None])
    )
    missing = copy.copy(observation)
    del missing["stall_age"]
    try:
        obs_to_model_input(missing, previous_actions, config(True))
    except ValueError as error:
        assert "obs['stall_age']" in str(error)
    else:
        raise AssertionError("missing stall_age observation was accepted")


def test_reset_context_uses_two_zero_initialized_branch_embeddings():
    env = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
    )
    context_config = config(False)
    context_config["reward_v2_reset_context_observation"] = True
    _, params = get_model_ready(
        jax.random.PRNGKey(3),
        context_config,
        env,
    )
    model_params = params["params"]
    for name in (
        "reward_v2_reset_context_actor_embedding",
        "reward_v2_reset_context_critic_embedding",
    ):
        values = np.asarray(model_params[name])
        assert values.shape == (2, FUSED_WIDTH)
        np.testing.assert_array_equal(values, np.zeros_like(values))

    observation = raw_obs()
    observation["reward_v2_reset_context"] = jnp.asarray(
        [[0.0, 1.0], [0.25, 0.75], [0.9, 0.1]], dtype=jnp.float32
    )
    model_input = obs_to_model_input(
        observation,
        jnp.zeros((3, 5), dtype=jnp.int32),
        context_config,
    )
    assert len(model_input) == 23
    np.testing.assert_array_equal(
        np.asarray(model_input[22]),
        np.asarray(observation["reward_v2_reset_context"]),
    )


def test_stall_age_receipt_is_carried_into_continuation_checkpoints():
    receipt = {
        "schema": "terra_v8_v61_stall_age_v3_prepared_v1",
        "source_checkpoint_sha256": "7" * 64,
    }
    rolling = {}
    final = {}
    _attach_stall_age_receipt(rolling, receipt)
    _attach_stall_age_receipt(final, receipt)
    assert rolling["stall_age_prepared_continuation"] == receipt
    assert final["stall_age_prepared_continuation"] == receipt
    assert rolling["stall_age_prepared_continuation"] is not receipt


def test_u14_sampler_conversion_keeps_history_and_clears_only_partial_window():
    names = [f"foundation-{index}" for index in range(5)] + [
        f"trench-{index}" for index in range(5)
    ]
    depths = (0, 1, 1, 2, 2) * 2
    labels = {
        name: {
            "family": name.split("-")[0],
            "branch_depth": "Composed",
            "curriculum_depth": depths[index],
        }
        for index, name in enumerate(names)
    }
    from utils.pooled_sampler import PooledConditionSampler, SamplerSettings

    source = PooledConditionSampler(
        names,
        SamplerSettings(
            rule="continuous_banded_v3",
            mastery_threshold=0.80,
            min_episodes=32,
        ),
        maps_per_condition=[96] * len(names),
        labels=labels,
    )
    source.start(13_950)
    state = source.state_dict()
    state["settings"]["rule"] = "continuous_banded_v2"
    state["probabilities"] = source._continuous_distribution_v2().tolist()
    state["current_window"]["updates"] = 50
    state["current_window"]["sampled_assignment_count"][0] = 102_400
    state["current_window"]["completed_episode_count"][0] = 13_725
    state["current_window"]["reset_exposure_count"][0] = 13_725
    state["current_window"]["transition_exposure_count"][0] = 3_276_800
    state["current_window"]["task_done_count"][0] = 8_810

    migrated, receipt = migrate_sampler_state(state)
    assert migrated["settings"]["rule"] == "continuous_banded_v3"
    assert migrated["current_window"]["updates"] == 0
    assert receipt["discarded_partial_window_updates"] == 50
    assert receipt["discarded_assignments"] == 102_400
    assert receipt["discarded_completed_episodes"] == 13_725
    assert receipt["discarded_transition_exposures"] == 3_276_800
    for key in (
        "conditions",
        "labels",
        "competence",
        "closed_window",
        "refresh",
        "numpy_rng",
        "mastery",
    ):
        assert migrated[key] == state[key]
