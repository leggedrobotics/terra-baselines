"""Relocation-distance channel and admissible-dig local map wiring.

Three claims: the flag-off path is byte-identical (same obs list, same
parameter tree), the flag-on path appends exactly [H, W] then width-12 entries
in that order and changes only the stem-conv and LocalMapNet fan-in, and a
Terra export without the keys fails loudly.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from terra.config import BatchConfig, MapsDimsConfig

from utils.models import get_model_ready
from utils.utils_ppo import obs_to_model_input


class Config(dict):
    __getattr__ = dict.__getitem__


def config(relocation: bool, admissible: bool):
    return Config(
        clip_action_maps=True,
        local_map_area_scale=1.0,
        action_logit_masking=False,
        stall_age_observation=False,
        reward_v2_reset_context_observation=False,
        trench_alignment_observation=True,
        relocation_distance_observation=relocation,
        admissible_dig_observation=admissible,
    )


def model_config(relocation: bool, admissible: bool):
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
        stall_age_observation=False,
        reward_v2_reset_context_observation=False,
        trench_alignment_observation=True,
        relocation_distance_observation=relocation,
        admissible_dig_observation=admissible,
    )


def raw_obs(batch=3, with_new_keys=True):
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
    obs = {
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
        "fresh_trench_dig_alignment_valid": jnp.ones((batch,), dtype=jnp.float32),
        "fresh_trench_dig_yaw_error": jnp.zeros((batch,), dtype=jnp.float32),
        "fresh_trench_dig_standoff_error": jnp.zeros((batch,), dtype=jnp.float32),
        **maps,
    }
    if with_new_keys:
        obs["relocation_distance_map"] = jnp.full(
            (batch, 64, 64), 1.25, dtype=jnp.float32
        )
        obs["local_map_admissible_dig"] = jnp.arange(
            batch * 12, dtype=jnp.int16
        ).reshape(batch, 12)
    return obs


def env():
    return SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
    )


def param_count(params):
    return sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params))


def test_flag_off_obs_list_is_unchanged():
    prev = jnp.zeros((3, 5), dtype=jnp.int32)
    off = obs_to_model_input(raw_obs(), prev, config(False, False))
    off_no_keys = obs_to_model_input(raw_obs(with_new_keys=False), prev, config(False, False))
    assert len(off) == 23  # 22 base + trench alignment
    assert len(off_no_keys) == 23
    for a, b in zip(off, off_no_keys):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_flag_on_appends_distance_then_admissible_in_order():
    prev = jnp.zeros((3, 5), dtype=jnp.int32)
    both = obs_to_model_input(raw_obs(), prev, config(True, True))
    assert len(both) == 25
    assert both[23].shape == (3, 64, 64)
    np.testing.assert_allclose(np.asarray(both[23]), 1.25)
    assert both[24].shape == (3, 12)
    np.testing.assert_array_equal(
        np.asarray(both[24]), np.arange(36).reshape(3, 12)
    )
    only_distance = obs_to_model_input(raw_obs(), prev, config(True, False))
    assert len(only_distance) == 24 and only_distance[23].shape == (3, 64, 64)
    only_admissible = obs_to_model_input(raw_obs(), prev, config(False, True))
    assert len(only_admissible) == 24 and only_admissible[23].shape == (3, 12)


def test_missing_terra_keys_fail_loudly():
    prev = jnp.zeros((3, 5), dtype=jnp.int32)
    with pytest.raises(ValueError, match="relocation_distance_map"):
        obs_to_model_input(raw_obs(with_new_keys=False), prev, config(True, False))
    with pytest.raises(ValueError, match="local_map_admissible_dig"):
        obs_to_model_input(raw_obs(with_new_keys=False), prev, config(False, True))


def test_parameter_delta_is_stem_and_local_map_fan_in_only():
    _, base = get_model_ready(jax.random.PRNGKey(0), model_config(False, False), env())
    model, both = get_model_ready(jax.random.PRNGKey(0), model_config(True, True), env())
    # Stem conv 3x3 x (+1 channel) x 24 = 216; LocalMapNet Dense_0 (+12) x 320 = 3840.
    assert param_count(both) - param_count(base) == 216 + 3840
    base_flat = {
        jax.tree_util.keystr(k): v.shape
        for k, v in jax.tree_util.tree_flatten_with_path(base)[0]
    }
    both_flat = {
        jax.tree_util.keystr(k): v.shape
        for k, v in jax.tree_util.tree_flatten_with_path(both)[0]
    }
    assert set(base_flat) == set(both_flat)
    changed = sorted(k for k in base_flat if base_flat[k] != both_flat[k])
    assert changed == [
        "['params']['local_map_net']['mlp']['layers_0']['kernel']",
        "['params']['maps_net']['cnn']['Conv_0']['kernel']",
    ]
    # Forward pass on real-shaped inputs is finite and reads the new entries.
    prev = jnp.zeros((3, 5), dtype=jnp.int32)
    obs = obs_to_model_input(raw_obs(), prev, config(True, True))
    value, logits = model.apply(both, obs)
    assert value.shape == (3, 1) and logits.shape == (3, 8)
    assert bool(jnp.all(jnp.isfinite(value))) and bool(jnp.all(jnp.isfinite(logits)))
    zeroed = list(obs)
    zeroed[23] = jnp.zeros_like(obs[23])
    zeroed[24] = jnp.zeros_like(obs[24])
    value_z, logits_z = model.apply(both, zeroed)
    assert not np.allclose(np.asarray(logits), np.asarray(logits_z))
