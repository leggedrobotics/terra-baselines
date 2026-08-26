"""Fresh-trench dig-alignment observation wiring (C0/T1 pilot, blocker 1).

Covers the three claims the pilot depends on: the flag-off path is unchanged,
the flag-on path appends exactly one width-3 entry carrying Terra's
[valid, yaw_error, standoff_error], and a bank without those keys fails loudly.
"""

import copy
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from terra.config import BatchConfig, MapsDimsConfig

from configs.training_configs import get_config
from utils.models import get_model_ready
from utils.utils_ppo import obs_to_model_input


class Config(dict):
    __getattr__ = dict.__getitem__


def test_metadata_preflight_uses_the_axis_v2_loader_signature(
    tmp_path, monkeypatch
):
    from train_mixed import _preflight_trench_alignment_metadata

    metadata_dir = tmp_path / "train/fixture/metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "trench_0.json").write_text(
        json.dumps({"axes_ABC": [], "trench_axes_count": 0})
    )
    monkeypatch.setenv("DATASET_PATH", str(tmp_path))

    validated = []
    env = SimpleNamespace(
        _validate_trench_alignment_metadata_requirements=validated.append
    )
    env_params = SimpleNamespace(_replace=lambda **kwargs: kwargs)
    _preflight_trench_alignment_metadata(
        env,
        env_params,
        [{"maps_path": "train/fixture"}],
    )

    assert validated == [{"enforce_trench_dig_alignment": True}]


def config(trench_alignment_observation: bool):
    return Config(
        clip_action_maps=True,
        local_map_area_scale=1.0,
        action_logit_masking=False,
        stall_age_observation=False,
        reward_v2_reset_context_observation=False,
        trench_alignment_observation=trench_alignment_observation,
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
        # Terra's top-level export (terra/env.py::_state_to_obs_dict).
        "fresh_trench_dig_alignment_valid": jnp.asarray(
            [1.0, 0.0, 0.0], dtype=jnp.float32
        ),
        "fresh_trench_dig_yaw_error": jnp.asarray(
            [0.0, 0.25, 1.0], dtype=jnp.float32
        ),
        "fresh_trench_dig_standoff_error": jnp.asarray(
            [0.0, -0.5, 0.75], dtype=jnp.float32
        ),
        **maps,
    }


def test_trench_alignment_observation_appends_one_width_three_entry():
    observation = raw_obs()
    previous_actions = jnp.zeros((3, 5), dtype=jnp.int32)

    baseline = obs_to_model_input(observation, previous_actions, config(False))
    treated = obs_to_model_input(observation, previous_actions, config(True))

    assert len(baseline) == 22
    assert len(treated) == len(baseline) + 1
    for index, entry in enumerate(baseline):
        np.testing.assert_array_equal(
            np.asarray(treated[index]), np.asarray(entry)
        )

    appended = np.asarray(treated[22])
    assert appended.shape == (3, 3)
    assert appended.dtype == np.float32
    np.testing.assert_array_equal(
        appended,
        np.stack(
            [
                np.asarray(observation["fresh_trench_dig_alignment_valid"]),
                np.asarray(observation["fresh_trench_dig_yaw_error"]),
                np.asarray(observation["fresh_trench_dig_standoff_error"]),
            ],
            axis=-1,
        ),
    )


def test_trench_alignment_observation_requires_the_terra_keys():
    observation = copy.copy(raw_obs())
    del observation["fresh_trench_dig_yaw_error"]

    with pytest.raises(ValueError) as error:
        obs_to_model_input(
            observation, jnp.zeros((3, 5), dtype=jnp.int32), config(True)
        )
    assert "fresh_trench_dig_yaw_error" in str(error.value)


def model_config(trench_alignment_observation: bool):
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
        action_logit_masking=False,
        local_map_area_scale=1.0,
        trench_alignment_observation=trench_alignment_observation,
    )


def test_encoder_consumes_the_width_three_vector_via_zero_init_embeddings():
    env = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
    )
    baseline_model, baseline_params = get_model_ready(
        jax.random.PRNGKey(1), model_config(False), env
    )
    treated_model, treated_params = get_model_ready(
        jax.random.PRNGKey(1), model_config(True), env
    )

    for name in (
        "trench_alignment_actor_embedding",
        "trench_alignment_critic_embedding",
    ):
        assert name not in baseline_params["params"]
        values = np.asarray(treated_params["params"][name])
        assert values.shape == (3, 704)
        np.testing.assert_array_equal(values, np.zeros_like(values))

    observation = raw_obs()
    previous_actions = jnp.zeros((3, 5), dtype=jnp.int32)
    baseline_value, baseline_logits = baseline_model.apply(
        baseline_params,
        obs_to_model_input(observation, previous_actions, config(False)),
    )
    treated_value, treated_logits = treated_model.apply(
        treated_params,
        obs_to_model_input(observation, previous_actions, config(True)),
    )
    # Zero init: the extra width is wired in but starts as an exact no-op.
    np.testing.assert_allclose(
        np.asarray(treated_value), np.asarray(baseline_value), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(treated_logits), np.asarray(baseline_logits), rtol=0, atol=0
    )

    # A non-zero embedding must actually move the heads, i.e. the vector is read.
    moved = jax.tree.map(lambda x: x, treated_params)
    moved["params"]["trench_alignment_actor_embedding"] = jnp.ones((3, 704))
    _, moved_logits = treated_model.apply(
        moved, obs_to_model_input(observation, previous_actions, config(True))
    )
    assert not np.allclose(np.asarray(moved_logits), np.asarray(treated_logits))


def test_pilot_arms_differ_only_in_the_gate():
    control = get_config("trench_align_c0_v1")
    treatment = get_config("trench_align_t1_v1")

    assert control.enforce_trench_dig_alignment is False
    assert treatment.enforce_trench_dig_alignment is True
    for preset in (control, treatment):
        assert preset.trench_alignment_observation is True
        assert preset.require_trench_alignment_metadata is True
        assert preset.agent_types == (0,)
        assert preset.action_types == (0,)
        assert [m.apply_trench_rewards for m in preset.maps] == [False]
        assert [m.rewards_type for m in preset.maps] == ["DENSE"]
    # One placeholder bank shared by both arms: repointing stays a one-liner.
    assert [m.maps_path for m in control.maps] == [
        m.maps_path for m in treatment.maps
    ]
    assert [m.max_steps_in_episode for m in control.maps] == [
        m.max_steps_in_episode for m in treatment.maps
    ]
    assert control.curriculum == treatment.curriculum


def fingerprint_config(**overrides):
    """A checkpoint-architecture dict as `checkpoint_treatment_fingerprint` reads it."""
    return {
        "name": "trench_align",
        "seed": 20260818,
        "model_size": "medium",
        "map_encoder": "resnet_spatial_8x8_se_sa_xattn",
        "carry_work_observation": True,
        **overrides,
    }


def test_evaluator_model_config_follows_the_checkpoint_architecture():
    """The checkpoint is the single source of truth; nothing is passed by hand."""
    from eval_fixed_bank import configure_for_bank
    from utils.utils_ppo import _config_option

    trained = SimpleNamespace(
        trench_alignment_observation=True,
        enforce_trench_dig_alignment=True,
        require_trench_alignment_metadata=True,
        num_minibatches=32,
    )
    treated = configure_for_bank(trained, "evaluation/gate_main/development", 608)
    assert _config_option(treated, "trench_alignment_observation", False) is True
    assert treated.enforce_trench_dig_alignment is True
    assert treated.require_trench_alignment_metadata is True

    # A pre-pilot checkpoint simply has no such field; it must read as off.
    legacy = configure_for_bank(
        SimpleNamespace(num_minibatches=32), "evaluation/main/development", 720
    )
    assert _config_option(legacy, "trench_alignment_observation", False) is False


def test_treatment_fingerprint_records_the_pilot_treatment_only_when_present():
    from eval_fixed_bank import checkpoint_treatment_fingerprint

    baseline = checkpoint_treatment_fingerprint(
        {"train_config": fingerprint_config()}
    )
    assert "trench_alignment_observation" not in baseline["contract"]["architecture"]
    assert "trench_dig_alignment" not in baseline["contract"]

    treated = checkpoint_treatment_fingerprint(
        {
            "train_config": fingerprint_config(
                trench_alignment_observation=True,
                enforce_trench_dig_alignment=True,
                require_trench_alignment_metadata=True,
            )
        }
    )
    assert treated["contract"]["architecture"]["trench_alignment_observation"] is True
    assert treated["contract"]["trench_dig_alignment"] == {
        "enforce_trench_dig_alignment": True,
        "require_trench_alignment_metadata": True,
    }
    assert treated["sha256"] != baseline["sha256"]


def test_architecture_validation_rejects_a_rebuild_without_the_alignment_flag():
    from train_mixed import _validate_checkpoint_architecture

    checkpoint = {"train_config": {"trench_alignment_observation": True}}
    _validate_checkpoint_architecture(
        checkpoint, SimpleNamespace(trench_alignment_observation=True)
    )
    with pytest.raises(ValueError) as error:
        _validate_checkpoint_architecture(checkpoint, SimpleNamespace())
    assert "trench_alignment_observation" in str(error.value)


def test_rebuilt_model_must_reproduce_the_checkpoint_parameters():
    from utils.models import validate_model_params_match

    env = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
    )
    _, treated_params = get_model_ready(
        jax.random.PRNGKey(1), model_config(True), env
    )
    _, baseline_params = get_model_ready(
        jax.random.PRNGKey(1), model_config(False), env
    )
    validate_model_params_match(treated_params, treated_params, "matched rebuild")
    validate_model_params_match(baseline_params, baseline_params, "matched rebuild")
    with pytest.raises(ValueError) as error:
        validate_model_params_match(
            baseline_params, treated_params, "flag-off rebuild"
        )
    assert "trench_alignment_actor_embedding" in str(error.value)
