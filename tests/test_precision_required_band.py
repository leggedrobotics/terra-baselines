"""Optional global precision input and zero-preserving GRU warm-start migration."""
import copy
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict
from terra.config import BatchConfig, MapsDimsConfig

from utils.models import PrecisionInputConv, get_model_ready
from utils.precision_required_band import STEM_KERNEL_PATH, migrate_precision_required_band_params
from utils.utils_ppo import obs_to_model_input


class Config(dict):
    __getattr__ = dict.__getitem__


def model_config(precision=False):
    return Config(
        clip_action_maps=True, loaded_max=127,
        local_map_normalization_bounds=(-16, 16), maps_net_normalization_bounds=(-10, 10),
        local_map_area_scale=1.0, model_core="mlp", model_size="medium",
        actor_core="gru", actor_gru_hidden_dim=64, num_prev_actions=5,
        map_encoder="resnet_spatial_8x8_se_sa_xattn",
        resnet_stage_channels=(8, 12, 16, 24), resnet_blocks_per_stage=(1, 1, 1, 1),
        encoder_compute_dtype="float32", attention_compute_dtype="float32",
        carry_work_observation=True, trench_alignment_observation=True,
        relocation_distance_observation=True, admissible_dig_observation=True,
        executable_dig_observation=True, precision_required_band_observation=precision,
        action_logit_masking=False,
    )


def model_env():
    return SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64)),
        executable_dig_observation=True,
    )


def raw_obs(prefix=(2, 3)):
    rng = np.random.default_rng(4)
    maps = {key: jnp.asarray(rng.integers(0, 2, size=prefix + (64, 64)), jnp.float32)
            for key in ("traversability_mask", "reachability_mask", "action_map", "target_map",
                        "padding_mask", "dumpability_mask", "interaction_mask")}
    local_names = ("action_neg", "action_pos", "target_neg", "target_pos", "dumpability",
                   "obstacles", "border_workspace", "edge_alignment_error", "border_diggable",
                   "admissible_dig")
    state = jnp.zeros(prefix + (4, 9), jnp.float32).at[..., 0, :].set(
        jnp.asarray([28, 31, 3, 2, 0, 4, 0, 0, .2]))
    return {
        **maps,
        **{f"local_map_{name}": jnp.asarray(rng.random(prefix + (12,)), jnp.float32)
           for name in local_names},
        "agent_states": state,
        "agent_active": jnp.zeros(prefix + (4,), jnp.int8).at[..., 0].set(1),
        "num_agents": jnp.ones(prefix, jnp.int32),
        "agent_width": jnp.full(prefix, 7, jnp.int32),
        "agent_height": jnp.full(prefix, 11, jnp.int32),
        "fresh_trench_dig_alignment_valid": jnp.ones(prefix),
        "fresh_trench_dig_yaw_error": jnp.full(prefix, .15),
        "fresh_trench_dig_standoff_error": jnp.zeros(prefix),
        "stall_age": jnp.full(prefix, .1),
        "reward_v2_reset_context": jnp.zeros(prefix + (2,)),
        "movement_feasibility": jnp.ones(prefix + (4,)),
        "previous_action_outcome": jnp.ones(prefix + (2,)),
        "relocation_distance_map": jnp.full(prefix + (64, 64), 1.25),
        "remaining_time": jnp.full(prefix, .6),
        "retained_work_context": jnp.zeros(prefix + (5,)),
        "precision_required_band": jnp.asarray(rng.integers(0, 2, size=prefix + (64, 64)), jnp.bool_),
        "action_mask": jnp.ones(prefix + (8,), jnp.bool_),
    }


def test_old_teacher_preprocessing_ignores_only_new_map():
    raw = raw_obs()
    previous = jnp.zeros((2, 3, 5), jnp.int32)
    config = model_config(False)
    old = obs_to_model_input(raw, previous, config)
    without_mask = dict(raw)
    without_mask.pop("precision_required_band")
    legacy = obs_to_model_input(without_mask, previous, config)
    assert len(old) == len(legacy)
    for first, second in zip(old, legacy):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(old[9], raw["local_map_border_workspace"])
    np.testing.assert_array_equal(old[10], raw["local_map_edge_alignment_error"])
    np.testing.assert_array_equal(old[11], raw["local_map_border_diggable"])


def test_append_after_existing_options_and_before_action_mask():
    raw = raw_obs((2,))
    previous = jnp.zeros((2, 5), jnp.int32)
    old_config = model_config(False)
    old_config.update(stall_age_observation=True, reward_v2_reset_context_observation=True,
                      movement_feasibility_observation=True, previous_outcome_observation=True,
                      time_observation_mode="remaining", retained_work_context_observation=True)
    old = obs_to_model_input(raw, previous, old_config)
    new_config = Config(old_config, precision_required_band_observation=True)
    new = obs_to_model_input(raw, previous, new_config)
    assert len(new) == len(old) + 1
    for first, second in zip(old, new[:-1]):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(new[-1], raw["precision_required_band"])
    # The separate action-mask interface always remains last.
    new_config.update(time_observation_mode="none", action_logit_masking=True)
    masked = obs_to_model_input(raw, previous, new_config)
    np.testing.assert_array_equal(masked[-2], raw["precision_required_band"])
    np.testing.assert_array_equal(masked[-1], raw["action_mask"])


def test_missing_or_wrong_shape_mask_fails():
    raw = raw_obs((1,))
    previous = jnp.zeros((1, 5), jnp.int32)
    raw.pop("precision_required_band")
    with pytest.raises(ValueError, match="precision_required_band"):
        obs_to_model_input(raw, previous, model_config(True))
    raw["precision_required_band"] = jnp.zeros((1, 12))
    with pytest.raises(ValueError, match="target_map shape"):
        obs_to_model_input(raw, previous, model_config(True))


def test_migration_rejects_unrelated_changes_and_keeps_source_untouched():
    source = FrozenDict({"params": {"maps_net": {"cnn": {"Conv_0": {
        "kernel": jnp.ones((3, 3, 12, 8))}}}, "other": jnp.ones(2)}})
    target = source.unfreeze()
    target["params"]["maps_net"]["cnn"]["Conv_0"]["kernel"] = jnp.ones((3, 3, 13, 8))
    migrated = migrate_precision_required_band_params(source, FrozenDict(target))
    assert isinstance(migrated, FrozenDict)
    assert flatten_dict(source)[STEM_KERNEL_PATH].shape == (3, 3, 12, 8)
    np.testing.assert_array_equal(flatten_dict(migrated)[STEM_KERNEL_PATH][:, :, :12],
                                  flatten_dict(source)[STEM_KERNEL_PATH])
    np.testing.assert_array_equal(flatten_dict(migrated)[STEM_KERNEL_PATH][:, :, 12:], 0)
    with pytest.raises(ValueError, match="exactly one"):
        migrate_precision_required_band_params(source, source)
    target["params"]["other"] = jnp.ones(3)
    with pytest.raises(ValueError, match="cannot change params/other"):
        migrate_precision_required_band_params(source, target)


def test_zero_precision_slice_receives_gradient_and_can_change_output():
    conv = PrecisionInputConv(features=4, compute_dtype=jnp.bfloat16)
    inputs = jnp.ones((2, 8, 8, 13), jnp.bfloat16)
    params = conv.init(jax.random.PRNGKey(3), inputs)
    params["params"]["kernel"] = params["params"]["kernel"].at[:, :, -1:].set(0)
    gradient = jax.grad(lambda p: conv.apply(p, inputs).astype(jnp.float32).sum())(params)
    assert bool(jnp.all(gradient["params"]["kernel"][:, :, -1:] > 0))
    before = conv.apply(params, inputs)
    changed = copy.deepcopy(params)
    changed["params"]["kernel"] = params["params"]["kernel"].at[:, :, -1:].set(.125)
    assert bool(jnp.any(conv.apply(changed, inputs) != before))


def assert_sequence_parity(source_model, source_params, target_model, initialized,
                           source_config, target_config):
    params = migrate_precision_required_band_params(source_params, initialized)
    source_flat, target_flat = flatten_dict(source_params), flatten_dict(params)
    for path in source_flat:
        if path == STEM_KERNEL_PATH:
            np.testing.assert_array_equal(target_flat[path][:, :, :-1], source_flat[path])
            np.testing.assert_array_equal(target_flat[path][:, :, -1:], 0)
        else:
            np.testing.assert_array_equal(target_flat[path], source_flat[path])
    raw = raw_obs()
    previous = jnp.arange(30, dtype=jnp.int32).reshape(2, 3, 5) % 8
    source_obs = obs_to_model_input(raw, previous, source_config)
    target_obs = obs_to_model_input(raw, previous, target_config)
    hidden = jnp.linspace(-.1, .1, 128).reshape(2, 64)
    done = jnp.asarray([[False, True, False], [False, False, False]])
    before = source_model.apply(source_params, source_obs, hidden, done, method="actor_sequence")
    after = target_model.apply(params, target_obs, hidden, done, method="actor_sequence")
    differences = []
    for old, new in zip(before, after):
        np.testing.assert_array_equal(new, old)
        differences.append(float(jnp.max(jnp.abs(new - old))))
    print("precision-mask migration value/logit/hidden max errors", differences)
    return params


def test_recurrent_outputs_preserved_with_nonzero_mask_and_terminal_reset():
    cfg = model_config(False)
    grown_cfg = model_config(True)
    source_model, source = get_model_ready(jax.random.PRNGKey(1), cfg, model_env())
    target_model, target = get_model_ready(jax.random.PRNGKey(2), grown_cfg, model_env())
    assert_sequence_parity(source_model, source, target_model, target, cfg, grown_cfg)


@pytest.mark.skipif(not os.environ.get("TERRA_PRECISION_SOURCE_CHECKPOINT"),
                    reason="optional check against the actual u110000 artifact")
def test_actual_u110000_checkpoint_migration():
    from utils.helpers import register_checkpoint_config_classes, load_pkl_object
    register_checkpoint_config_classes()
    checkpoint = load_pkl_object(os.environ["TERRA_PRECISION_SOURCE_CHECKPOINT"])
    source_cfg = checkpoint["train_config"]
    target_cfg = copy.copy(source_cfg)
    target_cfg.precision_required_band_observation = True
    source_model, _ = get_model_ready(jax.random.PRNGKey(1), source_cfg, model_env())
    target_model, initialized = get_model_ready(jax.random.PRNGKey(2), target_cfg, model_env())
    assert_sequence_parity(source_model, checkpoint["model"], target_model, initialized,
                           source_cfg, target_cfg)
