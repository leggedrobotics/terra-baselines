"""Objective/observation changes must be explicit while native Adam stays intact."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from terra.config import EnvConfig
from train_mixed import (
    _foundation_behavior_settings,
    _overlay_env_foundation_behavior,
    _r2_protocol_receipt,
    _validate_checkpoint_architecture,
    _validate_foundation_behavior_config,
    _validate_r2_resume_checkpoint,
)


def config(**changes):
    values = dict(
        reward_stage="reward_v2", gamma=0.9984,
        distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
        distance_sidecar_sha256="a" * 64, update_epochs=2, num_minibatches=32,
        admissible_dig_observation=True, executable_dig_observation=False,
        lateral_dig_cost=0.0, base_travel_cost=0.0, base_turn_cost=0.0,
        finetune_foundation_behavior=False, resume_from="parent.pkl", warm_start_from=None,
    )
    return SimpleNamespace(**(values | changes))


def checkpoint(saved_config=None):
    saved_config = saved_config or config()
    return dict(
        train_config=saved_config,
        r2_protocol_receipt=_r2_protocol_receipt(saved_config),
        optimizer_state={"sentinel": np.array([1.5, 2.5])},
        train_state_step=np.array(64_000), next_update=1000,
    )


def treatment(**changes):
    return config(**(dict(lateral_dig_cost=0.25, base_travel_cost=0.005,
                         base_turn_cost=0.02, executable_dig_observation=True) | changes))


def test_defaults_keep_the_legacy_receipt_and_environment_behavior():
    old = config()
    for name in _foundation_behavior_settings(old):
        delattr(old, name)
    assert _r2_protocol_receipt(old) == _r2_protocol_receipt(config())
    assert "foundation_behavior" not in _r2_protocol_receipt(old)
    env = EnvConfig()
    assert _overlay_env_foundation_behavior(env, old) == env
    _validate_r2_resume_checkpoint(checkpoint(old), _r2_protocol_receipt(config()), config())


def test_ordinary_resume_rejects_reward_or_affordance_changes():
    saved = checkpoint()
    for changed in (
        config(lateral_dig_cost=0.25), config(base_travel_cost=0.005),
        config(base_turn_cost=0.02), config(executable_dig_observation=True),
    ):
        with pytest.raises(ValueError, match="protocol receipt"):
            _validate_r2_resume_checkpoint(saved, _r2_protocol_receipt(changed), changed)


def test_explicit_finetune_preserves_optimizer_and_only_allows_behavior_changes():
    current = treatment(finetune_foundation_behavior=True)
    saved = checkpoint()
    optimizer = saved["optimizer_state"]
    _validate_foundation_behavior_config(current)
    _validate_r2_resume_checkpoint(saved, _r2_protocol_receipt(current), current)
    _validate_checkpoint_architecture(saved, current)
    assert saved["optimizer_state"] is optimizer
    np.testing.assert_array_equal(optimizer["sentinel"], [1.5, 2.5])
    assert saved["next_update"] == 1000
    assert saved["train_state_step"] == 64000
    bad = copy.deepcopy(saved)
    bad["r2_protocol_receipt"]["constants"]["shaping_weight"] = 2.0
    with pytest.raises(ValueError, match="protocol receipt"):
        _validate_r2_resume_checkpoint(bad, _r2_protocol_receipt(current), current)
    bad = copy.deepcopy(saved)
    bad["train_state_step"] = np.array(63999)
    with pytest.raises(ValueError, match="optimizer clock"):
        _validate_r2_resume_checkpoint(bad, _r2_protocol_receipt(current), current)
    bad = copy.deepcopy(saved)
    bad["train_config"].map_encoder = "resnet_spatial_8x8"
    with pytest.raises(ValueError, match="map_encoder"):
        _validate_checkpoint_architecture(bad, current)


def test_treatment_checkpoint_can_resume_normally_and_cannot_silently_disable_costs():
    current = treatment()
    saved = checkpoint(current)
    _validate_r2_resume_checkpoint(saved, _r2_protocol_receipt(current), current)
    _validate_checkpoint_architecture(saved, current)
    with pytest.raises(ValueError, match="protocol receipt"):
        _validate_r2_resume_checkpoint(saved, _r2_protocol_receipt(config()), config())


def test_executable_observation_requires_explicit_same_width_semantic_change():
    saved = checkpoint()
    with pytest.raises(ValueError, match="executable_dig_observation"):
        _validate_checkpoint_architecture(saved, treatment())
    _validate_checkpoint_architecture(saved, treatment(finetune_foundation_behavior=True))


def test_ordinary_resume_cannot_erase_cost_saved_only_in_environment():
    old = config()
    for name in _foundation_behavior_settings(old):
        delattr(old, name)
    saved = checkpoint(old)
    saved["env_config"] = EnvConfig()._replace(lateral_dig_cost=0.25)
    with pytest.raises(ValueError, match="lateral_dig_cost mismatch"):
        _validate_r2_resume_checkpoint(saved, _r2_protocol_receipt(config()), config())


def test_restored_environment_gets_all_four_selected_fields_only():
    env = EnvConfig()
    updated = _overlay_env_foundation_behavior(env, treatment())
    settings = _foundation_behavior_settings(treatment())
    for field in env._fields:
        if field in settings:
            assert getattr(updated, field) == settings[field]
        else:
            assert getattr(updated, field) == getattr(env, field)


@pytest.mark.parametrize("changes", [
    {"lateral_dig_cost": -0.01}, {"base_travel_cost": float("nan")},
    {"base_turn_cost": float("inf")},
    {"lateral_dig_cost": 0.1, "reward_stage": "dense_skill"},
    {"executable_dig_observation": True, "admissible_dig_observation": False},
    {"finetune_foundation_behavior": True, "resume_from": None},
    {"finetune_foundation_behavior": True, "warm_start_from": "old.pkl"},
])
def test_invalid_behavior_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        _validate_foundation_behavior_config(config(**changes))
