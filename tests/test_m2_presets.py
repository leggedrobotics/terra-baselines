"""Preset contracts after the agent-neutral relocation reward migration."""

import pytest

from configs import training_configs
from configs.training_configs import get_config, list_configs
from train_mixed import create_mixed_agent_env_config

M2_PRESETS = ("m2_wave_long", "m2_dose", "m2_dose_fast")
RELOCATION_PROGRESS_MULT = 1.5


def test_every_preset_uses_the_single_relocation_reward_knob():
    for name in list_configs():
        preset = get_config(name)
        assert "reward_multipliers" not in vars(preset)
        assert isinstance(preset.relocation_progress_mult, float)


def test_preset_multiplier_reaches_terra_env_config():
    multiplier = get_config("m2_dose").relocation_progress_mult
    env_cfg = create_mixed_agent_env_config(
        agent_types=(0,),
        action_types=(0,),
        relocation_progress_mult=multiplier,
    )
    assert env_cfg.relocation_progress_mult == RELOCATION_PROGRESS_MULT


def test_removed_nested_reward_config_fails_loudly(tmp_path, monkeypatch):
    config_path = tmp_path / "training_configs.yaml"
    config_path.write_text("""
stale:
  description: stale reward schema
  agent_types: [0]
  action_types: [0]
  reward_multipliers:
    excavator_relocate_dug_dirt_mult: 1.5
""".lstrip())
    monkeypatch.setattr(training_configs, "_get_yaml_path", lambda: config_path)
    monkeypatch.setattr(training_configs, "_TRAINING_CONFIGS", {})
    monkeypatch.setattr(training_configs, "_CONFIGS_LOADED", False)

    with pytest.raises(ValueError, match="removed reward_multipliers"):
        training_configs._load_configs_from_yaml()


def test_m2_promotion_rules_and_level_paths_are_unchanged():
    wave = get_config("m2_wave_long")
    dose = get_config("m2_dose")
    fast = get_config("m2_dose_fast")

    assert wave.relocation_progress_mult == RELOCATION_PROGRESS_MULT
    assert dose.relocation_progress_mult == RELOCATION_PROGRESS_MULT
    assert fast.relocation_progress_mult == RELOCATION_PROGRESS_MULT

    assert [level.maps_path for level in wave.maps] == [
        "train/L0",
        "train/L1",
        "train/L2",
        "train/L3",
    ]
    for arm in (dose, fast):
        assert [level.maps_path for level in arm.maps] == [
            "train/L0p",
            "train/L1p",
            "train/L2p",
            "train/L3p",
        ]

    assert wave.curriculum.increase_level_threshold == 3
    assert dose.curriculum.increase_level_threshold == 3
    assert fast.curriculum.increase_level_threshold == 2
    for arm in (wave, dose, fast):
        assert arm.curriculum.decrease_level_threshold == 3
        assert arm.curriculum.last_level_type == "none"
        assert all(level.max_steps_in_episode == 450 for level in arm.maps)
        assert arm.agent_types == (0,)
