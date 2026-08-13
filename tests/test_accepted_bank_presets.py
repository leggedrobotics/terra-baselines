from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from configs.training_configs import get_config
from train_mixed import (
    PER_ENV_RATCHET_DISABLED_THRESHOLD,
    _assert_pooled_level_contract,
    assign_curriculum_levels,
    pooled_sampler_settings,
    reset_exposure_histogram,
    transition_exposure_histogram,
)
from utils.accepted_bank import ARMS


@pytest.mark.parametrize("arm", ARMS)
def test_accepted_bank_presets_freeze_the_shared_contract(arm):
    config = get_config(arm)
    assert config.accepted_bank_arm == arm
    assert config.maps == []
    assert config.agent_types == (0,)
    assert config.action_types == (0,)
    assert config.relocation_progress_mult == 1.5
    assert config.curriculum.last_level_type == "none"
    assert (
        config.curriculum.increase_level_threshold >= PER_ENV_RATCHET_DISABLED_THRESHOLD
    )
    assert (
        config.curriculum.decrease_level_threshold >= PER_ENV_RATCHET_DISABLED_THRESHOLD
    )
    assert config.pooled_sampler.enabled


def test_generalist_arms_differ_only_in_sampler_rule():
    uniform = get_config("G-UNIFORM")
    adaptive = get_config("G-ADAPTIVE")
    assert uniform.agent_types == adaptive.agent_types
    assert uniform.action_types == adaptive.action_types
    assert uniform.relocation_progress_mult == adaptive.relocation_progress_mult
    assert uniform.curriculum == adaptive.curriculum

    uniform_sampler = vars(uniform.pooled_sampler).copy()
    adaptive_sampler = vars(adaptive.pooled_sampler).copy()
    assert uniform_sampler.pop("rule") == "uniform"
    assert adaptive_sampler.pop("rule") == "adaptive"
    assert uniform_sampler == adaptive_sampler


def test_v8_preset_uses_the_frozen_weighted_stage_sampler():
    config = get_config("G-V8-FIXED")
    assert config.accepted_bank_arm == "G-UNIFORM"
    assert config.maps == []
    assert config.pooled_sampler.enabled
    assert config.pooled_sampler.rule == "fixed"
    assert config.curriculum.last_level_type == "none"


def test_v8_v4_preset_selects_global_open_replay_sampler():
    config = get_config("G-V8-CONTINUOUS-V4")
    assert config.accepted_bank_arm == "G-UNIFORM"
    assert config.maps == []
    assert config.pooled_sampler.enabled
    assert config.pooled_sampler.rule == "continuous_banded_v4"
    assert config.pooled_sampler.max_mass == 0.15


@pytest.mark.parametrize("arm", ("F-SPECIALIST", "T-SPECIALIST"))
def test_family_specialists_use_the_frozen_uniform_treatment(arm):
    specialist = get_config(arm)
    generalist = get_config("G-UNIFORM")
    ignored = {"name", "description", "accepted_bank_arm"}
    specialist_treatment = {
        key: value for key, value in vars(specialist).items() if key not in ignored
    }
    generalist_treatment = {
        key: value for key, value in vars(generalist).items() if key not in ignored
    }
    assert specialist_treatment == generalist_treatment


def test_pooled_settings_are_read_from_the_effective_training_config():
    preset = get_config("G-ADAPTIVE")
    sampler = vars(preset.pooled_sampler).copy()
    sampler["seed"] = 11
    settings = pooled_sampler_settings(SimpleNamespace(pooled_sampler=sampler))
    assert settings.rule == "adaptive"
    assert settings.seed == 11
    assert settings.uniform_floor == 0.20


def test_reset_exposure_histogram_uses_each_transition_level():
    done = jnp.asarray([[False, True, False], [True, False, True]], dtype=jnp.bool_)
    levels = jnp.asarray([[0, 1, 2], [2, 2, 0]], dtype=jnp.int16)

    np.testing.assert_array_equal(
        reset_exposure_histogram(done, levels, num_stages=3),
        np.asarray([1, 1, 1], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="identical"):
        reset_exposure_histogram(done, levels[:, :2], num_stages=3)

    np.testing.assert_array_equal(
        transition_exposure_histogram(levels, num_stages=3),
        np.asarray([2, 1, 3], dtype=np.int32),
    )


def test_level_contract_accepts_any_condition_count_but_no_protocol_changes():
    level = {
        "maps_path": "train/condition",
        "max_steps_in_episode": 450,
        "rewards_type": 0,
        "apply_trench_rewards": False,
    }
    _assert_pooled_level_contract([level], 10**9, 10**9)
    _assert_pooled_level_contract(
        [level, {**level, "maps_path": "train/other"}],
        10**9,
        10**9,
    )
    for bad in (
        [{**level, "max_steps_in_episode": 449}],
        [{**level, "rewards_type": 1}],
        [{**level, "apply_trench_rewards": True}],
    ):
        with pytest.raises(ValueError, match="horizon=450"):
            _assert_pooled_level_contract(bad, 10**9, 10**9)
    with pytest.raises(ValueError, match="per-env ratchet"):
        _assert_pooled_level_contract([level], 3, 3)


def test_condition_assignment_preserves_shape_and_dtype():
    curriculum = SimpleNamespace(
        level=jnp.zeros((2, 4), dtype=jnp.int16),
        _replace=lambda **changes: SimpleNamespace(
            level=changes.get("level"),
        ),
    )
    env_cfg = SimpleNamespace(
        curriculum=curriculum,
        _replace=lambda **changes: SimpleNamespace(
            curriculum=changes["curriculum"],
        ),
    )
    assigned = assign_curriculum_levels(
        env_cfg,
        np.arange(8, dtype=np.int32).reshape(2, 4),
    )
    assert assigned.curriculum.level.dtype == jnp.int16
    np.testing.assert_array_equal(
        assigned.curriculum.level,
        np.arange(8).reshape(2, 4),
    )
    with pytest.raises(ValueError, match="condition assignment shape"):
        assign_curriculum_levels(env_cfg, np.zeros((8,), dtype=np.int32))


def test_shared_launcher_freezes_production_ppo_contract():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_accepted_bank_screen.sh"
    ).read_text()
    for argument in (
        ': "${SEED:?',
        '--seed "$SEED"',
        "--update_epochs 2",
        "--num_minibatches 32",
        "--lr 3e-4",
        "--log_eval_interval 0",
        "--keep_checkpoint_history",
        'FINITE_CHECK_INTERVAL="${FINITE_CHECK_INTERVAL:-10}"',
        '--finite_check_interval "$FINITE_CHECK_INTERVAL"',
        'CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"',
        '--checkpoint_interval "$CHECKPOINT_INTERVAL"',
        'CACHE_CLEAR_INTERVAL="${CACHE_CLEAR_INTERVAL:-1000}"',
        '--cache_clear_interval "$CACHE_CLEAR_INTERVAL"',
        'if [ "$#" -ne 4 ]',
    ):
        assert argument in script
    for arm in ARMS:
        assert arm in script
    assert '"$@"' not in script
