from types import SimpleNamespace

import numpy as np

from eval_fixed_bank import checkpoint_treatment_fingerprint, configure_for_bank
from train_mixed import (
    MixedAgentTrainConfig,
    PARTIAL_RESET_INITIAL_SHARE,
    PARTIAL_RESET_PHASE_UPDATES,
    PARTIAL_RESET_TOTAL_UPDATES,
    _validate_checkpoint_architecture,
    _validate_partial_reset_resume,
    partial_reset_curriculum_receipt,
    partial_reset_lane_tiers,
    partial_reset_schedule,
)
from utils.pooled_sampler import PooledConditionSampler, SamplerSettings


def test_partial_reset_schedule_has_three_fixed_windows_then_a_separate_fade():
    assert partial_reset_schedule(0).tiers == (1,)
    assert partial_reset_schedule(PARTIAL_RESET_PHASE_UPDATES).tiers == (2, 1)
    assert partial_reset_schedule(2 * PARTIAL_RESET_PHASE_UPDATES).tiers == (
        3,
        2,
        1,
    )
    hardest = partial_reset_schedule(3 * PARTIAL_RESET_PHASE_UPDATES - 1)
    assert hardest.share == PARTIAL_RESET_INITIAL_SHARE

    fade_start = partial_reset_schedule(3 * PARTIAL_RESET_PHASE_UPDATES)
    fade_mid = partial_reset_schedule(7 * PARTIAL_RESET_PHASE_UPDATES // 2)
    assert fade_start.tiers == fade_mid.tiers == (3, 2, 1)
    assert fade_start.share == PARTIAL_RESET_INITIAL_SHARE
    assert fade_mid.share == PARTIAL_RESET_INITIAL_SHARE / 2
    assert partial_reset_schedule(PARTIAL_RESET_TOTAL_UPDATES).share == 0.0

    lanes = partial_reset_lane_tiers((2, 8), fade_start, seed=7)
    assert np.count_nonzero(lanes) == 4
    assert set(lanes.reshape(-1)) <= {0, 1, 2, 3}
    np.testing.assert_array_equal(
        lanes,
        partial_reset_lane_tiers((2, 8), fade_start, seed=7),
    )

    cumulative_lanes = partial_reset_lane_tiers(
        (1, 1200),
        partial_reset_schedule(2 * PARTIAL_RESET_PHASE_UPDATES),
        seed=7,
    )
    assert np.mean(cumulative_lanes != 0) == PARTIAL_RESET_INITIAL_SHARE
    assert np.count_nonzero(cumulative_lanes == 1) == 100
    assert np.count_nonzero(cumulative_lanes == 2) == 100
    assert np.count_nonzero(cumulative_lanes == 3) == 100


def _continuous_sampler():
    names = tuple(f"condition-{index}" for index in range(8))
    families = ("foundation",) * 4 + ("trench",) * 4
    depths = (0, 1, 1, 2, 0, 1, 2, 2)
    labels = {
        name: {
            "family": families[index],
            "curriculum_depth": depths[index],
        }
        for index, name in enumerate(names)
    }
    settings = SamplerSettings(
        rule="continuous_banded_v3",
        update_interval=150,
        mastery_threshold=0.80,
        min_episodes=32,
        competence_ema=0.30,
        max_mass=0.15,
        seed=9,
    )
    return PooledConditionSampler(list(names), settings, labels=labels)


def test_partial_condition_sampling_respects_support_and_mastery_sees_full_only():
    sampler = _continuous_sampler()
    reset_tiers = np.resize(np.asarray([0, 1, 2, 3], dtype=np.int32), 400)
    support = np.ones((4, 8), dtype=bool)
    support[1:] = False
    support[1:, [0, 3, 4, 6]] = True

    levels = sampler.sample_levels_for_reset_tiers(reset_tiers, support)
    for tier in (1, 2, 3):
        assert np.all(support[tier, levels[reset_tiers == tier]])

    full_episodes = np.arange(1, 9, dtype=np.int64)
    full_successes = full_episodes // 2
    sampler.observe_exact_episode_counts(full_episodes, full_successes)
    state = sampler.state_dict()
    assert state["current_window"]["completed_episode_count"] == (
        full_episodes.tolist()
    )
    assert state["current_window"]["task_done_count"] == (
        full_successes.tolist()
    )
    assert sum(state["current_window"]["sampled_assignment_count"]) == 400

    mismatched = support.copy()
    mismatched[2, 0] = False
    try:
        sampler.sample_levels_for_reset_tiers(reset_tiers, mismatched)
    except ValueError as error:
        assert "one common condition support" in str(error)
    else:
        raise AssertionError("per-tier partial support drift was accepted")


def test_native_resume_uses_absolute_update_and_bank_digest_not_host_path():
    digest = "a" * 64
    source = SimpleNamespace(
        partial_reset_root="/source/partial-bank",
        partial_reset_bank_sha256=digest,
    )
    checkpoint = {
        "partial_reset_curriculum": partial_reset_curriculum_receipt(
            source, PARTIAL_RESET_PHASE_UPDATES
        )
    }
    relocated = SimpleNamespace(
        partial_reset_root="/relocated/same-bank",
        partial_reset_bank_sha256=digest,
        reward_v2_reset_context_observation=True,
    )
    _validate_partial_reset_resume(
        checkpoint,
        "resume",
        relocated,
        PARTIAL_RESET_PHASE_UPDATES,
    )

    wrong_bank = SimpleNamespace(
        partial_reset_root=relocated.partial_reset_root,
        partial_reset_bank_sha256="b" * 64,
    )
    try:
        _validate_partial_reset_resume(
            checkpoint,
            "resume",
            wrong_bank,
            PARTIAL_RESET_PHASE_UPDATES,
        )
    except ValueError as error:
        assert "schedule" in str(error)
    else:
        raise AssertionError("native resume accepted a different partial reset bank")

    later_checkpoint = {
        "train_config": relocated,
        "partial_reset_curriculum": partial_reset_curriculum_receipt(
            relocated, 2 * PARTIAL_RESET_PHASE_UPDATES
        ),
    }
    source_checkpoint = {
        "train_config": relocated,
        **checkpoint,
    }
    assert checkpoint_treatment_fingerprint(source_checkpoint) == (
        checkpoint_treatment_fingerprint(later_checkpoint)
    )
    context_control = {
        "train_config": SimpleNamespace(
            reward_v2_reset_context_observation=True,
        )
    }
    legacy_control = {
        "train_config": SimpleNamespace(
            reward_v2_reset_context_observation=False,
        )
    }
    assert checkpoint_treatment_fingerprint(context_control) != (
        checkpoint_treatment_fingerprint(legacy_control)
    )
    eval_config = configure_for_bank(relocated, "held-out", 4)
    assert eval_config.partial_reset_root is None
    assert eval_config.partial_reset_bank_sha256 is None
    assert eval_config.reward_v2_reset_context_observation is True

    try:
        _validate_checkpoint_architecture(
            {
                "train_config": SimpleNamespace(
                    reward_v2_reset_context_observation=False,
                )
            },
            relocated,
        )
    except ValueError as error:
        assert "reward_v2_reset_context_observation" in str(error)
    else:
        raise AssertionError("partial-reset resume accepted a legacy model tree")


def test_reset_context_supports_a_matched_control_and_rejects_masking():
    common = {
        "name": "reset-context-config-smoke",
        "num_devices": 1,
        "num_envs_per_device": 1,
        "num_steps": 1,
        "num_minibatches": 1,
        "total_timesteps": 1,
        "eval_episodes": 1,
        "reward_stage": "reward_v2",
        "carry_work_observation": True,
        "reward_v2_reset_context_observation": True,
    }
    control = MixedAgentTrainConfig(**common)
    assert control.partial_reset_root is None
    assert control.reward_v2_reset_context_observation is True

    try:
        MixedAgentTrainConfig(**common, warm_start_from="legacy.pkl")
    except ValueError as error:
        assert "fresh training or native --resume_from" in str(error)
    else:
        raise AssertionError("reset-context control accepted a warm start")

    resumed_control = MixedAgentTrainConfig(
        **common,
        resume_from="native-reset-context.pkl",
    )
    assert resumed_control.resume_from == "native-reset-context.pkl"

    try:
        MixedAgentTrainConfig(**common, action_logit_masking=True)
    except ValueError as error:
        assert "reset-context treatment is unmasked" in str(error)
    else:
        raise AssertionError("reset-context treatment accepted action masking")

    try:
        MixedAgentTrainConfig(
            **{
                **common,
                "reward_v2_reset_context_observation": False,
                "partial_reset_root": "/tmp/partial-reset-bank",
                "partial_reset_bank_sha256": "a" * 64,
                "accepted_bank": object(),
                "pooled_sampler": {
                    "enabled": True,
                    "rule": "continuous_banded_v3",
                },
            }
        )
    except ValueError as error:
        assert "--reward-v2-reset-context-observation" in str(error)
    else:
        raise AssertionError("partial reset bank accepted a missing context feature")

    partial = {
        **common,
        "partial_reset_root": "/tmp/partial-reset-bank",
        "partial_reset_bank_sha256": "a" * 64,
        "accepted_bank": object(),
        "pooled_sampler": {
            "enabled": True,
            "rule": "continuous_banded_v3",
        },
    }
    treatment = MixedAgentTrainConfig(**partial)
    assert treatment.partial_reset_root == "/tmp/partial-reset-bank"
    assert treatment.reward_v2_reset_context_observation is True
    try:
        MixedAgentTrainConfig(**partial, stall_age_observation=True)
    except ValueError as error:
        assert "partial-reset causal arm excludes" in str(error)
    else:
        raise AssertionError("partial-reset causal arm accepted stall age")
