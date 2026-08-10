from copy import deepcopy

import numpy as np
import pytest

from utils.pooled_sampler import PooledConditionSampler, SamplerSettings

NAMES = ("f0", "f1a", "f1b", "f2", "t0", "t1a", "t1b", "t2")
FAMILIES = ("foundation",) * 4 + ("trench",) * 4
DEPTHS = (0, 1, 1, 2, 0, 1, 1, 2)


def _sampler(seed=7, rule="continuous_banded_v1"):
    labels = {
        name: {
            "family": family,
            "curriculum_depth": depth,
            "branch_depth": "test-only",
        }
        for name, family, depth in zip(NAMES, FAMILIES, DEPTHS)
    }
    return PooledConditionSampler(
        list(NAMES),
        SamplerSettings(
            rule=rule,
            update_interval=150,
            mastery_threshold=0.80,
            min_episodes=32,
            competence_ema=0.30,
            seed=seed,
        ),
        maps_per_condition=[96] * len(NAMES),
        labels=labels,
    )


def _observe(sampler, rates, episodes=32):
    sampler.observe_episode_payload(
        {
            "schema": "terra_training_episode_aggregate_v2",
            "groups": [
                {
                    "primary_cell": name,
                    "episode_count": episodes,
                    "task_done_count": round(rate * episodes),
                }
                for name, rate in rates.items()
            ],
        }
    )


def _refresh(sampler, update, rates):
    _observe(sampler, rates)
    sampler.refresh(update)


def test_initial_distribution_is_family_balanced_full_support_and_banded():
    sampler = _sampler()
    probabilities = dict(zip(NAMES, sampler.probabilities))

    assert all(value > 0.0 for value in probabilities.values())
    assert sum(probabilities[name] for name in NAMES[:4]) == pytest.approx(0.5)
    assert sum(probabilities[name] for name in NAMES[4:]) == pytest.approx(0.5)
    for offset in (0, 4):
        # Per family: 10% uniform floor over four conditions, 75% on depth 0,
        # and 15% uniformly over depth 1. Depth 2 retains only the floor.
        assert probabilities[NAMES[offset]] == pytest.approx(0.5 * (0.10 / 4 + 0.75))
        assert probabilities[NAMES[offset + 1]] == pytest.approx(
            0.5 * (0.10 / 4 + 0.15 / 2)
        )
        assert probabilities[NAMES[offset + 2]] == pytest.approx(
            0.5 * (0.10 / 4 + 0.15 / 2)
        )
        assert probabilities[NAMES[offset + 3]] == pytest.approx(0.5 * 0.10 / 4)


def test_families_progress_independently_and_preview_cannot_skip_depth():
    sampler = _sampler()
    sampler.start(0)
    _refresh(
        sampler,
        150,
        {"f0": 1.0, "t0": 0.0, "f1a": 1.0, "f1b": 1.0},
    )
    receipt = sampler.receipt()["mastery"]
    mastered = dict(zip(NAMES, receipt["mastered"]))
    assert mastered["f0"] is True
    assert mastered["f1a"] is False  # measured preview stays locked
    assert mastered["f1b"] is False
    assert receipt["family_active_depth"] == {"foundation": 1, "trench": 0}

    probabilities = dict(zip(NAMES, sampler.probabilities))
    assert probabilities["f1a"] == pytest.approx(probabilities["f1b"])
    assert probabilities["f1a"] > probabilities["f0"]
    assert probabilities["t0"] > probabilities["t1a"]

    _refresh(sampler, 300, {"f1a": 1.0, "f1b": 1.0, "t0": 0.0})
    receipt = sampler.receipt()["mastery"]
    assert receipt["family_active_depth"] == {"foundation": 2, "trench": 0}


def test_active_depth_includes_mastered_siblings_and_low_ema_demotes():
    sampler = _sampler()
    sampler.start(0)
    _refresh(sampler, 150, {"f0": 1.0, "t0": 0.0})
    _refresh(sampler, 300, {"f1a": 1.0, "f1b": 0.0, "f0": 0.0})

    receipt = sampler.receipt()["mastery"]
    mastered = dict(zip(NAMES, receipt["mastered"]))
    assert mastered["f1a"] is True
    assert mastered["f1b"] is False
    assert receipt["family_active_depth"]["foundation"] == 1
    probabilities = dict(zip(NAMES, sampler.probabilities))
    assert probabilities["f1a"] == pytest.approx(probabilities["f1b"])
    assert mastered["f0"] is True  # EMA is 0.70 after one low window

    _refresh(sampler, 450, {"f0": 0.0, "f1a": 1.0, "f1b": 0.0})
    receipt = sampler.receipt()["mastery"]
    assert dict(zip(NAMES, receipt["mastered"]))["f0"] is False
    assert receipt["family_active_depth"]["foundation"] == 0


def test_no_next_depth_uses_ten_ninety_and_all_mastered_is_uniform():
    sampler = _sampler()
    sampler._mastered[:] = np.asarray(
        [True, True, True, False, True, True, True, False], dtype=bool
    )
    sampler._probabilities = sampler._continuous_distribution()
    probabilities = dict(zip(NAMES, sampler.probabilities))
    for offset in (0, 4):
        assert probabilities[NAMES[offset + 3]] == pytest.approx(
            0.5 * (0.10 / 4 + 0.90)
        )
        assert probabilities[NAMES[offset]] == pytest.approx(0.5 * 0.10 / 4)

    sampler._mastered[:] = True
    sampler._probabilities = sampler._continuous_distribution()
    np.testing.assert_allclose(sampler.probabilities, np.full(len(NAMES), 1 / 8))


def test_state_resume_preserves_mastery_ema_rng_and_all_exposure_axes():
    sampler = _sampler(seed=11)
    sampler.start(0)
    drawn = sampler.sample_levels((2, 128))
    counts = np.bincount(drawn.ravel(), minlength=len(NAMES))
    sampler.observe_reset_exposures(counts)
    sampler.observe_transition_exposures(counts * 32)
    _observe(sampler, {"f0": 1.0, "t0": 0.5})
    sampler.refresh(150)
    state = sampler.state_dict()

    restored = _sampler(seed=11)
    restored.restore_state_dict(deepcopy(state))
    assert restored.state_dict() == state
    np.testing.assert_array_equal(
        restored.sample_levels((3, 41)), sampler.sample_levels((3, 41))
    )
    assert restored.state_dict() == sampler.state_dict()

    old_state = deepcopy(state)
    old_state["schema"] = "terra_pooled_condition_sampler_state_v2"
    with pytest.raises(ValueError, match="unsupported"):
        _sampler(seed=11).restore_state_dict(old_state)


def test_held_out_payload_is_rejected_and_fixed_boundaries_are_enforced():
    sampler = _sampler()
    sampler.start(0)
    with pytest.raises(ValueError, match="training episode aggregates"):
        sampler.observe_episode_payload(
            {"schema": "terra_fixed_bank_eval_v4", "groups": []}
        )
    _observe(sampler, {"f0": 1.0})
    with pytest.raises(ValueError, match="fixed update boundary"):
        sampler.refresh(149)
    with pytest.raises(ValueError, match="cannot skip"):
        sampler.refresh(300)


def test_v2_initial_distribution_is_depth_weighted_with_full_support():
    sampler = _sampler(rule="continuous_banded_v2")
    probabilities = dict(zip(NAMES, sampler.probabilities))

    assert all(value > 0.0 for value in probabilities.values())
    assert sum(probabilities[name] for name in NAMES[:4]) == pytest.approx(0.5)
    assert sum(probabilities[name] for name in NAMES[4:]) == pytest.approx(0.5)
    for offset in (0, 4):
        # Per family: 10% uniform floor over four conditions and 90% over the
        # unmastered frontier weighted 2**(2 - depth): 4 + 2 + 2 + 1 = 9.
        assert probabilities[NAMES[offset]] == pytest.approx(
            0.5 * (0.10 / 4 + 0.90 * 4 / 9)
        )
        assert probabilities[NAMES[offset + 1]] == pytest.approx(
            0.5 * (0.10 / 4 + 0.90 * 2 / 9)
        )
        assert probabilities[NAMES[offset + 3]] == pytest.approx(
            0.5 * (0.10 / 4 + 0.90 * 1 / 9)
        )


def test_v2_straggler_cannot_pin_family_and_any_depth_graduates():
    sampler = _sampler(rule="continuous_banded_v2")
    sampler.start(0)
    _refresh(sampler, 150, {"f0": 1.0, "f1a": 1.0, "t0": 0.0})

    receipt = sampler.receipt()["mastery"]
    mastered = dict(zip(NAMES, receipt["mastered"]))
    assert mastered["f0"] is True
    assert mastered["f1a"] is True  # v1 would lock this measured preview
    assert receipt["role"][NAMES.index("f1a")] == "replay"
    assert receipt["role"][NAMES.index("f2")] == "frontier"

    probabilities = dict(zip(NAMES, sampler.probabilities))
    # Foundation frontier is now f1b (weight 2) and f2 (weight 1).
    assert probabilities["f1b"] == pytest.approx(0.5 * (0.10 / 4 + 0.90 * 2 / 3))
    assert probabilities["f2"] == pytest.approx(0.5 * (0.10 / 4 + 0.90 * 1 / 3))

    # The depth-2 cell graduates although its depth-1 sibling is unmastered.
    _refresh(sampler, 300, {"f2": 1.0})
    mastered = dict(zip(NAMES, sampler.receipt()["mastery"]["mastered"]))
    assert mastered["f2"] is True
    assert mastered["f1b"] is False
    probabilities = dict(zip(NAMES, sampler.probabilities))
    assert probabilities["f1b"] == pytest.approx(0.5 * (0.10 / 4 + 0.90))


def test_v2_restores_v1_checkpoint_and_recomputes_probabilities():
    source = _sampler(seed=11)
    source.start(0)
    drawn = source.sample_levels((2, 64))
    counts = np.bincount(drawn.ravel(), minlength=len(NAMES))
    source.observe_reset_exposures(counts)
    source.observe_transition_exposures(counts * 16)
    _refresh(source, 150, {"f0": 1.0, "f1a": 1.0, "t0": 1.0})
    state = source.state_dict()
    assert state["settings"]["rule"] == "continuous_banded_v1"

    migrated = _sampler(seed=11, rule="continuous_banded_v2")
    migrated.restore_state_dict(deepcopy(state))
    mastered = np.asarray(state["mastery"]["mastered"], dtype=bool)
    np.testing.assert_array_equal(migrated._mastered, mastered)
    np.testing.assert_allclose(
        migrated.probabilities,
        migrated._continuous_distribution_v2(mastered),
        rtol=0.0,
        atol=1e-15,
    )

    # The migrated state round-trips as a native v2 checkpoint.
    v2_state = migrated.state_dict()
    assert v2_state["settings"]["rule"] == "continuous_banded_v2"
    resumed = _sampler(seed=11, rule="continuous_banded_v2")
    resumed.restore_state_dict(deepcopy(v2_state))
    assert resumed.state_dict() == v2_state

    # Migration is one-way: a v1 sampler must reject a v2 checkpoint.
    with pytest.raises(ValueError, match="settings changed"):
        _sampler(seed=11).restore_state_dict(deepcopy(v2_state))
