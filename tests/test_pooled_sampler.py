import json
from copy import deepcopy

import numpy as np
import pytest

from utils.pooled_sampler import (
    PooledConditionSampler,
    SamplerSettings,
)


NAMES = [f"condition-{index}" for index in range(8)]


def _sampler(rule="adaptive", **overrides):
    settings = SamplerSettings(
        rule=rule,
        min_episodes=1,
        seed=7,
        **overrides,
    )
    labels = {
        name: {
            "family": "foundation" if index < 4 else "trench",
            "branch_depth": "Anchor" if index % 2 == 0 else "One-axis",
        }
        for index, name in enumerate(NAMES)
    }
    return PooledConditionSampler(
        NAMES,
        settings,
        maps_per_condition=[64] * len(NAMES),
        labels=labels,
    )


def _observe(sampler, completions, episodes=10):
    sampler.observe_episode_payload(
        {
            "groups": [
                {
                    "primary_cell": name,
                    "episode_count": episodes,
                    "combined_completion_sum": episodes * completion,
                }
                for name, completion in zip(NAMES, completions)
            ]
        }
    )


def test_uniform_control_never_refreshes_or_moves():
    sampler = _sampler(rule="uniform")
    sampler.start(0)
    _observe(sampler, [0.0] * 8)
    assert sampler.due(10_000)
    sampler.refresh(10_000)
    np.testing.assert_allclose(sampler.probabilities, np.full(8, 1 / 8))


def test_adaptive_sampler_focuses_on_the_solvable_frontier():
    sampler = _sampler(uniform_floor=0.2, mastery_threshold=0.75)
    sampler.start(0)
    _observe(sampler, [0.95, 0.60, 0.45, 0.35, 0.25, 0.15, 0.05, 0.0])
    sampler.refresh(1)
    probabilities = sampler.probabilities
    assert probabilities[1] > probabilities[7]  # frontier > stuck-at-zero
    assert probabilities[1] > probabilities[0]  # frontier > mastered
    assert probabilities.min() >= 0.2 / 8 - 1e-12
    assert probabilities.max() <= sampler.settings.max_mass + 1e-12
    assert probabilities.sum() == pytest.approx(1.0)


def test_mastered_condition_reopens_when_its_ema_falls():
    sampler = _sampler(competence_ema=1.0)
    sampler.start(0)
    _observe(sampler, [0.9] * 8)
    sampler.refresh(1)
    np.testing.assert_allclose(sampler.probabilities, np.full(8, 1 / 8))

    _observe(sampler, [0.5] + [0.9] * 7)
    sampler.refresh(2)
    assert sampler.probabilities[0] > sampler.probabilities[1]


def test_minimum_episode_count_blocks_thin_feedback():
    settings = SamplerSettings(
        rule="adaptive",
        min_episodes=20,
        max_mass=0.2,
    )
    sampler = PooledConditionSampler(NAMES, settings)
    sampler.start(0)
    _observe(sampler, [1.0] * 8, episodes=3)
    sampler.refresh(1)
    assert sampler.receipt()["competence"] == [None] * 8


def test_sampling_and_receipt_preserve_condition_exposure():
    sampler = _sampler(rule="uniform")
    drawn = sampler.sample_levels((4, 4096))
    counts = np.bincount(drawn.ravel(), minlength=8) / drawn.size
    np.testing.assert_allclose(counts, sampler.probabilities, atol=0.01)

    sampler.start(0)
    sampler.observe_reset_exposures(
        np.bincount(drawn.ravel(), minlength=len(NAMES))
    )
    _observe(sampler, [0.2] * 8)
    sampler.refresh(1)
    assert sampler.refreshes == 1
    receipt = json.loads(json.dumps(sampler.receipt()))
    assert receipt["schema"] == "terra_pooled_condition_sampler_v2"
    assert receipt["windows"]["current"]["completed_episode_mass"] == [
        None
    ] * 8
    assert receipt["windows"]["closed"]["completed_episode_count"] == [
        10
    ] * 8
    assert sum(receipt["windows"]["closed"]["sampled_assignment_count"]) == (
        drawn.size
    )


def test_assignments_resets_and_completed_episodes_are_not_conflated():
    sampler = _sampler(rule="uniform")
    drawn = sampler.sample_levels((2, 64))
    sampler.observe_reset_exposures(
        np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int64)
    )
    _observe(sampler, [0.1] * 8, episodes=2)
    receipt = sampler.receipt()["windows"]["current"]
    assert sum(receipt["sampled_assignment_count"]) == drawn.size
    assert sum(receipt["reset_exposure_count"]) == 36
    assert sum(receipt["completed_episode_count"]) == 16


def test_reset_exposure_shape_and_dtype_are_checked():
    sampler = _sampler(rule="uniform")
    with pytest.raises(ValueError, match="condition count"):
        sampler.observe_reset_exposures(np.zeros(7, dtype=np.int64))
    with pytest.raises(ValueError, match="nonnegative integers"):
        sampler.observe_reset_exposures(np.full(8, 0.5, dtype=np.float64))


def test_duplicate_conditions_and_infeasible_cap_fail_loudly():
    with pytest.raises(ValueError, match="one level per condition"):
        PooledConditionSampler(["same", "same"], SamplerSettings())
    with pytest.raises(ValueError, match="infeasible"):
        PooledConditionSampler(
            ["a", "b"],
            SamplerSettings(rule="adaptive", max_mass=0.15),
        )


def test_state_roundtrip_preserves_windows_probabilities_and_future_samples():
    sampler = _sampler(competence_ema=0.5)
    sampler.start(3)
    first = sampler.sample_levels((2, 64))
    sampler.observe_reset_exposures(
        np.bincount(first.ravel(), minlength=len(NAMES))
    )
    _observe(sampler, [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0])
    sampler.refresh(10)

    current = sampler.sample_levels((3, 17))
    sampler.observe_reset_exposures(
        np.bincount(current.ravel(), minlength=len(NAMES))
    )
    _observe(sampler, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.0, 0.0], episodes=3)

    state = sampler.state_dict()
    assert state["closed_window"]["completion_sum"][0] == pytest.approx(9.0)
    restored = _sampler(competence_ema=0.5)
    restored.restore_state_dict(deepcopy(state))
    assert restored.state_dict() == state

    for shape in ((1,), (4, 23), (2, 3, 5)):
        np.testing.assert_array_equal(
            restored.sample_levels(shape), sampler.sample_levels(shape)
        )
    assert restored.state_dict() == sampler.state_dict()

    _observe(sampler, [0.4] * len(NAMES), episodes=2)
    _observe(restored, [0.4] * len(NAMES), episodes=2)
    sampler.refresh(160)
    restored.refresh(160)
    assert restored.state_dict() == sampler.state_dict()


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    [
        ("schema", lambda state: state.update(schema="v2"), "unsupported"),
        (
            "conditions",
            lambda state: state["conditions"].reverse(),
            "conditions changed",
        ),
        (
            "settings",
            lambda state: state["settings"].update(temperature=0.5),
            "settings changed",
        ),
        (
            "maps",
            lambda state: state["maps_per_condition"].__setitem__(0, 63),
            "maps_per_condition changed",
        ),
        (
            "labels",
            lambda state: state["labels"][NAMES[0]].update(family="trench"),
            "labels changed",
        ),
    ],
)
def test_state_restore_rejects_contract_changes(field, mutate, message):
    del field
    state = _sampler().state_dict()
    mutate(state)
    with pytest.raises(ValueError, match=message):
        _sampler().restore_state_dict(state)


def test_state_restore_rejects_missing_fields_and_invalid_rng():
    state = _sampler().state_dict()
    del state["current_window"]
    with pytest.raises(ValueError, match="fields"):
        _sampler().restore_state_dict(state)

    state = _sampler().state_dict()
    state["numpy_rng"]["bit_generator"] = "MT19937"
    with pytest.raises(ValueError, match="RNG type"):
        _sampler().restore_state_dict(state)
