import json

import numpy as np
import pytest

from utils.pooled_sampler import (
    PooledConditionSampler,
    SamplerSettings,
    effective_sample_size,
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
    assert sampler.telemetry()["sampler/measured_conditions"] == 0.0


def test_sampling_and_telemetry_report_condition_and_branch_mass():
    sampler = _sampler(rule="uniform")
    drawn = sampler.sample_levels((4, 4096))
    counts = np.bincount(drawn.ravel(), minlength=8) / drawn.size
    np.testing.assert_allclose(counts, sampler.probabilities, atol=0.01)

    sampler.start(0)
    _observe(sampler, [0.2] * 8)
    sampler.refresh(1)
    metrics = sampler.telemetry()
    assert metrics["sampler/intended_ess"] == pytest.approx(
        effective_sample_size(sampler.probabilities)
    )
    assert metrics["sampler_family_q/foundation"] == pytest.approx(0.5)
    assert metrics["sampler_depth_q/Anchor"] == pytest.approx(0.5)
    assert metrics["sampler/window_episodes"] == 80
    assert json.loads(json.dumps(sampler.receipt()))["schema"] == (
        "terra_pooled_condition_sampler_v1"
    )


def test_duplicate_conditions_and_infeasible_cap_fail_loudly():
    with pytest.raises(ValueError, match="one level per condition"):
        PooledConditionSampler(["same", "same"], SamplerSettings())
    with pytest.raises(ValueError, match="infeasible"):
        PooledConditionSampler(
            ["a", "b"],
            SamplerSettings(rule="adaptive", max_mass=0.15),
        )
