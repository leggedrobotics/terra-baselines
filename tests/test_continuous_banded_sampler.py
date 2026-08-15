from copy import deepcopy

import numpy as np
import pytest

from utils.pooled_sampler import (
    PooledConditionSampler,
    SamplerSettings,
    effective_sample_size,
)

# The live V8 graph: 25 foundation + 22 trench conditions.
FOUNDATION_DEPTHS = (0,) + (1,) * 6 + (2,) * 18
TRENCH_DEPTHS = (0,) + (1,) * 7 + (2,) * 14


def _sampler(rule="continuous_banded_v3", seed=3):
    names = []
    labels = {}
    for family, depths in (
        ("foundation", FOUNDATION_DEPTHS),
        ("trench", TRENCH_DEPTHS),
    ):
        for position, depth in enumerate(depths):
            name = f"{family}-d{depth}-{position:02d}"
            names.append(name)
            labels[name] = {"family": family, "curriculum_depth": depth}
    return PooledConditionSampler(
        names,
        SamplerSettings(
            rule=rule,
            update_interval=150,
            mastery_threshold=0.80,
            min_episodes=32,
            competence_ema=0.30,
            seed=seed,
        ),
        maps_per_condition=[96] * len(names),
        labels=labels,
    )


def _set_mastery(sampler, mastered_names):
    mastered = set(mastered_names)
    sampler._mastered[:] = np.asarray(
        [name in mastered for name in sampler.names], dtype=bool
    )
    sampler._probabilities = sampler._distribution_for_rule(sampler.settings.rule)
    return dict(zip(sampler.names, sampler.probabilities))


def test_v3_reallocates_the_measured_u14_mastered_family_state():
    sampler = _sampler()
    foundation = [name for name in sampler.names if name.startswith("foundation")]
    trench = [name for name in sampler.names if name.startswith("trench")]

    # Exact u14 shape: 7/25 foundations and all 22 trenches mastered, leaving
    # five depth-1 and thirteen depth-2 foundations open.
    mastered = {foundation[0], foundation[1], *foundation[7:12], *trench}
    masses = _set_mastery(sampler, mastered)
    open_depth1 = foundation[2:7]
    open_depth2 = foundation[12:]

    assert len(open_depth1) == 5 and len(open_depth2) == 13
    assert all(masses[name] == pytest.approx(0.80 * 2 / 23) for name in open_depth1)
    assert all(masses[name] == pytest.approx(0.80 / 23) for name in open_depth2)
    assert all(masses[name] == pytest.approx(0.20 / 29) for name in mastered)
    assert sum(masses[name] for name in open_depth1 + open_depth2) == pytest.approx(
        0.80
    )
    assert sum(masses[name] for name in foundation if name in mastered) == (
        pytest.approx(0.20 * 7 / 29)
    )
    assert sum(masses[name] for name in trench) == pytest.approx(0.20 * 22 / 29)
    assert max(masses.values()) == pytest.approx(0.80 * 2 / 23)
    assert effective_sample_size(sampler.probabilities) == pytest.approx(
        24.21, abs=5e-3
    )


def test_v3_has_full_support_applies_the_cap_and_falls_back_when_complete():
    sampler = _sampler()
    initial = _set_mastery(sampler, ())
    assert min(initial.values()) > 0.0
    assert max(initial.values()) < sampler.settings.max_mass
    assert sum(initial.values()) == pytest.approx(1.0)

    one_open = _set_mastery(sampler, sampler.names[1:])
    assert one_open[sampler.names[0]] == pytest.approx(sampler.settings.max_mass)
    assert min(one_open.values()) > 0.0
    assert sum(one_open.values()) == pytest.approx(1.0)

    completed = _set_mastery(sampler, sampler.names)
    assert all(value == pytest.approx(1 / 47) for value in completed.values())


def test_v3_refresh_and_native_resume_preserve_state_and_rng():
    sampler = _sampler(seed=11)
    sampler.start(0)
    drawn = sampler.sample_levels((2, 128))
    counts = np.bincount(drawn.ravel(), minlength=len(sampler.names))
    sampler.observe_reset_exposures(counts)
    sampler.observe_transition_exposures(counts * 32)
    depth2 = next(name for name in sampler.names if "-d2-" in name)
    sampler.observe_episode_payload(
        {
            "schema": "terra_training_episode_aggregate_v2",
            "groups": [
                {
                    "primary_cell": depth2,
                    "episode_count": 32,
                    "task_done_count": 32,
                }
            ],
        }
    )
    sampler.refresh(150)

    receipt = sampler.receipt()
    index = sampler.names.index(depth2)
    assert receipt["mastery"]["mastered"][index] is True
    assert receipt["mastery"]["role"][index] == "replay"
    # With only one mastered condition, its nominal 20% replay share is capped.
    assert receipt["intended_mass"][index] == pytest.approx(sampler.settings.max_mass)

    state = sampler.state_dict()
    restored = _sampler(seed=11)
    restored.restore_state_dict(deepcopy(state))
    assert restored.state_dict() == state
    np.testing.assert_array_equal(
        restored.sample_levels((3, 41)), sampler.sample_levels((3, 41))
    )
    assert restored.state_dict() == sampler.state_dict()


def test_v2_source_migration_clears_only_the_partial_window():
    with pytest.raises(ValueError, match="sampler rule"):
        SamplerSettings(rule="continuous_banded_v2")

    source = _sampler(seed=11)
    source._probabilities = source._continuous_distribution_v2()
    boundary_state = source.state_dict()
    boundary_state["settings"]["rule"] = "continuous_banded_v2"
    with pytest.raises(ValueError, match="explicit offline"):
        _sampler(seed=11).restore_state_dict(deepcopy(boundary_state))

    source.start(13_950)
    drawn = source.sample_levels((2, 64))
    counts = np.bincount(drawn.ravel(), minlength=len(source.names))
    source.observe_reset_exposures(counts)
    source.observe_transition_exposures(counts * 16)
    source.observe_episode_payload(
        {"schema": "terra_training_episode_aggregate_v2", "groups": []}
    )
    source_state = source.state_dict()
    source_state["settings"]["rule"] = "continuous_banded_v2"

    migrated = _sampler(seed=11)
    with pytest.raises(ValueError, match="explicit offline"):
        migrated.restore_state_dict(deepcopy(source_state))
    migrated.restore_state_dict(deepcopy(source_state), clear_window_on_migration=True)
    state = migrated.state_dict()

    assert state["settings"]["rule"] == "continuous_banded_v3"
    assert state["current_window"]["updates"] == 0
    assert sum(state["current_window"]["sampled_assignment_count"]) == 0
    for key in (
        "conditions",
        "maps_per_condition",
        "labels",
        "competence",
        "closed_window",
        "refresh",
        "numpy_rng",
        "mastery",
    ):
        assert state[key] == source_state[key]
    mastered = np.asarray(state["mastery"]["mastered"], dtype=bool)
    probabilities = np.asarray(state["probabilities"])
    assert probabilities[~mastered].sum() == pytest.approx(1.0)
