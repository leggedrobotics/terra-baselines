from copy import deepcopy

import numpy as np
import pytest

from utils.pooled_sampler import (
    PooledConditionSampler,
    SamplerSettings,
    effective_sample_size,
)

NAMES = ("f0", "f1a", "f1b", "f2", "t0", "t1a", "t1b", "t2")
FAMILIES = ("foundation",) * 4 + ("trench",) * 4
DEPTHS = (0, 1, 1, 2, 0, 1, 1, 2)
# The live V8 graph shape: 25 foundation + 22 trench, depths {0: 2, 1: 13, 2: 32}.
V8_FOUNDATION_DEPTHS = (0,) + (1,) * 6 + (2,) * 18
V8_TRENCH_DEPTHS = (0,) + (1,) * 7 + (2,) * 14


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


def _v8_shaped_sampler(rule, seed=3):
    """One sampler with the live 47-condition V8 family/depth graph."""
    names = []
    labels = {}
    for family, depths in (
        ("foundation", V8_FOUNDATION_DEPTHS),
        ("trench", V8_TRENCH_DEPTHS),
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


def _masses(sampler, mastered_names=()):
    """Force one mastery state and return the rule's condition masses."""
    sampler._mastered[:] = np.asarray(
        [name in set(mastered_names) for name in sampler.names], dtype=bool
    )
    sampler._probabilities = sampler._distribution_for_rule(sampler.settings.rule)
    return dict(zip(sampler.names, sampler.probabilities))


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


def test_v3_is_exactly_v2_while_the_cap_is_inactive():
    v2 = _v8_shaped_sampler("continuous_banded_v2")
    v3 = _v8_shaped_sampler("continuous_banded_v3")
    foundation = [name for name in v3.names if name.startswith("foundation")]
    trench = [name for name in v3.names if name.startswith("trench")]

    for mastered in ((), (foundation[0], trench[0]), foundation[:8]):
        expected = _masses(v2, mastered)
        observed = _masses(v3, mastered)
        assert max(expected.values()) <= v3.settings.max_mass
        np.testing.assert_array_equal(v3.probabilities, v2.probabilities)
        assert observed == expected


def test_v3_caps_the_measured_u13_5_monopoly_state():
    v2 = _v8_shaped_sampler("continuous_banded_v2")
    v3 = _v8_shaped_sampler("continuous_banded_v3")
    foundation = [name for name in v3.names if name.startswith("foundation")]
    trench = [name for name in v3.names if name.startswith("trench")]
    # The u13.5k reward_v2_scratch state: every trench mastered except
    # trn-net4-side1-road, with 23 foundations (5 depth-1, 18 depth-2) open.
    straggler = trench[-1]
    mastered = (set(trench) - {straggler}) | {foundation[0], foundation[1]}

    v2_masses = _masses(v2, mastered)
    assert v2_masses[straggler] == pytest.approx(0.452, abs=5e-4)
    assert v2_masses[foundation[2]] == pytest.approx(0.0341, abs=5e-5)
    assert v2_masses[foundation[-1]] == pytest.approx(0.0181, abs=5e-5)
    assert effective_sample_size(v2.probabilities) == pytest.approx(4.62, abs=5e-3)

    v3_masses = _masses(v3, mastered)
    assert v3_masses[straggler] == pytest.approx(v3.settings.max_mass)
    assert max(v3_masses.values()) == pytest.approx(0.150, abs=5e-4)
    assert effective_sample_size(v3.probabilities) == pytest.approx(19.62, abs=5e-3)
    # The excess crosses the family boundary instead of staying in trench.
    assert sum(v3_masses[name] for name in foundation) == pytest.approx(0.776, abs=5e-4)
    assert sum(v3_masses[name] for name in trench) == pytest.approx(0.224, abs=5e-4)
    assert sum(v3_masses.values()) == pytest.approx(1.0)
    assert min(v3_masses.values()) > 0.0
    # Every condition keeps its v2 ordering; only the runaway cell is clipped.
    ranked = sorted(v3.names, key=lambda name: v2_masses[name])
    assert [v3_masses[name] for name in ranked] == sorted(
        v3_masses[name] for name in ranked
    )


def test_v3_migrates_only_at_an_empty_window_boundary():
    source = _sampler(seed=11, rule="continuous_banded_v2")
    source.start(0)
    drawn = source.sample_levels((2, 64))
    counts = np.bincount(drawn.ravel(), minlength=len(NAMES))
    source.observe_reset_exposures(counts)
    source.observe_transition_exposures(counts * 16)

    # Mid-window: exposure taken under v2 must not be counted under v3.
    _observe(source, {"f0": 1.0})
    with pytest.raises(ValueError, match="empty current window"):
        _sampler(seed=11, rule="continuous_banded_v3").restore_state_dict(
            deepcopy(source.state_dict())
        )

    # The refresh boundary closes that window and opens an empty one.
    source.refresh(150)
    v2_state = source.state_dict()
    migrated = _sampler(seed=11, rule="continuous_banded_v3")
    migrated.restore_state_dict(deepcopy(v2_state))
    mastered = np.asarray(v2_state["mastery"]["mastered"], dtype=bool)
    np.testing.assert_allclose(
        migrated.probabilities,
        migrated._continuous_distribution_v3(mastered),
        rtol=0.0,
        atol=1e-15,
    )
    v3_state = migrated.state_dict()
    assert v3_state["settings"]["rule"] == "continuous_banded_v3"
    for key in ("conditions", "labels", "competence", "closed_window", "refresh"):
        assert v3_state[key] == v2_state[key]

    # A native v3 checkpoint resumes unchanged, mid-window included.
    resumed = _sampler(seed=11, rule="continuous_banded_v3")
    resumed.restore_state_dict(deepcopy(v3_state))
    assert resumed.state_dict() == v3_state
    _observe(resumed, {"f0": 1.0})
    live = resumed.state_dict()
    again = _sampler(seed=11, rule="continuous_banded_v3")
    again.restore_state_dict(deepcopy(live))
    assert again.state_dict() == live

    # Migration never runs backwards.
    for older in ("continuous_banded_v1", "continuous_banded_v2"):
        with pytest.raises(ValueError, match="settings changed"):
            _sampler(seed=11, rule=older).restore_state_dict(deepcopy(v3_state))


def test_v1_and_v2_distributions_are_unchanged_beside_v3():
    mastered = {"f0", "f1a", "t0"}
    v1_masses = _masses(_sampler(), mastered)
    v2_masses = _masses(_sampler(rule="continuous_banded_v2"), mastered)

    # v1: both families sit at active depth 1 with a depth-2 preview, and the
    # active band still includes the already-mastered sibling f1a.
    assert v1_masses["f1a"] == pytest.approx(0.5 * (0.10 / 4 + 0.75 / 2))
    assert v1_masses["f1b"] == pytest.approx(0.5 * (0.10 / 4 + 0.75 / 2))
    assert v1_masses["f2"] == pytest.approx(0.5 * (0.10 / 4 + 0.15))
    assert v1_masses["f0"] == pytest.approx(0.5 * 0.10 / 4)
    assert v1_masses["t1a"] == pytest.approx(0.5 * (0.10 / 4 + 0.75 / 2))
    assert v1_masses["t2"] == pytest.approx(0.5 * (0.10 / 4 + 0.15))

    # v2: per-family pooled frontier weighted 2**(2 - depth), floor included.
    assert v2_masses["f1b"] == pytest.approx(0.5 * (0.10 / 4 + 0.90 * 2 / 3))
    assert v2_masses["f2"] == pytest.approx(0.5 * (0.10 / 4 + 0.90 * 1 / 3))
    assert v2_masses["t1a"] == pytest.approx(0.5 * (0.10 / 4 + 0.90 * 2 / 5))
    assert v2_masses["t2"] == pytest.approx(0.5 * (0.10 / 4 + 0.90 * 1 / 5))
    assert v2_masses["f1a"] == pytest.approx(0.5 * 0.10 / 4)
    for family_masses in (v1_masses, v2_masses):
        assert sum(family_masses[name] for name in NAMES[:4]) == pytest.approx(0.5)

    # The cap is v3-only: both older rules keep single conditions above it.
    cap = SamplerSettings().max_mass
    assert v1_masses["f1b"] > cap
    assert v2_masses["f1b"] > cap
    assert max(_masses(_sampler(rule="continuous_banded_v3"), mastered).values()) == (
        pytest.approx(cap)
    )
