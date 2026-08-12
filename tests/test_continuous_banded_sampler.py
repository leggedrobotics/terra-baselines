from copy import deepcopy

import numpy as np
import pytest

from utils.pooled_sampler import (
    MAX_CONDITION_MASS,
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


def test_v3_initial_distribution_pools_the_frontier_without_family_halves():
    v3_masses = _masses(_v8_shaped_sampler("continuous_banded_v3"))
    v2_masses = _masses(_v8_shaped_sampler("continuous_banded_v2"))
    foundation = [name for name in v3_masses if name.startswith("foundation")]
    trench = [name for name in v3_masses if name.startswith("trench")]

    # 10% uniform over all 47 conditions plus 90% over the pooled frontier
    # weighted 2**(2 - depth): 2*4 + 13*2 + 32*1 = 66.
    assert v3_masses[foundation[0]] == pytest.approx(0.10 / 47 + 0.90 * 4 / 66)
    assert v3_masses[foundation[1]] == pytest.approx(0.10 / 47 + 0.90 * 2 / 66)
    assert v3_masses[foundation[-1]] == pytest.approx(0.10 / 47 + 0.90 * 1 / 66)
    assert v3_masses[trench[1]] == pytest.approx(v3_masses[foundation[1]])
    assert sum(v3_masses.values()) == pytest.approx(1.0)
    assert all(value > 0.0 for value in v3_masses.values())
    assert max(v3_masses.values()) < MAX_CONDITION_MASS

    # v2 pins each family at exactly half the population; v3 has no family
    # boundary, so the split follows the pooled depth weights instead.
    assert sum(v2_masses[name] for name in foundation) == pytest.approx(0.5)
    assert sum(v3_masses[name] for name in foundation) == pytest.approx(
        25 * 0.10 / 47 + 0.90 * (4 + 6 * 2 + 18) / 66
    )
    assert sum(v3_masses[name] for name in trench) == pytest.approx(
        22 * 0.10 / 47 + 0.90 * (4 + 7 * 2 + 14) / 66
    )


def test_v3_pooled_frontier_ends_the_measured_last_cell_monopoly():
    v2 = _v8_shaped_sampler("continuous_banded_v2")
    v3 = _v8_shaped_sampler("continuous_banded_v3")
    foundation = [name for name in v3.names if name.startswith("foundation")]
    trench = [name for name in v3.names if name.startswith("trench")]
    # The measured reward_v2_scratch state: trench is one condition away from
    # mastered while 23 foundations (5 depth-1, 18 depth-2) remain unmastered.
    straggler = trench[-1]
    mastered = (set(trench) - {straggler}) | {foundation[0], foundation[1]}
    unmastered = [name for name in v3.names if name not in mastered]
    assert len(unmastered) == 24

    v2_masses = _masses(v2, mastered)
    # The measured monopoly: one condition took 45.2% of all sampling while
    # each unmastered depth-1 foundation took ~3.4%.
    assert v2_masses[straggler] == pytest.approx(0.5 * (0.10 / 22 + 0.90))
    assert v2_masses[straggler] == pytest.approx(0.452, abs=5e-4)
    assert v2_masses[foundation[2]] == pytest.approx(0.5 * (0.10 / 25 + 0.90 * 2 / 28))
    assert v2_masses[foundation[2]] == pytest.approx(0.034, abs=5e-4)

    v3_masses = _masses(v3, mastered)
    # Pooled frontier weights: 5*2 (depth 1) + 18 + 1 (depth 2) = 29.
    assert v3_masses[straggler] == pytest.approx(0.10 / 47 + 0.90 * 1 / 29)
    assert v3_masses[foundation[2]] == pytest.approx(0.10 / 47 + 0.90 * 2 / 29)
    mean_unmastered = sum(v3_masses[name] for name in unmastered) / len(unmastered)
    assert max(v3_masses.values()) <= 2.0 * mean_unmastered
    assert max(v3_masses.values()) <= MAX_CONDITION_MASS + 1e-12
    # Target ESS recovers instead of collapsing onto the one pinned cell.
    assert effective_sample_size(v2.probabilities) < 5.0
    assert effective_sample_size(v3.probabilities) > 20.0


def test_v3_gives_a_fully_mastered_family_only_its_uniform_floor():
    v2 = _v8_shaped_sampler("continuous_banded_v2")
    v3 = _v8_shaped_sampler("continuous_banded_v3")
    trench = [name for name in v3.names if name.startswith("trench")]
    foundation = [name for name in v3.names if name.startswith("foundation")]

    v2_masses = _masses(v2, trench)
    v3_masses = _masses(v3, trench)
    # v2 keeps half the population inside a family with nothing left to learn.
    assert sum(v2_masses[name] for name in trench) == pytest.approx(0.5)
    # v3 leaves it the uniform floor and pools the rest onto the real frontier.
    assert sum(v3_masses[name] for name in trench) == pytest.approx(0.10 * 22 / 47)
    for name in trench:
        assert v3_masses[name] == pytest.approx(0.10 / 47)
    # Foundation frontier weights: 4 + 6*2 + 18 = 34.
    assert v3_masses[foundation[0]] == pytest.approx(0.10 / 47 + 0.90 * 4 / 34)
    assert v3_masses[foundation[-1]] == pytest.approx(0.10 / 47 + 0.90 * 1 / 34)
    assert sum(v3_masses.values()) == pytest.approx(1.0)


def test_v3_caps_every_condition_and_spills_the_rest_into_replay():
    v3 = _v8_shaped_sampler("continuous_banded_v3")
    foundation = [name for name in v3.names if name.startswith("foundation")]
    for remaining in (5, 2):
        # The end-game: only unlearnable walls are left unmastered.
        wall = foundation[-remaining:]
        mastered = [name for name in v3.names if name not in set(wall)]
        masses = _masses(v3, mastered)
        # Uncapped, those cells would absorb nearly the whole frontier.
        assert 0.10 / 47 + 0.90 / remaining > MAX_CONDITION_MASS
        for name in wall:
            assert masses[name] == pytest.approx(MAX_CONDITION_MASS)
        replay = [masses[name] for name in mastered]
        assert min(replay) == pytest.approx(max(replay))
        assert min(replay) == pytest.approx(
            (1.0 - remaining * MAX_CONDITION_MASS) / len(mastered)
        )
        assert sum(masses.values()) == pytest.approx(1.0)
        assert max(masses.values()) <= MAX_CONDITION_MASS + 1e-12


def test_v3_cap_water_fills_the_excess_onto_the_uncapped_frontier():
    v3 = _v8_shaped_sampler("continuous_banded_v3")
    foundation = [name for name in v3.names if name.startswith("foundation")]
    # One depth-0 cell (weight 4) plus ten depth-2 cells (weight 1): the
    # depth-0 share would be 0.10/47 + 0.90*4/14 = 25.9% without the cap.
    wall = [foundation[0]] + foundation[-10:]
    mastered = [name for name in v3.names if name not in set(wall)]
    masses = _masses(v3, mastered)

    assert masses[foundation[0]] == pytest.approx(MAX_CONDITION_MASS)
    free_share = 0.10 / 47 + (1.0 - MAX_CONDITION_MASS - 46 * 0.10 / 47) / 10
    for name in wall[1:]:
        assert masses[name] == pytest.approx(free_share)
    assert free_share < MAX_CONDITION_MASS
    for name in mastered:
        assert masses[name] == pytest.approx(0.10 / 47)
    assert sum(masses.values()) == pytest.approx(1.0)


def test_v3_all_mastered_is_uniform_over_every_condition():
    v3 = _v8_shaped_sampler("continuous_banded_v3")
    v2 = _v8_shaped_sampler("continuous_banded_v2")
    _masses(v3, v3.names)
    _masses(v2, v2.names)
    np.testing.assert_allclose(v3.probabilities, np.full(47, 1.0 / 47))
    # v2 still splits the two families 50/50 even with nothing left to learn.
    assert v2.probabilities.max() == pytest.approx(0.5 / 22)


def test_v3_graduation_demotion_and_windows_match_v2():
    v2 = _sampler(rule="continuous_banded_v2")
    v3 = _sampler(rule="continuous_banded_v3")
    for sampler in (v2, v3):
        sampler.start(0)
        _refresh(sampler, 150, {"f0": 1.0, "f2": 1.0, "t0": 0.0})
        _refresh(sampler, 300, {"f0": 0.0, "f2": 0.0, "t0": 1.0})
        _refresh(sampler, 450, {"f0": 0.0})

    v2_receipt = v2.receipt()
    v3_receipt = v3.receipt()
    mastered = dict(zip(NAMES, v3_receipt["mastery"]["mastered"]))
    assert mastered["f2"] is True  # depth 2 graduates before its depth-1 peers
    assert mastered["f0"] is False  # EMA 0.49 demotes below 0.65
    assert v3_receipt["mastery"]["mastered"] == v2_receipt["mastery"]["mastered"]
    assert v3_receipt["mastery"]["role"] == v2_receipt["mastery"]["role"]
    assert v3_receipt["competence"] == v2_receipt["competence"]
    assert v3_receipt["windows"] == v2_receipt["windows"]
    assert v3.refreshes == v2.refreshes == 3
    # Family stays in the receipt as metadata but no longer moves any mass.
    assert v3_receipt["mastery"]["family_active_depth"] == (
        v2_receipt["mastery"]["family_active_depth"]
    )
    assert not np.allclose(v3.probabilities, v2.probabilities)


def test_v3_restores_v2_and_v1_checkpoints_and_rejects_downgrades():
    source = _sampler(seed=11, rule="continuous_banded_v2")
    source.start(0)
    drawn = source.sample_levels((2, 64))
    counts = np.bincount(drawn.ravel(), minlength=len(NAMES))
    source.observe_reset_exposures(counts)
    source.observe_transition_exposures(counts * 16)
    _refresh(source, 150, {"f0": 1.0, "f1a": 1.0, "t0": 1.0})
    v2_state = source.state_dict()
    assert v2_state["settings"]["rule"] == "continuous_banded_v2"

    migrated = _sampler(seed=11, rule="continuous_banded_v3")
    migrated.restore_state_dict(deepcopy(v2_state))
    mastered = np.asarray(v2_state["mastery"]["mastered"], dtype=bool)
    np.testing.assert_array_equal(migrated._mastered, mastered)
    np.testing.assert_allclose(
        migrated.probabilities,
        migrated._continuous_distribution_v3(mastered),
        rtol=0.0,
        atol=1e-15,
    )

    # Only the rule and its probability vector change; every other field of
    # the checkpoint survives the migration untouched.
    v3_state = migrated.state_dict()
    assert v3_state["settings"]["rule"] == "continuous_banded_v3"
    for key in (
        "conditions",
        "labels",
        "competence",
        "current_window",
        "closed_window",
        "refresh",
        "numpy_rng",
        "mastery",
    ):
        assert v3_state[key] == v2_state[key]
    resumed = _sampler(seed=11, rule="continuous_banded_v3")
    resumed.restore_state_dict(deepcopy(v3_state))
    assert resumed.state_dict() == v3_state

    # v1 chains to v3 through the same one-way migration.
    v1_source = _sampler(seed=11)
    v1_source.start(0)
    _refresh(v1_source, 150, {"f0": 1.0, "t0": 1.0})
    v1_state = v1_source.state_dict()
    chained = _sampler(seed=11, rule="continuous_banded_v3")
    chained.restore_state_dict(deepcopy(v1_state))
    np.testing.assert_allclose(
        chained.probabilities,
        chained._continuous_distribution_v3(
            np.asarray(v1_state["mastery"]["mastered"], dtype=bool)
        ),
        rtol=0.0,
        atol=1e-15,
    )

    # Migration never runs backwards: both older rules reject a v3 checkpoint.
    for older in ("continuous_banded_v1", "continuous_banded_v2"):
        with pytest.raises(ValueError, match="settings changed"):
            _sampler(seed=11, rule=older).restore_state_dict(deepcopy(v3_state))


def test_v1_and_v2_distributions_are_unchanged_and_uncapped_beside_v3():
    mastered = {"f0", "f1a", "t0"}
    v1_masses = _masses(_sampler(), mastered)
    v2_masses = _masses(_sampler(rule="continuous_banded_v2"), mastered)
    v3_masses = _masses(_sampler(rule="continuous_banded_v3"), mastered)

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

    # The per-condition cap is v3-only: v1 and v2 both keep single conditions
    # far above it, and only v3 clamps them.
    assert v1_masses["f1b"] > MAX_CONDITION_MASS
    assert v2_masses["f1b"] > MAX_CONDITION_MASS
    assert max(v3_masses.values()) == pytest.approx(MAX_CONDITION_MASS)
    for family_masses in (v1_masses, v2_masses):
        assert sum(family_masses[name] for name in NAMES[:4]) == pytest.approx(0.5)
    # v3 pools instead: the five unmastered cells share the frontier and the
    # three mastered cells split whatever the cap returns.
    assert v3_masses["f0"] == pytest.approx((1.0 - 5 * MAX_CONDITION_MASS) / 3)
