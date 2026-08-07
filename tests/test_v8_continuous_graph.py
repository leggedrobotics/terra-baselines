import hashlib
import json
import sys
from copy import deepcopy

import pytest

from scripts.verify_continuous_sampler_checkpoint import (
    main as verify_checkpoint_main,
    verify_sampler_state,
)
from utils.accepted_bank import (
    AcceptedLevel,
    V8_CONTINUOUS_GRAPH_PATH,
    _v8_continuous_graph,
    _v8_stage_selection,
)
from utils.pooled_sampler import PooledConditionSampler, SamplerSettings
from utils import helpers


def _graph_fixture():
    graph = json.loads(V8_CONTINUOUS_GRAPH_PATH.read_text())
    levels = []
    ids_by_depth = {depth: [] for depth in range(3)}
    for depth in range(3):
        for family, condition_ids in graph["depths"][str(depth)].items():
            for condition_id in condition_ids:
                levels.append(
                    AcceptedLevel(
                        condition_id,
                        family,
                        ("Anchor", "Nearby core", "One-axis")[depth],
                        condition_id,
                        96,
                    )
                )
                ids_by_depth[depth].append(condition_id)
    return graph, levels, ids_by_depth


def test_frozen_graph_covers_all_47_conditions_and_binds_its_sha():
    graph, levels, ids_by_depth = _graph_fixture()
    assert graph["siblings_ordered"] is False
    assert [len(ids_by_depth[depth]) for depth in range(3)] == [2, 13, 32]

    depths, digest = _v8_continuous_graph(
        tuple(levels),
        tuple(ids_by_depth[0]),
        tuple(ids_by_depth[1]),
        tuple(ids_by_depth[2]),
    )
    assert depths == tuple(depth for depth in range(3) for _ in ids_by_depth[depth])
    assert digest == hashlib.sha256(V8_CONTINUOUS_GRAPH_PATH.read_bytes()).hexdigest()


def test_continuous_profile_selects_full_support_without_a_fixed_vector():
    _, levels, ids_by_depth = _graph_fixture()
    selected, fixed_probabilities = _v8_stage_selection(
        levels,
        "full",
        tuple(ids_by_depth[2]),
        tuple(ids_by_depth[0]),
        tuple(ids_by_depth[1]),
        {
            "v7_geometry_mass_within_family": {
                "foundation": {
                    "slab": 0.25,
                    "irregular": 0.15,
                    "courtyard": 0.15,
                    "bearing_walls": 0.20,
                    "pads": 0.15,
                    "courtyard_pads": 0.10,
                },
                "trench": {
                    "straight": 0.15,
                    "dogleg": 0.15,
                    "tee": 0.20,
                    "cross": 0.10,
                    "double_t": 0.20,
                    "network3": 0.15,
                    "disconnected_pair": 0.05,
                },
            }
        },
        "continuous_banded_v1",
    )
    assert len(selected) == 47
    assert fixed_probabilities == ()


def test_real_graph_initial_mass_is_gradual_and_family_balanced(tmp_path, monkeypatch):
    _, levels, ids_by_depth = _graph_fixture()
    depth_by_id = {
        condition_id: depth
        for depth, condition_ids in ids_by_depth.items()
        for condition_id in condition_ids
    }
    sampler = PooledConditionSampler(
        [level.condition_id for level in levels],
        SamplerSettings(
            rule="continuous_banded_v1",
            update_interval=150,
            mastery_threshold=0.80,
            min_episodes=32,
            competence_ema=0.30,
        ),
        labels={
            level.condition_id: {
                "family": level.family,
                "curriculum_depth": depth_by_id[level.condition_id],
            }
            for level in levels
        },
    )
    for family in ("foundation", "trench"):
        assert sum(
            probability
            for probability, level in zip(sampler.probabilities, levels)
            if level.family == family
        ) == pytest.approx(0.5)
    depth_mass = {
        depth: sum(
            probability
            for probability, level in zip(sampler.probabilities, levels)
            if depth_by_id[level.condition_id] == depth
        )
        for depth in range(3)
    }
    assert depth_mass == pytest.approx(
        {0: 0.7542727273, 1: 0.1779090909, 2: 0.0678181818}
    )
    assert min(sampler.probabilities) > 0.0

    state = sampler.state_dict()
    receipt = verify_sampler_state(state)
    assert receipt["passed"] is True
    assert receipt["family_counts"] == {"foundation": 25, "trench": 22}
    assert receipt["depth_counts"] == {"0": 2, "1": 13, "2": 32}

    invalid = deepcopy(state)
    invalid["probabilities"][0] = 0.0
    with pytest.raises(ValueError):
        verify_sampler_state(invalid)

    periodic = tmp_path / "periodic.pkl"
    final = tmp_path / "final.pkl"
    output = tmp_path / "receipt.json"
    checkpoint = {"next_update": 1, "pooled_sampler_state": state}
    helpers.save_pkl_object(checkpoint, str(periodic))
    helpers.save_pkl_object(checkpoint, str(final))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_continuous_sampler_checkpoint.py",
            str(periodic),
            str(final),
            "--output",
            str(output),
        ],
    )
    verify_checkpoint_main()
    saved = json.loads(output.read_text())
    assert saved["passed"] is True
    assert len(saved["checkpoints"]) == 2
