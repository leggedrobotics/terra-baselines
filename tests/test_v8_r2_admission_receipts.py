from __future__ import annotations

import json

import numpy as np

from scripts.analysis.build_v8_r2_admission_receipts import (
    build_dominance,
    canonical_distance_tiles,
)


def test_canonical_distance_is_8_connected_and_permits_diagonal_corners():
    target = np.zeros((3, 3), dtype=np.int8)
    target[0, 0] = 1
    occupancy = np.zeros((3, 3), dtype=bool)
    occupancy[0, 1] = True
    occupancy[1, 0] = True

    distance = canonical_distance_tiles(target, occupancy)

    np.testing.assert_allclose(distance[1, 1], np.sqrt(2.0), atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(distance[occupancy], np.zeros(2))


def test_dominance_receipt_matches_frozen_proposal(tmp_path):
    receipt = build_dominance(
        tmp_path,
        distance_bound=2.5,
        gamma=0.9984,
        success_bonus=6.0,
        failure_penalty=1.0,
        alpha=1.0,
        beta=1.5,
        step_cost_total=1.0,
        horizon=450,
    )

    assert receipt["status"] == "passed"
    enumeration = receipt["enumeration"]
    np.testing.assert_allclose(
        enumeration["minimum_success_return"], 0.7710142504969686
    )
    np.testing.assert_allclose(
        enumeration["maximum_failure_return"], -0.8154758851969248
    )
    np.testing.assert_allclose(
        enumeration["minimum_success_bonus_strict_threshold"], 2.744
    )
    assert enumeration["minimum_success_step"] == 450
    assert json.loads((tmp_path / "dominance_receipt.json").read_text()) == receipt


def test_dwell_grid_exposes_implicit_time_pressure(tmp_path):
    receipt = build_dominance(
        tmp_path,
        distance_bound=2.5,
        gamma=0.9984,
        success_bonus=6.0,
        failure_penalty=1.0,
        alpha=1.0,
        beta=1.5,
        step_cost_total=1.0,
        horizon=450,
    )
    rows = {(row["Q"], row["P"]): row for row in receipt["dwell_grid"]}
    np.testing.assert_allclose(rows[(1.0, 2.5)]["implicit_dwell_cost_per_step"], 0.0136)
    np.testing.assert_allclose(
        rows[(0.0, 0.0)]["explicit_step_cost_per_step"], 1.0 / 450.0
    )
