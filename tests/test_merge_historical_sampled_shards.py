from pathlib import Path

import pytest

from scripts.analysis.merge_historical_sampled_shards import (
    EXPECTED_LABELS,
    EXPECTED_SEEDS,
    merge_payloads,
)


def _record(label, seed):
    return {
        "checkpoint_label": label,
        "seed": seed,
        "dataset": "development/M0",
        "mode": "sampled",
        "horizon": 450,
        "per_map": [
            {
                "target_mutation": False,
                "obstacle_mutation": False,
                "nonfinite_state": False,
                "terminal_reward_reconstruction_error": 0.0,
            }
        ],
    }


def _payload(records):
    return {
        "schema": "terra_historical_curriculum_audit_v1",
        "completion_contract": "legacy_implicit_buffer_v0",
        "observer_only": True,
        "source_revisions": {"terra": "old", "terra_baselines": "old"},
        "bank_root": "/bank",
        "mode": "sampled",
        "seeds": [],
        "horizon": 450,
        "numerical_tolerances": {"terminal_reward_reconstruction_atol": 1e-5},
        "reset_integrity": {"development/M0": {"passed": True}},
        "records": records,
    }


def test_merge_requires_and_orders_the_exact_sampled_grid(monkeypatch):
    monkeypatch.setattr(
        "scripts.analysis.merge_historical_sampled_shards.sha256_file",
        lambda _: "hash",
    )
    records = [
        _record(label, seed)
        for label in reversed(EXPECTED_LABELS)
        for seed in reversed(EXPECTED_SEEDS)
    ]
    payloads = [_payload(records[:8]), _payload(records[8:])]
    merged = merge_payloads(
        payloads,
        [Path("first.json"), Path("second.json")],
    )
    assert [
        (record["checkpoint_label"], record["seed"]) for record in merged["records"]
    ] == [(label, seed) for label in EXPECTED_LABELS for seed in EXPECTED_SEEDS]
    assert merged["seeds"] == list(EXPECTED_SEEDS)
    assert [row["record_count"] for row in merged["execution_shards"]] == [8, 8]


def test_merge_rejects_duplicate_or_incomplete_sampled_records(monkeypatch):
    monkeypatch.setattr(
        "scripts.analysis.merge_historical_sampled_shards.sha256_file",
        lambda _: "hash",
    )
    records = [
        _record(label, seed) for label in EXPECTED_LABELS for seed in EXPECTED_SEEDS
    ]
    with pytest.raises(RuntimeError, match="duplicate sampled record"):
        merge_payloads(
            [_payload(records), _payload([records[0]])],
            [Path("first.json"), Path("duplicate.json")],
        )
    with pytest.raises(RuntimeError, match="sampled grid mismatch"):
        merge_payloads(
            [_payload(records[:-1])],
            [Path("incomplete.json")],
        )
