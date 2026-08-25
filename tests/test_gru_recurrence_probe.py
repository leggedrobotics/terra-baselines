from __future__ import annotations

import numpy as np

from scripts.gru_recurrence_probe_v1.run_probe import (
    PADDING_COUNT,
    TARGET_COUNT,
    TARGET_SLOTS,
    analyze_lane,
    build_chunk_slots,
    first_paired_difference,
    source_identity,
    terminal_period,
)


def _hash_rows(values: list[int]) -> np.ndarray:
    rows = np.zeros((len(values), 32), dtype=np.uint8)
    rows[:, 0] = values
    return rows


def test_probe_chunk_preserves_canonical_target_positions_and_unique_padding():
    chunk, padding = build_chunk_slots(720)

    assert len(chunk) == 120
    assert len(set(chunk)) == 120
    assert len(padding) == PADDING_COUNT
    assert not set(padding).intersection(TARGET_SLOTS)
    for slot in TARGET_SLOTS:
        assert chunk[(slot - 1) % 120] == slot


def test_lane_analysis_distinguishes_input_recurrence_from_hidden_memory():
    active = np.ones(9, dtype=bool)
    actions = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    effects = np.asarray([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    material_changed = np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    input_hashes = _hash_rows([10, 11, 10, 11, 10, 11, 10, 11, 10])
    hidden_hashes = _hash_rows(list(range(9)))
    hidden_norms = np.arange(9, dtype=np.float32)
    logits = np.stack(
        [np.asarray([step, -step], dtype=np.float32) for step in range(9)]
    )

    result = analyze_lane(
        active=active,
        actions=actions,
        effects=effects,
        material_changed=material_changed,
        input_hashes=input_hashes,
        hidden_hashes=hidden_hashes,
        hidden_norms=hidden_norms,
        logits=logits,
    )

    assert result["unique_instantaneous_input_count"] == 2
    assert result["repeated_instantaneous_input_decisions"] == 7
    assert result["same_input_different_hidden_decisions"] == 7
    assert result["same_input_different_logits_decisions"] == 7
    assert result["longest_no_effect_streak"] == 7
    assert result["last_material_change_step"] == 1
    assert result["terminal_action_cycle"]["period"] == 2
    assert result["terminal_input_action_cycle"]["period"] == 2
    assert result["terminal_full_policy_state_cycle"] is None


def test_cycle_and_paired_divergence_are_one_based_and_require_common_activity():
    assert terminal_period([1, 2, 1, 2, 1, 2]) == {
        "period": 2,
        "repetitions": 3,
        "suffix_decisions": 6,
        "first_step": 1,
    }
    left = np.asarray([1, 1, 2, 3])
    right = np.asarray([1, 1, 9, 3])
    left_active = np.asarray([1, 1, 1, 1], dtype=bool)
    right_active = np.asarray([1, 1, 0, 0], dtype=bool)
    assert first_paired_difference(left, right, left_active, right_active) is None
    right_active[2] = True
    assert first_paired_difference(left, right, left_active, right_active) == 3


def test_archive_source_identity_reads_revision_marker_without_git(tmp_path):
    revision = "a" * 40
    source_root = tmp_path / "archive"
    module = source_root / "package" / "module.py"
    module.parent.mkdir(parents=True)
    module.write_text("pass\n")
    (source_root / "REVISION").write_text(revision + "\n")

    assert source_identity(str(module)) == {
        "root": str(source_root),
        "revision": revision,
        "source_form": "git_archive_with_revision_marker",
        "dirty": False,
    }
