"""A checkpoint continuation preserves and replaces only its replayed receipts."""

from types import SimpleNamespace

import pytest

from train_mixed import (
    _archive_replayed_episode_aggregate_receipts,
    _write_episode_aggregate_receipt,
)


def _write(config, update, marker):
    return _write_episode_aggregate_receipt(
        config,
        {"run_name": config.name, "update": update, "marker": marker},
    )


def test_resume_archives_only_this_runs_post_checkpoint_receipts(tmp_path):
    config = SimpleNamespace(checkpoint_dir=tmp_path, name="v2[gen]")
    output_dir = tmp_path / "episode_aggregates"
    completed = _write(config, 25_991, "before checkpoint")
    checkpoint_window = _write(config, 26_000, "checkpoint window")
    replayed = [
        _write(config, 26_001, "parent after checkpoint"),
        _write(config, 26_011, "parent last log before timeout"),
    ]
    unrelated_config = SimpleNamespace(checkpoint_dir=tmp_path, name="v2g")
    unrelated = _write(unrelated_config, 26_001, "unrelated run")
    similarly_named_config = SimpleNamespace(
        checkpoint_dir=tmp_path, name="v2[gen]_extra"
    )
    similarly_named = _write(similarly_named_config, 26_001, "different run")
    non_receipt = output_dir / "v2[gen]_update_latest.json"
    non_receipt.write_text("unrelated diagnostic\n")
    original_bytes = {
        path: path.read_bytes()
        for path in (completed, checkpoint_window, *replayed, unrelated,
                     similarly_named, non_receipt)
    }

    archive = _archive_replayed_episode_aggregate_receipts(config, 26_000)

    assert archive is not None
    assert archive.parent == output_dir
    assert {path.name for path in archive.iterdir()} == {
        path.name for path in replayed
    }
    for path in replayed:
        assert not path.exists()
        assert (archive / path.name).read_bytes() == original_bytes[path]
    for path in (completed, checkpoint_window, unrelated, similarly_named, non_receipt):
        assert path.read_bytes() == original_bytes[path]

    rewritten = _write(config, 26_001, "resumed trajectory")
    assert rewritten.read_bytes() != original_bytes[rewritten]
    assert (archive / rewritten.name).read_bytes() == original_bytes[rewritten]
    # Existing consumers use this nonrecursive glob, so archived windows do
    # not enter the canonical population or appear twice after replay.
    visible = list(output_dir.glob("*_update_*.json"))
    assert rewritten in visible
    assert not any(path.parent == archive for path in visible)


def test_repeated_resume_preserves_each_previous_attempt_in_a_unique_archive(tmp_path):
    config = SimpleNamespace(checkpoint_dir=tmp_path, name="same-run")
    receipt = _write(config, 501, "original parent")
    parent_bytes = receipt.read_bytes()
    first_archive = _archive_replayed_episode_aggregate_receipts(config, 500)
    assert _archive_replayed_episode_aggregate_receipts(config, 500) is None

    _write(config, 501, "first replay")
    first_replay_bytes = receipt.read_bytes()
    second_archive = _archive_replayed_episode_aggregate_receipts(config, 500)

    assert first_archive != second_archive
    assert (first_archive / receipt.name).read_bytes() == parent_bytes
    assert (second_archive / receipt.name).read_bytes() == first_replay_bytes
    _write(config, 501, "second replay")


def test_fresh_duplicate_receipt_still_fails_without_changing_original(tmp_path):
    config = SimpleNamespace(checkpoint_dir=tmp_path, name="fresh")
    receipt = _write(config, 1, "original")
    original = receipt.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        _write(config, 1, "duplicate")

    assert receipt.read_bytes() == original
    assert list(receipt.parent.iterdir()) == [receipt]


def test_resume_without_post_checkpoint_receipts_creates_no_archive(tmp_path):
    config = SimpleNamespace(checkpoint_dir=tmp_path, name="clean")
    assert _archive_replayed_episode_aggregate_receipts(config, 500) is None
    assert not (tmp_path / "episode_aggregates").exists()
    receipt = _write(config, 500, "checkpoint window")
    assert _archive_replayed_episode_aggregate_receipts(config, 500) is None
    assert list(receipt.parent.iterdir()) == [receipt]
