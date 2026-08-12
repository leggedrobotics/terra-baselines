import pickle
from pathlib import Path

import pytest

from utils import helpers


def test_save_pkl_object_replaces_only_after_a_complete_write(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.pkl"
    destination.write_bytes(b"previous-complete-checkpoint")

    def fail_after_partial_write(obj, output, protocol):
        del obj, protocol
        output.write(b"partial")
        raise RuntimeError("interrupted pickle")

    monkeypatch.setattr(pickle, "dump", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="interrupted pickle"):
        helpers.save_pkl_object({"model": 1}, str(destination))

    assert destination.read_bytes() == b"previous-complete-checkpoint"
    assert list(Path(tmp_path).glob(".checkpoint.pkl.*.tmp")) == []
