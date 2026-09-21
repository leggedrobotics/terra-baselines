"""Fixed evaluation must neither reset PPO nor lose a long run on eval failure."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.oracle_followup.run import evaluate_checkpoint


def complete_report(update, checkpoint):
    return {"horizon": 450, "deterministic": True, "checkpoint_update": update,
            "checkpoint": str(checkpoint),
            "per_map": [{"episode_id": str(i), "integrity_failure": False,
                         "integrity_unavailable": False, "terminated": True}
                        for i in range(608)]}


@pytest.mark.parametrize("visible,expected", [
    (None, "0"), ("2", "2"), ("2,3", "2"), ("GPU-allocated", "GPU-allocated"), ("", ""),
])
def test_milestone_uses_child_gpu_and_reuses_complete_panel(tmp_path, monkeypatch, visible, expected):
    calls = []
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    def evaluate(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == expected
        assert kwargs["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert kwargs["env"]["BANK_ROOT"] == str(tmp_path)
        Path(command[-1]).write_text(json.dumps(complete_report(10000, checkpoint)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.oracle_followup.run.subprocess.run", evaluate)
    checkpoint = tmp_path / "policy.pkl"
    evaluate_checkpoint(checkpoint, 9750, tmp_path, tmp_path)
    assert not calls
    evaluate_checkpoint(checkpoint, 10000, tmp_path, tmp_path)
    evaluate_checkpoint(checkpoint, 10000, tmp_path, tmp_path)
    assert len(calls) == 1
    assert json.loads((tmp_path / "u10000.status.json").read_text())["status"] == "PASS"


def test_interrupted_report_and_eval_failure_do_not_abort_training(tmp_path, monkeypatch):
    (tmp_path / "u10000.json").write_text('{"interrupted":')
    monkeypatch.setattr("scripts.oracle_followup.run.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=124))
    evaluate_checkpoint(tmp_path / "policy.pkl", 10000, tmp_path, tmp_path)
    assert list(tmp_path.glob("u10000.incomplete-*.json"))
    status = json.loads((tmp_path / "u10000.status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["returncode"] == 124
