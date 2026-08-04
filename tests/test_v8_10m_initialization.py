import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import v8_10m_initialization
from scripts import v8_10m_provisional_teacher


def test_identical_outputs_have_zero_transplant_damage():
    value = np.asarray([[1.0], [2.0]])
    logits = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    result = v8_10m_initialization.summarize_outputs(
        value, logits, value.copy(), logits.copy()
    )
    assert result["finite"] is True
    assert result["teacher_to_student_kl_mean"] == pytest.approx(0.0)
    assert result["deterministic_action_agreement"] == 1.0
    assert result["value_rmse"] == 0.0


def test_output_diagnostic_reports_known_policy_and_value_change():
    result = v8_10m_initialization.summarize_outputs(
        np.asarray([[0.0], [0.0]]),
        np.asarray([[8.0, 0.0], [0.0, 8.0]]),
        np.asarray([[3.0], [4.0]]),
        np.asarray([[0.0, 8.0], [0.0, 8.0]]),
    )
    assert result["teacher_to_student_kl_mean"] > 3.0
    assert result["deterministic_action_agreement"] == 0.5
    assert result["value_rmse"] == pytest.approx(np.sqrt(12.5))


def test_nonfinite_outputs_fail_loudly():
    with pytest.raises(ValueError, match="non-finite"):
        v8_10m_initialization.summarize_outputs(
            np.asarray([[0.0]]),
            np.asarray([[0.0, 1.0]]),
            np.asarray([[np.nan]]),
            np.asarray([[0.0, 1.0]]),
        )


def test_provisional_teacher_identity_is_explicit_and_hash_bound(tmp_path):
    checkpoint = tmp_path / "teacher_update_005000.pkl"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    inspection = tmp_path / "teacher_inspection.json"
    inspection.write_text(
        json.dumps(
            {
                "schema": "terra_v8_10m_provisional_teacher_v1",
                "passed": True,
                "provisional_teacher": True,
                "performance_mastery_gate_waived_by_user": True,
                "same_distribution": True,
                "finite_model_optimizer": True,
                "full_sampler_state_validated": True,
                "teacher_arm": "G-DEEP-XATTN-V8-DENSE-WARM",
                "teacher_checkpoint": str(checkpoint),
                "teacher_checkpoint_sha256": digest,
                "release_id": "terra_v8_v6_constraints_v7_adjacent_train96_v5",
                "terra_revision": "a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4",
                "curriculum_stage": "full",
                "reward_stage": "dense_skill",
            }
        )
    )
    record, path, admission = v8_10m_initialization.inspect_teacher_source(
        teacher_receipt=None,
        teacher_inspection=inspection,
        bank_root=tmp_path,
    )
    assert record["provisional_teacher"] is True
    assert path == inspection.resolve()
    assert admission == "provisional_inspection"


def test_provisional_teacher_requires_a_numbered_periodic_checkpoint():
    assert (
        v8_10m_provisional_teacher.checkpoint_update(Path("x_update_005000.pkl"))
        == 5000
    )
    with pytest.raises(ValueError, match="numbered periodic"):
        v8_10m_provisional_teacher.checkpoint_update(Path("x_FINAL.pkl"))
