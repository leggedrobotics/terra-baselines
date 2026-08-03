import numpy as np
import pytest

from scripts import v8_10m_initialization


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
