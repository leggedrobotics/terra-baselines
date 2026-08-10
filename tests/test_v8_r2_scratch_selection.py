import hashlib

import pytest

from scripts.euler_v8_r2_reward_v2 import select_promotion


def record(path, update, exact, macro, worst, episodes):
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_update": update,
        "summary": {
            "overall": {"successes": exact, "episodes": episodes},
            "graded": {
                "macro_completion": macro,
                "worst_condition_completion": worst,
            },
        },
    }


def test_promotion_selection_uses_combined_exact_then_macro(tmp_path):
    checkpoints = []
    for update in (1_000, 2_000, 3_000):
        path = tmp_path / f"checkpoint_{update}.pkl"
        path.write_bytes(f"checkpoint-{update}".encode())
        checkpoints.append(path)
    main = [
        record(checkpoints[0], 1_000, 500, 0.80, 0.20, 720),
        record(checkpoints[1], 2_000, 501, 0.95, 0.40, 720),
        record(checkpoints[2], 3_000, 500, 0.81, 0.30, 720),
    ]
    capability = [
        record(checkpoints[0], 1_000, 30, 0.80, 0.30, 32),
        record(checkpoints[1], 2_000, 28, 0.95, 0.50, 32),
        record(checkpoints[2], 3_000, 30, 0.81, 0.40, 32),
    ]

    selected = select_promotion.select_records(main, capability)

    assert selected["checkpoint_update"] == 3_000
    assert selected["exact_successes"] == 530
    assert selected["episodes"] == 752


def test_promotion_selection_rejects_panel_checkpoint_mismatch(tmp_path):
    first = tmp_path / "first.pkl"
    second = tmp_path / "second.pkl"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    main = [record(first, 1_000, 1, 0.1, 0.0, 720)]
    capability = [record(second, 1_000, 1, 0.1, 0.0, 32)]

    with pytest.raises(ValueError, match="different checkpoints"):
        select_promotion.select_records(main, capability)
