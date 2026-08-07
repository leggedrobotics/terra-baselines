import math

import numpy as np
import pytest

from scripts.create_wandb_human_workspace import workspace_spec
from utils.wandb_human import (
    BRANCH_DEPTHS,
    CONDITION_COLUMNS,
    FAMILIES,
    TRAINING_SCALAR_KEYS,
    condition_rows,
    curriculum_metrics,
    episode_metrics,
    fixed_eval_metrics,
    loss_metrics,
)


def _episode_payload(episodes=2):
    return {
        "totals": {
            "episode_count": episodes,
            "task_done_count": 1 if episodes else 0,
            "timeout_count": 1 if episodes else 0,
            "step_count": 6 if episodes else 0,
            "episodic_return_sum": 8.0 if episodes else 0.0,
            "agent_reward_sum": [4.0, 1.0, 0.0, 0.0],
            "terminal_reward_normalized_sum": 2.0 if episodes else 0.0,
            "trench_reward_sum": 0.5 if episodes else 0.0,
            "existence_reward_sum": 0.5 if episodes else 0.0,
            "productive_workspace_cycles": 4 if episodes else 0,
            "no_effect_action_count": 1 if episodes else 0,
            "action_counts": [1, 1, 1, 1, 1, 1, 0, 0] if episodes else [0] * 8,
            "dig_completion_sum": 1.4 if episodes else 0.0,
            "dump_purity_sum": 1.0 if episodes else 0.0,
            "dump_volume_completion_sum": 1.2 if episodes else 0.0,
            "combined_completion_sum": 1.3 if episodes else 0.0,
        },
        "rates": {
            "task_done_rate": 0.5 if episodes else None,
            "timeout_rate": 0.5 if episodes else None,
        },
        "groups": [
            {
                "primary_cell": "f0",
                "episode_count": 2,
                "task_done_count": 1,
                "combined_completion_sum": 1.2,
            },
            {
                "primary_cell": "t0",
                "episode_count": 1,
                "task_done_count": 0,
                "combined_completion_sum": 0.2,
            },
        ],
    }


def test_episode_metrics_are_rates_means_and_unknown_without_episodes():
    metrics = episode_metrics(_episode_payload(), include_trench_reward=True)
    assert metrics["train/episode_success_rate"] == 0.5
    assert metrics["behavior/mean_episode_length"] == 3.0
    assert metrics["behavior/absolute_completion"] == pytest.approx(0.65)
    assert metrics["reward/episode_return"] == 4.0
    assert metrics["reward/agent"] == 2.5
    assert metrics["reward/trench"] == 0.25
    assert metrics["behavior/action_fraction/forward"] == pytest.approx(1 / 6)
    assert metrics["behavior/action_fraction/no_op"] == 0.0

    empty = episode_metrics(_episode_payload(episodes=0), include_trench_reward=False)
    assert math.isnan(empty["train/episode_success_rate"])
    assert math.isnan(empty["train/episode_timeout_rate"])
    assert math.isnan(empty["behavior/absolute_completion"])
    assert "reward/trench" not in empty


def test_curriculum_population_and_condition_table_use_actual_exposure():
    names = ("f0", "f1", "t0", "t1")
    labels = {
        "f0": {"family": "foundation", "branch_depth": "Anchor"},
        "f1": {"family": "foundation", "branch_depth": "One-axis"},
        "t0": {"family": "trench", "branch_depth": "Anchor"},
        "t1": {"family": "trench", "branch_depth": "Composed"},
    }
    metrics = curriculum_metrics(
        np.array([[0, 0, 1], [2, 3, 3]]),
        names=names,
        labels=labels,
        probabilities=np.array([0.1, 0.2, 0.3, 0.4]),
        refreshes=3,
    )
    assert metrics["curriculum/population/foundation"] == 0.5
    assert metrics["curriculum/population/trench"] == 0.5
    assert metrics["curriculum/population/Anchor"] == 0.5
    assert metrics["curriculum/population/One-axis"] == pytest.approx(1 / 6)
    assert metrics["curriculum/population/Composed"] == pytest.approx(1 / 3)
    assert 0.0 <= metrics["curriculum/target_entropy_normalized"] <= 1.0

    rows = condition_rows(
        np.array([[0, 0, 1], [2, 3, 3]]),
        np.array([3, 0, 1, 0]),
        np.array([30, 10, 20, 40]),
        _episode_payload(),
        names=names,
        labels=labels,
        probabilities=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    table = {row[0]: dict(zip(CONDITION_COLUMNS, row)) for row in rows}
    assert table["f0"]["active_population_fraction"] == pytest.approx(1 / 3)
    assert table["f0"]["reset_exposure_fraction"] == 0.75
    assert table["f0"]["transition_exposure_fraction"] == 0.30
    assert table["f0"]["ended_episode_fraction"] == pytest.approx(2 / 3)
    assert table["f0"]["train_success_rate"] == 0.5
    assert table["t0"]["mean_absolute_completion"] == 0.2


def test_bounded_logging_schema_and_manual_workspace():
    metrics = loss_metrics(
        {
            "total_loss": np.array(3.0),
            "actor_loss": np.array(1.0),
            "value_loss": np.array(2.0),
            "entropy": np.array(0.4),
            "approx_kl": np.array(0.01),
            "clip_fraction": np.array(0.2),
            "explained_variance": np.array(0.3),
            "diagnostics/grad_global_norm": np.array(0.5),
            "kickstart/kl": np.array(0.6),
            "kickstart/value_mse": np.array(0.7),
            "diagnostics/params_all_finite": np.array(1.0),
        },
        entropy_coef=0.02,
        teacher_enabled=True,
        kickstart_kl_coef=0.8,
        kickstart_value_coef=0.4,
    )
    assert metrics["ppo/policy_loss"] == 1.0
    assert metrics["kickstart/kl"] == pytest.approx(0.6)
    assert not any(key.startswith("diagnostics/") for key in metrics)

    assert len(TRAINING_SCALAR_KEYS) <= 48
    assert {
        f"curriculum/population/{label}" for label in (*FAMILIES, *BRANCH_DEPTHS)
    }.issubset(TRAINING_SCALAR_KEYS)
    banned = ("integrity/", "curriculum_levels", "sampler_q/", "diagnostics/")
    assert not any(key.startswith(banned) for key in TRAINING_SCALAR_KEYS)

    spec = workspace_spec()
    assert spec["auto_generate_panels"] is False
    assert spec["visible_panel_count"] == 16
    assert spec["sections"][0]["name"] == "Task outcome"
    assert spec["sections"][0]["panels"][0]["title"] == "Fixed exact success"
    assert any(
        panel["title"] == "Action distribution"
        for section in spec["sections"]
        for panel in section["panels"]
    )
    assert {
        panel["x"] for section in spec["sections"] for panel in section["panels"]
    } == {"train/update", "eval/update"}
    assert all(
        panel["x"] == "train/update"
        for panel in spec["collapsed_details"]["line_panels"]
    )


def test_fixed_eval_metrics_recompute_exact_macro_and_worst():
    per_map = [
        {
            "primary_cell": "f0",
            "family": "foundation",
            "success": True,
            "terminal_absolute": 1.0,
        },
        {
            "primary_cell": "f0",
            "family": "foundation",
            "success": False,
            "terminal_absolute": 0.5,
        },
        {
            "primary_cell": "t0",
            "family": "trench",
            "success": False,
            "terminal_absolute": 0.0,
        },
        {
            "primary_cell": "t0",
            "family": "trench",
            "success": False,
            "terminal_absolute": 0.5,
        },
    ]
    record = {
        "checkpoint_update": 500,
        "per_map": per_map,
        "summary": {
            "overall": {"success_rate": 0.25},
            "graded": {
                "macro_completion": 0.5,
                "worst_condition_completion": 0.25,
            },
        },
    }
    metrics, rows = fixed_eval_metrics(record, "promotion")
    assert metrics["eval/promotion/exact_success_rate"] == 0.25
    assert metrics["eval/promotion/macro_completion"] == 0.5
    assert metrics["eval/promotion/worst_condition_completion"] == 0.25
    assert metrics["eval/promotion/zero_completion_rate"] == 0.25
    assert metrics["eval/promotion/foundation_exact_success_rate"] == 0.5
    assert len(rows) == 2
