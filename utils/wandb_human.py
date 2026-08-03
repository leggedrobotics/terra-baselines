"""Small, bounded W&B summaries for Terra training and fixed evaluation."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from utils.pooled_sampler import effective_sample_size, entropy

LOGGING_SCHEMA = "terra_wandb_human_v1"
FAMILIES = ("foundation", "trench")
BRANCH_DEPTHS = ("Anchor", "Nearby core", "One-axis", "Composed")
ACTION_NAMES = (
    "forward",
    "backward",
    "base_clockwise",
    "base_anticlockwise",
    "cabin_clockwise",
    "cabin_anticlockwise",
    "do",
    "no_op",
)

CONDITION_COLUMNS = (
    "condition",
    "family",
    "depth",
    "target_probability",
    "active_population_fraction",
    "reset_exposure_fraction",
    "ended_episode_fraction",
    "train_success_rate",
    "mean_absolute_completion",
)

EVAL_CONDITION_COLUMNS = (
    "condition",
    "family",
    "episodes",
    "exact_success_rate",
    "mean_completion",
    "zero_completion_rate",
)

TRAINING_SCALAR_KEYS = frozenset(
    {
        "train/episode_success_rate",
        "train/episode_timeout_rate",
        "train/ended_episodes",
        "train/update",
        "behavior/absolute_completion",
        "behavior/dig_completion",
        "behavior/dump_volume_completion",
        "behavior/dump_purity",
        "behavior/no_effect_action_rate",
        "behavior/mean_episode_length",
        "behavior/productive_workspace_cycles_per_episode",
        *{f"behavior/action_fraction/{name}" for name in ACTION_NAMES},
        "reward/episode_return",
        "reward/agent",
        "reward/terminal",
        "reward/trench",
        "reward/existence",
        *{f"curriculum/population/{label}" for label in (*FAMILIES, *BRANCH_DEPTHS)},
        "curriculum/target_ess",
        "curriculum/target_entropy_normalized",
        "curriculum/refreshes",
        "ppo/total_loss",
        "ppo/policy_loss",
        "ppo/value_loss",
        "ppo/entropy",
        "ppo/entropy_coef",
        "ppo/approx_kl",
        "ppo/clip_fraction",
        "ppo/explained_variance",
        "ppo/grad_norm",
        "kickstart/kl",
        "kickstart/value_mse",
        "kickstart/kl_coef",
        "kickstart/value_coef",
        "system/steps_per_second",
        "system/environment_steps",
    }
)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def episode_metrics(payload: dict, *, include_trench_reward: bool) -> dict[str, float]:
    """Return human-facing episode metrics from one aggregate receipt."""
    totals = payload["totals"]
    rates = payload["rates"]
    episodes = int(totals["episode_count"])
    steps = int(totals["step_count"])

    def per_episode(field: str) -> float:
        return _safe_rate(totals[field], episodes)

    metrics = {
        "train/ended_episodes": float(episodes),
        "train/episode_success_rate": (
            float(rates["task_done_rate"])
            if rates["task_done_rate"] is not None
            else float("nan")
        ),
        "train/episode_timeout_rate": (
            float(rates["timeout_rate"])
            if rates["timeout_rate"] is not None
            else float("nan")
        ),
        "behavior/absolute_completion": per_episode("combined_completion_sum"),
        "behavior/dig_completion": per_episode("dig_completion_sum"),
        "behavior/dump_volume_completion": per_episode("dump_volume_completion_sum"),
        "behavior/dump_purity": per_episode("dump_purity_sum"),
        "behavior/no_effect_action_rate": _safe_rate(
            totals["no_effect_action_count"], steps
        ),
        "behavior/mean_episode_length": _safe_rate(steps, episodes),
        "behavior/productive_workspace_cycles_per_episode": _safe_rate(
            totals["productive_workspace_cycles"], episodes
        ),
        "reward/episode_return": per_episode("episodic_return_sum"),
        "reward/agent": _safe_rate(sum(totals["agent_reward_sum"]), episodes),
        "reward/terminal": per_episode("terminal_reward_normalized_sum"),
        "reward/existence": per_episode("existence_reward_sum"),
    }
    action_counts = totals["action_counts"]
    if len(action_counts) != len(ACTION_NAMES):
        raise ValueError(
            f"expected {len(ACTION_NAMES)} action counts, got {len(action_counts)}"
        )
    action_total = sum(action_counts)
    metrics.update(
        {
            f"behavior/action_fraction/{name}": _safe_rate(count, action_total)
            for name, count in zip(ACTION_NAMES, action_counts)
        }
    )
    if include_trench_reward:
        metrics["reward/trench"] = per_episode("trench_reward_sum")
    return metrics


def loss_metrics(
    loss_info: dict,
    *,
    entropy_coef: float,
    teacher_enabled: bool,
    kickstart_kl_coef: float,
    kickstart_value_coef: float,
) -> dict[str, float]:
    """Select and rename the small PPO/kickstart metric set."""

    def scalar(key: str) -> float:
        return float(np.asarray(loss_info[key]))

    metrics = {
        "ppo/total_loss": scalar("total_loss"),
        "ppo/policy_loss": scalar("actor_loss"),
        "ppo/value_loss": scalar("value_loss"),
        "ppo/entropy": scalar("entropy"),
        "ppo/entropy_coef": float(entropy_coef),
        "ppo/approx_kl": scalar("approx_kl"),
        "ppo/clip_fraction": scalar("clip_fraction"),
        "ppo/explained_variance": scalar("explained_variance"),
        "ppo/grad_norm": scalar("diagnostics/grad_global_norm"),
    }
    if teacher_enabled:
        metrics.update(
            {
                "kickstart/kl": scalar("kickstart/kl"),
                "kickstart/value_mse": scalar("kickstart/value_mse"),
                "kickstart/kl_coef": float(kickstart_kl_coef),
                "kickstart/value_coef": float(kickstart_value_coef),
            }
        )
    return metrics


def curriculum_metrics(
    levels: np.ndarray,
    *,
    names: tuple[str, ...],
    labels: dict[str, dict[str, str]],
    probabilities: np.ndarray,
    refreshes: int,
) -> dict[str, float]:
    """Summarize the actual active population and intended sampler spread."""
    levels = np.asarray(levels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if levels.size == 0:
        raise ValueError("curriculum population cannot be empty")
    if probabilities.shape != (len(names),):
        raise ValueError("sampler probabilities must match condition names")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("sampler probabilities must be finite and nonnegative")
    if np.any((levels < 0) | (levels >= len(names))):
        raise ValueError("active curriculum level is outside the condition bank")
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("sampler probabilities must sum to one")

    active = np.bincount(levels, minlength=len(names)).astype(np.float64)
    active /= active.sum()

    metrics = {}
    for field, values in (("family", FAMILIES), ("branch_depth", BRANCH_DEPTHS)):
        masses = {value: 0.0 for value in values}
        for index, name in enumerate(names):
            label = labels.get(name, {}).get(field)
            if label in masses:
                masses[label] += float(active[index])
        if not math.isclose(sum(masses.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"every condition needs a recognized {field} label")
        for label, mass in masses.items():
            metrics[f"curriculum/population/{label}"] = mass

    normalized_entropy = (
        entropy(probabilities) / math.log(len(probabilities))
        if len(probabilities) > 1
        else 0.0
    )
    metrics.update(
        {
            "curriculum/target_ess": effective_sample_size(probabilities),
            "curriculum/target_entropy_normalized": normalized_entropy,
            "curriculum/refreshes": float(refreshes),
        }
    )
    return metrics


def condition_rows(
    levels: np.ndarray,
    reset_exposures: np.ndarray,
    episode_payload: dict,
    *,
    names: tuple[str, ...],
    labels: dict[str, dict[str, str]],
    probabilities: np.ndarray,
) -> list[list[object]]:
    """Build one aligned per-condition snapshot for the current PPO update."""
    levels = np.asarray(levels, dtype=np.int64).reshape(-1)
    resets = np.asarray(reset_exposures, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if resets.shape != (len(names),):
        raise ValueError("reset exposure counts must match condition names")
    if levels.size == 0 or np.any((levels < 0) | (levels >= len(names))):
        raise ValueError("active curriculum levels must index the condition bank")
    if np.any(resets < 0):
        raise ValueError("reset exposure counts must be nonnegative")
    if probabilities.shape != (len(names),):
        raise ValueError("sampler probabilities must match condition names")

    active = np.bincount(levels, minlength=len(names)).astype(np.float64)
    active /= active.sum()
    reset_total = int(resets.sum())

    episodes = defaultdict(int)
    successes = defaultdict(int)
    completion = defaultdict(float)
    for group in episode_payload.get("groups", ()):
        name = group.get("primary_cell")
        if name not in names:
            continue
        episodes[name] += int(group["episode_count"])
        successes[name] += int(group["task_done_count"])
        completion[name] += float(group["combined_completion_sum"])
    episode_total = sum(episodes.values())

    rows = []
    for index, name in enumerate(names):
        count = episodes[name]
        rows.append(
            [
                name,
                labels.get(name, {}).get("family", "unknown"),
                labels.get(name, {}).get("branch_depth", "unknown"),
                float(probabilities[index]),
                float(active[index]),
                _safe_rate(resets[index], reset_total),
                _safe_rate(count, episode_total),
                _safe_rate(successes[name], count),
                _safe_rate(completion[name], count),
            ]
        )
    return rows


def fixed_eval_metrics(
    record: dict, split: str
) -> tuple[dict[str, float], list[list[object]]]:
    """Recompute fixed-bank metrics from per-map rows and verify saved summaries."""
    per_map = record["per_map"]
    if not per_map:
        raise ValueError("fixed evaluation contains no scenarios")

    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in per_map:
        completion = float(row["terminal_absolute"])
        if not 0.0 <= completion <= 1.0 + 1e-6:
            raise ValueError(f"terminal completion is outside [0, 1]: {completion}")
        by_condition[row["primary_cell"]].append(row)
        by_family[row["family"]].append(row)

    def exact(rows: list[dict]) -> float:
        return float(np.mean([bool(row["success"]) for row in rows]))

    def mean_completion(rows: list[dict]) -> float:
        return float(np.mean([float(row["terminal_absolute"]) for row in rows]))

    condition_completion = {
        name: mean_completion(rows) for name, rows in by_condition.items()
    }
    exact_rate = exact(per_map)
    macro = float(np.mean(list(condition_completion.values())))
    worst = float(min(condition_completion.values()))
    zero = float(np.mean([float(row["terminal_absolute"]) <= 1e-12 for row in per_map]))

    prefix = f"eval/{split}"
    metrics = {
        "eval/update": float(record["checkpoint_update"]),
        f"{prefix}/exact_success_rate": exact_rate,
        f"{prefix}/macro_completion": macro,
        f"{prefix}/worst_condition_completion": worst,
        f"{prefix}/zero_completion_rate": zero,
    }
    for family, rows in sorted(by_family.items()):
        family_conditions = {row["primary_cell"] for row in rows}
        metrics[f"{prefix}/{family}_exact_success_rate"] = exact(rows)
        metrics[f"{prefix}/{family}_macro_completion"] = float(
            np.mean([condition_completion[name] for name in family_conditions])
        )

    summary = record["summary"]
    expected = {
        "exact": float(summary["overall"]["success_rate"]),
        "macro": float(summary["graded"]["macro_completion"]),
        "worst": float(summary["graded"]["worst_condition_completion"]),
    }
    observed = {"exact": exact_rate, "macro": macro, "worst": worst}
    for name, saved_value in expected.items():
        if not math.isclose(saved_value, observed[name], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"fixed evaluation {name} disagrees with saved summary: "
                f"recomputed={observed[name]}, saved={saved_value}"
            )

    rows = []
    for condition, condition_maps in sorted(by_condition.items()):
        families = {row["family"] for row in condition_maps}
        if len(families) != 1:
            raise ValueError(f"condition {condition!r} spans multiple families")
        rows.append(
            [
                condition,
                families.pop(),
                len(condition_maps),
                exact(condition_maps),
                condition_completion[condition],
                float(
                    np.mean(
                        [
                            float(row["terminal_absolute"]) <= 1e-12
                            for row in condition_maps
                        ]
                    )
                ),
            ]
        )
    return metrics, rows
