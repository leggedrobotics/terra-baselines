"""One host-side probability distribution over accepted map conditions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np


RULES = ("uniform", "adaptive")
_TOKEN = re.compile(r"[^0-9A-Za-z._-]+")


def metric_token(value: str) -> str:
    return _TOKEN.sub("_", value)


def entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-(positive * np.log(positive)).sum()) if positive.size else 0.0


def effective_sample_size(probabilities: np.ndarray) -> float:
    denominator = float(np.square(probabilities).sum())
    return 1.0 / denominator if denominator else 0.0


def _cap_distribution(probabilities: np.ndarray, cap: float) -> np.ndarray:
    """Water-fill a probability vector under one per-condition mass cap."""
    if cap * probabilities.size < 1.0 - 1e-12:
        raise ValueError(
            f"max_mass={cap} is infeasible for {probabilities.size} conditions"
        )
    weights = np.maximum(probabilities.astype(np.float64), 0.0)
    total = float(weights.sum())
    result = (
        weights / total
        if total > 0.0
        else np.full(probabilities.size, 1.0 / probabilities.size)
    )
    capped = np.zeros(probabilities.size, dtype=bool)
    for _ in range(probabilities.size):
        over = (~capped) & (result > cap + 1e-15)
        if not over.any():
            break
        capped |= over
        result[capped] = cap
        free = ~capped
        if not free.any():
            break
        remaining = 1.0 - cap * float(capped.sum())
        free_weight = float(weights[free].sum())
        result[free] = (
            remaining * weights[free] / free_weight
            if free_weight > 0.0
            else remaining / float(free.sum())
        )
    return result


@dataclass(frozen=True)
class SamplerSettings:
    rule: str = "uniform"
    update_interval: int = 150
    uniform_floor: float = 0.20
    mastery_threshold: float = 0.75
    temperature: float = 0.25
    min_episodes: int = 20
    competence_ema: float = 0.30
    max_mass: float = 0.15
    seed: int = 0

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"sampler rule must be one of {RULES}")
        if self.update_interval <= 0:
            raise ValueError("sampler update_interval must be positive")
        if not 0.0 <= self.uniform_floor <= 1.0:
            raise ValueError("sampler uniform_floor must be in [0, 1]")
        if not 0.0 <= self.mastery_threshold <= 1.0:
            raise ValueError("sampler mastery_threshold must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("sampler temperature must be positive")
        if self.min_episodes <= 0:
            raise ValueError("sampler min_episodes must be positive")
        if not 0.0 < self.competence_ema <= 1.0:
            raise ValueError("sampler competence_ema must be in (0, 1]")
        if not 0.0 < self.max_mass <= 1.0:
            raise ValueError("sampler max_mass must be in (0, 1]")


class PooledConditionSampler:
    """Uniform control or simple competence-frontier adaptive sampler.

    The adaptive arm uses

        q = uniform_floor * Uniform + (1 - uniform_floor) * Frontier.

    ``Frontier`` is a softmax over measured completion of conditions below the
    mastery threshold. Thus a condition that remains at zero cannot monopolize
    training, while the highest-competence unsolved conditions receive more
    exposure. Mastered conditions remain in the uniform floor and re-enter
    automatically if their EMA falls below the threshold.
    """

    def __init__(
        self,
        names: list[str],
        settings: SamplerSettings,
        *,
        maps_per_condition: list[int] | None = None,
        labels: dict[str, dict[str, str]] | None = None,
    ):
        if not names:
            raise ValueError("pooled sampler needs at least one condition")
        if len(names) != len(set(names)):
            raise ValueError("pooled sampler needs exactly one level per condition")
        self.names = tuple(names)
        self.settings = settings
        self._index = {name: index for index, name in enumerate(self.names)}
        self._count = len(self.names)
        if settings.rule == "adaptive" and settings.max_mass * self._count < 1.0:
            raise ValueError(
                f"adaptive max_mass={settings.max_mass} is infeasible for "
                f"{self._count} conditions"
            )
        self.maps_per_condition = tuple(
            maps_per_condition or [1] * self._count
        )
        if len(self.maps_per_condition) != self._count:
            raise ValueError("maps_per_condition must match the condition count")
        self.labels = labels or {}

        self._uniform = np.full(self._count, 1.0 / self._count, dtype=np.float64)
        self._probabilities = self._uniform.copy()
        self._competence = np.full(self._count, np.nan, dtype=np.float64)
        self._episodes = np.zeros(self._count, dtype=np.int64)
        self._completion_sum = np.zeros(self._count, dtype=np.float64)
        self._last_episodes = np.zeros(self._count, dtype=np.int64)
        self._assignments = np.zeros(self._count, dtype=np.int64)
        self._last_assignments = np.zeros(self._count, dtype=np.int64)
        self._reset_exposures = np.zeros(self._count, dtype=np.int64)
        self._last_reset_exposures = np.zeros(self._count, dtype=np.int64)
        self._window_updates = 0
        self._last_window_updates = 0
        self._has_closed_window = False
        self._last_refresh_update: int | None = None
        self._refreshes = 0
        self._rng = np.random.default_rng(settings.seed)

    @property
    def probabilities(self) -> np.ndarray:
        return self._probabilities.copy()

    def sample_levels(self, shape: tuple[int, ...]) -> np.ndarray:
        sample_count = int(np.prod(shape))
        samples = self._rng.choice(
            self._count, size=sample_count, p=self._probabilities
        )
        self._assignments += np.bincount(samples, minlength=self._count)
        return samples.reshape(shape).astype(np.int32)

    def observe_reset_exposures(self, counts: np.ndarray) -> None:
        """Count maps actually instantiated by reset, separately from episodes."""
        values = np.asarray(counts)
        if values.shape != (self._count,):
            raise ValueError(
                "reset exposure counts must match the condition count"
            )
        if values.dtype.kind not in "iub" or np.any(values < 0):
            raise ValueError(
                "reset exposure counts must be nonnegative integers"
            )
        self._reset_exposures += values.astype(np.int64)

    def start(self, update_index: int) -> None:
        if self._last_refresh_update is None:
            self._last_refresh_update = int(update_index)

    def due(self, update_index: int) -> bool:
        return (
            self._last_refresh_update is not None
            and update_index - self._last_refresh_update
            >= self.settings.update_interval
        )

    def observe_episode_payload(self, payload: dict) -> None:
        """Add one flushed aggregate receipt to the current sampler window."""
        for row in payload.get("groups", ()):
            index = self._index.get(row.get("primary_cell"))
            if index is None:
                continue
            episodes = int(row["episode_count"])
            self._episodes[index] += episodes
            self._completion_sum[index] += float(row["combined_completion_sum"])
        self._window_updates += 1

    def refresh(self, update_index: int) -> None:
        if self.settings.rule == "adaptive":
            alpha = self.settings.competence_ema
            for index, episodes in enumerate(self._episodes):
                if int(episodes) < self.settings.min_episodes:
                    continue
                observed = float(self._completion_sum[index]) / int(episodes)
                previous = self._competence[index]
                self._competence[index] = (
                    observed
                    if math.isnan(previous)
                    else (1.0 - alpha) * previous + alpha * observed
                )
            self._probabilities = self._adaptive_distribution()

        self._last_episodes = self._episodes
        self._last_assignments = self._assignments
        self._last_reset_exposures = self._reset_exposures
        self._episodes = np.zeros(self._count, dtype=np.int64)
        self._completion_sum = np.zeros(self._count, dtype=np.float64)
        self._assignments = np.zeros(self._count, dtype=np.int64)
        self._reset_exposures = np.zeros(self._count, dtype=np.int64)
        self._last_window_updates = self._window_updates
        self._window_updates = 0
        self._has_closed_window = True
        self._last_refresh_update = int(update_index)
        self._refreshes += 1

    def _adaptive_distribution(self) -> np.ndarray:
        competence = np.where(np.isnan(self._competence), 0.0, self._competence)
        unmastered = competence < self.settings.mastery_threshold
        if not unmastered.any():
            return self._uniform.copy()
        scores = np.full(self._count, -np.inf, dtype=np.float64)
        scores[unmastered] = (
            competence[unmastered] / self.settings.temperature
        )
        scores -= scores[unmastered].max()
        focus = np.where(np.isfinite(scores), np.exp(scores), 0.0)
        focus /= focus.sum()

        floor_share = self.settings.uniform_floor / self._count
        headroom = self.settings.max_mass - floor_share
        if self.settings.uniform_floor < 1.0 and headroom > 0.0:
            focus_cap = headroom / (1.0 - self.settings.uniform_floor)
            focus = _cap_distribution(focus, focus_cap)
        probabilities = (
            self.settings.uniform_floor * self._uniform
            + (1.0 - self.settings.uniform_floor) * focus
        )
        return probabilities / probabilities.sum()

    def _mass(self, counts: np.ndarray) -> np.ndarray:
        total = int(counts.sum())
        if total == 0:
            return np.full(self._count, np.nan, dtype=np.float64)
        return counts.astype(np.float64) / total

    def _label_mass(
        self, probabilities: np.ndarray, field: str
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for index, name in enumerate(self.names):
            value = self.labels.get(name, {}).get(field)
            if value:
                result[value] = result.get(value, 0.0) + float(
                    probabilities[index]
                )
        return result

    def telemetry(self) -> dict[str, float]:
        metrics = {
            "sampler/is_adaptive": float(self.settings.rule == "adaptive"),
            "sampler/refreshes": float(self._refreshes),
            "sampler/conditions": float(self._count),
            "sampler/intended_entropy": entropy(self._probabilities),
            "sampler/intended_ess": effective_sample_size(self._probabilities),
            "sampler/intended_mass_min": float(self._probabilities.min()),
            "sampler/intended_mass_max": float(self._probabilities.max()),
            "sampler/mastered_conditions": float(
                np.count_nonzero(
                    np.nan_to_num(self._competence, nan=-1.0)
                    >= self.settings.mastery_threshold
                )
            ),
            "sampler/measured_conditions": float(
                np.count_nonzero(~np.isnan(self._competence))
            ),
            "sampler/has_closed_window": float(self._has_closed_window),
            "sampler/current_window_updates": float(self._window_updates),
            "sampler/closed_window_updates": float(self._last_window_updates),
        }
        windows = (
            (
                "current",
                self._episodes,
                self._assignments,
                self._reset_exposures,
            ),
            (
                "closed",
                self._last_episodes,
                self._last_assignments,
                self._last_reset_exposures,
            ),
        )
        masses = {}
        for window, episodes, assignments, reset_exposures in windows:
            metrics[f"sampler/{window}_completed_episodes"] = float(
                episodes.sum()
            )
            metrics[f"sampler/{window}_sampled_assignments"] = float(
                assignments.sum()
            )
            metrics[f"sampler/{window}_reset_exposures"] = float(
                reset_exposures.sum()
            )
            for measure, counts in (
                ("completed_episode", episodes),
                ("assignment", assignments),
                ("reset_exposure", reset_exposures),
            ):
                mass = self._mass(counts)
                masses[(window, measure)] = mass
                if np.isfinite(mass).all():
                    prefix = f"sampler/{window}_{measure}"
                    metrics[f"{prefix}_entropy"] = entropy(mass)
                    metrics[f"{prefix}_ess"] = effective_sample_size(mass)
        for index, name in enumerate(self.names):
            token = metric_token(name)
            metrics[f"sampler_q/{token}"] = float(self._probabilities[index])
            if not math.isnan(self._competence[index]):
                metrics[f"sampler_competence/{token}"] = float(
                    self._competence[index]
                )
            for (window, measure), mass in masses.items():
                if np.isfinite(mass[index]):
                    metrics[
                        f"sampler_{measure}_{window}/{token}"
                    ] = float(mass[index])
        for field, prefix in (
            ("family", "sampler_family"),
            ("branch_depth", "sampler_depth"),
        ):
            for value, mass in self._label_mass(
                self._probabilities, field
            ).items():
                metrics[f"{prefix}_q/{metric_token(value)}"] = mass
            for (window, measure), distribution in masses.items():
                if np.isfinite(distribution).all():
                    for value, mass in self._label_mass(
                        distribution, field
                    ).items():
                        metrics[
                            f"{prefix}_{measure}_{window}/"
                            f"{metric_token(value)}"
                        ] = mass
        return metrics

    def receipt(self) -> dict:
        def window_receipt(
            updates: int,
            episodes: np.ndarray,
            assignments: np.ndarray,
            reset_exposures: np.ndarray,
        ) -> dict:
            def mass(counts: np.ndarray) -> list[float | None]:
                values = self._mass(counts)
                return [
                    None if math.isnan(value) else float(value)
                    for value in values
                ]

            return {
                "updates": updates,
                "completed_episode_count": episodes.tolist(),
                "completed_episode_mass": mass(episodes),
                "sampled_assignment_count": assignments.tolist(),
                "sampled_assignment_mass": mass(assignments),
                "reset_exposure_count": reset_exposures.tolist(),
                "reset_exposure_mass": mass(reset_exposures),
            }

        return {
            "schema": "terra_pooled_condition_sampler_v2",
            "rule": self.settings.rule,
            "settings": {
                field: getattr(self.settings, field)
                for field in self.settings.__dataclass_fields__
            },
            "conditions": list(self.names),
            "labels": self.labels,
            "maps_per_condition": list(self.maps_per_condition),
            "refreshes": self._refreshes,
            "intended_mass": [float(value) for value in self._probabilities],
            "windows": {
                "current": window_receipt(
                    self._window_updates,
                    self._episodes,
                    self._assignments,
                    self._reset_exposures,
                ),
                "closed": (
                    window_receipt(
                        self._last_window_updates,
                        self._last_episodes,
                        self._last_assignments,
                        self._last_reset_exposures,
                    )
                    if self._has_closed_window
                    else None
                ),
            },
            "competence": [
                None if math.isnan(value) else float(value)
                for value in self._competence
            ],
        }
