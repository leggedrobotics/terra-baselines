"""One host-side probability distribution over accepted map conditions."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


RULES = ("uniform", "adaptive")
STATE_SCHEMA = "terra_pooled_condition_sampler_state_v1"
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
        self.labels = deepcopy(labels or {})

        self._uniform = np.full(self._count, 1.0 / self._count, dtype=np.float64)
        self._probabilities = self._uniform.copy()
        self._competence = np.full(self._count, np.nan, dtype=np.float64)
        self._episodes = np.zeros(self._count, dtype=np.int64)
        self._completion_sum = np.zeros(self._count, dtype=np.float64)
        self._last_episodes = np.zeros(self._count, dtype=np.int64)
        self._last_completion_sum = np.zeros(self._count, dtype=np.float64)
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

    def state_dict(self) -> dict:
        """Return the exact mutable sampler state needed for a true resume."""

        def window(
            episodes: np.ndarray,
            completion_sum: np.ndarray,
            assignments: np.ndarray,
            reset_exposures: np.ndarray,
            updates: int,
        ) -> dict:
            return {
                "completed_episode_count": episodes.tolist(),
                "completion_sum": completion_sum.tolist(),
                "sampled_assignment_count": assignments.tolist(),
                "reset_exposure_count": reset_exposures.tolist(),
                "updates": int(updates),
            }

        return {
            "schema": STATE_SCHEMA,
            "conditions": list(self.names),
            "settings": {
                field: getattr(self.settings, field)
                for field in self.settings.__dataclass_fields__
            },
            "maps_per_condition": list(self.maps_per_condition),
            "labels": deepcopy(self.labels),
            "probabilities": self._probabilities.tolist(),
            "competence": [
                None if math.isnan(value) else float(value)
                for value in self._competence
            ],
            "current_window": window(
                self._episodes,
                self._completion_sum,
                self._assignments,
                self._reset_exposures,
                self._window_updates,
            ),
            "closed_window": window(
                self._last_episodes,
                self._last_completion_sum,
                self._last_assignments,
                self._last_reset_exposures,
                self._last_window_updates,
            ),
            "refresh": {
                "has_closed_window": bool(self._has_closed_window),
                "last_refresh_update": self._last_refresh_update,
                "refreshes": int(self._refreshes),
            },
            "numpy_rng": deepcopy(self._rng.bit_generator.state),
        }

    def restore_state_dict(self, state: dict) -> None:
        """Restore a sampler checkpoint after validating its full contract."""

        top_keys = {
            "schema",
            "conditions",
            "settings",
            "maps_per_condition",
            "labels",
            "probabilities",
            "competence",
            "current_window",
            "closed_window",
            "refresh",
            "numpy_rng",
        }
        if not isinstance(state, dict) or set(state) != top_keys:
            observed = sorted(state) if isinstance(state, dict) else type(state).__name__
            raise ValueError(
                "pooled sampler state fields do not match the v1 schema: "
                f"observed={observed}"
            )
        if state["schema"] != STATE_SCHEMA:
            raise ValueError(
                "unsupported pooled sampler state schema: "
                f"{state['schema']!r}"
            )

        expected_settings = {
            field: getattr(self.settings, field)
            for field in self.settings.__dataclass_fields__
        }
        contracts = (
            ("conditions", state["conditions"], list(self.names)),
            ("settings", state["settings"], expected_settings),
            (
                "maps_per_condition",
                state["maps_per_condition"],
                list(self.maps_per_condition),
            ),
            ("labels", state["labels"], self.labels),
        )
        for label, observed, expected in contracts:
            if observed != expected:
                raise ValueError(
                    f"pooled sampler {label} changed across resume: "
                    f"checkpoint={observed!r}, current={expected!r}"
                )

        def float_vector(
            value, label: str, *, allow_none: bool = False
        ) -> np.ndarray:
            if not isinstance(value, list) or len(value) != self._count:
                raise ValueError(
                    f"pooled sampler {label} must have {self._count} entries"
                )
            if allow_none:
                result = np.asarray(
                    [np.nan if item is None else item for item in value],
                    dtype=np.float64,
                )
                if np.any(~np.isfinite(result) & ~np.isnan(result)):
                    raise ValueError(f"pooled sampler {label} contains invalid values")
                return result
            result = np.asarray(value, dtype=np.float64)
            if not np.isfinite(result).all():
                raise ValueError(f"pooled sampler {label} must be finite")
            return result

        def count_vector(value, label: str) -> np.ndarray:
            if (
                not isinstance(value, list)
                or len(value) != self._count
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, np.integer))
                    for item in value
                )
            ):
                raise ValueError(
                    f"pooled sampler {label} must contain {self._count} integers"
                )
            result = np.asarray(value, dtype=np.int64)
            if np.any(result < 0):
                raise ValueError(f"pooled sampler {label} must be nonnegative")
            return result

        def nonnegative_int(value, label: str) -> int:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"pooled sampler {label} must be an integer")
            result = int(value)
            if result < 0:
                raise ValueError(f"pooled sampler {label} must be nonnegative")
            return result

        probabilities = float_vector(state["probabilities"], "probabilities")
        if np.any(probabilities < 0.0) or not np.isclose(
            probabilities.sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "pooled sampler probabilities must be nonnegative and sum to one"
            )
        competence = float_vector(
            state["competence"], "competence", allow_none=True
        )
        measured_competence = competence[~np.isnan(competence)]
        if np.any((measured_competence < 0.0) | (measured_competence > 1.0)):
            raise ValueError("pooled sampler competence must be in [0, 1]")

        window_keys = {
            "completed_episode_count",
            "completion_sum",
            "sampled_assignment_count",
            "reset_exposure_count",
            "updates",
        }

        def restore_window(value, label: str):
            if not isinstance(value, dict) or set(value) != window_keys:
                raise ValueError(
                    f"pooled sampler {label} fields do not match the v1 schema"
                )
            episodes = count_vector(
                value["completed_episode_count"],
                f"{label}.completed_episode_count",
            )
            completion_sum = float_vector(
                value["completion_sum"], f"{label}.completion_sum"
            )
            if np.any(completion_sum < 0.0):
                raise ValueError(
                    f"pooled sampler {label}.completion_sum must be nonnegative"
                )
            assignments = count_vector(
                value["sampled_assignment_count"],
                f"{label}.sampled_assignment_count",
            )
            reset_exposures = count_vector(
                value["reset_exposure_count"],
                f"{label}.reset_exposure_count",
            )
            updates = nonnegative_int(value["updates"], f"{label}.updates")
            return episodes, completion_sum, assignments, reset_exposures, updates

        current = restore_window(state["current_window"], "current_window")
        closed = restore_window(state["closed_window"], "closed_window")

        refresh = state["refresh"]
        refresh_keys = {"has_closed_window", "last_refresh_update", "refreshes"}
        if not isinstance(refresh, dict) or set(refresh) != refresh_keys:
            raise ValueError(
                "pooled sampler refresh fields do not match the v1 schema"
            )
        if not isinstance(refresh["has_closed_window"], bool):
            raise ValueError("pooled sampler has_closed_window must be boolean")
        last_refresh_update = refresh["last_refresh_update"]
        if last_refresh_update is not None:
            last_refresh_update = nonnegative_int(
                last_refresh_update, "last_refresh_update"
            )
        refreshes = nonnegative_int(refresh["refreshes"], "refreshes")
        if not refresh["has_closed_window"] and (
            closed[0].any()
            or closed[1].any()
            or closed[2].any()
            or closed[3].any()
            or closed[4] != 0
        ):
            raise ValueError(
                "pooled sampler has no closed window but closed-window state is nonzero"
            )

        rng_state = deepcopy(state["numpy_rng"])
        if (
            not isinstance(rng_state, dict)
            or rng_state.get("bit_generator")
            != self._rng.bit_generator.state.get("bit_generator")
        ):
            raise ValueError("pooled sampler NumPy RNG type changed across resume")
        restored_rng = np.random.default_rng()
        try:
            restored_rng.bit_generator.state = rng_state
        except (TypeError, ValueError) as error:
            raise ValueError("pooled sampler NumPy RNG state is invalid") from error

        self._probabilities = probabilities
        self._competence = competence
        (
            self._episodes,
            self._completion_sum,
            self._assignments,
            self._reset_exposures,
            self._window_updates,
        ) = current
        (
            self._last_episodes,
            self._last_completion_sum,
            self._last_assignments,
            self._last_reset_exposures,
            self._last_window_updates,
        ) = closed
        self._has_closed_window = refresh["has_closed_window"]
        self._last_refresh_update = last_refresh_update
        self._refreshes = refreshes
        self._rng = restored_rng

    @property
    def probabilities(self) -> np.ndarray:
        return self._probabilities.copy()

    @property
    def refreshes(self) -> int:
        return self._refreshes

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
        self._last_completion_sum = self._completion_sum
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
