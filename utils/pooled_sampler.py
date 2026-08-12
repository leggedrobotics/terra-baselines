"""One host-side probability distribution over accepted map conditions."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

RULES = (
    "uniform",
    "fixed",
    "adaptive",
    "continuous_banded_v1",
    "continuous_banded_v2",
    "continuous_banded_v3",
)
CONTINUOUS_RULES = (
    "continuous_banded_v1",
    "continuous_banded_v2",
    "continuous_banded_v3",
)
STATE_SCHEMA = "terra_pooled_condition_sampler_state_v2"
CONTINUOUS_STATE_SCHEMA = "terra_continuous_banded_sampler_state_v1"
CONTINUOUS_RECEIPT_SCHEMA = "terra_continuous_banded_sampler_v1"
FAMILIES = ("foundation", "trench")
ALL_FAMILY_FLOOR_MASS = 0.10
ACTIVE_DEPTH_MASS = 0.75
NEXT_DEPTH_MASS = 0.15
FRONTIER_MASS = 0.90
DEPTH_PRIORITY_BASE = 2.0
CONTINUOUS_MAX_MASS = 0.15
DEMOTION_THRESHOLD = 0.65


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
        if self.rule in CONTINUOUS_RULES and (
            self.update_interval != 150
            or self.mastery_threshold != 0.80
            or self.min_episodes != 32
            or self.competence_ema != 0.30
            or self.max_mass != CONTINUOUS_MAX_MASS
        ):
            raise ValueError(
                "continuous_banded samplers freeze interval=150, mastery=0.80, "
                "min_episodes=32, exact-success EMA alpha=0.30, and "
                f"max_mass={CONTINUOUS_MAX_MASS}"
            )


class PooledConditionSampler:
    """One host-side sampler for uniform, fixed, adaptive, or banded training.

    The adaptive arm uses

        q = uniform_floor * Uniform + (1 - uniform_floor) * Frontier.

    ``Frontier`` is a softmax over measured completion of conditions below the
    mastery threshold. Thus a condition that remains at zero cannot monopolize
    training, while the highest-competence unsolved conditions receive more
    exposure. Mastered conditions remain in the uniform floor and re-enter
    automatically if their EMA falls below the threshold.

    ``continuous_banded_v1`` instead keeps a 10% per-family floor across every
    condition, puts 75% on the family's shallowest incomplete depth, and 15%
    on the next depth. Foundation and trench each retain exactly half the
    population. Only exact completed training episodes update its mastery EMA.

    ``continuous_banded_v2`` removes the family-level depth gate that let a
    few stubborn conditions pin their whole family: the 90% frontier mass is
    spread over every unmastered condition, weighted
    ``DEPTH_PRIORITY_BASE ** (2 - depth)`` so shallow work still leads without
    starving any unmastered condition, and any condition with an eligible
    window may graduate regardless of depth. A v2 sampler may resume a v1
    checkpoint: mastery state carries over and the probability vector is
    recomputed under the v2 rule.

    ``continuous_banded_v3`` is v2 plus one per-condition mass cap. v2 keeps
    each family pinned at half the population, so a nearly-mastered family
    funnels its whole half onto its last unmastered condition: at u13.5k of
    the reward_v2_scratch run one cell held 45.2% of assignment mass. v3
    caps any condition at ``settings.max_mass`` and redistributes the excess
    proportionally over the uncapped conditions, ignoring family boundaries
    when it does so. While no condition exceeds the cap, v3 and v2 are the
    same distribution. Graduation, demotion, windows, and refresh boundaries
    are identical to v2, and a v3 sampler may resume a v1 or v2 checkpoint
    under a one-way migration taken at an empty window boundary.
    """

    def __init__(
        self,
        names: list[str],
        settings: SamplerSettings,
        *,
        maps_per_condition: list[int] | None = None,
        labels: dict[str, dict[str, object]] | None = None,
    ):
        if not names:
            raise ValueError("pooled sampler needs at least one condition")
        if len(names) != len(set(names)):
            raise ValueError("pooled sampler needs exactly one level per condition")
        self.names = tuple(names)
        self.settings = settings
        self._index = {name: index for index, name in enumerate(self.names)}
        self._count = len(self.names)
        if (
            settings.rule in ("adaptive", "continuous_banded_v3")
            and settings.max_mass * self._count < 1.0
        ):
            raise ValueError(
                f"{settings.rule} max_mass={settings.max_mass} is infeasible "
                f"for {self._count} conditions"
            )
        self.maps_per_condition = tuple(maps_per_condition or [1] * self._count)
        if len(self.maps_per_condition) != self._count:
            raise ValueError("maps_per_condition must match the condition count")
        self.labels = deepcopy(labels or {})
        raw_weights = [
            self.labels.get(name, {}).get("sampling_weight") for name in self.names
        ]
        if settings.rule == "fixed":
            if any(
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in raw_weights
            ):
                raise ValueError(
                    "sampling_weight labels must be finite and positive for every "
                    "condition"
                )
            weights = np.asarray(raw_weights, dtype=np.float64)
            self._uniform = weights / weights.sum()
        else:
            if any(value is not None for value in raw_weights):
                raise ValueError("sampling_weight labels require rule='fixed'")
            self._uniform = np.full(self._count, 1.0 / self._count, dtype=np.float64)
        self._depths = np.full(self._count, -1, dtype=np.int64)
        self._families = np.full(self._count, "", dtype=object)
        self._mastered = np.zeros(self._count, dtype=bool)
        if settings.rule in CONTINUOUS_RULES:
            for index, name in enumerate(self.names):
                label = self.labels.get(name, {})
                family = label.get("family")
                depth = label.get("curriculum_depth")
                if family not in FAMILIES:
                    raise ValueError(
                        f"continuous_banded condition {name!r} has invalid family"
                    )
                if isinstance(depth, bool) or depth not in (0, 1, 2):
                    raise ValueError(
                        f"continuous_banded condition {name!r} needs depth 0, 1, or 2"
                    )
                self._families[index] = family
                self._depths[index] = int(depth)
            for family in FAMILIES:
                family_mask = self._families == family
                for depth in range(3):
                    if not np.any(family_mask & (self._depths == depth)):
                        raise ValueError(
                            f"continuous_banded graph lacks {family} depth {depth}"
                        )
                if int(np.sum(family_mask & (self._depths == 0))) != 1:
                    raise ValueError(
                        f"continuous_banded {family} depth 0 must be a singleton"
                    )
        self._probabilities = (
            self._distribution_for_rule(settings.rule)
            if settings.rule in CONTINUOUS_RULES
            else self._uniform.copy()
        )
        self._competence = np.full(self._count, np.nan, dtype=np.float64)
        self._episodes = np.zeros(self._count, dtype=np.int64)
        self._completion_sum = np.zeros(self._count, dtype=np.float64)
        self._last_episodes = np.zeros(self._count, dtype=np.int64)
        self._last_completion_sum = np.zeros(self._count, dtype=np.float64)
        self._assignments = np.zeros(self._count, dtype=np.int64)
        self._last_assignments = np.zeros(self._count, dtype=np.int64)
        self._reset_exposures = np.zeros(self._count, dtype=np.int64)
        self._last_reset_exposures = np.zeros(self._count, dtype=np.int64)
        self._transition_exposures = np.zeros(self._count, dtype=np.int64)
        self._last_transition_exposures = np.zeros(self._count, dtype=np.int64)
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
            transition_exposures: np.ndarray,
            updates: int,
        ) -> dict:
            result = {
                "completed_episode_count": episodes.tolist(),
                "sampled_assignment_count": assignments.tolist(),
                "reset_exposure_count": reset_exposures.tolist(),
                "transition_exposure_count": transition_exposures.tolist(),
                "updates": int(updates),
            }
            if self.settings.rule in CONTINUOUS_RULES:
                if not np.all(completion_sum == np.floor(completion_sum)):
                    raise ValueError(
                        "continuous_banded task_done_count is not integral"
                    )
                result["task_done_count"] = completion_sum.astype(np.int64).tolist()
            else:
                result["completion_sum"] = completion_sum.tolist()
            return result

        result = {
            "schema": (
                CONTINUOUS_STATE_SCHEMA
                if self.settings.rule in CONTINUOUS_RULES
                else STATE_SCHEMA
            ),
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
                self._transition_exposures,
                self._window_updates,
            ),
            "closed_window": window(
                self._last_episodes,
                self._last_completion_sum,
                self._last_assignments,
                self._last_reset_exposures,
                self._last_transition_exposures,
                self._last_window_updates,
            ),
            "refresh": {
                "has_closed_window": bool(self._has_closed_window),
                "last_refresh_update": self._last_refresh_update,
                "refreshes": int(self._refreshes),
            },
            "numpy_rng": deepcopy(self._rng.bit_generator.state),
        }
        if self.settings.rule in CONTINUOUS_RULES:
            result["mastery"] = {
                "mastered": self._mastered.tolist(),
                "family": self._families.tolist(),
                "depth": self._depths.tolist(),
            }
        return result

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
        if self.settings.rule in CONTINUOUS_RULES:
            top_keys.add("mastery")
        if not isinstance(state, dict) or set(state) != top_keys:
            observed = (
                sorted(state) if isinstance(state, dict) else type(state).__name__
            )
            raise ValueError(
                "pooled sampler state fields do not match the active schema: "
                f"observed={observed}"
            )
        expected_schema = (
            CONTINUOUS_STATE_SCHEMA
            if self.settings.rule in CONTINUOUS_RULES
            else STATE_SCHEMA
        )
        if state["schema"] != expected_schema:
            raise ValueError(
                "unsupported pooled sampler state schema: " f"{state['schema']!r}"
            )

        expected_settings = {
            field: getattr(self.settings, field)
            for field in self.settings.__dataclass_fields__
        }
        stored_rule = (
            state["settings"].get("rule")
            if isinstance(state["settings"], dict)
            else None
        )
        migrating = (
            self.settings.rule in CONTINUOUS_RULES
            and stored_rule in CONTINUOUS_RULES
            and CONTINUOUS_RULES.index(stored_rule)
            < CONTINUOUS_RULES.index(self.settings.rule)
        )
        if migrating:
            # One-way rule migration: a newer continuous sampler may resume an
            # older continuous checkpoint (v1->v2, v2->v3, v1->v3), never the
            # reverse. Every other setting must still match exactly.
            expected_settings = {**expected_settings, "rule": stored_rule}
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

        def float_vector(value, label: str, *, allow_none: bool = False) -> np.ndarray:
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
                    isinstance(item, bool) or not isinstance(item, (int, np.integer))
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
        if self.settings.rule == "fixed" and not np.allclose(
            probabilities,
            self._uniform,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "fixed sampler checkpoint probabilities changed from frozen weights"
            )
        competence = float_vector(state["competence"], "competence", allow_none=True)
        measured_competence = competence[~np.isnan(competence)]
        if np.any((measured_competence < 0.0) | (measured_competence > 1.0)):
            raise ValueError("pooled sampler competence must be in [0, 1]")

        window_keys = {
            "completed_episode_count",
            "sampled_assignment_count",
            "reset_exposure_count",
            "transition_exposure_count",
            "updates",
        }
        sum_key = (
            "task_done_count"
            if self.settings.rule in CONTINUOUS_RULES
            else "completion_sum"
        )
        window_keys.add(sum_key)

        def restore_window(value, label: str):
            if not isinstance(value, dict) or set(value) != window_keys:
                raise ValueError(
                    f"pooled sampler {label} fields do not match the active schema"
                )
            episodes = count_vector(
                value["completed_episode_count"],
                f"{label}.completed_episode_count",
            )
            completion_sum = (
                count_vector(value[sum_key], f"{label}.{sum_key}").astype(np.float64)
                if self.settings.rule in CONTINUOUS_RULES
                else float_vector(value[sum_key], f"{label}.{sum_key}")
            )
            if np.any(completion_sum < 0.0):
                raise ValueError(
                    f"pooled sampler {label}.{sum_key} must be nonnegative"
                )
            if self.settings.rule in CONTINUOUS_RULES and np.any(
                completion_sum > episodes
            ):
                raise ValueError(
                    f"pooled sampler {label}.task_done_count exceeds episodes"
                )
            assignments = count_vector(
                value["sampled_assignment_count"],
                f"{label}.sampled_assignment_count",
            )
            reset_exposures = count_vector(
                value["reset_exposure_count"],
                f"{label}.reset_exposure_count",
            )
            transition_exposures = count_vector(
                value["transition_exposure_count"],
                f"{label}.transition_exposure_count",
            )
            updates = nonnegative_int(value["updates"], f"{label}.updates")
            return (
                episodes,
                completion_sum,
                assignments,
                reset_exposures,
                transition_exposures,
                updates,
            )

        current = restore_window(state["current_window"], "current_window")
        closed = restore_window(state["closed_window"], "closed_window")
        if migrating and (
            any(vector.any() for vector in current[:5]) or current[5] != 0
        ):
            # Windows drive graduation, so a window may never mix exposure
            # taken under two different rules: migrate at a refresh boundary.
            raise ValueError(
                "continuous_banded rule migration requires an empty current "
                "window; migrate at a refresh boundary, not mid-window"
            )

        refresh = state["refresh"]
        refresh_keys = {"has_closed_window", "last_refresh_update", "refreshes"}
        if not isinstance(refresh, dict) or set(refresh) != refresh_keys:
            raise ValueError("pooled sampler refresh fields do not match the v2 schema")
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
            or closed[4].any()
            or closed[5] != 0
        ):
            raise ValueError(
                "pooled sampler has no closed window but closed-window state is nonzero"
            )

        rng_state = deepcopy(state["numpy_rng"])
        if not isinstance(rng_state, dict) or rng_state.get(
            "bit_generator"
        ) != self._rng.bit_generator.state.get("bit_generator"):
            raise ValueError("pooled sampler NumPy RNG type changed across resume")
        restored_rng = np.random.default_rng()
        try:
            restored_rng.bit_generator.state = rng_state
        except (TypeError, ValueError) as error:
            raise ValueError("pooled sampler NumPy RNG state is invalid") from error

        mastered = self._mastered
        if self.settings.rule in CONTINUOUS_RULES:
            mastery = state["mastery"]
            mastery_keys = {
                "mastered",
                "family",
                "depth",
            }
            if not isinstance(mastery, dict) or set(mastery) != mastery_keys:
                raise ValueError(
                    "continuous_banded mastery fields do not match the v1 schema"
                )
            if (
                not isinstance(mastery["mastered"], list)
                or len(mastery["mastered"]) != self._count
                or any(not isinstance(value, bool) for value in mastery["mastered"])
            ):
                raise ValueError("continuous_banded mastered must be boolean")
            mastered = np.asarray(mastery["mastered"], dtype=bool)
            if (
                mastery["family"] != self._families.tolist()
                or mastery["depth"] != self._depths.tolist()
            ):
                raise ValueError("continuous_banded graph changed across resume")
            expected_probabilities = self._distribution_for_rule(
                stored_rule if migrating else self.settings.rule, mastered
            )
            if not np.allclose(
                probabilities, expected_probabilities, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    "continuous_banded checkpoint probabilities disagree with mastery"
                )
            if migrating:
                probabilities = self._distribution_for_rule(
                    self.settings.rule, mastered
                )

        self._probabilities = probabilities
        self._competence = competence
        (
            self._episodes,
            self._completion_sum,
            self._assignments,
            self._reset_exposures,
            self._transition_exposures,
            self._window_updates,
        ) = current
        (
            self._last_episodes,
            self._last_completion_sum,
            self._last_assignments,
            self._last_reset_exposures,
            self._last_transition_exposures,
            self._last_window_updates,
        ) = closed
        self._has_closed_window = refresh["has_closed_window"]
        self._last_refresh_update = last_refresh_update
        self._refreshes = refreshes
        self._rng = restored_rng
        self._mastered = mastered

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
            raise ValueError("reset exposure counts must match the condition count")
        if values.dtype.kind not in "iub" or np.any(values < 0):
            raise ValueError("reset exposure counts must be nonnegative integers")
        self._reset_exposures += values.astype(np.int64)

    def observe_transition_exposures(self, counts: np.ndarray) -> None:
        """Count actual policy transitions under each map condition."""
        values = np.asarray(counts)
        if values.shape != (self._count,):
            raise ValueError(
                "transition exposure counts must match the condition count"
            )
        if values.dtype.kind not in "iub" or np.any(values < 0):
            raise ValueError("transition exposure counts must be nonnegative integers")
        self._transition_exposures += values.astype(np.int64)

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
        if self.settings.rule in CONTINUOUS_RULES and payload.get("schema") != (
            "terra_training_episode_aggregate_v2"
        ):
            raise ValueError(
                "continuous_banded accepts only training episode aggregates; "
                "held-out evaluation must never feed the sampler"
            )
        for row in payload.get("groups", ()):
            index = self._index.get(row.get("primary_cell"))
            if index is None:
                continue
            episodes = int(row["episode_count"])
            self._episodes[index] += episodes
            self._completion_sum[index] += float(
                row[
                    (
                        "task_done_count"
                        if self.settings.rule in CONTINUOUS_RULES
                        else "combined_completion_sum"
                    )
                ]
            )
        self._window_updates += 1

    def refresh(self, update_index: int) -> None:
        if self.settings.rule in CONTINUOUS_RULES:
            if self._last_refresh_update is None or not self.due(update_index):
                raise ValueError(
                    "continuous_banded refresh must occur at its fixed update boundary"
                )
            if (
                update_index - self._last_refresh_update
                != self.settings.update_interval
            ):
                raise ValueError(
                    "continuous_banded cannot skip a fixed refresh boundary"
                )
            self._refresh_continuous_mastery()
            self._probabilities = self._distribution_for_rule(self.settings.rule)
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
        self._last_transition_exposures = self._transition_exposures
        self._episodes = np.zeros(self._count, dtype=np.int64)
        self._completion_sum = np.zeros(self._count, dtype=np.float64)
        self._assignments = np.zeros(self._count, dtype=np.int64)
        self._reset_exposures = np.zeros(self._count, dtype=np.int64)
        self._transition_exposures = np.zeros(self._count, dtype=np.int64)
        self._last_window_updates = self._window_updates
        self._window_updates = 0
        self._has_closed_window = True
        self._last_refresh_update = int(update_index)
        self._refreshes += 1

    def _active_depth(
        self, family: str, mastered: np.ndarray | None = None
    ) -> int | None:
        state = self._mastered if mastered is None else mastered
        unmastered = (self._families == family) & ~state
        if not unmastered.any():
            return None
        return int(self._depths[unmastered].min())

    def _refresh_continuous_mastery(self) -> None:
        active_depths = {family: self._active_depth(family) for family in FAMILIES}
        eligible = self._episodes >= self.settings.min_episodes
        for index in range(self._count):
            if not eligible[index]:
                continue
            episodes = int(self._episodes[index])
            observed = float(self._completion_sum[index]) / episodes
            if not 0.0 <= observed <= 1.0:
                raise ValueError(
                    "continuous_banded exact success rate is outside [0, 1]"
                )
            previous = self._competence[index]
            alpha = self.settings.competence_ema
            self._competence[index] = (
                observed
                if math.isnan(previous)
                else (1.0 - alpha) * previous + alpha * observed
            )
            if self._mastered[index]:
                if self._competence[index] < DEMOTION_THRESHOLD:
                    self._mastered[index] = False
                continue

            family = str(self._families[index])
            if (
                self.settings.rule == "continuous_banded_v1"
                and self._depths[index] != active_depths[family]
            ):
                # v1 preview is observable but remains locked until its
                # family reaches this depth. v2/v3 graduate any eligible cell.
                continue
            if self._competence[index] >= self.settings.mastery_threshold:
                self._mastered[index] = True

    def _continuous_distribution(
        self, mastered: np.ndarray | None = None
    ) -> np.ndarray:
        state = self._mastered if mastered is None else mastered
        probabilities = np.zeros(self._count, dtype=np.float64)
        for family in FAMILIES:
            family_mask = self._families == family
            active_depth = self._active_depth(family, state)
            if active_depth is None:
                probabilities[family_mask] = 0.5 / int(family_mask.sum())
                continue
            active = family_mask & (self._depths == active_depth)
            next_depth = active_depth + 1
            next_band = family_mask & (self._depths == next_depth)
            probabilities[family_mask] += (
                0.5 * ALL_FAMILY_FLOOR_MASS / int(family_mask.sum())
            )
            if next_band.any():
                active_mass = ACTIVE_DEPTH_MASS
                next_mass = NEXT_DEPTH_MASS
            else:
                active_mass = ACTIVE_DEPTH_MASS + NEXT_DEPTH_MASS
                next_mass = 0.0
            probabilities[active] += 0.5 * active_mass / int(active.sum())
            if next_mass:
                probabilities[next_band] += 0.5 * next_mass / int(next_band.sum())
        if np.any(probabilities <= 0.0) or not np.isclose(
            probabilities.sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "continuous_banded must retain positive support and unit mass"
            )
        return probabilities

    def _distribution_for_rule(
        self, rule: str, mastered: np.ndarray | None = None
    ) -> np.ndarray:
        if rule == "continuous_banded_v1":
            return self._continuous_distribution(mastered)
        if rule == "continuous_banded_v2":
            return self._continuous_distribution_v2(mastered)
        if rule == "continuous_banded_v3":
            return self._continuous_distribution_v3(mastered)
        raise ValueError(f"no banded distribution for rule {rule!r}")

    def _continuous_distribution_v2(
        self, mastered: np.ndarray | None = None
    ) -> np.ndarray:
        """Per-condition graduation: no family-level depth gate.

        Each family keeps the 10% all-condition floor and spends the
        remaining 90% on its unmastered conditions, weighted
        ``DEPTH_PRIORITY_BASE ** (2 - depth)`` so shallow work leads without
        starving any unmastered condition.
        """
        state = self._mastered if mastered is None else mastered
        probabilities = np.zeros(self._count, dtype=np.float64)
        for family in FAMILIES:
            family_mask = self._families == family
            frontier = family_mask & ~state
            if not frontier.any():
                probabilities[family_mask] = 0.5 / int(family_mask.sum())
                continue
            probabilities[family_mask] += (
                0.5 * ALL_FAMILY_FLOOR_MASS / int(family_mask.sum())
            )
            weights = np.zeros(self._count, dtype=np.float64)
            weights[frontier] = DEPTH_PRIORITY_BASE ** (
                2 - self._depths[frontier].astype(np.float64)
            )
            probabilities += 0.5 * FRONTIER_MASS * weights / weights.sum()
        if np.any(probabilities <= 0.0) or not np.isclose(
            probabilities.sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "continuous_banded must retain positive support and unit mass"
            )
        return probabilities

    def _continuous_distribution_v3(
        self, mastered: np.ndarray | None = None
    ) -> np.ndarray:
        """v2 under one per-condition mass cap.

        The v2 family halves stay: they keep both families in every window
        and bound how much of the population a family of unlearnable
        stragglers can collectively absorb. Only the runaway single condition
        is corrected - anything above ``settings.max_mass`` is water-filled
        back over the uncapped conditions in proportion to their v2 mass,
        which crosses family boundaries because the excess belongs to the
        population, not to the family that produced it. Below the cap v3 *is*
        v2, so the rule only acts in the regime it was added for.

        ``_cap_distribution`` raises rather than returning an over-cap vector
        when the cap is infeasible for the graph (``max_mass * count < 1``);
        that combination is rejected at construction.
        """
        probabilities = self._continuous_distribution_v2(mastered)
        if probabilities.max() <= self.settings.max_mass:
            return probabilities
        return _cap_distribution(probabilities, self.settings.max_mass)

    def _adaptive_distribution(self) -> np.ndarray:
        competence = np.where(np.isnan(self._competence), 0.0, self._competence)
        unmastered = competence < self.settings.mastery_threshold
        if not unmastered.any():
            return self._uniform.copy()
        scores = np.full(self._count, -np.inf, dtype=np.float64)
        scores[unmastered] = competence[unmastered] / self.settings.temperature
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
            transition_exposures: np.ndarray,
        ) -> dict:
            def mass(counts: np.ndarray) -> list[float | None]:
                values = self._mass(counts)
                return [None if math.isnan(value) else float(value) for value in values]

            return {
                "updates": updates,
                "completed_episode_count": episodes.tolist(),
                "completed_episode_mass": mass(episodes),
                "sampled_assignment_count": assignments.tolist(),
                "sampled_assignment_mass": mass(assignments),
                "reset_exposure_count": reset_exposures.tolist(),
                "reset_exposure_mass": mass(reset_exposures),
                "transition_exposure_count": transition_exposures.tolist(),
                "transition_exposure_mass": mass(transition_exposures),
            }

        result = {
            "schema": (
                CONTINUOUS_RECEIPT_SCHEMA
                if self.settings.rule in CONTINUOUS_RULES
                else "terra_pooled_condition_sampler_v2"
            ),
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
                    self._transition_exposures,
                ),
                "closed": (
                    window_receipt(
                        self._last_window_updates,
                        self._last_episodes,
                        self._last_assignments,
                        self._last_reset_exposures,
                        self._last_transition_exposures,
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
        if self.settings.rule in CONTINUOUS_RULES:
            role_by_condition = []
            family_active = {family: self._active_depth(family) for family in FAMILIES}
            for index in range(self._count):
                if self.settings.rule != "continuous_banded_v1":
                    # v2/v3 have no depth band: a condition is either on the
                    # pooled frontier or in mastered replay.
                    role_by_condition.append(
                        "replay" if self._mastered[index] else "frontier"
                    )
                    continue
                family = str(self._families[index])
                active_depth = family_active[family]
                if active_depth is None:
                    role = "all_mastered"
                elif self._depths[index] == active_depth:
                    role = "active"
                elif self._depths[index] == active_depth + 1:
                    role = "next"
                else:
                    role = "floor_only"
                role_by_condition.append(role)
            result["mastery"] = {
                "mastered": self._mastered.tolist(),
                "family_active_depth": family_active,
                "role": role_by_condition,
                "exact_success_ema": result["competence"],
            }
        return result
