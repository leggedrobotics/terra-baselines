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
    "continuous_banded_v3",
)
CONTINUOUS_RULES = ("continuous_banded_v3",)
LEGACY_SOURCE_RULE = "continuous_banded_v2"
STATE_SCHEMA = "terra_pooled_condition_sampler_state_v2"
CONTINUOUS_STATE_SCHEMA = "terra_continuous_banded_sampler_state_v1"
CONTINUOUS_RECEIPT_SCHEMA = "terra_continuous_banded_sampler_v1"
FAMILIES = ("foundation", "trench")
ALL_FAMILY_FLOOR_MASS = 0.10
FRONTIER_MASS = 0.90
DEPTH_PRIORITY_BASE = 2.0
CONTINUOUS_MAX_MASS = 0.15
OPEN_CONDITION_MASS = 0.80
MASTERED_REPLAY_MASS = 0.20
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

    ``continuous_banded_v2`` remains only as the tag on the selected
    update-14000 source checkpoint. Its private distribution is used by the
    one-off offline materializer to validate that source.

    ``continuous_banded_v3`` removes the semantic family quota. It assigns
    80% globally over open conditions using v2's depth weights and 20%
    uniformly over mastered replay, then applies the same per-condition cap.
    Family labels remain diagnostics only. The materializer explicitly clears
    the source's current window while producing a native v3 checkpoint; normal
    training accepts only native v3 state.
    """

    def __init__(
        self,
        names: list[str],
        settings: SamplerSettings,
        *,
        maps_per_condition: list[int] | None = None,
        labels: dict[str, dict[str, object]] | None = None,
        allow_sparse_depths: bool = False,
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
        self._allow_sparse_depths = bool(allow_sparse_depths)
        if self._allow_sparse_depths and settings.rule not in CONTINUOUS_RULES:
            raise ValueError(
                "allow_sparse_depths applies only to continuous_banded samplers"
            )
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
                if int(np.sum(family_mask & (self._depths == 0))) != 1:
                    raise ValueError(
                        f"continuous_banded {family} depth 0 must be a singleton"
                    )
                if not self._allow_sparse_depths:
                    for depth in range(3):
                        if not np.any(family_mask & (self._depths == depth)):
                            raise ValueError(
                                "continuous_banded graph lacks "
                                f"{family} depth {depth}"
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

    def restore_state_dict(
        self, state: dict, *, clear_window_on_migration: bool = False
    ) -> None:
        """Restore a sampler checkpoint after validating its full contract.

        ``clear_window_on_migration`` applies only to a one-way continuous rule
        migration whose checkpoint sits mid-window: instead of refusing, discard
        the partial window's exposure so no window mixes two rules. It is inert
        for a same-rule resume, which always keeps the partial window.
        """

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
            self.settings.rule == "continuous_banded_v3"
            and stored_rule == LEGACY_SOURCE_RULE
        )
        if migrating:
            # The selected update-14000 source is v2; its one-off materializer
            # migrates that state to v3. Every other setting must match exactly.
            if not clear_window_on_migration:
                raise ValueError(
                    "legacy v2 state is accepted only by the explicit offline "
                    "v3 checkpoint materializer"
                )
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
        if migrating:
            # Windows drive graduation, so a window may never mix exposure
            # taken under two different rules: migrate at a refresh boundary,
            # or discard the partial window explicitly. Discarding costs the
            # exposure taken since the last boundary and nothing else: mastery,
            # competence, the closed window, and the refresh grid all survive,
            # so the next boundary still lands on the checkpoint's schedule.
            current = (
                np.zeros(self._count, dtype=np.int64),
                np.zeros(self._count, dtype=np.float64),
                np.zeros(self._count, dtype=np.int64),
                np.zeros(self._count, dtype=np.int64),
                np.zeros(self._count, dtype=np.int64),
                0,
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

    def sample_levels_for_reset_tiers(
        self,
        reset_tiers: np.ndarray,
        supported_levels: np.ndarray,
    ) -> np.ndarray:
        """Sample full lanes normally and partial lanes only from valid sidecars."""
        tiers = np.asarray(reset_tiers)
        support = np.asarray(supported_levels)
        if tiers.dtype.kind not in "iu" or np.any((tiers < 0) | (tiers > 3)):
            raise ValueError("reset_tiers must contain integer tiers in [0, 3]")
        if support.shape != (4, self._count) or support.dtype.kind != "b":
            raise ValueError(
                "supported_levels must be boolean [4, condition_count]"
            )
        if not np.all(support[1:] == support[1]):
            raise ValueError(
                "partial reset tiers 1-3 must share one common condition support"
            )
        samples = self._rng.choice(
            self._count,
            size=tiers.size,
            p=self._probabilities,
        ).astype(np.int32)
        flat_tiers = tiers.reshape(-1)
        for tier in (1, 2, 3):
            lanes = np.flatnonzero(flat_tiers == tier)
            if not lanes.size:
                continue
            probabilities = np.where(
                support[tier], self._probabilities, 0.0
            ).astype(np.float64)
            mass = float(probabilities.sum())
            if mass <= 0.0:
                raise ValueError(f"partial reset tier {tier} has no supported condition")
            probabilities /= mass
            samples[lanes] = self._rng.choice(
                self._count,
                size=lanes.size,
                p=probabilities,
            )
        self._assignments += np.bincount(samples, minlength=self._count)
        return samples.reshape(tiers.shape)

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

    def observe_exact_episode_counts(
        self,
        episode_counts: np.ndarray,
        task_done_counts: np.ndarray,
    ) -> None:
        """Add full-start exact outcomes without mixing in shaped reset episodes."""
        if self.settings.rule not in CONTINUOUS_RULES:
            raise ValueError(
                "exact episode counts are supported only by continuous_banded"
            )
        episodes = np.asarray(episode_counts)
        successes = np.asarray(task_done_counts)
        for label, values in (
            ("episode_counts", episodes),
            ("task_done_counts", successes),
        ):
            if values.shape != (self._count,):
                raise ValueError(f"{label} must match the condition count")
            if values.dtype.kind not in "iub" or np.any(values < 0):
                raise ValueError(f"{label} must contain nonnegative integers")
        if np.any(successes > episodes):
            raise ValueError("task_done_counts cannot exceed episode_counts")
        self._episodes += episodes.astype(np.int64)
        self._completion_sum += successes.astype(np.float64)
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

    def _refresh_continuous_mastery(self) -> None:
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

            if self._competence[index] >= self.settings.mastery_threshold:
                self._mastered[index] = True

    def _distribution_for_rule(
        self, rule: str, mastered: np.ndarray | None = None
    ) -> np.ndarray:
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
        """Global open frontier plus mastered replay, under one condition cap."""
        state = self._mastered if mastered is None else mastered
        open_mask = ~state
        if not open_mask.any():
            return self._uniform.copy()

        probabilities = np.zeros(self._count, dtype=np.float64)
        open_weights = np.zeros(self._count, dtype=np.float64)
        open_weights[open_mask] = DEPTH_PRIORITY_BASE ** (
            2 - self._depths[open_mask].astype(np.float64)
        )
        if state.any():
            probabilities[open_mask] = (
                OPEN_CONDITION_MASS
                * open_weights[open_mask]
                / open_weights[open_mask].sum()
            )
            probabilities[state] = MASTERED_REPLAY_MASS / int(state.sum())
        else:
            probabilities = open_weights / open_weights.sum()

        if probabilities.max() > self.settings.max_mass:
            probabilities = _cap_distribution(probabilities, self.settings.max_mass)
        if np.any(probabilities <= 0.0) or not np.isclose(
            probabilities.sum(), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "continuous_banded_v3 must retain positive support and unit mass"
            )
        return probabilities

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
            role_by_condition = [
                "replay" if mastered else "frontier" for mastered in self._mastered
            ]
            family_active_depth = {}
            for family in FAMILIES:
                open_in_family = (self._families == family) & ~self._mastered
                family_active_depth[family] = (
                    int(self._depths[open_in_family].min())
                    if open_in_family.any()
                    else None
                )
            result["mastery"] = {
                "mastered": self._mastered.tolist(),
                # Diagnostic only: v3 does not use family to allocate mass.
                "family_active_depth": family_active_depth,
                "role": role_by_condition,
                "exact_success_ema": result["competence"],
            }
        return result
