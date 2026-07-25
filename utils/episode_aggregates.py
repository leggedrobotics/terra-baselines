"""Bounded, episode-complete training receipts for Terra PPO."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


NUM_ACTIONS = 8
MAX_AGENTS = 4
REWARD_COMPONENT_RTOL = 1e-5
TERMINATION_REASONS = (
    "task_done",
    "timeout",
    "task_done_and_timeout",
    "other",
)


class EpisodeStep(struct.PyTreeNode):
    done: jax.Array
    task_done: jax.Array
    timeout: jax.Array
    reward: jax.Array
    agent_rewards: jax.Array
    terminal_reward: jax.Array
    trench_reward: jax.Array
    existence_reward: jax.Array
    reward_normalizer: jax.Array
    action: jax.Array
    action_had_effect: jax.Array
    productive_workspace_cycle: jax.Array
    transition_mass_residual: jax.Array
    target_mutation: jax.Array
    obstacle_mutation: jax.Array
    dig_completion: jax.Array
    dump_purity: jax.Array
    dump_volume_completion: jax.Array
    combined_completion: jax.Array
    unloaded_completion: jax.Array
    accepted_dump_volume: jax.Array
    illegal_dump_volume: jax.Array


class EpisodeAccumulator(struct.PyTreeNode):
    family_id: jax.Array
    primary_cell_id: jax.Array
    stage_id: jax.Array
    episodic_return: jax.Array
    agent_reward_sum: jax.Array
    terminal_reward_normalized_sum: jax.Array
    terminal_reward_raw_sum: jax.Array
    trench_reward_sum: jax.Array
    existence_reward_sum: jax.Array
    step_count: jax.Array
    action_counts: jax.Array
    explicit_noop_count: jax.Array
    no_effect_action_count: jax.Array
    productive_workspace_cycles: jax.Array
    maximum_mass_residual: jax.Array
    mass_residual_violation_count: jax.Array
    target_mutation: jax.Array
    obstacle_mutation: jax.Array


class EpisodeAggregate(struct.PyTreeNode):
    episode_count: jax.Array
    task_done_count: jax.Array
    timeout_count: jax.Array
    episodic_return_sum: jax.Array
    agent_reward_sum: jax.Array
    terminal_reward_normalized_sum: jax.Array
    terminal_reward_raw_sum: jax.Array
    trench_reward_sum: jax.Array
    existence_reward_sum: jax.Array
    reward_component_sum: jax.Array
    reward_residual_abs_sum: jax.Array
    maximum_reward_residual: jax.Array
    reward_residual_violation_count: jax.Array
    step_count: jax.Array
    action_counts: jax.Array
    explicit_noop_count: jax.Array
    no_effect_action_count: jax.Array
    productive_workspace_cycles: jax.Array
    maximum_mass_residual: jax.Array
    mass_residual_violation_count: jax.Array
    target_mutation_count: jax.Array
    obstacle_mutation_count: jax.Array
    exact_dump_volume_sum: jax.Array
    accepted_dump_volume_sum: jax.Array
    buffer_only_dump_volume_sum: jax.Array
    illegal_dump_volume_sum: jax.Array
    dig_completion_sum: jax.Array
    dump_purity_sum: jax.Array
    dump_volume_completion_sum: jax.Array
    combined_completion_sum: jax.Array
    unloaded_completion_sum: jax.Array


_MAX_AGGREGATE_FIELDS = (
    "maximum_reward_residual",
    "maximum_mass_residual",
)


def new_episode_accumulator(
    family_id: jax.Array,
    primary_cell_id: jax.Array,
    stage_id: jax.Array,
) -> EpisodeAccumulator:
    """Create one persistent accumulator per parallel environment."""
    shape = jnp.asarray(family_id).shape
    zeros_float = jnp.zeros(shape, dtype=jnp.float32)
    zeros_int = jnp.zeros(shape, dtype=jnp.int32)
    zeros_bool = jnp.zeros(shape, dtype=jnp.bool_)
    return EpisodeAccumulator(
        family_id=jnp.asarray(family_id, dtype=jnp.int32),
        primary_cell_id=jnp.asarray(primary_cell_id, dtype=jnp.int32),
        stage_id=jnp.asarray(stage_id, dtype=jnp.int32),
        episodic_return=zeros_float,
        agent_reward_sum=jnp.zeros(shape + (MAX_AGENTS,), dtype=jnp.float32),
        terminal_reward_normalized_sum=zeros_float,
        terminal_reward_raw_sum=zeros_float,
        trench_reward_sum=zeros_float,
        existence_reward_sum=zeros_float,
        step_count=zeros_int,
        action_counts=jnp.zeros(shape + (NUM_ACTIONS,), dtype=jnp.int32),
        explicit_noop_count=zeros_int,
        no_effect_action_count=zeros_int,
        productive_workspace_cycles=zeros_int,
        maximum_mass_residual=zeros_int,
        mass_residual_violation_count=zeros_int,
        target_mutation=zeros_bool,
        obstacle_mutation=zeros_bool,
    )


def empty_episode_aggregate(group_count: int) -> EpisodeAggregate:
    """Create a bounded aggregate over stage/family/cell/reason groups."""
    shape = (group_count,)
    zeros_float = jnp.zeros(shape, dtype=jnp.float32)
    zeros_int = jnp.zeros(shape, dtype=jnp.int32)
    return EpisodeAggregate(
        episode_count=zeros_int,
        task_done_count=zeros_int,
        timeout_count=zeros_int,
        episodic_return_sum=zeros_float,
        agent_reward_sum=jnp.zeros(
            shape + (MAX_AGENTS,),
            dtype=jnp.float32,
        ),
        terminal_reward_normalized_sum=zeros_float,
        terminal_reward_raw_sum=zeros_float,
        trench_reward_sum=zeros_float,
        existence_reward_sum=zeros_float,
        reward_component_sum=zeros_float,
        reward_residual_abs_sum=zeros_float,
        maximum_reward_residual=zeros_float,
        reward_residual_violation_count=zeros_int,
        step_count=zeros_int,
        action_counts=jnp.zeros(
            shape + (NUM_ACTIONS,),
            dtype=jnp.int32,
        ),
        explicit_noop_count=zeros_int,
        no_effect_action_count=zeros_int,
        productive_workspace_cycles=zeros_int,
        maximum_mass_residual=zeros_int,
        mass_residual_violation_count=zeros_int,
        target_mutation_count=zeros_int,
        obstacle_mutation_count=zeros_int,
        exact_dump_volume_sum=zeros_float,
        accepted_dump_volume_sum=zeros_float,
        buffer_only_dump_volume_sum=zeros_float,
        illegal_dump_volume_sum=zeros_float,
        dig_completion_sum=zeros_float,
        dump_purity_sum=zeros_float,
        dump_volume_completion_sum=zeros_float,
        combined_completion_sum=zeros_float,
        unloaded_completion_sum=zeros_float,
    )


def _termination_reason(task_done: jax.Array, timeout: jax.Array) -> jax.Array:
    return jnp.where(
        jnp.logical_and(task_done, timeout),
        2,
        jnp.where(task_done, 0, jnp.where(timeout, 1, 3)),
    ).astype(jnp.int32)


def _masked_scatter_sum(
    group: jax.Array,
    mask: jax.Array,
    values: jax.Array,
    group_count: int,
) -> jax.Array:
    suffix = values.shape[1:]
    expanded_mask = mask.reshape(mask.shape + (1,) * len(suffix))
    weighted = jnp.where(expanded_mask, values, jnp.zeros_like(values))
    return (
        jnp.zeros(
            (group_count,) + suffix,
            dtype=values.dtype,
        )
        .at[group]
        .add(weighted)
    )


def _masked_scatter_max(
    group: jax.Array,
    mask: jax.Array,
    values: jax.Array,
    group_count: int,
) -> jax.Array:
    weighted = jnp.where(mask, values, jnp.zeros_like(values))
    return (
        jnp.zeros(
            (group_count,),
            dtype=values.dtype,
        )
        .at[group]
        .max(weighted)
    )


def _clear_terminal_environments(
    value: jax.Array,
    done: jax.Array,
) -> jax.Array:
    expanded_done = done.reshape(done.shape + (1,) * (value.ndim - done.ndim))
    return jnp.where(expanded_done, jnp.zeros_like(value), value)


def update_episode_aggregate(
    accumulator: EpisodeAccumulator,
    pending: EpisodeAggregate,
    step: EpisodeStep,
    *,
    next_family_id: jax.Array,
    next_primary_cell_id: jax.Array,
    next_stage_id: jax.Array,
    num_stages: int,
    num_families: int,
    num_primary_cells: int,
    reward_residual_rtol: float = REWARD_COMPONENT_RTOL,
) -> tuple[EpisodeAccumulator, EpisodeAggregate]:
    """Accumulate one step and emit only complete episodes into ``pending``."""
    trench_reward = jnp.nan_to_num(step.trench_reward, nan=0.0)
    updated = accumulator.replace(
        episodic_return=accumulator.episodic_return + step.reward,
        agent_reward_sum=accumulator.agent_reward_sum + step.agent_rewards,
        terminal_reward_normalized_sum=(
            accumulator.terminal_reward_normalized_sum + step.terminal_reward
        ),
        terminal_reward_raw_sum=(
            accumulator.terminal_reward_raw_sum
            + step.terminal_reward * step.reward_normalizer
        ),
        trench_reward_sum=accumulator.trench_reward_sum + trench_reward,
        existence_reward_sum=(accumulator.existence_reward_sum + step.existence_reward),
        step_count=accumulator.step_count + 1,
        action_counts=(
            accumulator.action_counts
            + jax.nn.one_hot(
                step.action,
                NUM_ACTIONS,
                dtype=jnp.int32,
            )
        ),
        explicit_noop_count=(
            accumulator.explicit_noop_count
            + (step.action == NUM_ACTIONS - 1).astype(jnp.int32)
        ),
        no_effect_action_count=(
            accumulator.no_effect_action_count
            + jnp.logical_not(step.action_had_effect).astype(jnp.int32)
        ),
        productive_workspace_cycles=(
            accumulator.productive_workspace_cycles
            + step.productive_workspace_cycle.astype(jnp.int32)
        ),
        maximum_mass_residual=jnp.maximum(
            accumulator.maximum_mass_residual,
            step.transition_mass_residual.astype(jnp.int32),
        ),
        mass_residual_violation_count=(
            accumulator.mass_residual_violation_count
            + (step.transition_mass_residual != 0).astype(jnp.int32)
        ),
        target_mutation=jnp.logical_or(
            accumulator.target_mutation,
            step.target_mutation,
        ),
        obstacle_mutation=jnp.logical_or(
            accumulator.obstacle_mutation,
            step.obstacle_mutation,
        ),
    )

    reason = _termination_reason(step.task_done, step.timeout)
    group = (
        (updated.stage_id * num_families + updated.family_id) * num_primary_cells
        + updated.primary_cell_id
    ) * len(TERMINATION_REASONS) + reason
    expected_group_count = (
        num_stages * num_families * num_primary_cells * len(TERMINATION_REASONS)
    )
    if pending.episode_count.shape[0] != expected_group_count:
        raise ValueError(
            "episode aggregate group count does not match the declared axes"
        )

    component_sum = (
        updated.agent_reward_sum.sum(axis=-1)
        + updated.terminal_reward_normalized_sum
        + updated.trench_reward_sum
        + updated.existence_reward_sum
    )
    reward_residual = jnp.abs(updated.episodic_return - component_sum)
    reward_residual_tolerance = reward_residual_rtol * jnp.maximum(
        jnp.float32(1.0),
        jnp.maximum(
            jnp.abs(updated.episodic_return),
            jnp.abs(component_sum),
        ),
    )
    done = step.done.astype(jnp.bool_)

    additions = EpisodeAggregate(
        episode_count=_masked_scatter_sum(
            group,
            done,
            jnp.ones_like(updated.step_count),
            expected_group_count,
        ),
        task_done_count=_masked_scatter_sum(
            group,
            done,
            step.task_done.astype(jnp.int32),
            expected_group_count,
        ),
        timeout_count=_masked_scatter_sum(
            group,
            done,
            step.timeout.astype(jnp.int32),
            expected_group_count,
        ),
        episodic_return_sum=_masked_scatter_sum(
            group,
            done,
            updated.episodic_return,
            expected_group_count,
        ),
        agent_reward_sum=_masked_scatter_sum(
            group,
            done,
            updated.agent_reward_sum,
            expected_group_count,
        ),
        terminal_reward_normalized_sum=_masked_scatter_sum(
            group,
            done,
            updated.terminal_reward_normalized_sum,
            expected_group_count,
        ),
        terminal_reward_raw_sum=_masked_scatter_sum(
            group,
            done,
            updated.terminal_reward_raw_sum,
            expected_group_count,
        ),
        trench_reward_sum=_masked_scatter_sum(
            group,
            done,
            updated.trench_reward_sum,
            expected_group_count,
        ),
        existence_reward_sum=_masked_scatter_sum(
            group,
            done,
            updated.existence_reward_sum,
            expected_group_count,
        ),
        reward_component_sum=_masked_scatter_sum(
            group,
            done,
            component_sum,
            expected_group_count,
        ),
        reward_residual_abs_sum=_masked_scatter_sum(
            group,
            done,
            reward_residual,
            expected_group_count,
        ),
        maximum_reward_residual=_masked_scatter_max(
            group,
            done,
            reward_residual,
            expected_group_count,
        ),
        reward_residual_violation_count=_masked_scatter_sum(
            group,
            done,
            (reward_residual > reward_residual_tolerance).astype(jnp.int32),
            expected_group_count,
        ),
        step_count=_masked_scatter_sum(
            group,
            done,
            updated.step_count,
            expected_group_count,
        ),
        action_counts=_masked_scatter_sum(
            group,
            done,
            updated.action_counts,
            expected_group_count,
        ),
        explicit_noop_count=_masked_scatter_sum(
            group,
            done,
            updated.explicit_noop_count,
            expected_group_count,
        ),
        no_effect_action_count=_masked_scatter_sum(
            group,
            done,
            updated.no_effect_action_count,
            expected_group_count,
        ),
        productive_workspace_cycles=_masked_scatter_sum(
            group,
            done,
            updated.productive_workspace_cycles,
            expected_group_count,
        ),
        maximum_mass_residual=_masked_scatter_max(
            group,
            done,
            updated.maximum_mass_residual,
            expected_group_count,
        ),
        mass_residual_violation_count=_masked_scatter_sum(
            group,
            done,
            updated.mass_residual_violation_count,
            expected_group_count,
        ),
        target_mutation_count=_masked_scatter_sum(
            group,
            done,
            updated.target_mutation.astype(jnp.int32),
            expected_group_count,
        ),
        obstacle_mutation_count=_masked_scatter_sum(
            group,
            done,
            updated.obstacle_mutation.astype(jnp.int32),
            expected_group_count,
        ),
        # Under exact_visible_dump_v1, exact and accepted are identical and
        # buffer-only volume is structurally zero.
        exact_dump_volume_sum=_masked_scatter_sum(
            group,
            done,
            step.accepted_dump_volume,
            expected_group_count,
        ),
        accepted_dump_volume_sum=_masked_scatter_sum(
            group,
            done,
            step.accepted_dump_volume,
            expected_group_count,
        ),
        buffer_only_dump_volume_sum=_masked_scatter_sum(
            group,
            done,
            jnp.zeros_like(step.accepted_dump_volume),
            expected_group_count,
        ),
        illegal_dump_volume_sum=_masked_scatter_sum(
            group,
            done,
            step.illegal_dump_volume,
            expected_group_count,
        ),
        dig_completion_sum=_masked_scatter_sum(
            group,
            done,
            step.dig_completion,
            expected_group_count,
        ),
        dump_purity_sum=_masked_scatter_sum(
            group,
            done,
            step.dump_purity,
            expected_group_count,
        ),
        dump_volume_completion_sum=_masked_scatter_sum(
            group,
            done,
            step.dump_volume_completion,
            expected_group_count,
        ),
        combined_completion_sum=_masked_scatter_sum(
            group,
            done,
            step.combined_completion,
            expected_group_count,
        ),
        unloaded_completion_sum=_masked_scatter_sum(
            group,
            done,
            step.unloaded_completion,
            expected_group_count,
        ),
    )
    pending = merge_episode_aggregates(pending, additions)

    reset_values = {
        field: _clear_terminal_environments(getattr(updated, field), done)
        for field in (
            "episodic_return",
            "agent_reward_sum",
            "terminal_reward_normalized_sum",
            "terminal_reward_raw_sum",
            "trench_reward_sum",
            "existence_reward_sum",
            "step_count",
            "action_counts",
            "explicit_noop_count",
            "no_effect_action_count",
            "productive_workspace_cycles",
            "maximum_mass_residual",
            "mass_residual_violation_count",
            "target_mutation",
            "obstacle_mutation",
        )
    }
    updated = updated.replace(
        family_id=jnp.where(done, next_family_id, updated.family_id),
        primary_cell_id=jnp.where(
            done,
            next_primary_cell_id,
            updated.primary_cell_id,
        ),
        stage_id=jnp.where(done, next_stage_id, updated.stage_id),
        **reset_values,
    )
    return updated, pending


def merge_episode_aggregates(
    left: EpisodeAggregate,
    right: EpisodeAggregate,
) -> EpisodeAggregate:
    values = {}
    for field in left.__dataclass_fields__:
        if field in _MAX_AGGREGATE_FIELDS:
            values[field] = jnp.maximum(
                getattr(left, field),
                getattr(right, field),
            )
        else:
            values[field] = getattr(left, field) + getattr(right, field)
    return EpisodeAggregate(**values)


def reduce_episode_aggregate(
    aggregate: EpisodeAggregate,
    axis_name: str,
) -> EpisodeAggregate:
    """Reduce all additive fields globally and maxima with pmax."""
    values = {}
    for field in aggregate.__dataclass_fields__:
        value = getattr(aggregate, field)
        if field in _MAX_AGGREGATE_FIELDS:
            values[field] = jax.lax.pmax(value, axis_name)
        else:
            values[field] = jax.lax.psum(value, axis_name)
    return EpisodeAggregate(**values)


def aggregate_to_payload(
    aggregate: EpisodeAggregate,
    *,
    family_names: Sequence[str],
    primary_cell_names: Sequence[str],
    stage_names: Sequence[str],
    update: int,
    run_name: str,
) -> dict:
    """Convert one host aggregate to a bounded machine-readable receipt."""
    arrays = {
        field: np.asarray(jax.device_get(getattr(aggregate, field)))
        for field in aggregate.__dataclass_fields__
    }
    rows = []
    reason_count = len(TERMINATION_REASONS)
    for group, episode_count in enumerate(arrays["episode_count"]):
        if int(episode_count) == 0:
            continue
        remainder, reason_id = divmod(group, reason_count)
        remainder, cell_id = divmod(
            remainder,
            len(primary_cell_names),
        )
        stage_id, family_id = divmod(remainder, len(family_names))
        row = {
            "stage_id": stage_id,
            "stage": stage_names[stage_id],
            "family_id": family_id,
            "family": family_names[family_id],
            "primary_cell_id": cell_id,
            "primary_cell": primary_cell_names[cell_id],
            "termination_reason": TERMINATION_REASONS[reason_id],
        }
        for field, values in arrays.items():
            value = values[group]
            row[field] = value.tolist() if value.ndim else value.item()
        rows.append(row)

    total_episodes = int(arrays["episode_count"].sum())
    total_steps = int(arrays["step_count"].sum())
    total_task_done = int(arrays["task_done_count"].sum())
    total_timeout = int(arrays["timeout_count"].sum())
    totals = {
        "episode_count": total_episodes,
        "task_done_count": total_task_done,
        "timeout_count": total_timeout,
        "step_count": total_steps,
        "episodic_return_sum": float(arrays["episodic_return_sum"].sum()),
        "agent_reward_sum": arrays["agent_reward_sum"].sum(axis=0).tolist(),
        "terminal_reward_normalized_sum": float(
            arrays["terminal_reward_normalized_sum"].sum()
        ),
        "terminal_reward_raw_sum": float(arrays["terminal_reward_raw_sum"].sum()),
        "trench_reward_sum": float(arrays["trench_reward_sum"].sum()),
        "existence_reward_sum": float(arrays["existence_reward_sum"].sum()),
        "reward_component_sum": float(arrays["reward_component_sum"].sum()),
        "action_counts": arrays["action_counts"].sum(axis=0).tolist(),
        "productive_workspace_cycles": int(arrays["productive_workspace_cycles"].sum()),
        "no_effect_action_count": int(arrays["no_effect_action_count"].sum()),
        "explicit_noop_count": int(arrays["explicit_noop_count"].sum()),
        "mass_residual_violation_count": int(
            arrays["mass_residual_violation_count"].sum()
        ),
        "maximum_mass_residual": int(arrays["maximum_mass_residual"].max(initial=0)),
        "target_mutation_count": int(arrays["target_mutation_count"].sum()),
        "obstacle_mutation_count": int(arrays["obstacle_mutation_count"].sum()),
        "reward_residual_violation_count": int(
            arrays["reward_residual_violation_count"].sum()
        ),
        "maximum_reward_residual": float(
            arrays["maximum_reward_residual"].max(initial=0.0)
        ),
        "exact_dump_volume_sum": float(arrays["exact_dump_volume_sum"].sum()),
        "accepted_dump_volume_sum": float(arrays["accepted_dump_volume_sum"].sum()),
        "buffer_only_dump_volume_sum": float(
            arrays["buffer_only_dump_volume_sum"].sum()
        ),
        "illegal_dump_volume_sum": float(arrays["illegal_dump_volume_sum"].sum()),
        "dig_completion_sum": float(arrays["dig_completion_sum"].sum()),
        "dump_purity_sum": float(arrays["dump_purity_sum"].sum()),
        "dump_volume_completion_sum": float(arrays["dump_volume_completion_sum"].sum()),
        "combined_completion_sum": float(arrays["combined_completion_sum"].sum()),
        "unloaded_completion_sum": float(arrays["unloaded_completion_sum"].sum()),
    }
    rates = {
        "task_done_rate": (
            total_task_done / total_episodes if total_episodes else None
        ),
        "timeout_rate": (total_timeout / total_episodes if total_episodes else None),
        "no_effect_action_rate": (
            totals["no_effect_action_count"] / total_steps if total_steps else None
        ),
        "productive_workspace_cycles_per_episode": (
            totals["productive_workspace_cycles"] / total_episodes
            if total_episodes
            else None
        ),
    }
    return {
        "schema": "terra_training_episode_aggregate_v1",
        "contract": "exact_visible_dump_v1",
        "numerical_tolerances": {
            "reward_component_relative": REWARD_COMPONENT_RTOL,
        },
        "run_name": run_name,
        "update": int(update),
        "axis_order": [
            "stage",
            "family",
            "primary_cell",
            "termination_reason",
        ],
        "family_names": list(family_names),
        "primary_cell_names": list(primary_cell_names),
        "stage_names": list(stage_names),
        "totals": totals,
        "rates": rates,
        "groups": rows,
    }


def assert_aggregate_integrity(payload: dict) -> None:
    """Hard fail before checkpointing if a completed episode is inconsistent."""
    totals = payload["totals"]
    failures = []
    for field in (
        "mass_residual_violation_count",
        "target_mutation_count",
        "obstacle_mutation_count",
        "reward_residual_violation_count",
    ):
        if totals[field]:
            failures.append(f"{field}={totals[field]}")
    if failures:
        raise RuntimeError(
            "Terra episode aggregate integrity failed: " + ", ".join(failures)
        )
