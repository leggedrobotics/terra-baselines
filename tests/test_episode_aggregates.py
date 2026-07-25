import jax.numpy as jnp
import numpy as np
import pytest

from utils.episode_aggregates import aggregate_to_payload
from utils.episode_aggregates import assert_aggregate_integrity
from utils.episode_aggregates import empty_episode_aggregate
from utils.episode_aggregates import EpisodeStep
from utils.episode_aggregates import merge_episode_aggregates
from utils.episode_aggregates import new_episode_accumulator
from utils.episode_aggregates import update_episode_aggregate


def _step(
    *,
    reward,
    done,
    task_done,
    timeout,
    agent_reward,
    terminal_reward=0.0,
    trench_reward=0.0,
    existence_reward=0.0,
):
    reward = jnp.asarray(reward, dtype=jnp.float32)
    batch = reward.shape[0]
    agent_rewards = jnp.zeros((batch, 4), dtype=jnp.float32)
    agent_rewards = agent_rewards.at[:, 0].set(
        jnp.asarray(agent_reward, dtype=jnp.float32)
    )
    zeros_float = jnp.zeros((batch,), dtype=jnp.float32)
    zeros_int = jnp.zeros((batch,), dtype=jnp.int32)
    zeros_bool = jnp.zeros((batch,), dtype=jnp.bool_)
    return EpisodeStep(
        done=jnp.asarray(done, dtype=jnp.bool_),
        task_done=jnp.asarray(task_done, dtype=jnp.bool_),
        timeout=jnp.asarray(timeout, dtype=jnp.bool_),
        reward=reward,
        agent_rewards=agent_rewards,
        terminal_reward=jnp.broadcast_to(
            jnp.asarray(terminal_reward, dtype=jnp.float32),
            (batch,),
        ),
        trench_reward=jnp.broadcast_to(
            jnp.asarray(trench_reward, dtype=jnp.float32),
            (batch,),
        ),
        existence_reward=jnp.broadcast_to(
            jnp.asarray(existence_reward, dtype=jnp.float32),
            (batch,),
        ),
        reward_normalizer=jnp.full((batch,), 10.0, dtype=jnp.float32),
        action=zeros_int,
        action_had_effect=jnp.ones((batch,), dtype=jnp.bool_),
        productive_workspace_cycle=zeros_int,
        transition_mass_residual=zeros_int,
        target_mutation=zeros_bool,
        obstacle_mutation=zeros_bool,
        dig_completion=jnp.ones((batch,), dtype=jnp.float32),
        dump_purity=jnp.ones((batch,), dtype=jnp.float32),
        dump_volume_completion=jnp.ones((batch,), dtype=jnp.float32),
        combined_completion=jnp.ones((batch,), dtype=jnp.float32),
        unloaded_completion=jnp.ones((batch,), dtype=jnp.float32),
        accepted_dump_volume=zeros_float,
        illegal_dump_volume=zeros_float,
    )


def _update(accumulator, pending, step, *, next_family=None):
    batch = step.done.shape[0]
    if next_family is None:
        next_family = jnp.ones((batch,), dtype=jnp.int32)
    return update_episode_aggregate(
        accumulator,
        pending,
        step,
        next_family_id=next_family,
        next_primary_cell_id=jnp.ones((batch,), dtype=jnp.int32),
        next_stage_id=jnp.zeros((batch,), dtype=jnp.int32),
        num_stages=1,
        num_families=2,
        num_primary_cells=2,
    )


def test_episode_sum_survives_two_rollout_windows_and_resets_only_on_done():
    accumulator = new_episode_accumulator(
        jnp.array([1]),
        jnp.array([1]),
        jnp.array([0]),
    )
    pending = empty_episode_aggregate(16)
    first = _step(
        reward=[1.0],
        done=[False],
        task_done=[False],
        timeout=[False],
        agent_reward=[0.8],
        existence_reward=0.2,
    )
    accumulator, pending = _update(accumulator, pending, first)
    assert int(pending.episode_count.sum()) == 0
    assert float(accumulator.episodic_return[0]) == pytest.approx(1.0)
    assert int(accumulator.step_count[0]) == 1

    second = _step(
        reward=[3.0],
        done=[True],
        task_done=[True],
        timeout=[False],
        agent_reward=[1.0],
        terminal_reward=1.8,
        existence_reward=0.2,
    )
    accumulator, pending = _update(
        accumulator,
        pending,
        second,
        next_family=jnp.array([0]),
    )

    assert int(pending.episode_count.sum()) == 1
    assert float(pending.episodic_return_sum.sum()) == pytest.approx(4.0)
    assert float(pending.reward_component_sum.sum()) == pytest.approx(4.0)
    assert int(pending.reward_residual_violation_count.sum()) == 0
    assert float(pending.terminal_reward_raw_sum.sum()) == pytest.approx(18.0)
    assert int(accumulator.step_count[0]) == 0
    assert float(accumulator.episodic_return[0]) == pytest.approx(0.0)
    assert int(accumulator.family_id[0]) == 0


def test_success_timeout_and_simultaneous_terminal_reasons_are_distinct():
    accumulator = new_episode_accumulator(
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
    )
    pending = empty_episode_aggregate(16)
    step = _step(
        reward=[1.0, 1.0, 1.0],
        done=[True, True, True],
        task_done=[True, False, True],
        timeout=[False, True, True],
        agent_reward=[1.0, 1.0, 1.0],
    )
    _, pending = _update(accumulator, pending, step)
    payload = aggregate_to_payload(
        pending,
        family_names=("unknown", "foundation"),
        primary_cell_names=("unknown", "easy"),
        stage_names=("F0",),
        update=2,
        run_name="fixture",
    )
    reason_counts = {
        row["termination_reason"]: row["episode_count"] for row in payload["groups"]
    }
    assert reason_counts == {
        "task_done": 1,
        "timeout": 1,
        "task_done_and_timeout": 1,
    }
    assert payload["totals"]["episode_count"] == 3
    assert payload["totals"]["task_done_count"] == 2
    assert payload["totals"]["timeout_count"] == 2


def test_shard_merge_matches_one_population_and_preserves_maxima():
    accumulator = new_episode_accumulator(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    pending = empty_episode_aggregate(16)
    step = _step(
        reward=[1.0, 2.0],
        done=[True, True],
        task_done=[True, False],
        timeout=[False, True],
        agent_reward=[1.0, 2.0],
    )
    step = step.replace(transition_mass_residual=jnp.array([0, 3], dtype=jnp.int32))
    _, population = _update(accumulator, pending, step)

    shards = []
    for index in range(2):
        shard_accumulator = new_episode_accumulator(
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
        )
        one = EpisodeStep(
            **{
                field: getattr(step, field)[index : index + 1]
                for field in step.__dataclass_fields__
            }
        )
        _, shard = _update(
            shard_accumulator,
            empty_episode_aggregate(16),
            one,
        )
        shards.append(shard)
    merged = merge_episode_aggregates(shards[0], shards[1])

    for field in population.__dataclass_fields__:
        np.testing.assert_allclose(
            getattr(merged, field),
            getattr(population, field),
        )
    assert int(merged.maximum_mass_residual.max()) == 3


def test_integrity_assertion_rejects_mass_or_reward_inconsistency():
    accumulator = new_episode_accumulator(
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    step = _step(
        reward=[2.0],
        done=[True],
        task_done=[True],
        timeout=[False],
        agent_reward=[1.0],
    ).replace(transition_mass_residual=jnp.array([1], dtype=jnp.int32))
    _, aggregate = _update(
        accumulator,
        empty_episode_aggregate(16),
        step,
    )
    payload = aggregate_to_payload(
        aggregate,
        family_names=("unknown", "foundation"),
        primary_cell_names=("unknown", "easy"),
        stage_names=("F0",),
        update=1,
        run_name="fixture",
    )
    with pytest.raises(RuntimeError, match="integrity failed"):
        assert_aggregate_integrity(payload)


def test_reward_reconstruction_uses_a_small_relative_tolerance():
    accumulator = new_episode_accumulator(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    step = _step(
        reward=[100.0, 100.0],
        done=[True, True],
        task_done=[True, True],
        timeout=[False, False],
        agent_reward=[99.9995, 99.998],
    )
    _, aggregate = _update(
        accumulator,
        empty_episode_aggregate(16),
        step,
    )
    assert int(aggregate.reward_residual_violation_count.sum()) == 1
