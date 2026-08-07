import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.config import EnvConfig, RewardStage
from train_mixed import (
    REWARD_ANNEAL_DURATION_UPDATES,
    _new_reward_anneal_state,
    _restore_reward_anneal_checkpoint,
    assign_terminal_reward_mix,
    create_mixed_agent_env_config,
    maybe_start_reward_anneal,
    reward_anneal_mix,
)


def _receipt(foundation_depth, trench_depth):
    return {
        "mastery": {
            "family_active_depth": {
                "foundation": foundation_depth,
                "trench": trench_depth,
            }
        }
    }


def test_reward_fade_trigger_is_one_way_and_none_means_fully_mastered():
    state = _new_reward_anneal_state()
    assert not maybe_start_reward_anneal(state, _receipt(2, 1), 100)
    assert maybe_start_reward_anneal(state, _receipt(2, 2), 101)
    assert state["started_update"] == 101
    assert reward_anneal_mix(state, 101) == 0.0
    assert reward_anneal_mix(state, 2601) == 0.5
    assert reward_anneal_mix(state, 5101) == 1.0
    assert reward_anneal_mix(state, 6000) == 1.0

    # Demotion affects map sampling, not the latched reward schedule.
    assert not maybe_start_reward_anneal(state, _receipt(0, 0), 102)
    assert state["started_update"] == 101

    one_finished = _new_reward_anneal_state()
    assert maybe_start_reward_anneal(one_finished, _receipt(None, 2), 200)
    both_finished = _new_reward_anneal_state()
    assert maybe_start_reward_anneal(both_finished, _receipt(None, None), 300)


def test_reward_fade_resume_is_keyed_to_absolute_update_and_env_mix():
    state = _new_reward_anneal_state()
    state["started_update"] = 100
    next_update = 600
    expected_last = reward_anneal_mix(state, next_update - 1)
    state["last_applied_mix"] = expected_last
    checkpoint = {
        "next_update": next_update,
        "reward_anneal_state": copy.deepcopy(state),
        "env_config": EnvConfig()._replace(
            terminal_reward_mix=jnp.float32(expected_last)
        ),
    }

    restored = _restore_reward_anneal_checkpoint(
        "annealed_objective", checkpoint, "resume", next_update
    )
    assert restored == state
    assert reward_anneal_mix(restored, next_update) == pytest.approx(
        (next_update - 100) / REWARD_ANNEAL_DURATION_UPDATES
    )

    stale_state = copy.deepcopy(checkpoint)
    stale_state["reward_anneal_state"]["last_applied_mix"] = 0.0
    with pytest.raises(ValueError, match="next_update"):
        _restore_reward_anneal_checkpoint(
            "annealed_objective", stale_state, "resume", next_update
        )

    stale_env = copy.deepcopy(checkpoint)
    stale_env["env_config"] = stale_env["env_config"]._replace(terminal_reward_mix=0.0)
    with pytest.raises(ValueError, match="env_config"):
        _restore_reward_anneal_checkpoint(
            "annealed_objective", stale_env, "resume", next_update
        )

    invalid = copy.deepcopy(checkpoint)
    invalid["reward_anneal_state"]["started_update"] = 1.5
    with pytest.raises(ValueError, match="started_update"):
        _restore_reward_anneal_checkpoint(
            "annealed_objective", invalid, "resume", next_update
        )


def test_reward_stage_and_mix_assignment_preserve_batched_shape():
    env_cfg = create_mixed_agent_env_config(
        agent_types=(0,),
        action_types=(0,),
        reward_stage="annealed_objective",
    )
    assert env_cfg.reward_stage == RewardStage.ANNEALED_OBJECTIVE
    batched = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(jnp.asarray(value), (2, 3) + np.shape(value)),
        env_cfg,
    )
    mixed = assign_terminal_reward_mix(batched, 0.25)
    assert mixed.terminal_reward_mix.shape == (2, 3)
    np.testing.assert_allclose(mixed.terminal_reward_mix, 0.25)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        assign_terminal_reward_mix(batched, 1.1)
