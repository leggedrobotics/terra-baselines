import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import eval_mcts
import eval_mixed
from eval_mcts import make_mcts_recurrent_fn, make_mcts_step_fn, rollout_episode
from terra.actions import TrackedAction
from terra.env import TimeStep


BATCH_SIZE = 2


def _observation():
    zeros_12 = jnp.zeros((BATCH_SIZE, 12))
    zeros_map = jnp.zeros((BATCH_SIZE, 4, 4))
    return {
        "agent_states": jnp.zeros((BATCH_SIZE, 4, 8)),
        "agent_active": jnp.ones((BATCH_SIZE, 4)),
        "num_agents": jnp.ones((BATCH_SIZE,), dtype=jnp.int32),
        "local_map_action_neg": zeros_12,
        "local_map_action_pos": zeros_12,
        "local_map_target_neg": zeros_12,
        "local_map_target_pos": zeros_12,
        "local_map_dumpability": zeros_12,
        "local_map_obstacles": zeros_12,
        "local_map_border_workspace": zeros_12,
        "local_map_edge_alignment_error": zeros_12,
        "local_map_border_diggable": zeros_12,
        "traversability_mask": zeros_map,
        "reachability_mask": zeros_map,
        "action_map": zeros_map,
        "target_map": zeros_map,
        "agent_width": jnp.ones((BATCH_SIZE,)),
        "agent_height": jnp.ones((BATCH_SIZE,)),
        "padding_mask": zeros_map,
        "dumpability_mask": jnp.ones((BATCH_SIZE, 4, 4)),
        "interaction_mask": zeros_map,
    }


def _timestep():
    return TimeStep(
        state=jnp.zeros((BATCH_SIZE,)),
        observation=_observation(),
        reward=jnp.zeros((BATCH_SIZE,)),
        done=jnp.zeros((BATCH_SIZE,), dtype=jnp.bool_),
        info={},
        env_cfg=jnp.zeros((BATCH_SIZE,)),
    )


class FakeModel:
    def apply(self, params, inputs):
        del params, inputs
        value = jnp.full((BATCH_SIZE, 1), 3.0)
        logits = jnp.zeros((BATCH_SIZE, 8))
        return value, logits


class FakeEnv:
    batch_cfg = SimpleNamespace(action_type=TrackedAction)

    def step(self, timestep, action, rng_keys):
        del action, rng_keys
        return timestep._replace(
            reward=jnp.array([5.0, 7.0]),
            done=jnp.array([True, False]),
        )


def _config():
    return SimpleNamespace(
        num_test_rollouts=BATCH_SIZE,
        num_simulations=4,
        gamma=0.9,
        clip_action_maps=True,
    )


class FakeRolloutEnv(FakeEnv):
    def reset(self, env_cfgs, rng_keys):
        del env_cfgs, rng_keys
        return _timestep()

    def step(self, timestep, action, rng_keys):
        del action, rng_keys
        next_state = timestep.state + 1
        done = jnp.array([next_state[0] >= 1, next_state[1] >= 2])
        task_done = jnp.array([next_state[0] >= 1, False])
        completion = jnp.array([1.0, 0.4])
        return timestep._replace(
            state=next_state,
            reward=jnp.array([1.0, 2.0]),
            done=done,
            info={
                "task_done": task_done,
                "reward_components": {
                    "total_dig_dump_completion": completion,
                    "dig_completion_total": completion,
                    "dump_completion_action_map": completion,
                },
            },
        )


class RolloutEpisodeAccountingTest(unittest.TestCase):
    def test_each_environment_contributes_only_its_first_episode(self):
        config = _config()
        config.num_prev_actions = 3
        env_cfgs = SimpleNamespace(
            tile_size=jnp.ones((BATCH_SIZE,)),
            agent=SimpleNamespace(move_tiles=jnp.ones((BATCH_SIZE,))),
        )
        cumulative_rewards, stats, _ = rollout_episode(
            FakeRolloutEnv(),
            FakeModel(),
            None,
            env_cfgs,
            config,
            max_frames=5,
            deterministic=True,
            seed=0,
            use_mcts=False,
        )

        np.testing.assert_array_equal(stats["episode_terminated_once"], [True, True])
        np.testing.assert_array_equal(stats["episode_done_once"], [True, False])
        np.testing.assert_array_equal(stats["episode_length"], [1, 2])
        np.testing.assert_allclose(cumulative_rewards[-1], [1.0, 4.0])

    def test_eval_mixed_reuses_the_authoritative_rollout(self):
        self.assertIs(eval_mixed.rollout_episode, rollout_episode)


@unittest.skipIf(eval_mcts.mctx is None, "mctx is not installed")
class MctsAbsorptionTest(unittest.TestCase):
    def test_recurrent_terminal_states_are_absorbing(self):
        recurrent_fn = make_mcts_recurrent_fn(FakeModel(), FakeEnv(), _config())
        previous_actions = jnp.array(
            [[1, 2, 3], [4, 5, 6]], dtype=jnp.int32
        )
        actions = jnp.array([7, 6], dtype=jnp.int32)

        output, embedding = recurrent_fn(
            None,
            jax.random.PRNGKey(0),
            actions,
            (_timestep(), previous_actions, jnp.array([False, True])),
        )

        np.testing.assert_allclose(output.reward, [5.0, 0.0])
        np.testing.assert_allclose(output.discount, [0.0, 0.0])
        np.testing.assert_allclose(output.value, [0.0, 0.0])
        np.testing.assert_array_equal(embedding[2], [True, True])
        np.testing.assert_array_equal(embedding[1][0], [0, 0, 0])

        output, _ = recurrent_fn(
            None,
            jax.random.PRNGKey(1),
            actions,
            (_timestep(), previous_actions, jnp.array([False, False])),
        )
        np.testing.assert_allclose(output.reward, [5.0, 7.0])
        np.testing.assert_allclose(output.discount, [0.0, 0.9])
        np.testing.assert_allclose(output.value, [0.0, 3.0])
        self.assertEqual(output.prior_logits.shape, (BATCH_SIZE, 8))

    def test_policy_step_runs_through_mctx(self):
        step_fn = make_mcts_step_fn(FakeModel(), FakeEnv(), _config())
        result = step_fn(
            None,
            jax.random.PRNGKey(0),
            _timestep(),
            jnp.zeros((BATCH_SIZE, 3), dtype=jnp.int32),
        )
        jax.block_until_ready(result[3])

        _, next_timestep, previous_actions, actions, ppo_actions, mcts_actions = result
        self.assertEqual(actions.shape, (BATCH_SIZE,))
        self.assertEqual(ppo_actions.shape, (BATCH_SIZE,))
        self.assertEqual(mcts_actions.shape, (BATCH_SIZE,))
        np.testing.assert_array_equal(next_timestep.done, [True, False])
        np.testing.assert_array_equal(previous_actions[0], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
