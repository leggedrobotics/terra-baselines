import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import eval_ppo


class _FakeState(NamedTuple):
    env_steps: jax.Array
    episode_index: jax.Array


class _FakeTimeStep(NamedTuple):
    state: _FakeState
    observation: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


class _FakeEnv:
    batch_cfg = SimpleNamespace(action_type=object())

    def reset(self, env_params, rng):
        del env_params
        num_envs = rng.shape[0]
        return _FakeTimeStep(
            state=_FakeState(
                env_steps=jnp.zeros((num_envs,), dtype=jnp.int32),
                episode_index=jnp.zeros((num_envs,), dtype=jnp.int32),
            ),
            observation=jnp.zeros((num_envs, 1), dtype=jnp.float32),
            reward=jnp.zeros((num_envs,), dtype=jnp.float32),
            done=jnp.zeros((num_envs,), dtype=jnp.bool_),
            info={"task_done": jnp.zeros((num_envs,), dtype=jnp.bool_)},
        )

    def step(self, timestep, action, rng):
        del action, rng
        env_index = jnp.arange(timestep.state.env_steps.shape[0])
        next_steps = timestep.state.env_steps + 1
        episode_index = timestep.state.episode_index

        env0_success = env_index == 0
        env1_timeout = jnp.logical_and(
            env_index == 1,
            jnp.logical_and(episode_index == 0, next_steps >= 2),
        )
        env1_later_success = jnp.logical_and(
            env_index == 1,
            episode_index > 0,
        )
        task_done = jnp.logical_or(env0_success, env1_later_success)
        done = jnp.logical_or(task_done, env1_timeout)

        return _FakeTimeStep(
            state=_FakeState(
                env_steps=jnp.where(done, 0, next_steps),
                episode_index=episode_index + done.astype(jnp.int32),
            ),
            observation=timestep.observation,
            reward=jnp.zeros_like(timestep.reward),
            done=done,
            info={"task_done": task_done},
        )


def _fake_select_action(train_state, observation, prev_actions, rng, config):
    del train_state, prev_actions, rng, config
    num_envs = observation.shape[0]
    actions = jnp.zeros((num_envs,), dtype=jnp.int32)
    zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
    return actions, zeros, zeros, None


class EvalPpoMetricTest(unittest.TestCase):
    def setUp(self):
        self.env = _FakeEnv()
        self.config = SimpleNamespace(
            num_envs_per_device=3,
            num_rollouts_eval=3,
            num_prev_actions=1,
        )
        eval_ppo._SINGLE_DEVICE_STEP_CACHE.clear()
        eval_ppo._PMAPPED_STEP_CACHE.clear()
        eval_ppo._PMAPPED_ROLLOUT_CACHE.clear()

    def _assert_rollout_stats(self, stats):
        # Env 0 succeeds on every step. Env 1 first times out, then succeeds;
        # env 2 remains censored throughout the three-step horizon.
        self.assertEqual(int(stats.positive_terminations), 4)
        self.assertEqual(int(stats.terminations), 5)
        self.assertEqual(int(stats.episodes), 5)
        self.assertEqual(int(stats.initial_episode_successes), 1)
        self.assertEqual(int(stats.initial_episode_terminations), 2)
        self.assertEqual(int(stats.positive_terminations_steps), 4)

    def test_success_rate_is_nan_without_outcomes_and_bounded_for_valid_counts(self):
        self.assertTrue(np.isnan(float(eval_ppo.episode_success_rate(0, 0))))
        self.assertAlmostEqual(float(eval_ppo.episode_success_rate(3, 4)), 0.75)
        np.testing.assert_array_equal(
            np.asarray(
                eval_ppo._success_events(
                    done=jnp.array([False, True]),
                    task_done=jnp.array([True, True]),
                )
            ),
            np.array([False, True]),
        )

    def test_all_rollout_paths_track_initial_episodes_and_true_lengths(self):
        with patch("eval_ppo.select_action_ppo", _fake_select_action), patch(
            "eval_ppo.wrap_action", lambda action, action_type: action
        ):
            direct_stats = eval_ppo._rollout_impl(
                jax.random.PRNGKey(0),
                self.env,
                None,
                jnp.asarray(0),
                self.config,
            )
            self._assert_rollout_stats(direct_stats)

            single_stats = eval_ppo.rollout_single_device(
                jax.random.PRNGKey(1),
                self.env,
                None,
                jnp.asarray(0),
                self.config,
            )
            self._assert_rollout_stats(single_stats)

            reset_keys = jax.random.split(jax.random.PRNGKey(2), 3)
            timestep = self.env.reset(None, reset_keys)
            pmapped_timestep = jtu.tree_map(lambda x: x[None], timestep)
            pmapped_stats = eval_ppo.rollout_from_timestep(
                jax.random.split(jax.random.PRNGKey(3), 1),
                self.env,
                pmapped_timestep,
                jnp.zeros((1,), dtype=jnp.float32),
                self.config,
            )
            aggregate = eval_ppo.aggregate_device_stats(pmapped_stats)
            self._assert_rollout_stats(aggregate)

            two_devices = jtu.tree_map(
                lambda x: jnp.stack([jnp.asarray(x), jnp.asarray(x)]),
                direct_stats,
            )
            aggregate = eval_ppo.aggregate_device_stats(two_devices)
            self.assertEqual(int(aggregate.positive_terminations), 8)
            self.assertEqual(int(aggregate.initial_episode_successes), 2)
            self.assertEqual(int(aggregate.initial_episode_terminations), 4)


if __name__ == "__main__":
    unittest.main()
