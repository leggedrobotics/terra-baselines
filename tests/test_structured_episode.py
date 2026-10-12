"""Trainer integration at success/failure boundaries and the following reset."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from terra.structured_actions import StructuredTimeConfig, StructuredTransition
from train_structured import environment_kernels, initial_runner, make_rollout
from utils.structured_ppo import duration_gae
import test_structured_dead_ends as fixtures


class StructuredEpisodeRolloutTests(unittest.TestCase):
    def test_success_and_cap_failure_reward_once_then_reset_context_and_gae(self):
        for complete, expected_reward in ((True, 7.6), (False, -1.)):
            with self.subTest(complete=complete), ExitStack() as mocks:
                initial = fixtures.fixture(loaded=0)._replace(env_steps=jnp.int32(9))
                model_config = dict(actor_core="gru", actor_gru_hidden_dim=2, num_prev_actions=5)

                def transition(state, action, *, clock, **kwargs):
                    outcome = fixtures.wait_transition(state, action, clock=clock)
                    # Only the initial lane reaches a terminal boundary; its
                    # fresh replacement must remain incomplete on the next step.
                    terrain = jnp.where((state.env_steps == 9) & complete,
                                        state.world.target_map.map, state.world.action_map.map)
                    next_state = outcome.state._replace(world=state.world._replace(
                        action_map=state.world.action_map._replace(map=terrain)))
                    return StructuredTransition(next_state, outcome.reward, jnp.float32(10.), outcome.info)

                def forward(model, params, obs, previous, hidden, config):
                    _, logits, hidden = fixtures.policy_forward(model, params, obs, previous, hidden, config)
                    return jnp.zeros(previous.shape[:-1]), logits, hidden

                for name, replacement in (
                        ("LocalMapWrapper.wrap", lambda state, **kwargs: state),
                        ("TerraEnv._state_to_obs_dict", lambda state: {"loaded": state.agent.loaded[0]}),
                        ("structured_action_masks", fixtures.blocked_masks),
                        ("structured_transition", transition),
                        ("forward_step", forward)):
                    mocks.enter_context(patch("train_structured." + name, side_effect=replacement))
                observe, advance = environment_kernels(
                    initial.env_cfg, {}, time_budget_s=100., decision_limit=10,
                    timing=StructuredTimeConfig(), allow_wait=False,
                    success_time_bonus=2., success_decision_bonus=1.)
                bank = jax.tree.map(lambda x: jnp.asarray(x)[None], initial)
                runner = jax.tree.map(jnp.asarray,
                                     initial_runner(bank, model_config, jax.random.PRNGKey(1)))
                runner = runner._replace(elapsed_s=jnp.array([10.]), hidden=jnp.ones((1, 2)) * 9,
                                         previous_types=jnp.ones((1, 5), jnp.int32))

                def fresh(data, key, state):
                    return state._replace(env_steps=jnp.int32(0), world=state.world._replace(
                        action_map=state.world.action_map._replace(map=jnp.zeros_like(state.world.action_map.map))))

                rollout = make_rollout(None, model_config, observe, advance, fresh, num_steps=2)
                runner, transitions, _, stats = jax.jit(rollout)(None, runner, None)
                np.testing.assert_array_equal(transitions.done, [[True, False]])
                np.testing.assert_array_equal(transitions.task_done, [[complete, False]])
                np.testing.assert_allclose(transitions.reward, [[expected_reward, 0.]])
                np.testing.assert_array_equal(transitions.prev_actions[:, 1], np.zeros((1, 5)))
                np.testing.assert_array_equal(runner.hidden, [[1., 1.]])
                np.testing.assert_array_equal(runner.elapsed_s, [10.])
                self.assertEqual(int(runner.state.env_steps[0]), 1)
                self.assertEqual(int(stats["episodes_bulk"]), 1)
                self.assertEqual(int(stats["success_bulk"]), int(complete))
                self.assertEqual(int(stats["loaded_dead_ends_bulk"]), 0)
                # A large value after reset cannot leak through the first
                # terminal transition or repeat its terminal bonus.
                advantages, _ = duration_gae(transitions, jnp.array([900.]), gamma=1.)
                np.testing.assert_allclose(advantages, [[expected_reward, 900.]])

    def test_disabled_deadline_keeps_time_reference_in_trainer(self):
        state = fixtures.fixture(loaded=0)
        with patch("train_structured.structured_transition", side_effect=fixtures.wait_transition):
            _, advance = environment_kernels(
                state.env_cfg, {}, time_budget_s=100., decision_limit=10,
                timing=StructuredTimeConfig(), allow_wait=False, time_limit=False,
                success_time_bonus=2., success_decision_bonus=1.)
            from terra.structured_actions import StructuredAction, StructuredClock
            action = StructuredAction(jnp.int32(7))
            result = jax.jit(advance)(state, action, StructuredClock(), jnp.float32(1000.))
            self.assertFalse(bool(result[5]))
            self.assertEqual(float(result[1]), 0.)
            self.assertEqual(float(result[8]), 100.)
            completed = state._replace(world=state.world._replace(action_map=state.world.target_map))
            result = jax.jit(advance)(completed, action, StructuredClock(), jnp.float32(1000.))
            self.assertTrue(bool(result[6]))
            self.assertAlmostEqual(float(result[1]), 6.9, places=5)


if __name__ == "__main__":
    unittest.main()
