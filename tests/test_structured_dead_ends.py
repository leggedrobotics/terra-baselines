"""Loaded dead-end rollout wiring; native geometry has its own Terra tests."""

from collections import namedtuple
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from terra.structured_actions import StructuredClock, StructuredTimeConfig, StructuredTransition
from train_structured import (
    environment_kernels, episode_log, initial_runner, make_rollout, parse_args,
    validate_resume_training,
)
from utils.structured_ppo import duration_gae, sample_action


AgentConfig = namedtuple("AgentConfig", "angles_base angles_cabin")
Config = namedtuple("Config", "tile_size agent enforce_foundation_border_alignment")
Agent = namedtuple("Agent", "num_agents loaded agent_type action_type")
Map = namedtuple("Map", "map")
World = namedtuple("World", "target_map action_map")


class State(namedtuple("State", "env_cfg agent world env_steps")):
    def _get_current_agent_state(self):
        return self.agent

    def _get_task_completion(self, action, target):
        return {"absolute_completion": jnp.mean(action == target)}


def fixture(loaded=1):
    return State(Config(.5, AgentConfig(12, 12), False),
                 Agent(jnp.int32(1), jnp.array([loaded]), jnp.array([0]), jnp.array([0])),
                 World(Map(jnp.array([[-1]])), Map(jnp.array([[0]]))), jnp.int32(0))


def blocked_masks(state):
    return dict(action_mask=jnp.array([False] * 4 + [True] * 4),
                move_mask=jnp.zeros((2, 5), bool), turn_mask=jnp.zeros((2, 6), bool),
                do_mask=jnp.zeros(12, bool))


def wait_transition(state, action, *, clock, **kwargs):
    """The only sampled action is native WAIT: unchanged state, one decision."""
    info = dict(time_visit_open=clock.visit_open, time_moved=clock.moved,
                action_had_effect=jnp.bool_(False), material_or_load_changed=jnp.bool_(False),
                transition_mass_residual=jnp.int32(0), target_mutation=jnp.bool_(False),
                obstacle_mutation=jnp.bool_(False))
    return StructuredTransition(state._replace(env_steps=state.env_steps + 1),
                                jnp.float32(0), jnp.float32(0), info)


def policy_forward(model, params, obs, previous, hidden, config):
    shape = previous.shape[:-1]
    logits = dict(action=jnp.zeros(shape + (8,)), move=jnp.zeros(shape + (2, 5)),
                  turn=jnp.zeros(shape + (2, 6)), do=jnp.zeros(shape + (12,)))
    return jnp.where(obs["loaded"] > 0, 2., 100.), logits, hidden + 1


class LoadedDeadEndTrainerTests(unittest.TestCase):
    def setUp(self):
        self.mocks = ExitStack()
        self.addCleanup(self.mocks.close)
        for name, replacement in (
                ("LocalMapWrapper.wrap", lambda state, **kwargs: state),
                ("TerraEnv._state_to_obs_dict", lambda state: {"loaded": state.agent.loaded[0]}),
                ("structured_action_masks", blocked_masks),
                ("structured_transition", wait_transition),
                ("forward_step", policy_forward)):
            self.mocks.enter_context(patch("train_structured." + name, side_effect=replacement))
        self.state = fixture()

    def kernels(self, enabled):
        return environment_kernels(self.state.env_cfg, {}, time_budget_s=100., decision_limit=10,
                                   timing=StructuredTimeConfig(), allow_wait=False,
                                   terminate_loaded_dead_ends=enabled)

    def test_enabled_fails_after_one_zero_duration_wait_disabled_preserves_old_behavior(self):
        for enabled in (True, False):
            observe, advance = self.kernels(enabled)
            _, masks, _ = observe(self.state)
            self.assertEqual(bool(masks["loaded_dead_end"]), enabled)
            np.testing.assert_array_equal(masks["action_mask"], [False] * 7 + [True])
            _, logits, _ = policy_forward(None, None, {"loaded": jnp.int32(1)},
                                          jnp.zeros(5), jnp.zeros(2), {})
            action, _ = sample_action(logits, masks, jax.random.PRNGKey(0))
            self.assertEqual(int(action.action), 7)
            output = jax.jit(advance)(self.state, action, StructuredClock(), jnp.float32(3.),
                                      masks["loaded_dead_end"])
            state, reward, duration, elapsed, _, done, success = output[:7]
            self.assertEqual(int(state.env_steps), 1)
            self.assertEqual(float(duration), 0.)
            self.assertEqual(float(elapsed), 3.)
            self.assertEqual(bool(done), enabled)
            self.assertFalse(bool(success))
            self.assertEqual(float(reward), -1. if enabled else 0.)

    def test_rollout_resets_and_zero_duration_terminal_never_bootstraps(self):
        config = dict(actor_core="gru", actor_gru_hidden_dim=2, num_prev_actions=5)
        observe, advance = self.kernels(True)
        bank = jax.tree.map(lambda value: jnp.asarray(value)[None], self.state)
        runner = jax.tree.map(jnp.asarray, initial_runner(bank, config, jax.random.PRNGKey(1)))
        runner = runner._replace(hidden=jnp.ones((1, 2)) * 9)

        def fresh(data, key, state):
            return state._replace(env_steps=jnp.int32(0),
                                  agent=state.agent._replace(loaded=jnp.zeros_like(state.agent.loaded)))

        rollout = make_rollout(None, config, observe, advance, fresh, num_steps=2)
        runner, transitions, _, stats = jax.jit(rollout)(None, runner, None)
        np.testing.assert_array_equal(transitions.done, [[True, False]])
        np.testing.assert_array_equal(transitions.reward, [[-1., 0.]])
        np.testing.assert_array_equal(transitions.duration_s, [[0., 0.]])
        np.testing.assert_array_equal(runner.hidden, [[1., 1.]])
        self.assertEqual(int(runner.state.env_steps[0]), 1)
        self.assertEqual(int(runner.state.agent.loaded[0, 0]), 0)
        advantages, _ = duration_gae(transitions, jnp.array([900.]))
        np.testing.assert_allclose(advantages, [[-3., 800.]])
        log = episode_log(stats)
        self.assertEqual(log["episode/count"], 1)
        self.assertEqual(log["episode/loaded_dead_ends"], 1)
        self.assertEqual(log["episode/bulk_loaded_dead_end_rate"], 1.)
        self.assertIsNone(log["episode/precision_loaded_dead_end_rate"])


class LoadedDeadEndProtocolTests(unittest.TestCase):
    def test_flag_defaults_on_and_can_explicitly_preserve_old_behavior(self):
        argv = ["--initial-states", "bank.pkl", "--output", "out"]
        self.assertTrue(parse_args(argv).terminate_loaded_dead_ends)
        self.assertFalse(parse_args(argv + ["--no-terminate-loaded-dead-ends"]).terminate_loaded_dead_ends)

    def test_resume_rejects_silent_protocol_changes_including_old_checkpoints(self):
        for saved in ({}, {"terminate_loaded_dead_ends": False}):
            validate_resume_training(saved, {"terminate_loaded_dead_ends": False})
            with self.assertRaisesRegex(ValueError, "terminate_loaded_dead_ends"):
                validate_resume_training(saved, {"terminate_loaded_dead_ends": True})
        validate_resume_training({"terminate_loaded_dead_ends": True}, {"terminate_loaded_dead_ends": True})
        with self.assertRaisesRegex(ValueError, "terminate_loaded_dead_ends"):
            validate_resume_training({"terminate_loaded_dead_ends": True}, {"terminate_loaded_dead_ends": False})


if __name__ == "__main__":
    unittest.main()
