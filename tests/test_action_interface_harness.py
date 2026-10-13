"""Matched interfaces share native rules, episode handling and saved protocol."""

from collections import namedtuple
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from terra.structured_actions import StructuredAction, StructuredClock, StructuredTimeConfig
from train_structured import (
    checkpoint_action_interface, environment_kernels, initial_runner, make_rollout,
    parse_args, policy_snapshot, snapshot_log, validate_resume_training,
)
import test_structured_dead_ends as fixtures


def state_with_cabin(loaded=1, cabin=3):
    state = fixtures.fixture(loaded)
    Agent = namedtuple("AgentWithCabin", fixtures.Agent._fields + ("angle_cabin",))
    return state._replace(agent=Agent(*state.agent, jnp.array([cabin], jnp.int32)))


def available_masks(state):
    return dict(action_mask=jnp.ones(8, bool),
                move_mask=jnp.ones((2, 5), bool), turn_mask=jnp.ones((2, 6), bool),
                do_mask=jnp.arange(12) == 5)


class ActionInterfaceHarnessTests(unittest.TestCase):
    def test_defaults_and_saved_protocol_reject_cross_interface_reuse(self):
        argv = ["--initial-states", "bank.pkl", "--output", "out"]
        self.assertEqual(parse_args(argv).action_interface, "structured")
        args = parse_args(argv + ["--action-interface", "primitive8", "--save-initial-checkpoint"])
        self.assertEqual(args.action_interface, "primitive8")
        self.assertTrue(args.save_initial_checkpoint)
        old = dict(model_config={}, training_config={})
        self.assertEqual(checkpoint_action_interface(old), "structured")
        with self.assertRaisesRegex(ValueError, "requested primitive8"):
            checkpoint_action_interface(old, "primitive8")
        saved = dict(action_interface="primitive8", model_config=dict(action_interface="primitive8"),
                     training_config=dict(action_interface="primitive8"))
        self.assertEqual(checkpoint_action_interface(saved, "primitive8"), "primitive8")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            checkpoint_action_interface({**saved, "training_config": {}})
        common = dict(terminate_loaded_dead_ends=False)
        validate_resume_training(common, common)
        with self.assertRaisesRegex(ValueError, "action_interface"):
            validate_resume_training(common, {**common, "action_interface": "primitive8"})
        validate_resume_training({**common, "action_interface": "primitive8", "save_initial_checkpoint": True},
                                 {**common, "action_interface": "primitive8", "save_initial_checkpoint": False})

    def test_current_heading_legality_does_not_become_a_false_loaded_dead_end(self):
        state = state_with_cabin()
        with ExitStack() as mocks:
            for name, replacement in (
                    ("LocalMapWrapper.wrap", lambda state, **kwargs: state),
                    ("TerraEnv._state_to_obs_dict", lambda state: {}),
                    ("structured_action_masks", available_masks)):
                mocks.enter_context(patch("train_structured." + name, side_effect=replacement))
            for interface in ("structured", "primitive8"):
                observe, _ = environment_kernels(
                    state.env_cfg, dict(action_interface=interface), time_budget_s=100., decision_limit=2,
                    timing=StructuredTimeConfig(), allow_wait=False)
                _, masks, _ = observe(state)
                self.assertFalse(bool(masks["loaded_dead_end"]))
                self.assertEqual(bool(masks["action_mask"][6]), interface == "structured")
                self.assertEqual(bool(masks["action_mask"][4]), interface == "primitive8")
                self.assertTrue(bool(masks["do_mask"][5]))
                _, at_target, _ = observe(state._replace(agent=state.agent._replace(angle_cabin=jnp.array([5]))))
                self.assertTrue(bool(at_target["action_mask"][6]))
            with patch("train_structured.structured_action_masks", side_effect=lambda s: {
                    **available_masks(s), "do_mask": jnp.zeros(12, bool)}):
                _, masks, _ = observe(state)
                self.assertTrue(bool(masks["loaded_dead_end"]))
                np.testing.assert_array_equal(masks["action_mask"], np.arange(8) == 7)

    def test_primitive_rollout_routes_cabin_actions_and_keeps_terminal_reset(self):
        state = state_with_cabin()
        config = dict(action_interface="primitive8", actor_core="gru", actor_gru_hidden_dim=2,
                      num_prev_actions=5)

        def forward(model, params, obs, previous, hidden, config):
            value, logits, next_hidden = fixtures.policy_forward(model, params, obs, previous, hidden, config)
            logits["action"] = logits["action"].at[..., 4].set(100.)
            return value, logits, next_hidden

        with ExitStack() as mocks:
            for name, replacement in (
                    ("LocalMapWrapper.wrap", lambda state, **kwargs: state),
                    ("TerraEnv._state_to_obs_dict", lambda state: {"loaded": state.agent.loaded[0]}),
                    ("structured_action_masks", available_masks),
                    ("structured_transition", fixtures.wait_transition),
                    ("forward_step", forward)):
                mocks.enter_context(patch("train_structured." + name, side_effect=replacement))
            observe, advance = environment_kernels(
                state.env_cfg, config, time_budget_s=100., decision_limit=2,
                timing=StructuredTimeConfig(), allow_wait=False)
            bank = jax.tree.map(lambda value: jnp.asarray(value)[None], state)
            runner = jax.tree.map(jnp.asarray, initial_runner(bank, config, jax.random.PRNGKey(1)))
            fresh = lambda data, key, state: state._replace(env_steps=jnp.int32(0))
            rollout = make_rollout(None, config, observe, advance, fresh, num_steps=2, diagnostic_lanes=1)
            runner, transition, _, stats = jax.jit(rollout)(None, runner, None)
            np.testing.assert_array_equal(transition.action.action, [[4, 4]])
            np.testing.assert_array_equal(transition.action.amount, [[1, 1]])
            np.testing.assert_array_equal(transition.action.heading, [[-1, -1]])
            np.testing.assert_array_equal(transition.done, [[False, True]])
            np.testing.assert_allclose(transition.reward, [[0., -1.]])
            np.testing.assert_array_equal(runner.hidden, [[0., 0.]])
            self.assertEqual(stats["snapshot_move_probs"].shape, (1, 2, 5))
            self.assertEqual(int(stats["snapshot_action"][0]), 4)

    def test_snapshot_groups_use_actual_counts_and_type_weighted_argument_entropy(self):
        snapshot = dict(
            action_probs=np.array([[0., 0., 0., 0., 0., 0., 1., 0.],
                                   [0.5, 0.5, 0., 0., 0., 0., 0., 0.]]),
            action_mask=np.ones((2, 8), bool), precision=np.ones(2, bool),
            loaded=np.zeros(2, bool), dug_fraction=np.array([.9, .9]),
            move_probs=np.ones((2, 2, 5)) / 5, turn_probs=np.ones((2, 2, 6)) / 6,
            do_probs=np.array([[1.] + [0.] * 11, [1. / 12] * 12]))
        log = snapshot_log(snapshot, "structured")
        prefix = "policy_sample/precision/empty/finishing/"
        self.assertEqual(log[prefix + "count"], 2)
        self.assertEqual(log[prefix + "p_do"], .5)
        self.assertEqual(log[prefix + "do_mass"], 1.)
        self.assertEqual(log[prefix + "entropy_do"], 0.)
        self.assertNotIn(prefix + "entropy_turn", log)
        primitive = snapshot_log(snapshot, "primitive8")
        self.assertFalse(any("entropy" in key for key in primitive))


if __name__ == "__main__":
    unittest.main()
