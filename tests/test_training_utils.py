import unittest
from dataclasses import asdict
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from terra.config import EnvConfig
from train import TrainConfig
from train_mixed import (
    MixedAgentTrainConfig,
    _backfill_terminal_rewards,
    _strip_checkpoint_env_axis,
    _num_agents_from_env_params,
    _validate_checkpoint_architecture,
    _validate_checkpoint_history_width,
    _validate_resume_update,
)
from utils.helpers import replicate_checkpoint_env_config


class TrainingAccountingTest(unittest.TestCase):
    def test_global_step_accounting(self):
        config = MixedAgentTrainConfig(
            name="test",
            num_devices=1,
            num_envs_per_device=8,
            num_steps=4,
            num_minibatches=2,
            total_timesteps=96,
        )
        self.assertEqual(config.env_steps_per_update, 32)
        self.assertEqual(config.num_updates, 3)
        self.assertEqual(config.actual_total_timesteps, 96)
        self.assertIn("cache_clear_interval", asdict(config))
        self.assertIn("map_encoder", asdict(config))

        standard = TrainConfig(
            name="test",
            num_devices=1,
            num_envs_per_device=8,
            num_steps=4,
            num_minibatches=2,
            total_timesteps=96,
        )
        self.assertEqual(standard.num_updates, 3)
        self.assertEqual(standard.actual_total_timesteps, 96)

    def test_invalid_agent_action_lengths_fail(self):
        with self.assertRaises(ValueError):
            MixedAgentTrainConfig(
                name="test",
                num_devices=1,
                num_envs_per_device=8,
                num_minibatches=2,
                total_timesteps=32,
                agent_types_override=(0, 1),
                action_types_override=(0,),
            )

    def test_new_configs_store_canonical_encoder_names(self):
        config = MixedAgentTrainConfig(
            name="test",
            num_devices=1,
            num_envs_per_device=8,
            num_steps=4,
            num_minibatches=2,
            total_timesteps=32,
            map_encoder="resnet_spatial_v2",
        )
        self.assertEqual(config.map_encoder, "resnet_spatial_8x8")

    def test_four_device_accounting_does_not_divide_twice(self):
        with patch("train_mixed.jax.local_device_count", return_value=4), patch(
            "train_mixed.jax.devices", return_value=["cpu"] * 4
        ):
            config = MixedAgentTrainConfig(
                name="test",
                num_devices=4,
                num_envs_per_device=8,
                num_steps=4,
                num_minibatches=2,
                total_timesteps=384,
            )
        self.assertEqual(config.env_steps_per_update, 128)
        self.assertEqual(config.num_updates, 3)
        self.assertEqual(config.actual_total_timesteps, 384)


class TerminalBackfillTest(unittest.TestCase):
    def test_backfill_is_causal_and_stays_within_episode(self):
        rewards = jnp.zeros((6, 1), dtype=jnp.float32)
        terminal = jnp.array([[0], [0], [5], [0], [7], [99]], dtype=jnp.float32)
        done = jnp.array([[False], [False], [True], [False], [True], [False]])
        num_agents = jnp.array([3], dtype=jnp.int32)

        result = _backfill_terminal_rewards(rewards, terminal, done, num_agents)

        np.testing.assert_allclose(
            np.asarray(result[:, 0]),
            np.array([5, 5, 0, 7, 0, 0], dtype=np.float32),
        )


class CheckpointCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            map_encoder="atari",
            model_core="mlp",
            model_size="base",
            num_prev_actions=10,
        )

    def test_legacy_checkpoint_defaults_to_atari(self):
        _validate_checkpoint_architecture({"train_config": {}}, self.config)

    def test_encoder_mismatch_fails_before_model_init(self):
        checkpoint = {"train_config": {"map_encoder": "resnet_global_pool"}}
        with self.assertRaisesRegex(ValueError, "map_encoder"):
            _validate_checkpoint_architecture(checkpoint, self.config)

        spatial_config = SimpleNamespace(
            map_encoder="resnet_spatial_8x8",
            model_core="mlp",
            model_size="base",
        )
        with self.assertRaisesRegex(ValueError, "map_encoder"):
            _validate_checkpoint_architecture(checkpoint, spatial_config)

        # Historical names are aliases, not new architectures.
        _validate_checkpoint_architecture(
            {"train_config": {"map_encoder": "resnet_delayed"}},
            SimpleNamespace(
                map_encoder="resnet_global_pool",
                model_core="mlp",
                model_size="base",
            ),
        )
        _validate_checkpoint_architecture(
            {"train_config": {"map_encoder": "resnet_spatial_v2"}},
            spatial_config,
        )

    def test_history_width_mismatch_fails(self):
        checkpoint = {"train_config": {"num_prev_actions": 20}}
        with self.assertRaisesRegex(ValueError, "action-history width"):
            _validate_checkpoint_history_width(checkpoint, self.config)

    def test_resume_update_must_leave_work(self):
        _validate_resume_update(0, 3)
        _validate_resume_update(2, 3)
        with self.assertRaises(ValueError):
            _validate_resume_update(-1, 3)
        with self.assertRaises(ValueError):
            _validate_resume_update(3, 3)

    def test_checkpoint_env_axis_keeps_agent_vectors(self):
        class MiniEnvConfig(NamedTuple):
            agent_types: jnp.ndarray
            action_types: jnp.ndarray
            max_steps: jnp.ndarray

        batched = MiniEnvConfig(
            agent_types=jnp.array([[0, 2], [0, 2], [0, 2]]),
            action_types=jnp.array([[0, 0], [0, 0], [0, 0]]),
            max_steps=jnp.array([100, 100, 100]),
        )
        scalar = _strip_checkpoint_env_axis(batched, num_envs_per_device=3)
        np.testing.assert_array_equal(np.asarray(scalar.agent_types), [0, 2])
        np.testing.assert_array_equal(np.asarray(scalar.action_types), [0, 0])
        self.assertEqual(np.asarray(scalar.max_steps).shape, ())

        # Scalarizing an already-scalar two-agent config must be idempotent,
        # even when num_envs_per_device happens to equal num_agents.
        scalar_again = _strip_checkpoint_env_axis(scalar, num_envs_per_device=2)
        np.testing.assert_array_equal(np.asarray(scalar_again.agent_types), [0, 2])
        np.testing.assert_array_equal(np.asarray(scalar_again.action_types), [0, 0])

    def test_real_env_config_tuple_layout_round_trips(self):
        config = EnvConfig()._replace(
            agent_types=(0, 2),
            action_types=(0, 0),
        )
        batched = jtu.tree_map(
            lambda x: jnp.asarray(x)[None, None].repeat(2, 0).repeat(3, 1),
            config,
        )
        self.assertIsInstance(batched.agent_types, tuple)
        self.assertEqual(_num_agents_from_env_params(batched), 2)

        scalar = _strip_checkpoint_env_axis(batched, num_envs_per_device=3)
        np.testing.assert_array_equal(np.asarray(scalar.agent_types), [0, 2])
        np.testing.assert_array_equal(np.asarray(scalar.action_types), [0, 0])
        self.assertEqual(_num_agents_from_env_params(scalar), 2)

        scalar_again = _strip_checkpoint_env_axis(scalar, num_envs_per_device=2)
        np.testing.assert_array_equal(np.asarray(scalar_again.agent_types), [0, 2])
        np.testing.assert_array_equal(np.asarray(scalar_again.action_types), [0, 0])

        replicated = replicate_checkpoint_env_config(scalar, n_envs=4)
        np.testing.assert_array_equal(
            np.asarray(replicated.agent_types),
            np.tile(np.array([[0, 2]]), (4, 1)),
        )
        self.assertEqual(np.asarray(replicated.tile_size).shape, (4,))

        legacy_replicated = replicate_checkpoint_env_config(batched, n_envs=4)
        np.testing.assert_array_equal(
            np.asarray(legacy_replicated.agent_types),
            np.tile(np.array([[0, 2]]), (4, 1)),
        )


if __name__ == "__main__":
    unittest.main()
