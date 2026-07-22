import unittest
from dataclasses import asdict
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from flax.training.train_state import TrainState

from terra.actions import TrackedAction
from terra.config import BatchConfig, EnvConfig, MapsDimsConfig, RewardsType
from train import (
    Transition as BaseTransition,
    TrainConfig,
    ppo_update_networks,
    downsample_teacher_obs,
)
from train_mixed import (
    MixedAgentTrainConfig,
    kickstart_coef_schedule,
    _backfill_terminal_rewards,
    _strip_checkpoint_env_axis,
    _num_agents_from_env_params,
    _validate_checkpoint_architecture,
    _validate_checkpoint_history_width,
    _validate_resume_update,
)
from utils.helpers import checkpoint_batch_config, replicate_checkpoint_env_config
from utils.models import get_model_ready


class _PPOConfig(dict):
    """Dict config that supports attribute access for ppo_update_networks."""

    __getattr__ = dict.__getitem__


def _model_config(num_prev_actions=5, map_edge=64):
    return _PPOConfig(
        clip_action_maps=True,
        loaded_max=6,
        local_map_normalization_bounds=(-1, 1),
        map_encoder="atari",
        maps_net_normalization_bounds=(-1, 1),
        model_core="mlp",
        model_size="base",
        num_prev_actions=num_prev_actions,
    )


def _ppo_config(use_value_clip=True, flat_minibatch_shuffle=False, teacher_obs_downsample=1):
    return _PPOConfig(
        clip_eps=0.2,
        vf_coef=2.0,
        ent_coef=0.01,
        clip_action_maps=True,
        use_value_clip=use_value_clip,
        flat_minibatch_shuffle=flat_minibatch_shuffle,
        teacher_obs_downsample=teacher_obs_downsample,
    )


def _make_train_state(rng_seed=0, num_prev_actions=5, map_edge=64):
    env = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=map_edge))
    )
    model, params = get_model_ready(
        jax.random.PRNGKey(rng_seed), _model_config(num_prev_actions, map_edge), env
    )
    train_state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(1e-3)
    )
    return train_state, env


def _make_transition(shape_prefix, env, num_prev_actions, old_value, log_prob=0.0):
    """Build a train.Transition whose obs/prev_actions match obs_to_model_input."""
    edge = env.batch_cfg.maps_dims.maps_edge_length
    angles = env.batch_cfg.agent.angles_cabin
    num_state_obs = env.batch_cfg.agent.num_state_obs
    max_agents = 4

    def m(*trailing):
        return jnp.zeros(shape_prefix + trailing, dtype=jnp.float32)

    grid = lambda: m(edge, edge)
    local = lambda: m(angles)
    agent_active = jnp.zeros(shape_prefix + (max_agents,), dtype=jnp.int8)
    agent_active = agent_active.at[..., 0].set(1)
    obs = {
        "agent_states": m(max_agents, num_state_obs),
        "agent_active": agent_active,
        "num_agents": jnp.ones(shape_prefix, dtype=jnp.int32),
        "local_map_action_neg": local(),
        "local_map_action_pos": local(),
        "local_map_target_neg": local(),
        "local_map_target_pos": local(),
        "local_map_dumpability": local(),
        "local_map_obstacles": local(),
        "local_map_border_workspace": local(),
        "local_map_edge_alignment_error": local(),
        "local_map_border_diggable": local(),
        "traversability_mask": grid(),
        "reachability_mask": grid(),
        "action_map": grid(),
        "target_map": grid(),
        "agent_width": jnp.zeros(shape_prefix, dtype=jnp.int32),
        "agent_height": jnp.zeros(shape_prefix, dtype=jnp.int32),
        "padding_mask": grid(),
        "dumpability_mask": grid(),
        "interaction_mask": grid(),
    }
    return BaseTransition(
        done=jnp.zeros(shape_prefix, dtype=jnp.bool_),
        task_done=jnp.zeros(shape_prefix, dtype=jnp.bool_),
        action=jnp.zeros(shape_prefix, dtype=jnp.int32),
        value=jnp.full(shape_prefix, old_value, dtype=jnp.float32),
        reward=jnp.zeros(shape_prefix, dtype=jnp.float32),
        log_prob=jnp.full(shape_prefix, log_prob, dtype=jnp.float32),
        obs=obs,
        prev_actions=jnp.zeros(shape_prefix + (num_prev_actions,), dtype=jnp.int32),
        prev_reward=jnp.zeros(shape_prefix, dtype=jnp.float32),
    )


def _run_ppo_update(config, train_state, transitions, advantages, targets, **kwargs):
    """Run ppo_update_networks under a size-1 'devices' axis (mirrors pmap)."""

    def _f(ts, tr, adv, tgt):
        return ppo_update_networks(ts, tr, adv, tgt, config, **kwargs)

    add = lambda x: jnp.asarray(x)[None]
    _, info = jax.vmap(_f, axis_name="devices")(
        jtu.tree_map(add, train_state),
        jtu.tree_map(add, transitions),
        advantages[None],
        targets[None],
    )
    return jtu.tree_map(lambda x: x[0], info)


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

    def test_checkpoint_batch_config_restores_saved_map_curriculum(self):
        config = SimpleNamespace(
            curriculum_levels_override=[
                {
                    "maps_path": "foundations",
                    "max_steps_in_episode": 450,
                    "rewards_type": RewardsType.DENSE,
                    "apply_trench_rewards": False,
                }
            ],
            curriculum_increase_level_threshold=3,
            curriculum_decrease_level_threshold=7,
            curriculum_last_level_type="none",
        )

        batch_config = checkpoint_batch_config(config, TrackedAction)

        self.assertIs(batch_config.action_type, TrackedAction)
        self.assertEqual(batch_config.curriculum_global.levels, config.curriculum_levels_override)
        self.assertEqual(batch_config.curriculum_global.increase_level_threshold, 3)
        self.assertEqual(batch_config.curriculum_global.decrease_level_threshold, 7)
        self.assertEqual(batch_config.curriculum_global.last_level_type, "none")
        self.assertEqual(tuple(batch_config.curriculum_global), (3, 7))

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

        se_config = MixedAgentTrainConfig(
            name="test",
            num_devices=1,
            num_envs_per_device=8,
            num_steps=4,
            num_minibatches=2,
            total_timesteps=32,
            map_encoder="resnet_spatial_v3",
        )
        self.assertEqual(se_config.map_encoder, "resnet_spatial_8x8_se")

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


class ValueClipToggleTest(unittest.TestCase):
    def test_value_clip_on_off_differ_when_clipping_binds(self):
        train_state, env = _make_train_state(rng_seed=0)
        # Old value far from the near-zero fresh prediction and target 0 makes
        # the clipped-max value loss much larger than the plain MSE.
        transitions = _make_transition((2, 3), env, 5, old_value=5.0)
        advantages = jnp.ones((2, 3), dtype=jnp.float32)
        targets = jnp.zeros((2, 3), dtype=jnp.float32)

        info_clip = _run_ppo_update(
            _ppo_config(use_value_clip=True), train_state, transitions, advantages, targets
        )
        info_noclip = _run_ppo_update(
            _ppo_config(use_value_clip=False), train_state, transitions, advantages, targets
        )

        for info in (info_clip, info_noclip):
            self.assertTrue(bool(jnp.isfinite(info["total_loss"])))
            self.assertTrue(bool(jnp.isfinite(info["value_loss"])))
        # Clipping binds: clipped value loss is much larger than the plain MSE.
        self.assertGreater(
            float(info_clip["value_loss"]), float(info_noclip["value_loss"]) + 1.0
        )
        # No teacher -> no kickstart keys leak into update_info.
        self.assertNotIn("kickstart/kl", info_clip)

        # Default path is numerically the historical clipped objective:
        # recompute both value losses from a direct forward pass.
        from utils.utils_ppo import obs_to_model_input, policy

        config = _ppo_config()
        obs_flat = jtu.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
            transitions.obs,
        )
        prev_actions_flat = jnp.reshape(
            transitions.prev_actions, (-1, transitions.prev_actions.shape[-1])
        )
        model_obs = obs_to_model_input(dict(obs_flat), prev_actions_flat, config)
        value, _ = policy(train_state.apply_fn, train_state.params, model_obs)
        value = jnp.reshape(value[:, 0], transitions.value.shape)
        clipped_pred = transitions.value + (value - transitions.value).clip(
            -config.clip_eps, config.clip_eps
        )
        expected_clip = 0.5 * jnp.maximum(
            jnp.square(value - targets), jnp.square(clipped_pred - targets)
        ).mean()
        expected_noclip = 0.5 * jnp.square(value - targets).mean()
        self.assertAlmostEqual(
            float(info_clip["value_loss"]), float(expected_clip), places=5
        )
        self.assertAlmostEqual(
            float(info_noclip["value_loss"]), float(expected_noclip), places=5
        )


class FlatMinibatchShuffleTest(unittest.TestCase):
    def test_flat_update_produces_finite_losses(self):
        train_state, env = _make_train_state(rng_seed=0)
        # Flat layout: a single [n_samples] axis, no seq dimension.
        transitions = _make_transition((6,), env, 5, old_value=0.3)
        advantages = jax.random.normal(jax.random.PRNGKey(3), (6,))
        targets = jax.random.normal(jax.random.PRNGKey(4), (6,))

        info = _run_ppo_update(
            _ppo_config(flat_minibatch_shuffle=True),
            train_state,
            transitions,
            advantages,
            targets,
        )
        for key in ("total_loss", "value_loss", "actor_loss", "entropy"):
            self.assertTrue(bool(jnp.isfinite(info[key])), key)

    def test_blocked_default_path_still_runs(self):
        train_state, env = _make_train_state(rng_seed=0)
        transitions = _make_transition((2, 3), env, 5, old_value=0.3)
        advantages = jax.random.normal(jax.random.PRNGKey(3), (2, 3))
        targets = jax.random.normal(jax.random.PRNGKey(4), (2, 3))
        info = _run_ppo_update(
            _ppo_config(flat_minibatch_shuffle=False),
            train_state,
            transitions,
            advantages,
            targets,
        )
        self.assertTrue(bool(jnp.isfinite(info["total_loss"])))

    def test_flat_shuffle_consumes_every_sample_exactly_once(self):
        # Mirror the flat minibatch preparation: flatten [seq, env] -> [seq*env],
        # permute over all samples, reshape into minibatches, and confirm the
        # union of minibatch samples is exactly the full sample set.
        num_steps, num_envs, num_minibatches = 4, 6, 3
        n_samples = num_steps * num_envs
        # A [seq, env] field carrying a unique id per sample.
        ids = jnp.arange(n_samples, dtype=jnp.int32).reshape(num_steps, num_envs)
        flat = jnp.reshape(ids, (n_samples,) + ids.shape[2:])
        permutation = jax.random.permutation(jax.random.PRNGKey(0), n_samples)
        shuffled = jnp.take(flat, permutation, axis=0)
        minibatches = jnp.reshape(shuffled, (num_minibatches, -1) + shuffled.shape[1:])
        union = np.sort(np.asarray(minibatches).reshape(-1))
        np.testing.assert_array_equal(union, np.arange(n_samples))
        self.assertEqual(n_samples % num_minibatches, 0)


class KickstartScheduleTest(unittest.TestCase):
    def test_cosine_anneals_to_zero_and_clamps(self):
        self.assertEqual(kickstart_coef_schedule(0, 1.0, 100), 1.0)
        self.assertAlmostEqual(kickstart_coef_schedule(50, 1.0, 100), 0.5, places=6)
        self.assertEqual(kickstart_coef_schedule(100, 1.0, 100), 0.0)
        self.assertEqual(kickstart_coef_schedule(150, 1.0, 100), 0.0)  # clamped
        self.assertEqual(kickstart_coef_schedule(0, 0.5, 500), 0.5)
        self.assertAlmostEqual(kickstart_coef_schedule(250, 0.5, 500), 0.25, places=6)
        # A non-positive window is inert.
        self.assertEqual(kickstart_coef_schedule(5, 1.0, 0), 0.0)


class KickstartLossTermsTest(unittest.TestCase):
    def test_no_kickstart_terms_when_teacher_is_none(self):
        train_state, env = _make_train_state(rng_seed=0)
        transitions = _make_transition((2, 3), env, 5, old_value=0.3)
        advantages = jax.random.normal(jax.random.PRNGKey(3), (2, 3))
        targets = jax.random.normal(jax.random.PRNGKey(4), (2, 3))
        info = _run_ppo_update(
            _ppo_config(), train_state, transitions, advantages, targets
        )
        self.assertNotIn("kickstart/kl", info)
        self.assertNotIn("kickstart/value_mse", info)

    def test_identical_teacher_gives_zero_kickstart_terms(self):
        train_state, env = _make_train_state(rng_seed=0)
        transitions = _make_transition((2, 3), env, 5, old_value=0.3)
        advantages = jax.random.normal(jax.random.PRNGKey(3), (2, 3))
        targets = jax.random.normal(jax.random.PRNGKey(4), (2, 3))

        base_info = _run_ppo_update(
            _ppo_config(), train_state, transitions, advantages, targets
        )
        teacher_info = _run_ppo_update(
            _ppo_config(),
            train_state,
            transitions,
            advantages,
            targets,
            teacher_apply_fn=train_state.apply_fn,
            teacher_params=train_state.params,
            kickstart_kl_coef=1.0,
            kickstart_value_coef=1.0,
        )
        # Teacher == student -> KL and value MSE are ~0, total loss unchanged.
        self.assertIn("kickstart/kl", teacher_info)
        self.assertIn("kickstart/value_mse", teacher_info)
        self.assertLess(float(jnp.abs(teacher_info["kickstart/kl"])), 1e-5)
        self.assertLess(float(jnp.abs(teacher_info["kickstart/value_mse"])), 1e-6)
        self.assertAlmostEqual(
            float(teacher_info["total_loss"]), float(base_info["total_loss"]), places=5
        )

    def test_different_teacher_gives_positive_kl(self):
        train_state, env = _make_train_state(rng_seed=0)
        teacher_state, _ = _make_train_state(rng_seed=999)
        transitions = _make_transition((2, 3), env, 5, old_value=0.3)
        advantages = jax.random.normal(jax.random.PRNGKey(3), (2, 3))
        targets = jax.random.normal(jax.random.PRNGKey(4), (2, 3))

        info = _run_ppo_update(
            _ppo_config(),
            train_state,
            transitions,
            advantages,
            targets,
            teacher_apply_fn=teacher_state.apply_fn,
            teacher_params=teacher_state.params,
            kickstart_kl_coef=1.0,
            kickstart_value_coef=0.5,
        )
        self.assertTrue(bool(jnp.isfinite(info["kickstart/kl"])))
        self.assertGreater(float(info["kickstart/kl"]), 0.0)


class ResolutionScalingConfigTest(unittest.TestCase):
    """F15: resolution-scaling config plumbing and checkpoint arch validation."""

    def _base_kwargs(self, **extra):
        kwargs = dict(
            name="test",
            num_devices=1,
            num_envs_per_device=8,
            num_steps=4,
            num_minibatches=2,
            total_timesteps=32,
        )
        kwargs.update(extra)
        return kwargs

    def test_loaded_max_override_folds_into_loaded_max(self):
        config = MixedAgentTrainConfig(**self._base_kwargs(loaded_max_override=400))
        self.assertEqual(config.loaded_max, 400)
        # Default (no override) keeps the historical value.
        default = MixedAgentTrainConfig(**self._base_kwargs())
        self.assertEqual(default.loaded_max, 100)

    def test_teacher_downsample_requires_teacher(self):
        with self.assertRaisesRegex(ValueError, "teacher_obs_downsample"):
            MixedAgentTrainConfig(**self._base_kwargs(teacher_obs_downsample=2))
        with self.assertRaises(ValueError):
            MixedAgentTrainConfig(**self._base_kwargs(teacher_obs_downsample=0))
        # Valid together with a teacher.
        config = MixedAgentTrainConfig(
            **self._base_kwargs(
                teacher_obs_downsample=2, teacher_checkpoint="/tmp/does-not-exist.pkl"
            )
        )
        self.assertEqual(config.teacher_obs_downsample, 2)

    def test_stage_overrides_stored_as_tuples(self):
        config = MixedAgentTrainConfig(
            **self._base_kwargs(
                resnet_stage_channels=(16, 32, 48, 64, 64),
                resnet_blocks_per_stage=(1, 1, 2, 2, 2),
            )
        )
        self.assertEqual(config.resnet_stage_channels, (16, 32, 48, 64, 64))
        self.assertEqual(config.resnet_blocks_per_stage, (1, 1, 2, 2, 2))

    def test_arch_validation_flags_stage_override_mismatch(self):
        # Checkpoint trained with the default preset (no stage override) vs a
        # 5-stage current config: must be flagged.
        checkpoint = {"train_config": {"map_encoder": "resnet_spatial_8x8"}}
        current = SimpleNamespace(
            map_encoder="resnet_spatial_8x8",
            model_core="mlp",
            model_size="base",
            critic_hidden_dims=None,
            resnet_stage_channels=(16, 32, 48, 64, 64),
            resnet_blocks_per_stage=(1, 1, 2, 2, 2),
        )
        with self.assertRaisesRegex(ValueError, "resnet_stage_channels"):
            _validate_checkpoint_architecture(checkpoint, current)

    def test_arch_validation_matches_equal_stage_overrides(self):
        # List vs tuple must read as equal; matching stage specs pass.
        checkpoint = {
            "train_config": {
                "map_encoder": "resnet_spatial_8x8",
                "resnet_stage_channels": [16, 32, 48, 64, 64],
                "resnet_blocks_per_stage": [1, 1, 2, 2, 2],
            }
        }
        current = SimpleNamespace(
            map_encoder="resnet_spatial_8x8",
            model_core="mlp",
            model_size="base",
            critic_hidden_dims=None,
            resnet_stage_channels=(16, 32, 48, 64, 64),
            resnet_blocks_per_stage=(1, 1, 2, 2, 2),
        )
        _validate_checkpoint_architecture(checkpoint, current)

    def test_five_stage_override_builds_128_policy(self):
        # Parsing -> model: the config tuples drive get_model_ready to build a
        # 5-stage trunk that keeps a 128x128 input at an 8x8 readout.
        env = SimpleNamespace(
            batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=128))
        )
        config = _PPOConfig(
            clip_action_maps=False,
            loaded_max=6,
            local_map_normalization_bounds=(-1, 1),
            map_encoder="resnet_spatial_8x8",
            maps_net_normalization_bounds=(-1, 1),
            model_core="mlp",
            model_size="base",
            num_prev_actions=5,
            resnet_stage_channels=(16, 32, 48, 64, 64),
            resnet_blocks_per_stage=(1, 1, 2, 2, 2),
        )
        model, params = get_model_ready(jax.random.PRNGKey(0), config, env)
        # The flatten readout Dense reads 8*8*64 rows -> the 5th stage kept 8x8.
        dense_kernels = [
            leaf for leaf in jax.tree_util.tree_leaves(params)
            if getattr(leaf, "ndim", 0) == 2
        ]
        self.assertTrue(any(k.shape[0] == 8 * 8 * 64 for k in dense_kernels))


class TeacherObsDownsampleTest(unittest.TestCase):
    """F15: the teacher-forward-only obs subsample transform."""

    def _crafted_obs(self):
        # 4x4 global maps with distinct values so subsampling is verifiable.
        base = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4)
        obs = [None] * 22
        agent_states = jnp.zeros((1, 4, 8), dtype=jnp.float32)
        agent_states = agent_states.at[0, 0, 0].set(6.0)  # pos_x
        agent_states = agent_states.at[0, 0, 1].set(3.0)  # pos_y
        agent_states = agent_states.at[0, 0, 2].set(9.0)  # angle_base (untouched)
        obs[0] = agent_states
        for i in range(1, 22):
            obs[i] = jnp.zeros((1,), dtype=jnp.float32)
        for idx in (12, 13, 14, 15, 18, 19, 20):
            obs[idx] = base
        obs[16] = jnp.array([10], dtype=jnp.int32)
        obs[17] = jnp.array([18], dtype=jnp.int32)
        return obs

    def test_downsample_two_subsamples_maps_and_halves_positions(self):
        obs = self._crafted_obs()
        out = downsample_teacher_obs(obs, 2)
        expected_map = np.array([[[0, 2], [8, 10]]], dtype=np.float32)
        for idx in (12, 13, 14, 15, 18, 19, 20):
            np.testing.assert_array_equal(np.asarray(out[idx]), expected_map)
        # pos_x/pos_y integer-divided by 2 -> [3, 1].
        np.testing.assert_array_equal(
            np.asarray(out[0][0, 0, 0:2]), np.array([3.0, 1.0], dtype=np.float32)
        )
        # angle_base untouched.
        self.assertEqual(float(out[0][0, 0, 2]), 9.0)
        # agent width/height halved.
        self.assertEqual(int(out[16][0]), 5)
        self.assertEqual(int(out[17][0]), 9)

    def test_downsample_one_is_identity(self):
        obs = self._crafted_obs()
        self.assertIs(downsample_teacher_obs(obs, 1), obs)

    def test_downsampled_teacher_matches_native_64_forward(self):
        # A 64-world teacher applied to a 128-world obs (downsampled x2) must give
        # the SAME output as applying it to the equivalent native-64 obs.
        env64 = SimpleNamespace(
            batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
        )
        teacher, params = get_model_ready(
            jax.random.PRNGKey(0), _model_config(5, 64), env64
        )
        # Build a native-64 obs and its 128 "up-sampled" counterpart (repeat each
        # map cell 2x2, positions/dims x2) so downsample x2 recovers the 64 obs.
        rng = jax.random.PRNGKey(1)
        native = _real_obs_like(env64, rng)
        upsampled = _upsample_obs(native, 2)
        from utils.utils_ppo import policy

        v_native, pi_native = policy(teacher.apply, params, native)
        down = downsample_teacher_obs(upsampled, 2)
        v_down, pi_down = policy(teacher.apply, params, down)
        np.testing.assert_allclose(
            np.asarray(v_native), np.asarray(v_down), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(pi_native.logits_parameter()),
            np.asarray(pi_down.logits_parameter()),
            atol=1e-5,
        )


def _real_obs_like(env, rng):
    """A single-batch model-input obs with random in-range global maps."""
    edge = env.batch_cfg.maps_dims.maps_edge_length
    angles = env.batch_cfg.agent.angles_cabin
    num_state_obs = env.batch_cfg.agent.num_state_obs
    max_agents = 4
    keys = jax.random.split(rng, 8)
    agent_states = jnp.zeros((1, max_agents, num_state_obs), dtype=jnp.float32)
    # Even positions so //2 then the 64-world embedding stays exact under x2.
    agent_states = agent_states.at[0, 0, 0].set(4.0).at[0, 0, 1].set(6.0)
    obs = [agent_states]
    obs.append(jnp.zeros((1, max_agents), dtype=jnp.int8).at[:, 0].set(1))
    obs.append(jnp.ones((1,), dtype=jnp.int32))
    obs += [jnp.zeros((1, angles), dtype=jnp.float32) for _ in range(9)]
    obs += [
        jax.random.randint(keys[i], (1, edge, edge), 0, 3).astype(jnp.float32)
        for i in range(4)
    ]
    obs += [jnp.array([9], dtype=jnp.int32), jnp.array([5], dtype=jnp.int32)]
    obs += [
        jax.random.randint(keys[4 + i], (1, edge, edge), 0, 2).astype(jnp.float32)
        for i in range(3)
    ]
    obs += [jnp.zeros((1, 5), dtype=jnp.int32)]
    return obs


def _upsample_obs(obs, factor):
    """Inverse of the teacher downsample: tile maps factor x factor, scale dims."""
    s = factor
    up = list(obs)
    for idx in (12, 13, 14, 15, 18, 19, 20):
        up[idx] = jnp.repeat(jnp.repeat(obs[idx], s, axis=-1), s, axis=-2)
    agent_states = obs[0]
    scaled_pos = agent_states[..., 0:2] * s
    up[0] = agent_states.at[..., 0:2].set(scaled_pos)
    up[16] = obs[16] * s
    up[17] = obs[17] * s
    return up


class RegisterCheckpointConfigClassesTest(unittest.TestCase):
    def test_does_not_overwrite_main_defined_config_class(self):
        # Regression for Euler job 7862190: a derived per-run trainer running
        # as __main__ defines its own MixedAgentTrainConfig; the teacher-load
        # path calls register_checkpoint_config_classes(), and overwriting the
        # __main__ class makes pickling the FINAL checkpoint fail with
        # "it's not the same object as __main__.MixedAgentTrainConfig".
        import pickle
        import sys
        import types

        from utils.helpers import register_checkpoint_config_classes

        class MainConfig:
            pass

        fake_main = types.ModuleType("__main__")
        fake_main.MixedAgentTrainConfig = MainConfig
        real_main = sys.modules["__main__"]
        sys.modules["__main__"] = fake_main
        try:
            register_checkpoint_config_classes()
            # __main__'s own class must be untouched...
            self.assertIs(fake_main.MixedAgentTrainConfig, MainConfig)
            # ...and the missing name must have been filled in.
            self.assertTrue(hasattr(fake_main, "TrainConfig"))
        finally:
            sys.modules["__main__"] = real_main


if __name__ == "__main__":
    unittest.main()
