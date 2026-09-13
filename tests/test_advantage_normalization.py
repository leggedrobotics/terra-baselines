"""Run with JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4."""
import ast
from dataclasses import asdict
from pathlib import Path
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from flax.training.train_state import TrainState

import train
import train_mixed
from train import Transition, _normalize_ppo_advantages, ppo_update_networks
from train_mixed import MixedAgentTrainConfig, _validate_advantage_normalization_resume


FOUR_CPUS = len(jax.local_devices()) >= 4 and jax.default_backend() == "cpu"


def _toy_apply(params, obs):
    return obs @ params["value"], obs @ params["actor"]


class AdvantageNormalizationTest(unittest.TestCase):
    def test_default_arithmetic_is_unchanged(self):
        values = jnp.asarray([[1.0, 2.0, 5.0], [-3.0, 0.0, 8.0]])
        expected = jax.jit(lambda x: (x - x.mean()) / (x.std() + 1e-8))(values)
        actual = jax.jit(_normalize_ppo_advantages)(values)
        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(FOUR_CPUS, "requires four forced CPU devices")
    def test_global_statistics_match_pooled_population_including_between_shard_variance(self):
        devices = jax.local_devices()[:4]
        global_norm = jax.pmap(lambda x: _normalize_ppo_advantages(x, True),
                               axis_name="devices", devices=devices)
        local_norm = jax.pmap(_normalize_ppo_advantages, axis_name="devices", devices=devices)
        # Each shard alone is constant, but their pooled variance is nonzero.
        values = np.broadcast_to(np.array([-8, -2, 1, 9], dtype=np.float32)[:, None, None], (4, 2, 3))
        expected = (values - values.mean()) / (values.std() + 1e-8)
        np.testing.assert_allclose(global_norm(values), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(local_norm(values), np.zeros_like(values))
        np.testing.assert_array_equal(global_norm(np.ones_like(values) * 7), np.zeros_like(values))
        shifted = np.arange(24, dtype=np.float32).reshape(4, 2, 3) + 1_000_000
        expected = (shifted - shifted.mean()) / (shifted.std() + 1e-8)
        np.testing.assert_allclose(global_norm(shifted), expected, rtol=2e-6, atol=2e-6)

    @unittest.skipUnless(FOUR_CPUS, "requires four forced CPU devices")
    def test_distributed_ppo_matches_merged_minibatches_after_64_adam_steps(self):
        rng = np.random.default_rng(17)
        shape = (4, 32, 2, 2)  # devices, minibatches, envs, time
        features = rng.normal(0, .5, shape + (3,)).astype(np.float32)
        features[..., 0] += np.arange(4, dtype=np.float32)[:, None, None, None]
        advantages = rng.normal(0, .1, shape).astype(np.float32)
        advantages += np.array([-7, -2, 1, 9], dtype=np.float32)[:, None, None, None]
        params = {"actor": jnp.array([[.1, -.1], [.2, -.2], [-.1, .1]]),
                  "value": jnp.array([[.1], [-.2], [.05]])}
        values, logits = _toy_apply(params, jnp.asarray(features))
        actions = rng.integers(0, 2, shape, dtype=np.int32)
        log_probs = jnp.take_along_axis(jax.nn.log_softmax(logits),
                                        jnp.asarray(actions[..., None]), axis=-1)[..., 0]
        zeros = jnp.zeros(shape)
        transitions = Transition(
            done=zeros.astype(bool), task_done=zeros.astype(bool), action=jnp.asarray(actions),
            value=values[..., 0], reward=zeros, log_prob=log_probs,
            obs={"features": jnp.asarray(features)}, prev_actions=jnp.zeros(shape + (1,), dtype=jnp.int32),
            prev_reward=zeros,
        )
        batch = transitions, jnp.asarray(advantages), values[..., 0] + advantages
        state = TrainState.create(apply_fn=_toy_apply, params=params,
                                  tx=optax.chain(optax.clip_by_global_norm(.5),
                                                 optax.adam(3e-4, eps=1e-5)))

        def run(batch, devices, global_norm):
            config = SimpleNamespace(clip_eps=.2, vf_coef=2., ent_coef=.01,
                                     global_minibatch_advantage_norm=global_norm)

            def updates(state, batch):
                def epoch(state, _):
                    return jax.lax.scan(
                        lambda state, mb: ppo_update_networks(state, *mb, config),
                        state, batch,
                    )
                return jax.lax.scan(epoch, state, None, length=2)

            mapped = jax.pmap(updates, axis_name="devices", devices=devices)
            return mapped(jax.device_put_replicated(state, devices), batch)

        def merge_shards(x):
            return x.swapaxes(0, 1).reshape((1, 32, 8, 2) + x.shape[4:])

        # Keep the actual categorical PPO loss, gradient pmean, clipping and
        # Adam path. Only replace the expensive map-observation adapter.
        with patch("train.obs_to_model_input", side_effect=lambda obs, *_: obs["features"]):
            distributed, dist_info = run(batch, jax.local_devices()[:4], True)
            merged, merged_info = run(jtu.tree_map(merge_shards, batch), jax.local_devices()[:1], False)
            jax.block_until_ready((distributed, merged))
        for a, b in zip(jtu.tree_leaves(distributed), jtu.tree_leaves(merged)):
            np.testing.assert_allclose(a, np.broadcast_to(b, a.shape), rtol=3e-5, atol=2e-7)
        np.testing.assert_array_equal(distributed.step, np.full(4, 64))
        np.testing.assert_array_equal(distributed.opt_state[1][0].count, np.full(4, 64))
        for field in ("total_loss", "actor_loss", "value_loss", "entropy", "approx_kl",
                      "clip_fraction", "diagnostics/grad_global_norm"):
            np.testing.assert_allclose(dist_info[field], np.broadcast_to(merged_info[field], dist_info[field].shape),
                                       rtol=3e-5, atol=2e-6, err_msg=field)

    def test_cli_metadata_and_native_resume_keep_the_mode_explicit(self):
        module = ast.parse(Path(train_mixed.__file__).read_text())
        main = module.body[-1]
        self.assertIsInstance(main, ast.If)
        cli = compile(ast.Module(body=main.body, type_ignores=[]), train_mixed.__file__, "exec")
        captured = []
        for enabled in (False, True):
            argv = ["train_mixed.py", "--num_devices", "1", "--num_envs_per_device", "64",
                    "--num_steps", "2", "--num_minibatches", "32", "--update_epochs", "2",
                    "--total_timesteps", "128"]
            if enabled:
                argv.append("--global_minibatch_advantage_norm")
            namespace = vars(train_mixed).copy()
            namespace["train_mixed_agents"] = captured.append
            with patch("sys.argv", argv):
                exec(cli, namespace)
            config = captured[-1]
            self.assertEqual(config.global_minibatch_advantage_norm, enabled)
            self.assertEqual(asdict(pickle.loads(pickle.dumps(config)))["global_minibatch_advantage_norm"], enabled)
            self.assertEqual(config.update_epochs * config.num_minibatches, 64)
        local, global_config = captured
        for saved in ({}, {"global_minibatch_advantage_norm": False}):
            _validate_advantage_normalization_resume({"train_config": saved}, local)
            _validate_advantage_normalization_resume({"train_config": saved}, global_config)
        for saved in (global_config, asdict(global_config)):
            _validate_advantage_normalization_resume({"train_config": saved}, global_config)
            with self.assertRaisesRegex(ValueError, "retain --global_minibatch_advantage_norm"):
                _validate_advantage_normalization_resume({"train_config": saved}, local)
        self.assertFalse(MixedAgentTrainConfig.global_minibatch_advantage_norm)
        self.assertFalse(train.TrainConfig.global_minibatch_advantage_norm)


if __name__ == "__main__":
    unittest.main()
