import os
import subprocess
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from terra.structured_actions import StructuredAction
from utils.models import get_model_ready
from utils.structured_ppo import (
    StructuredRollout, entropy_components, joint_log_prob, policy_diagnostics,
    policy_masks, ppo_update, sample_action,
)
from utils.utils_ppo import obs_to_model_input
from test_recurrent_actor import _config, _env, _observation
from test_structured_ppo import logits_and_masks


class PrimitiveInterfaceTests(unittest.TestCase):
    def test_fixed_request_masks_keep_cabin_and_full_native_unload_checks(self):
        _, masks = logits_and_masks((2,))
        masks["action_mask"] = jnp.ones((2, 8), bool)
        masks["move_mask"] = masks["move_mask"].at[0, :, 4].set(False)
        masks["turn_mask"] = masks["turn_mask"].at[0, :, 0].set(False)
        masks["do_mask"] = jnp.zeros((2, 12), bool).at[:, 9].set(True)
        result = jax.jit(lambda m, h: policy_masks(m, action_interface="primitive8", current_heading=h))(
            masks, jnp.array([3, 9]))
        np.testing.assert_array_equal(result["action_mask"],
            [[False, False, False, False, True, True, False, False],
             [True, True, True, True, True, True, True, False]])
        np.testing.assert_array_equal(result["do_mask"], masks["do_mask"])
        # A different unload heading remains legal; current-heading DO cannot
        # be used as a certificate that a loaded machine is trapped.
        self.assertTrue(bool(result["do_mask"][0].any()))
        with self.assertRaisesRegex(ValueError, "require current_heading"):
            policy_masks(masks, action_interface="primitive8")
        with self.assertRaisesRegex(ValueError, "batch shape"):
            policy_masks(masks, action_interface="primitive8", current_heading=jnp.array([[3], [9]]))
        with self.assertRaisesRegex(ValueError, "unknown action interface"):
            policy_masks(masks, action_interface="wrong")
        empty = policy_masks({k: jnp.zeros_like(v) for k, v in masks.items()},
                             action_interface="primitive8", current_heading=jnp.array([0, 0]))
        np.testing.assert_array_equal(empty["action_mask"], np.eye(8, dtype=bool)[[7, 7]])

    def test_primitive_law_and_gradients_ignore_argument_heads(self):
        logits, masks = logits_and_masks((8,))
        masks["action_mask"] = jnp.ones((8, 8), bool)
        masks = policy_masks(masks, True, action_interface="primitive8", current_heading=jnp.arange(8))
        logits["action"] = jnp.eye(8) * 100.
        action, logp = sample_action(logits, masks, jax.random.PRNGKey(2), True,
                                    action_interface="primitive8")
        np.testing.assert_array_equal(action.action, np.arange(8))
        np.testing.assert_array_equal(action.amount, [5, 5, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(action.heading, np.full(8, -1))
        np.testing.assert_allclose(logp, 0., atol=1e-6)
        logits["action"] = jnp.zeros((8, 8))
        loss = lambda x: (joint_log_prob(x, masks, action, action_interface="primitive8").sum()
                          + sum(v.sum() for v in entropy_components(x, masks, action_interface="primitive8").values()))
        grads = jax.grad(loss)(logits)
        for name in ("move", "turn", "do"):
            np.testing.assert_array_equal(grads[name], np.zeros_like(grads[name]))
        lp = joint_log_prob(logits, masks, action, action_interface="primitive8")
        np.testing.assert_allclose(jnp.exp(lp).sum(), 1., atol=1e-6)
        entropies = entropy_components(logits, masks, action_interface="primitive8")
        np.testing.assert_allclose(entropies["action"], np.log(8), atol=1e-6)
        for name in ("move", "turn", "do"):
            np.testing.assert_array_equal(entropies[name], np.zeros(8))
        diagnostics = policy_diagnostics(logits, masks, action_interface="primitive8")
        for key, value in diagnostics.items():
            if not key.startswith("head_action_"):
                self.assertEqual(float(value), 0.)
        sampled, _ = jax.jit(lambda x, m: sample_action(x, m, jax.random.PRNGKey(3),
            action_interface="primitive8"))(logits, masks)
        np.testing.assert_array_equal(sampled.amount, np.where(sampled.action < 2, 5, 1))
        np.testing.assert_array_equal(sampled.heading, np.full(8, -1))

    def test_diagnostics_condition_on_head_and_pool_devices_before_division(self):
        logits, masks = logits_and_masks()
        metrics = policy_diagnostics(logits, masks)
        for name, count, probability in (("action", 5, 1.), ("move", 5, .4),
                                          ("turn", 6, .4), ("do", 12, .2)):
            self.assertAlmostEqual(float(metrics[f"head_{name}_conditional_entropy"]), np.log(count), places=6)
            self.assertAlmostEqual(float(metrics[f"head_{name}_normalized_entropy"]), 1., places=6)
            self.assertAlmostEqual(float(metrics[f"head_{name}_valid_count"]), count, places=5)
            self.assertAlmostEqual(float(metrics[f"head_{name}_probability"]), probability, places=6)
        # Head DO is inactive on one device and has one choice on the other.
        # Pmean of local conditional means would incorrectly halve its count.
        code = '''
import jax
import jax.numpy as jnp
import numpy as np
from utils.structured_ppo import policy_diagnostics, policy_masks
from test_structured_ppo import logits_and_masks
assert jax.local_device_count() == 2
logits, masks = logits_and_masks((2, 3))
masks["do_mask"] = jnp.zeros((2, 3, 12), bool).at[1, :, 7].set(True)
masks = policy_masks(masks)
expected = policy_diagnostics(logits, masks)
actual = jax.pmap(lambda l, m: policy_diagnostics(l, m, axis_name="d"), axis_name="d")(logits, masks)
for key, value in expected.items():
    np.testing.assert_allclose(actual[key], np.repeat(value, 2), atol=1e-6, err_msg=key)
assert float(expected["head_do_valid_count"]) == 1.
assert float(expected["head_do_normalized_entropy"]) == 0.
assert float(expected["head_do_prob_7"]) == 1.
'''
        env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=60,
                       capture_output=True, text=True)

    def test_primitive_ppo_updates_type_head_without_argument_updates(self):
        env, cfg = _env(), _config()
        cfg.update(structured_actions=True, action_interface="primitive8")
        model, params = get_model_ready(jax.random.PRNGKey(1), cfg, env)
        # The interface changes only PPO/action interpretation, not the model
        # tree or RNG path used to initialize the shared backbone and type head.
        structured_cfg = type(cfg)(cfg, action_interface="structured")
        _, paired = get_model_ready(jax.random.PRNGKey(1), structured_cfg, env)
        for before, after in zip(jax.tree.leaves(params), jax.tree.leaves(paired)):
            np.testing.assert_array_equal(before, after)
        obs = _observation((2, 3), env)
        obs["structured_context"] = jnp.zeros((2, 3, 19))
        prev, hidden = jnp.zeros((2, 3, 5), jnp.int32), jnp.zeros((2, 64))
        done = jnp.zeros((2, 3), bool)
        values, logits, _ = model.apply(params, obs_to_model_input(obs, prev, cfg), hidden, done,
                                       method="actor_sequence")
        _, masks = logits_and_masks((2, 3))
        masks["action_mask"] = jnp.ones((2, 3, 8), bool)
        masks = policy_masks(masks, action_interface="primitive8", current_heading=jnp.zeros((2, 3), jnp.int32))
        action, old_lp = sample_action(logits, masks, jax.random.PRNGKey(4), action_interface="primitive8")
        rollout = StructuredRollout(obs, masks, prev, action, old_lp, values[..., 0],
            jnp.ones((2, 3)), jnp.ones((2, 3)) * 30, done, done)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(3e-4))
        updated, metrics = jax.jit(lambda s: ppo_update(s, model, rollout,
            jnp.array([[1., 2., 3.], [-1., -2., -3.]]), values[..., 0] + .25, hidden, cfg))(state)
        self.assertTrue(bool(metrics["grads_finite"]))
        self.assertAlmostEqual(float(metrics["approx_kl"]), 0., places=5)
        self.assertGreater(float(metrics["head_action_grad_norm"]), 0.)
        for name in ("move", "turn", "do"):
            self.assertEqual(float(metrics[f"head_{name}_grad_norm"]), 0.)
            for before, after in zip(jax.tree.leaves(params["params"]["structured_heads"][name]),
                                     jax.tree.leaves(updated.params["params"]["structured_heads"][name])):
                np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
