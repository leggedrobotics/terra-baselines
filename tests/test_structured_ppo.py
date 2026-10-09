import tempfile
import unittest
from pathlib import Path
import pickle
import subprocess
import sys
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from terra.structured_actions import StructuredAction
from utils.models import get_model_ready
from utils.structured_ppo import (
    StructuredRollout, duration_gae, entropy_components, forward_step,
    joint_log_prob, policy_masks, ppo_update, sample_action, warm_start_params,
)
from utils.utils_ppo import obs_to_model_input
from test_recurrent_actor import _config, _env, _observation


def logits_and_masks(shape=()):
    logits = dict(action=jnp.zeros(shape+(8,)), move=jnp.zeros(shape+(2,5)),
                  turn=jnp.zeros(shape+(2,6)), do=jnp.zeros(shape+(12,)))
    masks = dict(action_mask=jnp.ones(shape+(8,), bool), move_mask=jnp.ones(shape+(2,5), bool),
                 turn_mask=jnp.ones(shape+(2,6), bool), do_mask=jnp.ones(shape+(12,), bool))
    return logits, policy_masks(masks)


class StructuredDistributionTests(unittest.TestCase):
    def test_joint_law_and_exact_entropy(self):
        logits, masks = logits_and_masks()
        probabilities = []
        for kind, count in ((0,5), (1,5), (2,6), (3,6), (6,12)):
            for argument in range(count):
                action = StructuredAction(jnp.int32(kind), jnp.int32(argument+1 if kind != 6 else 1),
                                          jnp.int32(argument if kind == 6 else -1))
                probabilities.append(float(jnp.exp(joint_log_prob(logits, masks, action))))
        self.assertAlmostEqual(sum(probabilities), 1., places=6)
        expected = -sum(p*np.log(p) for p in probabilities)
        self.assertAlmostEqual(float(sum(entropy_components(logits, masks).values())), expected, places=5)
        selected_move = StructuredAction(jnp.int32(1), jnp.int32(3), jnp.int32(-1))
        grad = jax.grad(lambda values: joint_log_prob(values, masks, selected_move))(logits)
        self.assertTrue(np.all(np.asarray(grad["do"]) == 0))
        self.assertTrue(np.all(np.asarray(grad["turn"]) == 0))
        self.assertTrue(np.all(np.asarray(grad["move"][0]) == 0))
        # An inactive argument never contributes to a probability or gradient.
        wait_masks = {k: jnp.zeros_like(v) for k,v in masks.items()}
        wait_masks = policy_masks(wait_masks)
        action = StructuredAction(jnp.int32(7), jnp.int32(6), jnp.int32(11))
        grad = jax.grad(lambda values: joint_log_prob(values, wait_masks, action))(logits)
        self.assertTrue(all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(grad)))

    def test_masks_sampling_and_empty_rows(self):
        logits, masks = logits_and_masks((256,))
        masks["move_mask"] = masks["move_mask"].at[..., :4].set(False)
        masks["turn_mask"] = jnp.zeros_like(masks["turn_mask"])
        masks["do_mask"] = masks["do_mask"].at[..., 1:].set(False)
        masks = policy_masks(masks)
        action, logp = jax.jit(sample_action)(logits, masks, jax.random.PRNGKey(4))
        self.assertTrue(np.all(np.isin(action.action, [0,1,6])))
        self.assertTrue(np.all(np.asarray(action.amount)[np.asarray(action.action)<2] == 5))
        self.assertTrue(np.all(np.asarray(action.heading)[np.asarray(action.action)==6] == 0))
        self.assertTrue(np.isfinite(logp).all())
        empty = policy_masks({k: jnp.zeros_like(v) for k,v in masks.items()})
        fallback, logp = sample_action(logits, empty, jax.random.PRNGKey(5))
        self.assertTrue(np.all(np.asarray(fallback.action) == 7))
        self.assertTrue(np.all(np.asarray(logp) == 0))

    def test_duration_gae_and_terminal_boundary(self):
        r = StructuredRollout({}, {}, None, None, None, jnp.array([[1.,2.,3.]]),
            jnp.array([[4.,5.,6.]]), jnp.array([[30.,60.,0.]]),
            jnp.array([[False,True,False]]), jnp.zeros((1,3), bool))
        adv, targets = duration_gae(r, jnp.array([7.]), gamma=.9, gae_lambda=.8, reference_s=30.)
        # t2 has no elapsed time; t1 ends an episode and cannot see t2.
        np.testing.assert_allclose(adv, [[4+.9*2-1 + .9*.8*(5-2), 5-2, 6+7-3]], rtol=1e-6)
        np.testing.assert_allclose(targets, adv+r.value)
        r = r.replace(done=jnp.zeros((1,3), bool))
        adv, _ = duration_gae(r, jnp.array([7.]), gamma=.9, gae_lambda=.8, reference_s=30.)
        self.assertAlmostEqual(float(adv[0,1]), 5+.9**2*3-2 + .9**2*.8**2*10, places=5)


class StructuredModelTests(unittest.TestCase):
    def test_gru_rollout_replay_and_update(self):
        env = _env()
        cfg = _config()
        cfg.update(structured_actions=True)
        model, params = get_model_ready(jax.random.PRNGKey(1), cfg, env)
        params = jax.device_put(params, jax.devices()[0])
        obs = _observation((2,3), env)
        obs["structured_context"] = jnp.zeros((2,3,19))
        prev = jnp.zeros((2,3,5), jnp.int32)
        done = jnp.array([[False,True,False], [False,False,False]])
        hidden = jnp.zeros((2,64))
        inputs = obs_to_model_input(obs, prev, cfg)
        values, logits, final_hidden = model.apply(params, inputs, hidden, done, method="actor_sequence")
        carry = hidden
        steps = []
        for index in range(3):
            _, step_logits, carry = model.apply(params, [x[:,index] for x in inputs], carry, method="actor_step")
            steps.append(step_logits)
            carry = jnp.where(done[:,index,None], 0., carry)
        expected = jax.tree.map(lambda *x: jnp.stack(x, 1), *steps)
        for actual, want in zip(jax.tree.leaves(logits), jax.tree.leaves(expected)):
            np.testing.assert_allclose(actual, want, atol=2e-3)
        np.testing.assert_allclose(final_hidden, carry, atol=2e-3)
        _, masks = logits_and_masks((2,3))
        action, old_lp = sample_action(logits, masks, jax.random.PRNGKey(3))
        rollout = StructuredRollout(obs, masks, prev, action, old_lp, values[...,0],
            jnp.ones((2,3)), jnp.ones((2,3))*30, done, jnp.zeros_like(done))
        state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(3e-4))
        update = jax.jit(lambda state: ppo_update(state, model, rollout,
            jnp.array([[1.,2.,3.],[-1.,-2.,-3.]]), values[...,0]+.25, hidden, cfg))
        state, metrics = update(state)
        self.assertTrue(bool(metrics["grads_finite"]))
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves((state.params,state.opt_state,metrics))))
        self.assertAlmostEqual(float(metrics["approx_kl"]), 0., places=5)

    def test_legacy_migration_preserves_encoder_and_gru(self):
        env = _env()
        cfg = _config()
        _, old = get_model_ready(jax.random.PRNGKey(0), cfg, env)
        new_cfg = type(cfg)(cfg, structured_actions=True)
        _, initialized = get_model_ready(jax.random.PRNGKey(1), new_cfg, env)
        grown = warm_start_params(initialized, old)
        for name in old["params"]:
            if name != "mlp_v":
                for before, after in zip(jax.tree.leaves(old["params"][name]), jax.tree.leaves(grown["params"][name])):
                    np.testing.assert_array_equal(before, after)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(old["params"]["mlp_v"]), jax.tree.leaves(grown["params"]["mlp_v"]))))
        damaged = dict(old, params=dict(old["params"]))
        damaged["params"].pop("actor_gru")
        with self.assertRaisesRegex(ValueError, "missing existing"):
            warm_start_params(initialized, damaged)


class StructuredCheckpointTests(unittest.TestCase):
    def test_legacy_main_config_unpickles_from_new_entrypoint(self):
        from train_structured import load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pkl"
            code = ("import pickle,sys\nclass MixedAgentTrainConfig: pass\n"
                    "config=MixedAgentTrainConfig();config.actor_core='gru'\n"
                    "pickle.dump({'train_config':config},open(sys.argv[1],'wb'))\n")
            subprocess.run([sys.executable, "-c", code, str(path)], check=True)
            checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint["train_config"].actor_core, "gru")

    def test_restore_only_broadcasts_matching_appended_config_defaults(self):
        from train_structured import restore_runner
        Config = namedtuple("Config", "new_field")
        State = namedtuple("State", "env_cfg terrain")
        expected = State(Config(jnp.zeros(2)), jnp.zeros((2,3)))
        restored = restore_runner(State(Config(0.), jnp.ones((2,3))), expected)
        np.testing.assert_array_equal(restored.env_cfg.new_field, jnp.zeros(2))
        np.testing.assert_array_equal(restored.terrain, jnp.ones((2,3)))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            restore_runner(State(Config(1.), jnp.ones((2,3))), expected)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            restore_runner(State(Config(0.), jnp.ones(3)), expected)


if __name__ == "__main__":
    unittest.main()
