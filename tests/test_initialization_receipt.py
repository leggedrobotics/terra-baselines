"""Startup provenance distinguishes parameter initialization from Adam resume."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import NamedTuple
import unittest

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.initialization_receipt import tree_fingerprint, write_initialization_receipt


class ResetState(NamedTuple):
    env_steps: object


class ResetTimestep(NamedTuple):
    state: ResetState
    observation: object


class InitializationReceiptTests(unittest.TestCase):
    def test_fingerprint_covers_values_dtype_shape_path_and_container(self):
        base = {"a": np.array([1, 2], dtype=np.float32), "b": np.array(3)}
        expected = tree_fingerprint(base)["sha256"]
        self.assertEqual(expected, tree_fingerprint(dict(reversed(list(base.items()))))["sha256"])
        alternatives = [
            {**base, "a": np.array([1, 3], dtype=np.float32)},
            {**base, "a": np.array([1, 2], dtype=np.int32)},
            {**base, "a": np.array([[1, 2]], dtype=np.float32)},
            {"renamed": base["a"], "b": base["b"]},
            {**base, "empty": []},
        ]
        for alternative in alternatives:
            with self.subTest(alternative=alternative):
                self.assertNotEqual(expected, tree_fingerprint(alternative)["sha256"])
        self.assertNotEqual(tree_fingerprint([np.array(1)])["sha256"],
                            tree_fingerprint((np.array(1),))["sha256"])

    def test_typed_keys_and_bfloat16_have_real_numeric_checks(self):
        key = jax.random.key(12)
        receipt = tree_fingerprint({"key": key})
        self.assertTrue(receipt["all_finite"])
        self.assertEqual(receipt, tree_fingerprint({"key": jax.random.key(12)}))
        self.assertNotEqual(receipt["sha256"],
                            tree_fingerprint({"key": jax.random.key_data(key)})["sha256"])
        values = jnp.array([0, 1, jnp.nan, jnp.inf], dtype=jnp.bfloat16)
        receipt = tree_fingerprint(values)
        self.assertEqual(receipt["nonfinite_count"], 2)
        self.assertFalse(receipt["all_finite"])
        self.assertFalse(receipt["all_zero"])
        self.assertTrue(tree_fingerprint(jnp.zeros(2, dtype=jnp.bfloat16))["all_zero"])

    def test_scratch_and_pretrained_keep_fresh_adam_then_resume_own_state(self):
        config = SimpleNamespace(name="test", seed=17, num_devices=1,
                                 num_envs_per_device=2, teacher_checkpoint="teacher.pkl")
        teacher = {"weight": jnp.array([1., 2.])}
        tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4))
        scratch = TrainState.create(apply_fn=lambda *a: None,
                                    params={"weight": jnp.array([-.5, .25])}, tx=tx)
        pretrained = scratch.replace(params=teacher)
        timestep = ResetTimestep(ResetState(jnp.zeros((1, 2), dtype=jnp.int32)),
                                 {"map": jnp.zeros((1, 2, 4, 4))})
        common = dict(config=config, timestep=timestep,
                      rollout_rng=jax.random.split(jax.random.PRNGKey(17), 1),
                      reset_rng=jax.random.split(jax.random.PRNGKey(3), 2),
                      teacher_params=teacher,
                      initial_history={"actions": jnp.zeros((1, 2, 5))})
        with tempfile.TemporaryDirectory() as directory:
            scratch_path = Path(directory) / "scratch.json"
            fresh = write_initialization_receipt(
                scratch_path, checkpoint_mode=None, checkpoint_path=None,
                optimizer_restored=False, next_update=0, train_state=scratch, **common)
            warm = write_initialization_receipt(
                Path(directory) / "pretrained.json", checkpoint_mode="warm_start",
                checkpoint_path="teacher.pkl", optimizer_restored=False,
                next_update=0, train_state=pretrained, **common)
            self.assertEqual(json.loads(scratch_path.read_text()), fresh)
            self.assertEqual(fresh["checkpoint_mode"], "scratch")
            self.assertEqual(warm["checkpoint_mode"], "warm_start")
            self.assertEqual(warm["model"], warm["teacher_model"])
            self.assertNotEqual(fresh["model"]["sha256"], warm["model"]["sha256"])
            for key in ["optimizer", "timestep", "rollout_rng", "reset_rng", "initial_history"]:
                self.assertEqual(fresh[key], warm[key], key)
            self.assertTrue(fresh["fresh_optimizer"])
            self.assertTrue(fresh["optimizer"]["all_zero"])
            self.assertEqual(fresh["train_state_step"], 0)
            self.assertEqual(fresh["optimizer_counters"][0]["value"], 0)
            self.assertTrue(fresh["initial_env_steps_all_zero"])

            # Emulate two full PPO updates of this run: native resume must keep
            # the resulting Adam moments and count, instead of freshening them.
            trained = pretrained
            for _ in range(128):
                trained = trained.apply_gradients(grads={"weight": jnp.array([.1, -.2])})
            resumed = write_initialization_receipt(
                Path(directory) / "resume.json", checkpoint_mode="resume",
                checkpoint_path="own_update_000002.pkl", optimizer_restored=True,
                next_update=2, train_state=trained, **common)
            self.assertFalse(resumed["fresh_optimizer"])
            self.assertEqual(resumed["next_update"], 2)
            self.assertEqual(resumed["train_state_step"], 128)
            self.assertEqual(resumed["optimizer_counters"][0]["value"], 128)
            self.assertFalse(resumed["optimizer_counters_all_zero"])
            self.assertFalse(resumed["optimizer"]["all_zero"])
            self.assertTrue(resumed["optimizer"]["all_finite"])

            with self.assertRaises(FileExistsError):
                write_initialization_receipt(
                    scratch_path, checkpoint_mode=None, checkpoint_path=None,
                    optimizer_restored=False, next_update=0, train_state=scratch, **common)
            self.assertEqual(json.loads(scratch_path.read_text()), fresh)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["pretrained.json", "resume.json", "scratch.json"])


if __name__ == "__main__":
    unittest.main()
