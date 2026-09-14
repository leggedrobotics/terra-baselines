"""Teacher regularization must preserve native Adam and its absolute clock."""
from dataclasses import asdict
import pickle
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from train_mixed import (
    MixedAgentTrainConfig,
    _make_training_optimizer,
    _validate_teacher_resume,
    kickstart_coef_schedule,
)


class NativeTeacherResumeTests(unittest.TestCase):
    def test_fixed_bank_clears_teacher_origin_without_mutating_training_config(self):
        from eval_fixed_bank import configure_for_bank

        trained = MixedAgentTrainConfig(
            name="native-teacher", teacher_checkpoint="teacher.pkl",
            kickstart_start_update=15000, kickstart_lr_warmup_updates=0,
        )
        original = asdict(trained)
        evaluated = configure_for_bank(trained, "validation/all", 64)
        self.assertIsNone(evaluated.teacher_checkpoint)
        self.assertEqual(evaluated.kickstart_start_update, 0)
        self.assertEqual(asdict(trained), original)
        # Exercise the same dataclass validation that runtime reconstruction
        # performs, rather than checking only the copied field values.
        evaluated.__post_init__()
        self.assertEqual(evaluated.num_envs_per_device, 64)

    def test_resume_guard_preserves_teacher_protocol_and_allows_relocated_file(self):
        config = MixedAgentTrainConfig(name="resume", teacher_checkpoint="local/teacher.pkl",
                                      kickstart_start_update=15000,
                                      kickstart_kl_anneal_updates=20000,
                                      kickstart_lr_warmup_updates=0)
        checkpoint = {"train_config": pickle.loads(pickle.dumps(config))}
        config.teacher_checkpoint = "cluster/teacher.pkl"
        _validate_teacher_resume(checkpoint, config)
        fields = ["kickstart_start_update", "kickstart_kl_coef",
                  "kickstart_kl_anneal_updates", "kickstart_value_coef",
                  "kickstart_value_anneal_updates", "kickstart_lr_warmup_updates",
                  "teacher_obs_downsample"]
        for field in fields:
            with self.subTest(field=field):
                live = SimpleNamespace(**asdict(config))
                setattr(live, field, 0 if field == "kickstart_start_update"
                        else getattr(live, field) + 1)
                with self.assertRaisesRegex(ValueError, field):
                    _validate_teacher_resume(checkpoint, live)
        config.teacher_checkpoint = None
        with self.assertRaisesRegex(ValueError, "teacher_checkpoint"):
            _validate_teacher_resume(checkpoint, config)

    def test_teacher_free_parent_and_legacy_zero_origin(self):
        config = MixedAgentTrainConfig(name="new-teacher", teacher_checkpoint="teacher.pkl",
                                      kickstart_start_update=15000,
                                      kickstart_lr_warmup_updates=0)
        _validate_teacher_resume({"train_config": {"teacher_checkpoint": None}}, config)
        _validate_teacher_resume({}, config)
        config.kickstart_start_update = 0
        config.kickstart_lr_warmup_updates = 100
        # Legacy teacher configs did not save an origin; zero is their clock.
        _validate_teacher_resume({"train_config": {"teacher_checkpoint": "old/path.pkl"}}, config)
        config.kickstart_start_update = 15000
        with self.assertRaisesRegex(ValueError, "kickstart_start_update"):
            _validate_teacher_resume({"train_config": {"teacher_checkpoint": "old/path.pkl"}}, config)

    def test_native_origin_and_checkpointed_continuation(self):
        config = MixedAgentTrainConfig(name="test", teacher_checkpoint="teacher.pkl",
                                      kickstart_start_update=15000,
                                      kickstart_lr_warmup_updates=0)
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(asdict(restored)["kickstart_start_update"], 15000)
        for update, expected in [(14999, 1.), (15000, 1.), (15750, .5),
                                 (16500, 0.), (17000, 0.)]:
            self.assertAlmostEqual(kickstart_coef_schedule(
                update, config.kickstart_kl_coef, config.kickstart_kl_anneal_updates,
                restored.kickstart_start_update), expected)
        self.assertAlmostEqual(kickstart_coef_schedule(
            15250, restored.kickstart_value_coef,
            restored.kickstart_value_anneal_updates, restored.kickstart_start_update), .25)
        # Default origin preserves existing teacher schedules, including resumes.
        self.assertEqual(kickstart_coef_schedule(15000, 1., 1500), 0.)
        self.assertEqual(kickstart_coef_schedule(0, 1., 1500), 1.)
        self.assertEqual(kickstart_coef_schedule(15000, 1., 0, 15000), 0.)

        # The foundation continuation uses a longer anchor-relative window.
        # Saving again at u17500 must not restart that window at the new resume.
        restored.kickstart_kl_anneal_updates = 20000
        checkpoint = pickle.loads(pickle.dumps({"config": restored, "next_update": 17500}))
        resumed = checkpoint["config"]
        self.assertAlmostEqual(kickstart_coef_schedule(
            checkpoint["next_update"], resumed.kickstart_kl_coef,
            resumed.kickstart_kl_anneal_updates, resumed.kickstart_start_update),
            .9619397662556434)
        self.assertAlmostEqual(kickstart_coef_schedule(
            25000, resumed.kickstart_kl_coef, resumed.kickstart_kl_anneal_updates,
            resumed.kickstart_start_update), .5)
        self.assertEqual(kickstart_coef_schedule(
            35000, resumed.kickstart_kl_coef, resumed.kickstart_kl_anneal_updates,
            resumed.kickstart_start_update), 0.)

    def test_invalid_origin_and_warmup(self):
        for kwargs in [{"kickstart_start_update": -1},
                       {"kickstart_start_update": 15000},
                       {"kickstart_lr_warmup_updates": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MixedAgentTrainConfig(name="test", **kwargs)
        self.assertEqual(MixedAgentTrainConfig(name="test").kickstart_start_update, 0)
        self.assertEqual(MixedAgentTrainConfig(name="test").kickstart_lr_warmup_updates, 100)

    def test_constant_adam_restores_with_teacher_and_no_warmup(self):
        config = MixedAgentTrainConfig(name="test")
        params = {"weight": jnp.array([.5, -1.], dtype=jnp.float32)}
        # This is exactly the pre-change no-teacher optimizer structure.
        old_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                             optax.adam(config.lr, eps=1e-5))
        old = TrainState.create(apply_fn=lambda *a: None, params=params, tx=old_tx)
        old = old.apply_gradients(grads={"weight": jnp.array([.1, -.2])})
        # Match the native u15000 update/Adam clock while preserving real moments.
        count = jnp.array(15000 * 64, dtype=jnp.int32)
        adam, scale = old.opt_state[1]
        old = old.replace(step=count, opt_state=(old.opt_state[0],
                          (adam._replace(count=count), scale)))
        saved = pickle.loads(pickle.dumps({"params": old.params,
                            "optimizer_state": old.opt_state, "train_state_step": old.step}))
        config.teacher_checkpoint = "teacher.pkl"
        config.kickstart_start_update = 15000
        config.kickstart_lr_warmup_updates = 0
        new = TrainState.create(apply_fn=old.apply_fn, params=saved["params"],
                                tx=_make_training_optimizer(config))
        self.assertEqual(jax.tree_util.tree_structure(new.opt_state),
                         jax.tree_util.tree_structure(saved["optimizer_state"]))
        new = new.replace(opt_state=saved["optimizer_state"], step=saved["train_state_step"])
        grads = {"weight": jnp.array([-.3, .4])}
        expected = old.apply_gradients(grads=grads)
        actual = new.apply_gradients(grads=grads)
        self.assertEqual(int(actual.step), 960001)
        for x, y in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_array_equal(x, y)

    def test_positive_warmup_retains_existing_schedule_and_resume_state(self):
        config = MixedAgentTrainConfig(name="test", teacher_checkpoint="teacher.pkl")
        n = config.kickstart_lr_warmup_updates * config.update_epochs * config.num_minibatches
        legacy_lr = optax.join_schedules([
            optax.linear_schedule(config.lr/3, config.lr, n),
            optax.constant_schedule(config.lr)], [n])
        old_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                             optax.adam(legacy_lr, eps=1e-5))
        new_tx = _make_training_optimizer(config)
        p = {"w": jnp.array([1.])}
        old_state = old_tx.init(p)
        # Existing scheduled teacher resumes keep both Adam and LR counters.
        old_state = jax.tree_util.tree_map(
            lambda x: jnp.full_like(x, 100) if jnp.issubdtype(x.dtype, jnp.integer) else x,
            old_state)
        self.assertEqual(jax.tree_util.tree_structure(old_state),
                         jax.tree_util.tree_structure(new_tx.init(p)))
        expected = old_tx.update({"w": jnp.array([.2])}, old_state, p)
        actual = new_tx.update({"w": jnp.array([.2])}, old_state, p)
        for x, y in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
