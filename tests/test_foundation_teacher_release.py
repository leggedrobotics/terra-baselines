"""One teacher-release clock; unchanged PPO except foundation row weights."""
import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.jax_utils import replicate
from flax.training.train_state import TrainState

from train import Transition, ppo_update_networks
from train_mixed import (
    foundation_teacher_release_coef, kickstart_coef_schedule,
    restore_foundation_teacher_release,
)
from test_task_teachers import ROUTE, _config, _raw_obs
from utils.task_teachers import FAMILY_KEY


def _parent_and_config():
    cfg = _config(resume_from="parent.pkl", kickstart_kl_anneal_updates=20000,
                  task_teacher_family_ids=ROUTE)
    parent = {"train_config": copy.deepcopy(cfg), "next_update": 5000,
              "optimizer_state": ("unchanged native Adam",)}
    cfg.foundation_teacher_release_updates = 1250
    return parent, cfg


def test_release_boundaries_and_native_continuation_keep_original_clock():
    parent, cfg = _parent_and_config()
    state = restore_foundation_teacher_release(cfg, parent, "resume", 5000)
    start = kickstart_coef_schedule(5000, 1., 20000)
    assert start == pytest.approx(.8535533905932737)
    for update, fraction in [(5000, 1.), (5625, .5), (6249, 1 / 1250),
                             (6250, 0.), (7500, 0.)]:
        assert foundation_teacher_release_coef(state, update) == pytest.approx(start * fraction)
    saved = pickle.loads(pickle.dumps(dict(parent, train_config=copy.deepcopy(cfg),
                 next_update=5600, foundation_teacher_release_state=state)))
    cfg.foundation_teacher_release_updates = 0  # normal resume restores, never disables
    resumed = restore_foundation_teacher_release(cfg, saved, "resume", 5600)
    assert resumed == state
    assert cfg.foundation_teacher_release_updates == 1250
    assert foundation_teacher_release_coef(resumed, 5600) == pytest.approx(start * .52)
    assert saved["optimizer_state"] == parent["optimizer_state"]
    saved["next_update"] = 7500
    cfg.foundation_teacher_release_updates = 0
    released = restore_foundation_teacher_release(cfg, saved, "resume", 7500)
    assert released == state
    assert foundation_teacher_release_coef(released, 7500) == 0.
    # The independent trench coefficient retains its original absolute cosine.
    assert kickstart_coef_schedule(6250, cfg.kickstart_kl_coef,
                                  cfg.kickstart_kl_anneal_updates) == pytest.approx(.7777851165098011)


def test_release_rejects_restarts_and_eval_copy_keeps_training_metadata():
    from eval_fixed_bank import checkpoint_treatment_fingerprint, configure_for_bank
    from utils.helpers import checkpoint_evaluation_config

    parent, cfg = _parent_and_config()
    state = restore_foundation_teacher_release(cfg, parent, "resume", 5000)
    saved = dict(parent, train_config=copy.deepcopy(cfg), next_update=5500,
                 foundation_teacher_release_state=state)
    cfg.foundation_teacher_release_updates = 1000
    with pytest.raises(ValueError, match="duration changed"):
        restore_foundation_teacher_release(cfg, saved, "resume", 5500)
    cfg.foundation_teacher_release_updates = 0
    with pytest.raises(ValueError, match="saved Adam and clock"):
        restore_foundation_teacher_release(cfg, saved, "resume", 5000)
    missing = dict(saved)
    del missing["foundation_teacher_release_state"]
    with pytest.raises(ValueError, match="missing its saved state"):
        restore_foundation_teacher_release(cfg, missing, "resume", 5500)
    assert restore_foundation_teacher_release(cfg, parent, "resume", 5000) is None

    evaluated = configure_for_bank(checkpoint_evaluation_config(saved), "validation/all", 64)
    evaluated.__post_init__()
    assert evaluated.foundation_teacher_release_updates == 0
    assert saved["train_config"].foundation_teacher_release_updates == 1250
    fingerprint = checkpoint_treatment_fingerprint(saved)
    assert fingerprint["contract"]["foundation_teacher_release"] == state
    assert checkpoint_treatment_fingerprint(parent)["sha256"] != fingerprint["sha256"]


def _actual_update(foundation_coef=None, trench_coef=.7, *, dual=True, teacher_nan=False):
    devices, samples = jax.local_device_count(), 4
    n = devices * samples
    raw = _raw_obs(n)
    raw[FAMILY_KEY] = jnp.where(jnp.arange(n) < 3, ROUTE["foundation"], ROUTE["trench"])
    logits = jnp.arange(n * 8, dtype=jnp.float32).reshape(n, 8) / 13
    # Distinct action distributions for both families and across devices.
    logits = logits * jnp.arange(1, n + 1)[:, None]
    if teacher_nan:
        logits = jnp.full_like(logits, jnp.nan)
    cfg = _config(num_envs_per_device=2, num_minibatches=1, num_steps=2,
                  ent_coef=0., vf_coef=0., task_teacher_family_ids=ROUTE,
                  flat_minibatch_shuffle=True)
    if not dual:
        cfg.trench_teacher_checkpoint = None

    def student_apply(params, obs):
        return jnp.zeros((obs[0].shape[0], 1)), jnp.broadcast_to(params, (obs[0].shape[0], 8))

    def teacher_apply(params, obs, history=None):
        return jnp.zeros((samples, 1)), params

    state = TrainState.create(apply_fn=student_apply, params=jnp.zeros(8), tx=optax.sgd(1.))
    shape = (devices, samples)
    tr = Transition(done=jnp.zeros(shape, bool), task_done=jnp.zeros(shape, bool),
        action=jnp.zeros(shape, jnp.int32), value=jnp.zeros(shape),
        reward=jnp.zeros(shape), log_prob=jnp.full(shape, -np.log(8)),
        obs=jax.tree_util.tree_map(lambda x: x.reshape(shape + x.shape[1:]), raw),
        prev_actions=jnp.zeros(shape + (5,)), prev_reward=jnp.zeros(shape))

    def update(state, tr, teacher_params):
        return ppo_update_networks(state, tr, jnp.zeros_like(tr.value), jnp.zeros_like(tr.value),
            cfg, teacher_apply_fn=teacher_apply, teacher_params=teacher_params,
            kickstart_kl_coef=trench_coef, foundation_kickstart_kl_coef=foundation_coef)

    updated, info = jax.pmap(update, axis_name="devices")(
        replicate(state), tr, logits.reshape(devices, samples, 8))
    return updated, info, logits, raw[FAMILY_KEY]


def test_disabled_release_matches_legacy_scalar_loss_and_actual_gradient():
    dual, dual_info, _, _ = _actual_update()
    scalar, scalar_info, logits, _ = _actual_update(dual=False)
    np.testing.assert_array_equal(dual.params, scalar.params)
    for key in ("total_loss", "kickstart/kl", "diagnostics/grad_global_norm"):
        np.testing.assert_array_equal(dual_info[key], scalar_info[key])
    expected_gradient = .7 * (jnp.full_like(logits, 1 / 8) - jax.nn.softmax(logits)).mean(0)
    np.testing.assert_allclose(dual.params[0], -expected_gradient, atol=1e-7)


def test_foundation_zeroing_preserves_trench_weight_counts_and_inactive_fast_path():
    updated, info, logits, family = _actual_update(foundation_coef=0.)
    weights = jnp.where(family == ROUTE["foundation"], 0., .7)
    expected = (weights[:, None] * (jnp.full_like(logits, 1 / 8) - jax.nn.softmax(logits))).mean(0)
    np.testing.assert_allclose(updated.params[0], -expected, atol=1e-7)
    logp = jax.nn.log_softmax(logits)
    kl = (jnp.exp(logp) * (logp + np.log(8))).sum(-1)
    np.testing.assert_allclose(info["total_loss"], (weights * kl).mean(), atol=1e-7)
    np.testing.assert_allclose(info["kickstart/foundation_kl_sum"], kl[:3].sum(), atol=1e-7)
    np.testing.assert_array_equal(info["kickstart/foundation_selected_count"], 3)
    np.testing.assert_array_equal(info["kickstart/trench_selected_count"], len(family) - 3)
    np.testing.assert_array_equal(info["kickstart/foundation_kl_coef"], 0.)
    np.testing.assert_allclose(info["kickstart/trench_kl_coef"], .7)
    inactive, zero_info, _, _ = _actual_update(foundation_coef=0., trench_coef=0., teacher_nan=True)
    np.testing.assert_array_equal(inactive.params, 0.)
    np.testing.assert_array_equal(zero_info["total_loss"], 0.)
    np.testing.assert_array_equal(zero_info["diagnostics/teacher_logits_finite_fraction"], 1.)
