"""Load-bearing contracts for the mixed cutting-space experiment."""

from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.config import EnvConfig
from terra.maps_buffer import MapsBuffer
from train_mixed import (
    PrecisionMapsBuffer, precision_lane_flags, resolve_pull_direction_training_slots,
)
from utils.task_teachers import (
    CACHED_LOGITS_KEY, CACHED_VALUE_KEY, ELIGIBLE_KEY, masked_teacher_kl,
    recurrent_teacher_rollout_observation, reset_recurrent_teacher_hidden,
)


def test_cached_recurrent_ppo_gradient_matches_global_selected_rows():
    import optax
    from flax.jax_utils import replicate
    from flax.training.train_state import TrainState
    from train import Transition, ppo_update_networks
    from train_mixed import MixedAgentTrainConfig

    devices = jax.local_device_count()
    shape = (devices, 2, 2)
    teacher_logits = jnp.arange(devices * 32, dtype=jnp.float32).reshape(shape + (8,)) / 7
    eligible = (jnp.arange(devices * 4).reshape(shape) < 3)
    obs = {"signal": jnp.zeros(shape + (1,)), CACHED_LOGITS_KEY: teacher_logits,
           CACHED_VALUE_KEY: jnp.zeros(shape + (1,)), ELIGIBLE_KEY: eligible}
    cfg = MixedAgentTrainConfig(
        name="cached-gru", num_devices=devices, actor_core="gru", num_steps=2,
        num_envs_per_device=2, num_minibatches=2, ent_coef=0, vf_coef=0,
        teacher_checkpoint="frozen.pkl", recurrent_teacher=True, kickstart_value_coef=0,
    )
    def student(params, model_obs, hidden, dones, method):
        assert method == "actor_sequence"
        return jnp.zeros(dones.shape + (1,)), jnp.broadcast_to(params, dones.shape + (8,)), hidden
    def forbidden_teacher(*_):
        raise AssertionError("PPO must reuse the pre-action cached recurrent teacher logits")
    state = TrainState.create(apply_fn=student, params=jnp.zeros(8), tx=optax.sgd(1.0))
    transition = Transition(
        done=jnp.zeros(shape, bool), task_done=jnp.zeros(shape, bool),
        action=jnp.zeros(shape, jnp.int32), value=jnp.zeros(shape), reward=jnp.zeros(shape),
        log_prob=jnp.full(shape, -np.log(8)), obs=obs,
        prev_actions=jnp.zeros(shape + (5,), jnp.int32), prev_reward=jnp.zeros(shape),
    )
    def update(st, tr):
        return ppo_update_networks(
            st, tr, jnp.zeros_like(tr.value), jnp.zeros_like(tr.value), cfg,
            actor_hidden_init=jnp.zeros((2, 64)), teacher_apply_fn=forbidden_teacher,
            kickstart_kl_coef=1., kickstart_value_coef=0.,
        )
    with mock.patch("train.obs_to_model_input", side_effect=lambda raw, *_: [raw["signal"]]):
        updated, info = jax.pmap(update, axis_name="devices")(replicate(state), transition)
    expected = jnp.sum(jnp.where(eligible[..., None], 1 / 8 - jax.nn.softmax(teacher_logits), 0), axis=(0, 1, 2)) / eligible.sum()
    np.testing.assert_allclose(updated.params, jnp.broadcast_to(-expected, (devices, 8)), atol=2e-7)
    np.testing.assert_array_equal(info["kickstart/value_mse"], 0)


def test_teacher_carry_is_pre_action_independent_frozen_and_resets_only_done_lanes():
    def apply(params, obs, hidden, method):
        assert method == "actor_step"
        next_hidden = hidden + obs + params
        return next_hidden, jnp.concatenate((next_hidden, -next_hidden), -1), next_hidden

    hidden = jnp.array([[4.], [9.]])
    student_hidden = jnp.array([[100.], [200.]])
    history = jnp.zeros((2, 5), dtype=jnp.int32)
    raw = {"signal": jnp.array([[1.], [2.]])}
    with mock.patch("utils.task_teachers.obs_to_model_input", side_effect=lambda obs, *_: obs["signal"]):
        def advance(params):
            return recurrent_teacher_rollout_observation(
                raw, history, hidden, apply, params, {},
                jnp.array([False, True]), jnp.array([17, 17]), jnp.array([17]),
            )
        cached, advanced = advance(jnp.float32(3))
        np.testing.assert_array_equal(advanced, [[8.], [14.]])
        np.testing.assert_array_equal(cached[ELIGIBLE_KEY], [True, False])
        np.testing.assert_array_equal(student_hidden, [[100.], [200.]])
        reset = reset_recurrent_teacher_hidden(advanced, jnp.array([True, False]))
        np.testing.assert_array_equal(reset, [[0.], [14.]])
        next_cached, next_hidden = recurrent_teacher_rollout_observation(
            raw, history, reset, apply, jnp.float32(3), {},
            jnp.array([False, False]), jnp.array([99, 17]), jnp.array([17]),
        )
        np.testing.assert_array_equal(next_hidden, [[4.], [19.]])
        np.testing.assert_array_equal(next_cached[ELIGIBLE_KEY], [False, True])
        # A reset does not relabel the just-completed transition.
        np.testing.assert_array_equal(cached[ELIGIBLE_KEY], [True, False])
        grad = jax.grad(lambda p: advance(p)[0][CACHED_LOGITS_KEY].sum() + advance(p)[1].sum())(jnp.float32(3))
        assert float(grad) == 0.0


def test_kl_mask_all_excluded_is_finite_zero_with_zero_student_gradient():
    teacher_logits = jnp.array([[1., -1.], [-2., 2.], [3., -3.]])
    def loss(student_logits, eligible):
        teacher_logp = jax.nn.log_softmax(teacher_logits)
        per_row = jnp.sum(jnp.exp(teacher_logp) * (teacher_logp - jax.nn.log_softmax(student_logits)), -1)
        return masked_teacher_kl(per_row, eligible)
    student = jnp.zeros((3, 2))
    excluded = jnp.zeros(3, dtype=bool)
    assert float(loss(student, excluded)) == 0.0
    np.testing.assert_array_equal(jax.grad(loss)(student, excluded), 0)
    selected = jnp.array([True, False, False])
    grad = jax.grad(loss)(student, selected)
    assert np.any(np.asarray(grad[0]) != 0)
    np.testing.assert_array_equal(grad[1:], 0)

    # Named collectives have the same reduction contract under vmap and pmap.
    # Compare the actual per-device loss-gradient pmean with a pooled reference.
    signals = jnp.array([[1., 2., 3.], [4., 5., 6.]])
    def device_update(weight, signal, mask):
        value, gradient = jax.value_and_grad(lambda w: masked_teacher_kl(
            (w * signal - 1) ** 2, mask, axis_name="devices",
        ))(weight)
        return jax.lax.pmean(value, "devices"), jax.lax.pmean(gradient, "devices")
    distributed = jax.vmap(device_update, in_axes=(None, 0, 0), axis_name="devices")
    for masks in (
        jnp.array([[True, False, False], [True, True, True]]),
        jnp.array([[False, False, False], [True, True, False]]),
        jnp.zeros((2, 3), dtype=bool),
    ):
        values, gradients = distributed(jnp.float32(.3), signals, masks)
        expected_value, expected_gradient = jax.value_and_grad(lambda w: masked_teacher_kl(
            (w * signals - 1) ** 2, masks,
        ))(jnp.float32(.3))
        np.testing.assert_allclose(values, expected_value, atol=1e-6)
        np.testing.assert_allclose(gradients, expected_gradient, atol=1e-6)


def test_native_sampler_precision_pool_and_provenance_match_while_bulk_is_unchanged():
    grids = jnp.zeros((1, 4, 2, 2), dtype=jnp.int32)
    base = MapsBuffer.new(
        maps=grids, padding_mask=grids, dumpability_masks_init=grids,
        action_maps=grids, distance_maps=grids,
        trench_axes=jnp.zeros((1, 4, 8, 8)), trench_types=jnp.zeros((1, 4)),
        foundation_border_axes=jnp.zeros((1, 4, 8, 7)), foundation_border_types=jnp.zeros((1, 4)),
        slot_indices=jnp.array([[41, 42, 43, 44]]),
    )
    buffer = PrecisionMapsBuffer(*base)
    buffer.precision_indices = jnp.array([1, 3])
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    bulk = EnvConfig(enforce_foundation_border_alignment=False)
    precise = bulk._replace(enforce_foundation_border_alignment=True)
    old = jax.vmap(lambda k: base._select_index(k, bulk)[1])(keys)
    new = jax.vmap(lambda k: buffer._select_index(k, bulk)[1])(keys)
    np.testing.assert_array_equal(old, new)
    indices = jax.vmap(lambda k: buffer._select_index(k, precise)[1])(keys)
    slots = jax.vmap(lambda k: buffer.get_map_provenance(k, precise)[0])(keys)
    assert set(np.asarray(indices).tolist()) == {1, 3}
    np.testing.assert_array_equal(slots, base.slot_indices[0, indices])
    np.testing.assert_array_equal(precision_lane_flags(2, 4, .5), [[True, True, False, False]] * 2)


def test_resume_reconstructs_recipe_without_the_original_json_and_rejects_changed_slots():
    saved = {"train_config": {"precision_episode_fraction": .5, "precision_slots": [3, 7], "teacher_slots": []}}
    config = SimpleNamespace(resume_from="resume.pkl", pull_direction_training_slots="/missing/old.json",
                             precision_episode_fraction=0., precision_slots=None, teacher_slots=None)
    resolve_pull_direction_training_slots(config)
    resolve_pull_direction_training_slots(config, saved)
    assert config.precision_episode_fraction == .5
    assert config.precision_slots == [3, 7] and config.teacher_slots == []
    config.precision_slots = [3]
    with pytest.raises(ValueError, match="precision_slots"):
        resolve_pull_direction_training_slots(config, saved)
