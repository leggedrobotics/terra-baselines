"""Imitation must preserve on-policy PPO, pre-action inputs, and its release clock."""
import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.jax_utils import replicate
from flax.training.train_state import TrainState

from test_task_teachers import _raw_obs
from train import Transition, ppo_update_networks
from train_mixed import MixedAgentTrainConfig
from utils.demonstrations import (
    demonstration_coefficient, demonstration_global_transitions, load_demonstrations,
    restore_demonstration_schedule, sample_demonstrations,
)
from utils.utils_ppo import obs_to_model_input


def _config(**kwargs):
    fields = dict(name="demonstration-test", num_envs_per_device=2, num_steps=2,
                  num_minibatches=1, num_prev_actions=5, actor_core="mlp",
                  agent_types_override=(0,), action_types_override=(0,),
                  executable_dig_observation=True, admissible_dig_observation=True,
                  time_observation_mode="remaining", local_map_area_scale=2.,
                  flat_minibatch_shuffle=True)
    fields.update(kwargs)
    return MixedAgentTrainConfig(**fields)


def _write_bank(path, lengths=(2, 8), **overrides):
    n = sum(lengths)
    action = (np.arange(n, dtype=np.int32) + 1) % 8
    history = np.zeros((n, 5), dtype=np.int32)
    offset = 0
    for length in lengths:
        for i in range(offset + 1, offset + length):
            history[i] = np.roll(history[i-1], 1)
            history[i, 0] = action[i-1]
        offset += length
    obs = {key: np.asarray(value) for key, value in _raw_obs(n).items()}
    obs["remaining_time"] = np.concatenate([1 - np.arange(x) / 450 for x in lengths]).astype(np.float32)
    # A row marker survives raw sampling and the actual preprocessing path.
    obs["agent_height"] = np.arange(n, dtype=np.int32)
    data = {"split": np.asarray("train"), "actions": action,
            "previous_actions": history, "episode_id": np.repeat(np.arange(len(lengths)), lengths),
            "group": np.asarray(["expert"] * len(lengths)),
            "condition": np.asarray(["test-condition"] * len(lengths)),
            "source_id": np.asarray([f"source-{i}" for i in range(len(lengths))]),
            "group_names": np.asarray(["expert"]), "group_weights": np.asarray([1.], np.float32),
            "supervision_mask": np.ones(n, bool), "parent_probs": np.zeros((n, 8), np.float32),
            **{f"obs/{key}": value for key, value in obs.items()}}
    data.update(overrides)
    np.savez(path, **data)
    return data


def test_training_only_sampling_is_episode_then_step_uniform_and_preserves_input_history(tmp_path):
    path = tmp_path / "training.npz"
    original = _write_bank(path)
    cfg = _config()
    data, counts = load_demonstrations([path], cfg, 2)
    assert counts == dict(episodes=2, transitions=10, episodes_per_file=[2], eligible_steps=10,
                         groups={"expert": dict(weight=1., episodes=2, conditions=1, sources=2, eligible_steps=10)})
    data = jax.tree_util.tree_map(jnp.asarray, data)
    sampled = jax.jit(lambda key: sample_demonstrations(data, key, 40000))(jax.random.PRNGKey(13))
    index = np.asarray(sampled["obs"]["agent_height"])
    assert abs(np.mean(index < 2) - .5) < .01  # transition-uniform would give .2
    np.testing.assert_allclose(np.bincount(index, minlength=10) / len(index),
                               [1/4]*2 + [1/16]*8, atol=.006)
    np.testing.assert_array_equal(sampled["previous_actions"], original["previous_actions"][index])
    model_input = obs_to_model_input(sampled["obs"], sampled["previous_actions"], cfg)
    np.testing.assert_array_equal(model_input[21], original["previous_actions"][index])
    np.testing.assert_array_equal(model_input[-1][:, 0], original["obs/remaining_time"][index])
    np.testing.assert_array_equal(model_input[-2], original["obs/local_map_admissible_dig"][index] / 2)
    second = tmp_path / "junction.npz"
    _write_bank(second, lengths=(3,))
    merged, merged_counts = load_demonstrations([path, second], cfg, 2)
    assert merged_counts["episodes"] == 3 and merged_counts["transitions"] == 13
    assert merged_counts["episodes_per_file"] == [2, 1]
    np.testing.assert_array_equal(merged["starts"], [0, 2, 10])
    np.testing.assert_array_equal(merged["lengths"], [2, 8, 3])
    for invalid in ({"split": np.asarray("evaluation")},
                    {"previous_actions": np.ones((10, 5), np.int32)},
                    {"episode_id": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])}):
        _write_bank(path, **invalid)
        with pytest.raises(ValueError):
            load_demonstrations([path], cfg, 2)


def test_group_condition_source_episode_sampling_excludes_prefix_without_resetting_history(tmp_path):
    path = tmp_path / "balanced.npz"
    lengths = (2, 8, 3, 4, 5, 7, 6, 9)
    groups = np.asarray(["foundation"]*4 + ["trench"] + ["expert"]*3)
    mask = np.ones(sum(lengths), bool)
    prefix_start = sum(lengths[:5])
    mask[prefix_start:prefix_start+3] = False
    probabilities = np.zeros((sum(lengths), 8), np.float32)
    probabilities[:sum(lengths[:5])] = .125
    original = _write_bank(path, lengths=lengths, group=groups,
        condition=np.asarray(["f-A", "f-A", "f-A", "f-B", "t-A", "h-A", "h-A", "h-B"]),
        source_id=np.asarray(["f0", "f0", "f1", "f2", "t0", "h0", "h1", "h2"]),
        group_names=np.asarray(["foundation", "trench", "expert"]),
        group_weights=np.asarray([.5, .2, .3], np.float32),
        supervision_mask=mask, parent_probs=probabilities)
    data, counts = load_demonstrations([path], _config(), 2)
    expected = np.asarray([.0625, .0625, .125, .25, .2, .1, .1, .1])
    np.testing.assert_allclose(np.exp(data["episode_log_probs"]), expected, atol=1e-7)
    assert counts["eligible_steps"] == sum(lengths)-3
    assert counts["groups"]["foundation"]["sources"] == 3
    sampled = jax.jit(lambda bank, key: sample_demonstrations(bank, key, 40000))(data, jax.random.PRNGKey(9))
    index = np.asarray(sampled["obs"]["agent_height"])
    episode = np.searchsorted(np.cumsum(lengths), index, side="right")
    np.testing.assert_allclose(np.bincount(episode, minlength=8)/len(index), expected, atol=.007)
    assert mask[index].all()
    np.testing.assert_array_equal(sampled["previous_actions"], original["previous_actions"][index])
    np.testing.assert_array_equal(sampled["parent_probs"], probabilities[index])
    assert np.any(np.asarray(sampled["previous_actions"])[index == prefix_start+3])
    model_input = obs_to_model_input(sampled["obs"], sampled["previous_actions"], _config())
    np.testing.assert_array_equal(model_input[-1][:, 0], original["obs/remaining_time"][index])


def test_grouped_bank_rejects_missing_groups_invalid_soft_targets_and_empty_supervision(tmp_path):
    path = tmp_path / "invalid.npz"
    for invalid in (
        dict(group_weights=np.asarray([.8], np.float32)),
        dict(group_names=np.asarray(["foundation", "expert"]), group_weights=np.asarray([.5, .5], np.float32)),
        dict(group=np.asarray(["foundation"]*2), group_names=np.asarray(["foundation"])),
        dict(supervision_mask=np.asarray([False]*2 + [True]*8)),
        dict(supervision_mask=np.ones(10, np.int32)),
    ):
        _write_bank(path, **invalid)
        with pytest.raises(ValueError):
            load_demonstrations([path], _config(), 2)


@pytest.mark.parametrize("group", ["foundation", "trench"])
def test_retention_requires_normalized_parent_targets_only_on_eligible_rows(tmp_path, group):
    path = tmp_path / "retention.npz"
    probabilities = np.full((10, 8), .125, np.float32)
    probabilities[0] = 0.  # Unsupervised prefix retains its original history.
    mask = np.ones(10, bool)
    mask[0] = False
    _write_bank(path, group=np.asarray([group]*2), group_names=np.asarray([group]),
                parent_probs=probabilities, supervision_mask=mask)
    data, _ = load_demonstrations([path], _config(), 2)
    np.testing.assert_array_equal(data["parent_probs"], probabilities)
    assert 0 not in data["eligible_indices"]
    for invalid in (np.zeros(8, np.float32), np.full(8, .25, np.float32)):
        broken = probabilities.copy()
        broken[1] = invalid
        _write_bank(path, group=np.asarray([group]*2), group_names=np.asarray([group]),
                    parent_probs=broken, supervision_mask=mask)
        with pytest.raises(ValueError, match=f"eligible {group} rows require cached selected-parent"):
            load_demonstrations([path], _config(), 2)
    for invalid in ([-.1] + [1.1/7]*7, [np.nan] + [.125]*7):
        broken = probabilities.copy()
        broken[1] = np.asarray(invalid, np.float32)
        _write_bank(path, group=np.asarray([group]*2), group_names=np.asarray([group]),
                    parent_probs=broken, supervision_mask=mask)
        with pytest.raises(ValueError, match="finite nonnegative float32"):
            load_demonstrations([path], _config(), 2)


@pytest.mark.parametrize("group", ["foundation", "trench", "expert", "recovery"])
def test_extra_conditions_preserve_hard_source_mass_and_retention_condition_balance(tmp_path, group):
    path = tmp_path / "sources.npz"

    def load(sources, conditions):
        n = len(sources)
        _write_bank(path, lengths=(1,)*n, group=np.asarray([group]*n),
                    group_names=np.asarray([group]), source_id=np.asarray(sources),
                    condition=np.asarray(conditions), parent_probs=np.full((n, 8), .125, np.float32))
        return load_demonstrations([path], _config(), 2)[0]

    baseline = load(["geometry-A", "geometry-B"], ["side-A", "side-A"])
    np.testing.assert_allclose(np.exp(baseline["episode_log_probs"]), [.5, .5], atol=1e-7)
    # Only A gains a repeated episode and another disposal condition.
    expanded = load(["geometry-A"]*3 + ["geometry-B"],
                    ["side-A", "side-A", "side-B", "side-A"])
    if group in ("expert", "recovery"):
        expected, source_a_mass = [.125, .125, .25, .5], .5
    else:
        # Ordinary retention keeps the original condition-first balancing.
        expected, source_a_mass = [.125, .125, .5, .25], .75
    probabilities = np.exp(expanded["episode_log_probs"])
    np.testing.assert_allclose(probabilities, expected, atol=1e-7)
    np.testing.assert_allclose(probabilities[:3].sum(), source_a_mass, atol=1e-7)
    sampled = jax.jit(lambda data, key: sample_demonstrations(data, key, 20000))(
        expanded, jax.random.PRNGKey(14))
    assert abs(np.mean(np.asarray(sampled["obs"]["agent_height"]) < 3) - source_a_mass) < .015


def test_release_restores_absolute_experience_and_rejects_missing_active_inputs(tmp_path):
    path = str(tmp_path / "train.npz")
    parent_cfg = _config(num_envs_per_device=4)
    parent = dict(train_config=parent_cfg, next_update=25000, optimizer_state=("native Adam",))
    cfg = _config(resume_from="parent.pkl", demonstration_npz=[path], demonstration_coef=.05,
                  demonstration_fade_transitions=4000)
    state = restore_demonstration_schedule(cfg, parent, "resume", 25000)
    assert state["origin_transitions"] == 25000 * parent_cfg.env_steps_per_update
    # A reduced finite smoke layout still starts at the actual parent's experience.
    assert demonstration_coefficient(state, state["origin_transitions"]) == .05
    halfway = 25000 + 2000 // cfg.env_steps_per_update
    now = demonstration_global_transitions(state, halfway, cfg.env_steps_per_update)
    assert demonstration_coefficient(state, now) == pytest.approx(.025)
    saved = pickle.loads(pickle.dumps(dict(train_config=copy.deepcopy(cfg), next_update=halfway,
                                          optimizer_state=parent["optimizer_state"], demonstration_state=state)))
    resumed_cfg = _config(resume_from="continuation.pkl", demonstration_npz=[path])
    resumed = restore_demonstration_schedule(resumed_cfg, saved, "resume", halfway)
    assert resumed == state
    assert resumed_cfg.demonstration_coef == .05
    assert resumed_cfg.demonstration_fade_transitions == 4000
    assert saved["optimizer_state"] == parent["optimizer_state"]
    missing_cfg = _config(resume_from="continuation.pkl")
    with pytest.raises(ValueError, match="explicit"):
        restore_demonstration_schedule(missing_cfg, saved, "resume", halfway)
    with pytest.raises(ValueError, match="missing its saved schedule"):
        restore_demonstration_schedule(resumed_cfg, {k:v for k,v in saved.items() if k != "demonstration_state"}, "resume", halfway)
    changed = copy.deepcopy(resumed_cfg)
    changed.demonstration_fade_transitions = 8000
    with pytest.raises(ValueError, match="fade changed"):
        restore_demonstration_schedule(changed, saved, "resume", halfway)
    end = 25000 + 4000 // cfg.env_steps_per_update
    saved["next_update"] = end + 100
    expired = restore_demonstration_schedule(missing_cfg, saved, "resume", end + 100)
    assert expired == state
    assert demonstration_coefficient(expired, demonstration_global_transitions(expired, end, cfg.env_steps_per_update)) == 0.
    assert not (tmp_path / "train.npz").exists()  # expired restore does not read the input

    from eval_fixed_bank import configure_for_bank, checkpoint_treatment_fingerprint
    from utils.helpers import checkpoint_evaluation_config
    evaluation = configure_for_bank(checkpoint_evaluation_config(saved), "validation/all", 2)
    evaluation.__post_init__()
    assert evaluation.demonstration_npz is None and evaluation.demonstration_coef == 0
    assert checkpoint_treatment_fingerprint(saved)["contract"]["demonstrations"] == state


def test_replicated_bank_is_a_dynamic_pmap_argument_with_identical_sampling(tmp_path):
    path = tmp_path / "training.npz"
    _write_bank(path)
    data, _ = load_demonstrations([path], _config(), 2)
    devices = jax.local_devices()
    bank = replicate(data, devices)
    steps = jnp.arange(len(devices), dtype=jnp.int32) + 1600000

    def sample_step(step, bank, active):
        if not active:
            return None
        key = jax.random.fold_in(jax.random.PRNGKey(42), step)
        key = jax.random.fold_in(key, jax.lax.axis_index("devices"))
        return sample_demonstrations(bank, key, 32)

    sampler = jax.pmap(sample_step, axis_name="devices", static_broadcasted_argnums=(2,))
    actual = sampler(steps, bank, True)
    old_bank = jax.tree_util.tree_map(jnp.asarray, data)
    expected = jax.pmap(lambda step: sample_step(step, old_bank, True), axis_name="devices")(steps)
    for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(left, right)
    # Raw bank arrays are traced as runtime arguments, never closed-over constants.
    traced = jax.make_jaxpr(lambda step, bank: sampler(step, bank, True))(steps, bank)
    assert not traced.consts
    assert sampler(steps, None, False) is None


def _actual_update(coefficient=None, poison=False, live_learning=False,
                   group_ids=(2, 2, 2, 2), parent_probability=None):
    cfg = _config(ent_coef=0., vf_coef=2. if live_learning else 0.)
    devices, samples = jax.local_device_count(), 4
    shape = (devices, samples)
    raw = _raw_obs(devices * samples)
    raw["remaining_time"] = jnp.ones(devices * samples)
    raw = jax.tree_util.tree_map(lambda x: x.reshape(shape + x.shape[1:]), raw)
    tr = Transition(done=jnp.zeros(shape, bool), task_done=jnp.zeros(shape, bool),
                    action=jnp.broadcast_to(jnp.arange(samples)[None], shape),
                    value=jnp.zeros(shape), reward=jnp.zeros(shape),
                    log_prob=jnp.full(shape, -np.log(8)), obs=raw,
                    prev_actions=jnp.zeros(shape + (5,)), prev_reward=jnp.zeros(shape))
    history = jnp.broadcast_to(jnp.arange(1, samples+1)[None, :, None], shape + (5,))
    demo_raw = dict(raw)
    if poison:
        demo_raw["remaining_time"] = jnp.full(shape, jnp.nan)
    group_id = jnp.broadcast_to(jnp.asarray(group_ids)[None], shape)
    probability = jnp.asarray(parent_probability if parent_probability is not None else [1/8]*8, jnp.float32)
    probabilities = jnp.where(group_id[..., None] < 2, probability, 0.)
    if poison:
        probabilities = jnp.full(shape + (8,), jnp.nan)
    batch = dict(obs=demo_raw, previous_actions=history,
                 actions=jnp.broadcast_to(jnp.arange(samples)[None], shape),
                 group_id=group_id, parent_probs=probabilities)

    def student(params, obs):
        signal = obs[21][:, 0] + obs[-1][:, 0]
        value = jnp.broadcast_to(params["critic"], (len(signal), 1))
        logits = params["actor"][None] * signal[:, None]
        return value, logits

    state = TrainState.create(apply_fn=student,
                             params=dict(actor=jnp.zeros(8), critic=jnp.float32(0)), tx=optax.adam(.01))

    def update(state, tr, batch):
        advantages = jnp.arange(samples, dtype=jnp.float32) if live_learning else jnp.zeros_like(tr.value)
        targets = jnp.ones_like(tr.value) if live_learning else jnp.zeros_like(tr.value)
        return ppo_update_networks(state, tr, advantages, targets, cfg,
                                   demonstration_batch=None if coefficient is None else batch,
                                   demonstration_coef=0. if coefficient is None else coefficient)
    return jax.pmap(update, axis_name="devices")(replicate(state), tr, batch)


def test_zero_coefficient_keeps_native_update_identical_and_skips_poisoned_demo_forward():
    control, control_info = _actual_update(live_learning=True)
    inactive, inactive_info = _actual_update(0., poison=True, live_learning=True, group_ids=(0, 0, 1, 2))
    assert np.any(control.params["actor"] != 0) and np.all(control.params["critic"] != 0)
    for expected, actual in zip(jax.tree_util.tree_leaves(control), jax.tree_util.tree_leaves(inactive)):
        np.testing.assert_array_equal(actual, expected)
    for key, expected in control_info.items():
        np.testing.assert_array_equal(inactive_info[key], expected)
    assert np.all(inactive_info["imitation/cross_entropy"] == 0)


@pytest.mark.parametrize("group_id", [2, 3])
def test_positive_actor_imitation_is_finite_and_has_no_demonstration_critic_target(group_id):
    state, info = _actual_update(.05, group_ids=(group_id,)*4)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((state, info)))
    np.testing.assert_allclose(info["imitation/cross_entropy"], np.log(8), atol=1e-6)
    np.testing.assert_allclose(info["total_loss"], .05 * np.log(8), atol=1e-6)
    np.testing.assert_allclose(info["imitation/action_accuracy"], .25)
    np.testing.assert_array_equal(state.params["critic"], 0.)
    assert np.all(state.params["actor"][:, :4] > 0)
    assert np.all(state.params["actor"][:, 4:] < 0)


@pytest.mark.parametrize("group_id, group", [(0, "foundation"), (1, "trench")])
def test_cached_retention_kl_is_zero_at_parent_and_pulls_toward_shifted_parent(group_id, group):
    control, _ = _actual_update()
    retained, retained_info = _actual_update(.05, group_ids=(group_id,)*4)
    for expected, actual in zip(jax.tree_util.tree_leaves(control), jax.tree_util.tree_leaves(retained)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(retained_info[f"imitation/{group}/loss"], 0., atol=1e-7)
    np.testing.assert_allclose(retained_info[f"imitation/{group}/target_entropy"], np.log(8), atol=1e-6)
    np.testing.assert_array_equal(retained_info["imitation/cross_entropy"], 0.)
    shifted, shifted_info = _actual_update(.05, group_ids=(group_id,)*4, parent_probability=[.5, .25, .25, 0, 0, 0, 0, 0])
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves((shifted, shifted_info)))
    expected_kl = np.log(8) + .5*np.log(.5) + .5*np.log(.25)
    np.testing.assert_allclose(shifted_info[f"imitation/{group}/loss"], expected_kl, atol=1e-6)
    np.testing.assert_allclose(shifted_info["total_loss"], .05*expected_kl, atol=1e-6)
    np.testing.assert_array_equal(shifted.params["critic"], 0.)
    assert np.all(shifted.params["actor"][:, :3] > 0)


def test_mixed_soft_retention_and_hard_correction_apply_group_weights_only_once():
    state, mixed = _actual_update(.05, group_ids=(0, 1, 2, 3))
    for group in ("foundation", "trench", "expert", "recovery"):
        np.testing.assert_allclose(mixed[f"imitation/{group}/sample_fraction"], .25)
        expected_loss = 0. if group in ("foundation", "trench") else np.log(8)
        np.testing.assert_allclose(mixed[f"imitation/{group}/loss"], expected_loss, atol=1e-6)
    np.testing.assert_allclose(mixed["total_loss"], .05*.5*np.log(8), atol=1e-6)
    np.testing.assert_allclose(mixed["imitation/cross_entropy"], np.log(8), atol=1e-6)
    np.testing.assert_allclose(mixed["imitation/action_accuracy"], 0.)
    np.testing.assert_array_equal(state.params["critic"], 0.)
    assert np.all(state.params["actor"][:, 2:4] > 0)
