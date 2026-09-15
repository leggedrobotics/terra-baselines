"""Claim-driven checks for native observations, routing, gradients and resume."""

import copy
from dataclasses import asdict
import os
import pickle
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.jax_utils import replicate
from flax.training.train_state import TrainState

from train import Transition, ppo_update_networks
from train_mixed import MixedAgentTrainConfig, _validate_teacher_resume
from utils.task_teachers import (
    FAMILY_KEY, LEGACY_DIG_KEY, bind_task_teacher_checkpoints,
    finalize_task_teacher_metrics, legacy_teacher_admissible_dig,
    load_task_teacher_checkpoint, make_task_teacher_apply_fn,
    native_task_teacher_obs, resolve_task_teacher_families,
    task_teacher_rollout_observation, validate_task_teacher_configs,
    validate_task_teacher_resume,
)
from utils.utils_ppo import obs_to_model_input


ROUTE = {"foundation": 3, "trench": 1}


def _config(**kwargs):
    fields = dict(name="task-teacher-test", teacher_checkpoint="foundation.pkl",
                  trench_teacher_checkpoint="trench.pkl", kickstart_value_coef=0,
                  kickstart_lr_warmup_updates=0, num_prev_actions=5,
                  executable_dig_observation=True, admissible_dig_observation=True,
                  agent_types_override=(0,), action_types_override=(0,))
    fields.update(kwargs)
    return MixedAgentTrainConfig(**fields)


def _teacher_config(executable=True, **kwargs):
    fields = dict(num_prev_actions=5, admissible_dig_observation=True,
                  executable_dig_observation=executable, actor_core="mlp",
                  clip_action_maps=True, local_map_area_scale=1.0)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _raw_obs(n):
    local = ("local_map_action_neg", "local_map_action_pos", "local_map_target_neg",
             "local_map_target_pos", "local_map_dumpability", "local_map_obstacles",
             "local_map_border_workspace", "local_map_edge_alignment_error",
             "local_map_border_diggable", "local_map_admissible_dig")
    maps = ("traversability_mask", "reachability_mask", "action_map", "target_map",
            "padding_mask", "dumpability_mask", "interaction_mask")
    obs = {key: jnp.zeros((n, 12)) for key in local}
    obs.update({key: jnp.zeros((n, 2, 2)) for key in maps})
    obs.update(agent_states=jnp.zeros((n, 4, 9)), agent_active=jnp.ones((n, 4)),
               num_agents=jnp.ones(n), agent_width=jnp.ones(n), agent_height=jnp.ones(n),
               reward_v2_reset_context=jnp.ones((n, 2)),
               movement_feasibility=jnp.ones((n, 4)))
    obs["local_map_admissible_dig"] = jnp.arange(n, dtype=jnp.float32)[:, None] + jnp.ones((n, 12))
    obs[LEGACY_DIG_KEY] = obs["local_map_admissible_dig"] + 9
    obs[FAMILY_KEY] = jnp.asarray(np.resize([3, 1, 1, 3], n), dtype=jnp.int32)
    return obs


def _teacher_apply(params, obs):
    # The last feature is native admissible digging; previous action is also
    # sample-dependent so a broken trajectory permutation changes the result.
    signal = obs[-1][:, :1] + obs[21][:, :1]
    logits = signal * params[None, :]
    return signal, logits


def _forward():
    return make_task_teacher_apply_fn(
        _teacher_apply, _teacher_apply, _teacher_config(True), _teacher_config(False), ROUTE,
    )


def _params():
    return {"foundation": jnp.arange(8, dtype=jnp.float32) / 9,
            "trench": -jnp.arange(8, dtype=jnp.float32) / 13}


def test_known_family_ids_come_from_actual_names():
    assert resolve_task_teacher_families(("unknown", "trench", "unused", "foundation"),
                                        [[3, 1], [1, 3]]) == ROUTE
    for ids in ([[0, 1, 3]], [[1]], [[1, 2, 3]]):
        with pytest.raises(ValueError, match="provenance"):
            resolve_task_teacher_families(("unknown", "trench", "unused", "foundation"), ids)


@pytest.mark.parametrize("override", [
    {"teacher_checkpoint": None}, {"teacher_obs_downsample": 2},
    {"kickstart_value_coef": .5}, {"actor_core": "gru"},
    {"agent_types_override": (0, 0)}, {"action_types_override": (1,)},
    {"executable_dig_observation": False}, {"action_logit_masking": True},
])
def test_narrow_mode_rejects_unsupported_interfaces(override):
    with pytest.raises(ValueError):
        _config(**override)


def test_native_config_checks_history_and_uses_each_teachers_preprocessing():
    raw = _raw_obs(3)
    history = jnp.zeros((3, 5))
    teacher = _teacher_config(False, local_map_area_scale=2.0,
                              reward_v2_reset_context_observation=True,
                              movement_feasibility_observation=True)
    actual = native_task_teacher_obs(raw, history, teacher)
    expected_raw = dict(raw, local_map_admissible_dig=raw[LEGACY_DIG_KEY])
    expected = obs_to_model_input(expected_raw, history, teacher)
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(actual[-1], raw[LEGACY_DIG_KEY] / 2)
    assert len(actual) == 25  # proves no fixed obs[25] replacement
    validate_task_teacher_configs(_config(), _teacher_config(True), teacher)
    teacher.num_prev_actions = 10
    with pytest.raises(ValueError, match="num_prev_actions"):
        validate_task_teacher_configs(_config(), _teacher_config(True), teacher)


def test_teacher_without_admissible_uses_its_native_reset_context_and_carry():
    teacher = _teacher_config(False, admissible_dig_observation=False,
                              carry_work_observation=True,
                              reward_v2_reset_context_observation=True)
    validate_task_teacher_configs(_config(), teacher, _teacher_config(False))
    raw = _raw_obs(2)
    del raw[LEGACY_DIG_KEY]
    del raw["local_map_admissible_dig"]
    raw["reward_v2_reset_context"] = jnp.array([[0., .82], [.25, .64]])
    raw["agent_states"] = raw["agent_states"].at[:, 0, 8].set(jnp.array([.03, .07]))
    actual = native_task_teacher_obs(raw, jnp.zeros((2, 5)), teacher)
    assert len(actual) == 23
    np.testing.assert_array_equal(actual[22], raw["reward_v2_reset_context"])
    np.testing.assert_array_equal(actual[0][..., 8], raw["agent_states"][..., 8])
    missing = dict(raw)
    del missing["reward_v2_reset_context"]
    with pytest.raises(ValueError, match="reward_v2_reset_context"):
        native_task_teacher_obs(missing, jnp.zeros((2, 5)), teacher)


@pytest.mark.parametrize("flat", [False, True])
def test_routing_survives_both_real_ppo_shuffle_layouts(flat):
    n, seq, envs = 12, 3, 4
    raw = _raw_obs(n)
    history = jnp.arange(n * 5, dtype=jnp.float32).reshape(n, 5) / 10
    original = _forward()(_params(), raw, history)
    batch = jax.tree_util.tree_map(lambda x: x.reshape((seq, envs) + x.shape[1:]), (raw, history))
    if flat:
        batch = jax.tree_util.tree_map(lambda x: x.reshape((n,) + x.shape[2:]), batch)
        indices = np.array([7, 4, 11, 1, 6, 10, 5, 0, 3, 9, 8, 2])
        shuffled = jax.tree_util.tree_map(lambda x: x[indices], batch)
    else:
        indices = np.arange(n).reshape(seq, envs).T[[2, 0, 3, 1]].reshape(-1)
        shuffled = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1)[jnp.array([2, 0, 3, 1])], batch)
        shuffled = jax.tree_util.tree_map(lambda x: x.reshape((n,) + x.shape[2:]), shuffled)
    actual = _forward()(_params(), *shuffled)
    for x, y in zip(actual, original):
        np.testing.assert_array_equal(x, y[indices])
    expected_f = _teacher_apply(_params()["foundation"], native_task_teacher_obs(raw, history, _teacher_config(True)))[1]
    expected_t = _teacher_apply(_params()["trench"], native_task_teacher_obs(raw, history, _teacher_config(False)))[1]
    expected = jnp.where((raw[FAMILY_KEY] == 3)[:, None], expected_f, expected_t)
    np.testing.assert_array_equal(original[1], expected)
    frozen_grads = jax.grad(lambda p: _forward()(p, raw, history)[1].sum())(_params())
    for leaf in jax.tree_util.tree_leaves(frozen_grads):
        np.testing.assert_array_equal(leaf, 0)
    wrong = dict(raw, **{FAMILY_KEY: raw[FAMILY_KEY].at[0].set(0)})
    assert bool(jnp.isnan(_forward()(_params(), wrong, history)[1][0]).all())


@pytest.mark.parametrize("flat", [False, True])
def test_actual_ppo_gradient_and_global_selected_counts(flat):
    devices, samples = jax.local_device_count(), 4
    n = devices * samples
    raw, history = _raw_obs(n), jnp.zeros((n, 5))
    # Deliberately unequal family exposure between devices.
    raw[FAMILY_KEY] = jnp.where(jnp.arange(n) < 3, 3, 1).astype(jnp.int32)
    teacher_logits = _forward()(_params(), raw, history)[1]
    expected_gradient = (jnp.ones_like(teacher_logits) / 8 - jax.nn.softmax(teacher_logits)).mean(0)
    cfg = _config(num_envs_per_device=2, num_minibatches=1, num_steps=2,
                  ent_coef=0, vf_coef=0, task_teacher_family_ids=ROUTE,
                  flat_minibatch_shuffle=flat)
    def student_apply(params, obs):
        return jnp.zeros((obs[0].shape[0], 1)), jnp.broadcast_to(params, (obs[0].shape[0], 8))
    state = TrainState.create(apply_fn=student_apply, params=jnp.zeros(8), tx=optax.sgd(1.0))
    shape = (devices, 4) if flat else (devices, 2, 2)
    batched_raw = jax.tree_util.tree_map(lambda x: x.reshape(shape + x.shape[1:]), raw)
    transition = Transition(done=jnp.zeros(shape, bool), task_done=jnp.zeros(shape, bool),
                            action=jnp.zeros(shape, jnp.int32), value=jnp.zeros(shape),
                            reward=jnp.zeros(shape), log_prob=jnp.full(shape, -np.log(8)),
                            obs=batched_raw, prev_actions=history.reshape(shape + (5,)),
                            prev_reward=jnp.zeros(shape))
    def update(state, tr):
        return ppo_update_networks(state, tr, jnp.zeros_like(tr.value), jnp.zeros_like(tr.value),
                                   cfg, teacher_apply_fn=_forward(), teacher_params=_params(),
                                   kickstart_kl_coef=1., kickstart_value_coef=0.)
    updated, info = jax.pmap(update, axis_name="devices")(replicate(state), transition)
    np.testing.assert_allclose(np.asarray(updated.params),
                               np.broadcast_to(-np.asarray(expected_gradient), (devices, 8)),
                               atol=2e-7, rtol=2e-6)
    assert np.all(np.asarray(info["kickstart/foundation_selected_count"]) == 3)
    assert np.all(np.asarray(info["kickstart/trench_selected_count"]) == n - 3)
    assert np.all(np.isfinite(np.asarray(info["kickstart/kl"])))


def test_conditional_kl_uses_weighted_sums_even_for_rare_family():
    # Four minibatches, two epochs: the foundation has only one sample per
    # rollout. Averaging per-minibatch means or clamping count to 1 is wrong.
    metrics = {"kickstart/foundation_kl_sum": jnp.float32(.75),
               "kickstart/foundation_selected_count": jnp.float32(.25),
               "kickstart/trench_kl_sum": jnp.float32(0),
               "kickstart/trench_selected_count": jnp.float32(0)}
    result = finalize_task_teacher_metrics(metrics, 4)
    assert float(result["kickstart/foundation_kl"]) == 3
    assert float(result["kickstart/foundation_selected_count"]) == 1
    assert float(result["kickstart/trench_kl"]) == 0


def test_terminal_transition_retains_old_family_then_next_transition_switches():
    from test_episode_aggregates import _step, _update
    from utils.episode_aggregates import new_episode_accumulator, empty_episode_aggregate

    accumulator = new_episode_accumulator(jnp.array([1, 0]), jnp.ones(2, jnp.int32),
                                           jnp.zeros(2, jnp.int32))
    pending = empty_episode_aggregate(16)
    with mock.patch("utils.task_teachers.legacy_teacher_admissible_dig",
                    side_effect=lambda state: jnp.full((12,), state)):
        recorded = task_teacher_rollout_observation({}, jnp.array([2, 3]), accumulator.family_id)
        ended = _step(reward=[0., 0.], done=[True, False], task_done=[False, False],
                      timeout=[True, False], agent_reward=[0., 0.])
        accumulator, _ = _update(accumulator, pending, ended, next_family=jnp.array([0, 1]))
        next_recorded = task_teacher_rollout_observation({}, jnp.array([8, 3]), accumulator.family_id)
    np.testing.assert_array_equal(recorded[FAMILY_KEY], [1, 0])
    np.testing.assert_array_equal(next_recorded[FAMILY_KEY], [0, 0])
    np.testing.assert_array_equal(recorded[LEGACY_DIG_KEY][:, 0], [2, 3])
    np.testing.assert_array_equal(next_recorded[LEGACY_DIG_KEY][:, 0], [8, 3])


def test_dual_resume_binds_bytes_and_roles_but_allows_relocated_paths(tmp_path):
    f, t = tmp_path / "f.pkl", tmp_path / "t.pkl"
    f.write_bytes(pickle.dumps({"model": [1]}))
    t.write_bytes(pickle.dumps({"model": [2]}))
    config = _config(teacher_checkpoint=str(f), trench_teacher_checkpoint=str(t),
                     task_teacher_family_ids=ROUTE)
    bind_task_teacher_checkpoints(config)
    checkpoint = {"train_config": copy.deepcopy(config)}
    moved = tmp_path / "moved.pkl"
    moved.write_bytes(f.read_bytes())
    config.teacher_checkpoint = str(moved)
    bind_task_teacher_checkpoints(config)
    _validate_teacher_resume(checkpoint, config)
    validate_task_teacher_resume(checkpoint, config, require_families=True)
    assert load_task_teacher_checkpoint(moved, config.teacher_checkpoint_sha256) == {"model": [1]}
    config.task_teacher_family_ids = {"foundation": 1, "trench": 3}
    with pytest.raises(ValueError, match="family_ids"):
        validate_task_teacher_resume(checkpoint, config, require_families=True)
    config.task_teacher_family_ids = ROUTE
    moved.write_bytes(t.read_bytes())
    with pytest.raises(ValueError, match="bytes"):
        bind_task_teacher_checkpoints(config)
    with pytest.raises(ValueError, match="bytes"):
        load_task_teacher_checkpoint(moved, config.teacher_checkpoint_sha256)
    config.trench_teacher_checkpoint = None
    with pytest.raises(ValueError, match="mode"):
        _validate_teacher_resume(checkpoint, config)
    single = {"train_config": {"teacher_checkpoint": "old.pkl"}}
    with pytest.raises(ValueError, match="mode"):
        _validate_teacher_resume(single, checkpoint["train_config"])


def test_fixed_bank_clears_dual_flags_without_changing_saved_config():
    from eval_fixed_bank import configure_for_bank
    config = _config(task_teacher_family_ids=ROUTE, teacher_checkpoint_sha256="a" * 64,
                     trench_teacher_checkpoint_sha256="b" * 64)
    saved = asdict(config)
    evaluated = configure_for_bank(config, "validation/all", 64)
    evaluated.__post_init__()
    assert evaluated.teacher_checkpoint is None
    assert evaluated.trench_teacher_checkpoint is None
    assert evaluated.teacher_checkpoint_sha256 is None
    assert evaluated.trench_teacher_checkpoint_sha256 is None
    assert evaluated.task_teacher_family_ids is None
    assert asdict(config) == saved


def test_real_state_legacy_feature_matches_native_wrapper_and_differs_from_executable():
    from terra.env import TerraEnv
    from terra.tests.test_foundation_behavior import foundation, _pose
    from terra.wrappers import LocalMapWrapper

    state = _pose(foundation.__wrapped__(), loaded=3, cabin=2)
    # Compare compiled geometry to compiled native geometry: eager sin/cos
    # rounding can change boundary cells and is not the training execution.
    native_legacy = jax.jit(lambda s: LocalMapWrapper.wrap(s, executable_dig_observation=False))(state)
    native_executable = jax.jit(lambda s: LocalMapWrapper.wrap(s, executable_dig_observation=True))(state)
    legacy = jax.jit(legacy_teacher_admissible_dig)(state)
    np.testing.assert_array_equal(legacy, native_legacy.world.local_map_admissible_dig.map)
    assert np.asarray(legacy).sum() > 0
    assert np.asarray(native_executable.world.local_map_admissible_dig.map).sum() == 0
    raw = TerraEnv._state_to_obs_dict(native_executable)
    batched_raw = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), raw)
    batched_state = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)
    old_family = jnp.array([3, 1], jnp.int32)
    recorded = jax.jit(task_teacher_rollout_observation)(batched_raw, batched_state, old_family)
    np.testing.assert_array_equal(recorded[FAMILY_KEY], old_family)
    np.testing.assert_array_equal(recorded[LEGACY_DIG_KEY], np.stack([legacy, legacy]))
    assert FAMILY_KEY not in batched_raw and LEGACY_DIG_KEY not in batched_raw
    history = jnp.zeros((2, 5))
    expected_raw = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]),
                                        TerraEnv._state_to_obs_dict(native_legacy))
    actual = native_task_teacher_obs(recorded, history, _teacher_config(False))
    expected = obs_to_model_input(expected_raw, history, _teacher_config(False))
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(x, y)

    # The student input remains its original executable observation list.
    for x, y in zip(obs_to_model_input(recorded, history, _teacher_config(True)),
                    obs_to_model_input(batched_raw, history, _teacher_config(True))):
        np.testing.assert_array_equal(x, y)


@pytest.mark.skipif(not os.environ.get("TERRA_LEGACY_FF_TEACHER_CHECKPOINT"),
                    reason="optional recovered native FF checkpoint is not installed")
def test_recovered_native_ff_logits_match_evaluation_with_real_reset_and_carry_context():
    from terra.config import BatchConfig, MapsDimsConfig
    from terra.env import TerraEnv
    from terra.state import State
    from terra.tests.test_foundation_behavior import foundation
    from terra.wrappers import LocalMapWrapper
    from utils import helpers
    from utils.models import get_model_ready, validate_model_params_match

    helpers.register_checkpoint_config_classes()
    saved = load_task_teacher_checkpoint(
        os.environ["TERRA_LEGACY_FF_TEACHER_CHECKPOINT"],
        "2fe5d23c86cc7702b188d33ca1ca9a42066a9a2515150e8795f8c640bbbeb4af",
    )
    teacher = helpers.checkpoint_evaluation_config(saved)
    assert teacher.carry_work_observation and teacher.reward_v2_reset_context_observation
    assert not teacher.admissible_dig_observation
    assert not teacher.executable_dig_observation
    assert not teacher.relocation_distance_observation
    assert not teacher.trench_alignment_observation
    validate_task_teacher_configs(_config(), teacher, _teacher_config(False))

    full = foundation.__wrapped__()
    target = np.asarray(full.world.target_map.map)
    action = np.zeros_like(target)
    # A mass-balanced partial start: 20 dug units, 17 accepted, 3 carried.
    dug = np.argwhere(target < 0)[:20]
    accepted = np.argwhere(target > 0)[:17]
    action[dug[:, 0], dug[:, 1]] = -1
    action[accepted[:, 0], accepted[:, 1]] = 1
    agent_state = full._get_current_agent_state()._replace(
        loaded=jnp.array([3], dtype=jnp.int8), carry_relocation_credit=jnp.float32(3),
    )
    partial = State.new(
        jax.random.PRNGKey(17), full.env_cfg._replace(reset_tier=jnp.int32(1)),
        target, np.zeros_like(target),
        -97.0 * np.ones((4, 8), np.float32), np.int32(-1),
        -97.0 * np.ones((64, 3), np.float32), np.int32(-1),
        np.ones_like(target, dtype=bool), action,
        distance_map_override=np.ones_like(target, dtype=np.float32),
        initial_agent=full._set_current_agent_state(agent_state).agent,
    )
    states = jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y]), full, partial)
    def observations(state, executable):
        return TerraEnv._state_to_obs_dict(TerraEnv.wrap_state(
            state, executable_dig_observation=executable,
        ))
    native_raw = jax.jit(jax.vmap(lambda s: observations(s, False)))(states)
    student_raw = jax.jit(jax.vmap(lambda s: observations(s, True)))(states)
    v0 = float(full._required_excavation_volume())
    np.testing.assert_allclose(native_raw["reward_v2_reset_context"],
                               [[0, 1], [20/v0, (v0-17)/v0]], rtol=1e-6)
    np.testing.assert_allclose(native_raw["agent_states"][:, 0, 8], [0, 3/v0], rtol=1e-6)
    # These values are produced even though the student does not request the
    # carry/reset-context model features. Never replace a full-start H/V0 by 0.
    assert not _config().reward_v2_reset_context_observation
    history = jnp.array([[0, 0, 0, 0, 0], [7, 0, 6, 4, 2]], dtype=jnp.int32)
    native_input = obs_to_model_input(native_raw, history, teacher)
    teacher_input = native_task_teacher_obs(student_raw, history, teacher)
    assert len(native_input) == len(teacher_input) == 23
    for actual, expected in zip(teacher_input, native_input):
        np.testing.assert_array_equal(actual, expected)

    model_env = SimpleNamespace(
        batch_cfg=BatchConfig()._replace(maps_dims=MapsDimsConfig(maps_edge_length=64)),
        executable_dig_observation=False,
    )
    model, initialized = get_model_ready(jax.random.PRNGKey(8), teacher, model_env)
    validate_model_params_match(initialized, saved["model"], "recovered FF teacher")
    native_apply = jax.jit(model.apply)
    native_value, native_logits = native_apply(saved["model"], native_input)
    adapted_value, adapted_logits = native_apply(saved["model"], teacher_input)
    np.testing.assert_array_equal(adapted_value, native_value)
    np.testing.assert_array_equal(adapted_logits, native_logits)
    assert np.isfinite(np.asarray(native_logits)).all()

    student_raw[FAMILY_KEY] = jnp.array([ROUTE["foundation"]] * 2, jnp.int32)
    student_raw[LEGACY_DIG_KEY] = native_raw["local_map_admissible_dig"]
    def other_teacher(params, obs):
        return jnp.zeros((2, 1)), jnp.zeros((2, 8))
    routed = make_task_teacher_apply_fn(native_apply, other_teacher,
                                       teacher, _teacher_config(False), ROUTE)
    routed_value, routed_logits = routed({"foundation": saved["model"], "trench": {}},
                                        student_raw, history)
    np.testing.assert_array_equal(routed_value, native_value)
    np.testing.assert_array_equal(routed_logits, native_logits)
