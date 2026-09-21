"""Retained-cost history is observable without resetting the native policy."""

import copy
from dataclasses import replace
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
from terra.config import BatchConfig, EnvConfig, MapsDimsConfig
from terra.env import TerraEnv

from utils.models import get_model_ready
from utils.retained_work_context import (
    RETAINED_WORK_PARAMETER_NAMES,
    migrate_retained_work_context_checkpoint,
)
from utils.utils_ppo import obs_to_model_input


class Config(dict):
    __getattr__ = dict.__getitem__


@pytest.mark.parametrize("changes", [
    {"resume_from": None}, {"warm_start_from": "parent.pkl"},
    {"resume_update": 105000}, {"retained_work_context_observation": False},
    {"migrate_remaining_time": True}, {"grow_actor_capacity": True},
    {"action_logit_masking": True}, {"actor_core": "gru"},
])
def test_context_migration_configuration_requires_only_native_context_growth(changes):
    from train_mixed import MixedAgentTrainConfig
    kwargs = dict(name="context-test", resume_from="parent.pkl",
                  time_observation_mode="remaining", actor_residual_head=True,
                  retained_work_context_observation=True, migrate_retained_work_context=True)
    MixedAgentTrainConfig(**kwargs)
    with pytest.raises(ValueError):
        MixedAgentTrainConfig(**(kwargs | changes))


def test_native_context_and_retained_cost_ramp_share_resume_and_evaluation_contracts():
    from eval_fixed_bank import checkpoint_treatment_fingerprint
    from train_mixed import (
        MixedAgentTrainConfig, _r2_protocol_receipt,
        _validate_checkpoint_architecture, _validate_r2_resume_checkpoint,
    )
    from utils.behavior_cost_ramp import ramp_costs, restore_behavior_cost_ramp
    from utils.helpers import (
        checkpoint_evaluation_config, checkpoint_foundation_behavior,
        checkpoint_retained_work_costs, overlay_foundation_behavior,
    )

    cfg = MixedAgentTrainConfig(
        name="native-parent", num_devices=1, reward_stage="reward_v2",
        carry_work_observation=True, time_observation_mode="remaining",
        actor_residual_head=True, resume_from="parent.pkl", num_minibatches=32,
        distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
        distance_sidecar_sha256="a" * 64,
    )
    model = {"params": {"sentinel": jnp.array([1., 2.])}}
    optimizer = optax.adam(3e-4).init(model)
    optimizer = (optimizer[0]._replace(count=jnp.int32(105000 * 64)), *optimizer[1:])
    checkpoint = dict(train_config=cfg, model=model, optimizer_state=optimizer,
                      env_config=EnvConfig(), next_update=105000,
                      train_state_step=np.array(105000 * 64),
                      r2_protocol_receipt=_r2_protocol_receipt(cfg))
    target = replace(cfg, retained_work_context_observation=True,
                     lateral_dig_cost=.125, retained_work_setup_cost=.0025,
                     retained_work_travel_cost=.0025, retained_work_turn_cost=.01,
                     finetune_foundation_behavior=True, behavior_cost_ramp_updates=2500)
    with pytest.raises(ValueError, match="retained_work_context_observation"):
        _validate_checkpoint_architecture(checkpoint, target)
    target.migrate_retained_work_context = True
    _validate_checkpoint_architecture(checkpoint, target)
    _validate_r2_resume_checkpoint(checkpoint, _r2_protocol_receipt(target), target)
    parent_behavior = {**checkpoint_foundation_behavior(checkpoint),
                       **checkpoint_retained_work_costs(checkpoint)}
    ramp = restore_behavior_cost_ramp(target, checkpoint, "resume", 105000, parent_behavior)
    rebuilt = copy.deepcopy(model)
    for name in RETAINED_WORK_PARAMETER_NAMES:
        rebuilt["params"][name] = jnp.zeros((5, 704))
    migrated = migrate_retained_work_context_checkpoint(checkpoint, rebuilt)
    assert migrated["train_config"].retained_work_context_observation
    assert not migrated["train_config"].migrate_retained_work_context
    saved_cfg = replace(target, migrate_retained_work_context=False,
                        finetune_foundation_behavior=False)
    resumed = {
        **migrated, "train_config": saved_cfg, "next_update": 106250,
        "train_state_step": np.array(106250 * 64), "behavior_cost_ramp_state": ramp,
        "optimizer_state": (migrated["optimizer_state"][0]._replace(count=jnp.int32(106250 * 64)),
                            *migrated["optimizer_state"][1:]),
        "r2_protocol_receipt": _r2_protocol_receipt(saved_cfg),
        "env_config": overlay_foundation_behavior(EnvConfig(), ramp_costs(ramp, 106250)),
    }
    _validate_checkpoint_architecture(resumed, saved_cfg)
    _validate_r2_resume_checkpoint(resumed, _r2_protocol_receipt(saved_cfg), saved_cfg)
    behavior = {**checkpoint_foundation_behavior(resumed), **checkpoint_retained_work_costs(resumed)}
    assert restore_behavior_cost_ramp(saved_cfg, resumed, "resume", 106250, behavior) == ramp
    evaluated = checkpoint_evaluation_config(resumed)
    assert evaluated.retained_work_context_observation and not evaluated.migrate_retained_work_context
    assert evaluated.retained_work_setup_cost == pytest.approx(.00125)
    contract = checkpoint_treatment_fingerprint(resumed)["contract"]
    assert contract["architecture"]["retained_work_context_observation"] is True
    assert contract["retained_work_costs"]["retained_work_setup_cost"] == pytest.approx(.00125)
    with pytest.raises(ValueError, match="retained_work_context_observation"):
        _validate_checkpoint_architecture(resumed, replace(saved_cfg, retained_work_context_observation=False))


def test_retained_context_has_reset_validity_and_uses_acting_slot():
    state = NS(
        agent=NS(current_agent=0),
        retained_work_pose=jnp.array([[0, 0, 0], [63, 31, 1], [2, 3, 0], [0, 0, 0]]),
        retained_work_events=jnp.array([0, 2, 1, 0]),
        world=NS(target_map=NS(map=jnp.zeros((64, 32)))),
        env_cfg=NS(agent=NS(angles_base=4)),
    )
    np.testing.assert_array_equal(TerraEnv._retained_work_context(state), np.zeros(5))
    state.agent.current_agent = 1
    np.testing.assert_allclose(
        TerraEnv._retained_work_context(state), [1, 1, 1, 0, 1], atol=1e-7,
    )
    state.agent.current_agent = 2
    np.testing.assert_allclose(
        TerraEnv._retained_work_context(state), [2 / 63, 3 / 31, 0, 1, 1], atol=1e-7,
    )
    state.retained_work_events = jnp.zeros(4, jnp.int32)
    np.testing.assert_array_equal(TerraEnv._retained_work_context(state), np.zeros(5))


def test_native_context_migration_preserves_policy_adam_and_exposes_history():
    cfg = Config(
        clip_action_maps=True, loaded_max=100,
        local_map_normalization_bounds=(-16, 16), maps_net_normalization_bounds=(-10, 10),
        model_core="mlp", model_size="medium", num_prev_actions=5,
        map_encoder="resnet_spatial_8x8", resnet_stage_channels=(2, 4, 4, 4),
        resnet_blocks_per_stage=(1, 1, 1, 1), time_observation_mode="remaining",
    )
    env = NS(batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64)))
    source_model, source_params = get_model_ready(jax.random.PRNGKey(1), cfg, env)
    target_cfg = Config(cfg, retained_work_context_observation=True)
    target_model, rebuilt = get_model_ready(jax.random.PRNGKey(2), target_cfg, env)
    observations = {
        **{key: jnp.zeros((2, 64, 64), jnp.float32) for key in (
            "traversability_mask", "reachability_mask", "action_map", "target_map",
            "padding_mask", "dumpability_mask", "interaction_mask")},
        **{key: jnp.zeros((2, 12), jnp.float32) for key in (
            "local_map_action_neg", "local_map_action_pos", "local_map_target_neg",
            "local_map_target_pos", "local_map_dumpability", "local_map_obstacles",
            "local_map_border_workspace", "local_map_edge_alignment_error", "local_map_border_diggable")},
        "agent_states": jnp.zeros((2, 4, 9), jnp.float32),
        "agent_active": jnp.tile(jnp.array([1, 0, 0, 0], jnp.int8), (2, 1)),
        "num_agents": jnp.ones(2, jnp.int32), "agent_width": jnp.ones(2, jnp.int32),
        "agent_height": jnp.ones(2, jnp.int32), "remaining_time": jnp.array([.8, .8]),
        "retained_work_context": jnp.array([[0, 0, 0, 0, 0], [.7, .5, 1, 0, 1]]),
    }
    history = jnp.zeros((2, 5), jnp.int32)
    old_input = obs_to_model_input(observations, history, cfg)
    context_input = obs_to_model_input(observations, history, target_cfg)
    zero_input = obs_to_model_input(
        {**observations, "retained_work_context": jnp.zeros((2, 5))}, history, target_cfg,
    )
    assert len(context_input) == len(old_input) + 1
    np.testing.assert_array_equal(context_input[-2], old_input[-1])
    tx = optax.chain(optax.clip_by_global_norm(.5), optax.adam(3e-4, eps=1e-5))
    old = TrainState.create(apply_fn=source_model.apply, params=source_params, tx=tx)
    old = old.apply_gradients(grads=jax.tree.map(lambda p: jnp.full_like(p, .0001), old.params))
    checkpoint = dict(model=old.params, optimizer_state=old.opt_state,
                      train_state_step=old.step, next_update=105000, train_config=cfg)
    migrated = migrate_retained_work_context_checkpoint(checkpoint, rebuilt)
    assert migrated["next_update"] == 105000
    np.testing.assert_array_equal(migrated["train_state_step"], old.step)
    for key in ("model", "optimizer_state"):
        before = dict(jax.tree_util.tree_flatten_with_path(checkpoint[key])[0])
        after = dict(jax.tree_util.tree_flatten_with_path(migrated[key])[0])
        for path, leaf in before.items():
            np.testing.assert_array_equal(after[path], leaf)
    assert jax.tree.structure(migrated["model"]) == jax.tree.structure(rebuilt)
    for name in RETAINED_WORK_PARAMETER_NAMES:
        np.testing.assert_array_equal(migrated["model"]["params"][name], np.zeros((5, 704)))
        for moment in ("mu", "nu"):
            slots = getattr(migrated["optimizer_state"][1][0], moment)
            np.testing.assert_array_equal(slots["params"][name], np.zeros((5, 704)))
    expected = source_model.apply(old.params, old_input)
    for model_input in (context_input, zero_input):
        for target, source in zip(target_model.apply(migrated["model"], model_input), expected):
            np.testing.assert_array_equal(target, source)

    def loss(model, params, inputs):
        values, logits = model.apply(params, inputs)
        return jnp.mean(jnp.square(values - 1)) + jnp.mean(logits * jnp.arange(8))

    old_grads = jax.jit(jax.grad(lambda p: loss(source_model, p, old_input)))(old.params)
    zero_grads = jax.jit(jax.grad(lambda p: loss(target_model, p, zero_input)))(migrated["model"])
    context_grads = jax.jit(jax.grad(lambda p: loss(target_model, p, context_input)))(migrated["model"])
    for path, value in flatten_dict(old_grads).items():
        np.testing.assert_allclose(flatten_dict(zero_grads)[path], value, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(flatten_dict(context_grads)[path], value, atol=1e-7, rtol=1e-6)
    for name in RETAINED_WORK_PARAMETER_NAMES:
        np.testing.assert_array_equal(zero_grads["params"][name], np.zeros((5, 704)))
        assert np.any(np.asarray(context_grads["params"][name]) != 0)
    new = TrainState.create(apply_fn=target_model.apply, params=migrated["model"], tx=tx).replace(
        opt_state=migrated["optimizer_state"], step=migrated["train_state_step"])
    old_updated = old.apply_gradients(grads=old_grads)
    zero_updated = new.apply_gradients(grads=zero_grads)
    for path, leaf in flatten_dict(old_updated.params).items():
        np.testing.assert_allclose(flatten_dict(zero_updated.params)[path], leaf, atol=1e-7, rtol=1e-6)
    changed = new.apply_gradients(grads=context_grads)
    values, logits = target_model.apply(changed.params, context_input)
    assert np.any(np.asarray(values[0]) != np.asarray(values[1]))
    assert np.any(np.asarray(logits[0]) != np.asarray(logits[1]))
    with pytest.raises(ValueError, match="without context"):
        migrate_retained_work_context_checkpoint(migrated, rebuilt)
    with pytest.raises(ValueError, match="native checkpoint field optimizer_state"):
        migrate_retained_work_context_checkpoint(
            {k: v for k, v in checkpoint.items() if k != "optimizer_state"}, rebuilt,
        )
    with pytest.raises(ValueError, match="requires Terra"):
        obs_to_model_input({k: v for k, v in observations.items() if k != "retained_work_context"}, history, target_cfg)
    with pytest.raises(ValueError, match="width 5"):
        obs_to_model_input({**observations, "retained_work_context": jnp.zeros((2, 4))}, history, target_cfg)
