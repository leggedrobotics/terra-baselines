"""Function-preserving time migration, including native Adam continuation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
from terra.config import BatchConfig, MapsDimsConfig

from utils.models import get_model_ready
from utils.remaining_time import TIME_PARAMETER_NAMES, migrate_remaining_time_checkpoint
from utils.utils_ppo import obs_to_model_input


class Config(dict):
    __getattr__ = dict.__getitem__


def test_time_migration_preserves_outputs_slots_and_constant_input_updates():
    # Same 704-wide fused representation as the broad policy, with a small
    # spatial trunk for the CPU contract check; the native parent gets a GPU smoke.
    cfg = Config(
        clip_action_maps=True, loaded_max=100,
        local_map_normalization_bounds=(-16, 16), maps_net_normalization_bounds=(-10,10),
        model_core="mlp", model_size="medium", num_prev_actions=5,
        map_encoder="resnet_spatial_8x8", resnet_stage_channels=(2,4,4,4),
        resnet_blocks_per_stage=(1,1,1,1), time_observation_mode="none",
    )
    env = SimpleNamespace(batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64)))
    source_model, source_params = get_model_ready(jax.random.PRNGKey(1),cfg,env)
    target_cfg = Config(cfg,time_observation_mode="constant")
    target_model, rebuilt = get_model_ready(jax.random.PRNGKey(2),target_cfg,env)
    observations = {
        **{key:jnp.zeros((2,64,64),jnp.float32) for key in (
            "traversability_mask","reachability_mask","action_map","target_map",
            "padding_mask","dumpability_mask","interaction_mask")},
        **{key:jnp.zeros((2,12),jnp.float32) for key in (
            "local_map_action_neg","local_map_action_pos","local_map_target_neg",
            "local_map_target_pos","local_map_dumpability","local_map_obstacles",
            "local_map_border_workspace","local_map_edge_alignment_error","local_map_border_diggable")},
        "agent_states":jnp.zeros((2,4,9),jnp.float32),
        "agent_active":jnp.tile(jnp.array([1,0,0,0],jnp.int8),(2,1)),
        "num_agents":jnp.ones(2,jnp.int32), "agent_width":jnp.ones(2,jnp.int32),
        "agent_height":jnp.ones(2,jnp.int32), "remaining_time":jnp.array([.9,.1]),
    }
    previous_actions = jnp.zeros((2,5),jnp.int32)
    with pytest.raises(ValueError, match="unmasked"):
        obs_to_model_input(observations, previous_actions,
                           Config(target_cfg, action_logit_masking=True))
    old_input = obs_to_model_input(observations,previous_actions,cfg)
    constant_input = obs_to_model_input(observations,previous_actions,target_cfg)
    timed_input = obs_to_model_input(observations,previous_actions,Config(cfg,time_observation_mode="remaining"))
    tx = optax.chain(optax.clip_by_global_norm(.5),optax.adam(3e-4,eps=1e-5))
    old = TrainState.create(apply_fn=source_model.apply,params=source_params,tx=tx)
    old = old.apply_gradients(grads=jax.tree.map(lambda p:jnp.full_like(p,.0001),old.params))
    checkpoint = dict(model=old.params,optimizer_state=old.opt_state,
                      train_state_step=old.step,next_update=5000,train_config=cfg)
    migrated = migrate_remaining_time_checkpoint(checkpoint,rebuilt,"constant")
    assert migrated['next_update']==checkpoint['next_update']
    np.testing.assert_array_equal(migrated['train_state_step'],old.step)
    for tree_key in ('model','optimizer_state'):
        old_leaves = dict(jax.tree_util.tree_flatten_with_path(checkpoint[tree_key])[0])
        new_leaves = dict(jax.tree_util.tree_flatten_with_path(migrated[tree_key])[0])
        for path,leaf in old_leaves.items():
            np.testing.assert_array_equal(new_leaves[path],leaf)
    assert jax.tree.structure(migrated['model'])==jax.tree.structure(rebuilt)
    for name in TIME_PARAMETER_NAMES:
        np.testing.assert_array_equal(migrated['model']['params'][name],jnp.zeros(704))
        np.testing.assert_array_equal(migrated['optimizer_state'][1][0].mu['params'][name],jnp.zeros(704))
        np.testing.assert_array_equal(migrated['optimizer_state'][1][0].nu['params'][name],jnp.zeros(704))
    expected = source_model.apply(old.params,old_input)
    for target_input in (constant_input,timed_input):
        actual = target_model.apply(migrated['model'],target_input)
        for target,source in zip(actual,expected):
            np.testing.assert_array_equal(target,source)

    def loss(model,params,inputs):
        values,logits = model.apply(params,inputs)
        return jnp.mean(jnp.square(values-1)) + jnp.mean(logits*jnp.arange(8))
    old_grads = jax.jit(jax.grad(lambda p:loss(source_model,p,old_input)))(old.params)
    constant_grads = jax.jit(jax.grad(lambda p:loss(target_model,p,constant_input)))(migrated['model'])
    time_grads = jax.jit(jax.grad(lambda p:loss(target_model,p,timed_input)))(migrated['model'])
    for path,value in flatten_dict(old_grads).items():
        np.testing.assert_allclose(flatten_dict(constant_grads)[path],value,atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(flatten_dict(time_grads)[path],value,atol=1e-7,rtol=1e-6)
    for name in TIME_PARAMETER_NAMES:
        np.testing.assert_array_equal(constant_grads['params'][name],jnp.zeros(704))
        assert np.any(np.asarray(time_grads['params'][name])!=0)
    new = TrainState.create(apply_fn=target_model.apply,params=migrated['model'],tx=tx).replace(
        opt_state=migrated['optimizer_state'],step=migrated['train_state_step'])
    old_updated = old.apply_gradients(grads=old_grads)
    new_updated = new.apply_gradients(grads=constant_grads)
    for path,leaf in flatten_dict(old_updated.params).items():
        np.testing.assert_allclose(flatten_dict(new_updated.params)[path],leaf,atol=1e-7,rtol=1e-6)
    for path,leaf in jax.tree_util.tree_flatten_with_path(old_updated.opt_state)[0]:
        np.testing.assert_allclose(dict(jax.tree_util.tree_flatten_with_path(new_updated.opt_state)[0])[path],leaf,
                                   atol=1e-7,rtol=1e-6)
    # After one real-time gradient update, both heads distinguish two otherwise
    # identical states with different remaining budgets.
    timed_updated = new.apply_gradients(grads=time_grads)
    values,logits = target_model.apply(timed_updated.params,timed_input)
    assert np.any(np.asarray(values[0]) != np.asarray(values[1]))
    assert np.any(np.asarray(logits[0]) != np.asarray(logits[1]))
    with pytest.raises(ValueError,match="source with time mode none"):
        migrate_remaining_time_checkpoint(migrated,rebuilt,"remaining")
    with pytest.raises(ValueError,match="native checkpoint field optimizer_state"):
        migrate_remaining_time_checkpoint({k:v for k,v in checkpoint.items() if k!='optimizer_state'},rebuilt,"remaining")
