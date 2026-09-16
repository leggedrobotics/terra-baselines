"""Native growth keeps the policy and Adam history, then learns new capacity."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from terra.config import BatchConfig, MapsDimsConfig

from utils.actor_capacity import (
    migrate_actor_capacity_checkpoint, without_actor_capacity,
)
from utils.models import ActorResidualHead, get_model_ready
from utils.remaining_time import migrate_remaining_time_checkpoint


class Config(dict):
    __getattr__ = dict.__getitem__


def test_native_time_and_actor_growth_preserves_parent():
    cfg = Config(
        clip_action_maps=True, loaded_max=100,
        local_map_normalization_bounds=(-16, 16), maps_net_normalization_bounds=(-10, 10),
        model_core='mlp', model_size='medium', num_prev_actions=5,
        map_encoder='resnet_spatial_8x8', resnet_stage_channels=(2, 4, 4, 4),
        resnet_blocks_per_stage=(1, 1, 1, 1), time_observation_mode='none',
        actor_residual_head=False,
    )
    env = SimpleNamespace(batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64)))
    source_model, params = get_model_ready(jax.random.PRNGKey(1), cfg, env)
    target_model, rebuilt = get_model_ready(jax.random.PRNGKey(2),
        Config(cfg, time_observation_mode='remaining', actor_residual_head=True), env)
    observations = [
        jnp.zeros((2, 4, 9)),
        jnp.tile(jnp.array([1, 0, 0, 0]), (2, 1)), jnp.ones((2,), jnp.int32),
        *[jnp.zeros((2, 12)) for _ in range(9)],
        *[jnp.zeros((2, 64, 64)) for _ in range(4)],
        jnp.ones((2,), jnp.int32), jnp.ones((2,), jnp.int32),
        *[jnp.zeros((2, 64, 64)) for _ in range(3)],
        jnp.zeros((2, 5), jnp.int32),
    ]
    tx = optax.chain(optax.clip_by_global_norm(.5), optax.adam(3e-4, eps=1e-5))
    optimizer = tx.init(params)
    # Nonzero moments and a mature count: do not accidentally test only fresh Adam.
    optimizer = jax.tree.map(
        lambda state: state._replace(
            count=jnp.int32(320000),
            mu=jax.tree.map(lambda x: jnp.full_like(x, .0001), state.mu),
            nu=jax.tree.map(lambda x: jnp.full_like(x, .001), state.nu),
        ) if isinstance(state, optax.ScaleByAdamState) else state,
        optimizer, is_leaf=lambda state: isinstance(state, optax.ScaleByAdamState))
    checkpoint = dict(model=params, optimizer_state=optimizer,
                      train_state_step=320000, next_update=5000, train_config=cfg)
    timed = migrate_remaining_time_checkpoint(checkpoint, without_actor_capacity(rebuilt), 'remaining')
    grown = migrate_actor_capacity_checkpoint(timed, rebuilt)
    assert grown['next_update'] == 5000
    assert grown['train_state_step'] == 320000
    assert grown['actor_capacity_migration']['added_parameters'] == 627720
    assert grown['train_config']['actor_residual_head']
    assert grown['train_config']['time_observation_mode'] == 'remaining'
    for name in ('model', 'optimizer_state'):
        new_leaves = dict(jax.tree_util.tree_flatten_with_path(grown[name])[0])
        for path, value in jax.tree_util.tree_flatten_with_path(checkpoint[name])[0]:
            np.testing.assert_array_equal(new_leaves[path], value)
    for moments in (grown['optimizer_state'][1][0].mu, grown['optimizer_state'][1][0].nu):
        for value in jax.tree.leaves(moments['params']['actor_residual_head']):
            np.testing.assert_array_equal(value, jnp.zeros_like(value))
    expected = source_model.apply(params, observations)
    actual = target_model.apply(grown['model'], observations + [jnp.array([[.9], [.1]])])
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match='source without the residual head'):
        migrate_actor_capacity_checkpoint(grown, rebuilt)
    with pytest.raises(ValueError, match='may only add the named residual head'):
        migrate_actor_capacity_checkpoint(checkpoint, rebuilt)


def test_new_head_learns_and_unblocks_hidden_gradients():
    model = ActorResidualHead(num_actions=8)
    inputs = jax.random.normal(jax.random.PRNGKey(7), (4, 704))
    params = model.init(jax.random.PRNGKey(8), inputs)
    np.testing.assert_array_equal(model.apply(params, inputs), jnp.zeros((4, 8)))
    targets = jnp.array([0, 1, 2, 3])

    def loss(parameters):
        logits = model.apply(parameters, inputs)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(4), targets])

    gradient = jax.jit(jax.grad(loss))
    first = gradient(params)
    assert np.linalg.norm(np.asarray(first['params']['Dense_2']['kernel'])) > 0
    np.testing.assert_array_equal(first['params']['Dense_0']['kernel'],
                                  jnp.zeros_like(first['params']['Dense_0']['kernel']))
    tx = optax.adam(3e-4)
    updates, _ = tx.update(first, tx.init(params), params)
    updated = optax.apply_updates(params, updates)
    assert float(loss(updated)) < float(loss(params))
    second = gradient(updated)
    assert np.linalg.norm(np.asarray(second['params']['Dense_0']['kernel'])) > 0
    assert np.linalg.norm(np.asarray(second['params']['Dense_1']['kernel'])) > 0
