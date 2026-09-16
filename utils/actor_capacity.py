"""One-way native Adam migration for the parallel residual actor head."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict


ACTOR_CAPACITY_SUBTREE = "actor_residual_head"


def without_actor_capacity(params):
    """Temporary shape target when composing time then actor migrations."""
    mutable = params.unfreeze() if isinstance(params, FrozenDict) else copy.deepcopy(params)
    mutable['params'].pop(ACTOR_CAPACITY_SUBTREE, None)
    return FrozenDict(mutable) if isinstance(params, FrozenDict) else mutable


def migrate_actor_capacity_checkpoint(checkpoint, rebuilt_params):
    """Add exactly one 704→512→512→8 head, retaining all native optimizer state.

    Hidden weights come from the freshly rebuilt target; its final projection
    must be zero so the trained actor is preserved. New Adam moments are zero,
    but the Adam count is deliberately retained along with the old moments.
    """
    saved_config = checkpoint.get('train_config', {})
    enabled = (saved_config.get('actor_residual_head', False)
               if isinstance(saved_config, dict)
               else getattr(saved_config, 'actor_residual_head', False))
    if enabled:
        raise ValueError('actor capacity migration requires a source without the residual head')
    for key in ('model', 'optimizer_state', 'train_state_step', 'next_update'):
        if key not in checkpoint:
            raise ValueError(f'actor capacity migration requires native checkpoint field {key}')
    source = checkpoint['model']
    if ACTOR_CAPACITY_SUBTREE in source['params']:
        raise ValueError('source already contains actor residual parameters')
    if ACTOR_CAPACITY_SUBTREE not in rebuilt_params['params']:
        raise ValueError('rebuilt model has no actor residual head')
    head = rebuilt_params['params'][ACTOR_CAPACITY_SUBTREE]
    shapes = {path: jnp.shape(value) for path, value in flatten_dict(head).items()}
    expected = {
        ('Dense_0', 'kernel'): (704, 512), ('Dense_0', 'bias'): (512,),
        ('Dense_1', 'kernel'): (512, 512), ('Dense_1', 'bias'): (512,),
        ('Dense_2', 'kernel'): (512, 8), ('Dense_2', 'bias'): (8,),
    }
    if shapes != expected:
        raise ValueError('actor capacity migration expects the named 704→512→512→8 head')
    if any(np.any(np.asarray(value) != 0) for value in head['Dense_2'].values()):
        raise ValueError('actor residual output projection must be exactly zero')
    source_shapes = {p: jnp.shape(v) for p, v in flatten_dict(source).items()}
    remainder_shapes = {p: jnp.shape(v) for p, v in
                        flatten_dict(without_actor_capacity(rebuilt_params)).items()}
    if source_shapes != remainder_shapes:
        raise ValueError('actor capacity migration may only add the named residual head')

    def add_head(tree, *, moments):
        mutable = tree.unfreeze() if isinstance(tree, FrozenDict) else copy.deepcopy(tree)
        if ACTOR_CAPACITY_SUBTREE in mutable['params']:
            raise ValueError('optimizer already contains actor residual parameters')
        mutable['params'][ACTOR_CAPACITY_SUBTREE] = (
            jax.tree.map(jnp.zeros_like, head) if moments else copy.deepcopy(head))
        return FrozenDict(mutable) if isinstance(tree, FrozenDict) else mutable

    count = 0

    def grow_adam(state):
        nonlocal count
        if not isinstance(state, optax.ScaleByAdamState):
            return state
        count += 1
        return state._replace(mu=add_head(state.mu, moments=True),
                              nu=add_head(state.nu, moments=True))

    optimizer = jax.tree.map(grow_adam, checkpoint['optimizer_state'],
                            is_leaf=lambda state: isinstance(state, optax.ScaleByAdamState))
    if count != 1:
        raise ValueError(f'actor capacity migration requires exactly one Adam state, got {count}')
    config = copy.copy(saved_config)
    if isinstance(config, dict):
        config['actor_residual_head'] = True
    else:
        setattr(config, 'actor_residual_head', True)
    return {
        **checkpoint, 'model': add_head(source, moments=False),
        'optimizer_state': optimizer, 'train_config': config,
        'actor_capacity_migration': {
            'origin_update': int(checkpoint['next_update']),
            'new_subtree': ACTOR_CAPACITY_SUBTREE,
            'hidden_dims': [512, 512],
            'added_parameters': sum(int(v.size) for v in jax.tree.leaves(head)),
        },
    }
