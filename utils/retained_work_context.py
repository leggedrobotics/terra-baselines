"""Native checkpoint migration for observable retained-work costs."""

import copy

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict


RETAINED_WORK_PARAMETER_NAMES = (
    "retained_work_context_actor_embedding",
    "retained_work_context_critic_embedding",
)


def _add_zero_context_parameters(tree):
    mutable = tree.unfreeze() if isinstance(tree, FrozenDict) else copy.deepcopy(tree)
    if any(name in mutable["params"] for name in RETAINED_WORK_PARAMETER_NAMES):
        raise ValueError("source already contains retained-work context parameters")
    for name in RETAINED_WORK_PARAMETER_NAMES:
        mutable["params"][name] = jnp.zeros((5, 704), dtype=jnp.float32)
    return FrozenDict(mutable) if isinstance(tree, FrozenDict) else mutable


def migrate_retained_work_context_checkpoint(checkpoint, rebuilt_params):
    """Add two zero projections and Adam slots while preserving native clocks.

    A resumed checkpoint that already has the context uses ordinary native
    resume. This one-way migration cannot alter time input, actor capacity or
    any existing model shape.
    """
    saved_config = checkpoint.get("train_config", {})
    enabled = (
        saved_config.get("retained_work_context_observation", False)
        if isinstance(saved_config, dict)
        else getattr(saved_config, "retained_work_context_observation", False)
    )
    if enabled:
        raise ValueError("retained-work migration requires a source without context")
    for key in ("model", "optimizer_state", "train_state_step", "next_update"):
        if key not in checkpoint:
            raise ValueError(f"retained-work migration requires native checkpoint field {key}")
    model = _add_zero_context_parameters(checkpoint["model"])
    grown_shapes = {k: jnp.shape(v) for k, v in flatten_dict(model).items()}
    target_shapes = {k: jnp.shape(v) for k, v in flatten_dict(rebuilt_params).items()}
    if grown_shapes != target_shapes:
        raise ValueError("retained-work migration may only add the two context embeddings")

    adam_count = 0

    def grow_adam(state):
        nonlocal adam_count
        if not isinstance(state, optax.ScaleByAdamState):
            return state
        adam_count += 1
        return state._replace(
            mu=_add_zero_context_parameters(state.mu),
            nu=_add_zero_context_parameters(state.nu),
        )

    optimizer = jax.tree.map(
        grow_adam, checkpoint["optimizer_state"],
        is_leaf=lambda state: isinstance(state, optax.ScaleByAdamState),
    )
    if adam_count != 1:
        raise ValueError(f"retained-work migration requires exactly one Adam state, got {adam_count}")
    config = copy.copy(saved_config)
    if isinstance(config, dict):
        config["retained_work_context_observation"] = True
    else:
        setattr(config, "retained_work_context_observation", True)
    return {
        **checkpoint,
        "model": model,
        "optimizer_state": optimizer,
        "train_config": config,
        "retained_work_context_migration": {
            "origin_update": int(checkpoint["next_update"]),
            "new_parameters": list(RETAINED_WORK_PARAMETER_NAMES),
            "features": ["previous_x", "previous_y", "heading_sin", "heading_cos", "valid"],
        },
    }
