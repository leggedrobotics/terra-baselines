"""The optional time-aware heads and their one-way native Adam migration."""

import copy

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict


TIME_PARAMETER_NAMES = (
    "remaining_time_actor_embedding",
    "remaining_time_critic_embedding",
)


def validate_time_observation_mode(mode):
    if mode not in ("none", "remaining", "constant"):
        raise ValueError(
            f"Unsupported time_observation_mode={mode!r}; "
            "expected none, remaining, or constant"
        )
    return mode


def _add_zero_time_parameters(tree):
    mutable = tree.unfreeze() if isinstance(tree, FrozenDict) else copy.deepcopy(tree)
    params = mutable["params"]
    if any(name in params for name in TIME_PARAMETER_NAMES):
        raise ValueError("source already contains remaining-time parameters")
    for name in TIME_PARAMETER_NAMES:
        params[name] = jnp.zeros((704,), dtype=jnp.float32)
    return FrozenDict(mutable) if isinstance(tree, FrozenDict) else mutable


def migrate_remaining_time_checkpoint(checkpoint, rebuilt_params, mode):
    """Add only the two zero time leaves; preserve every old parameter/Adam slot.

    This is an explicit migration from a native no-time checkpoint. Ordinary
    resume must instead use the saved time mode and its unchanged parameter tree.
    The caller keeps the checkpoint's native next_update and train_state_step.
    """
    validate_time_observation_mode(mode)
    if mode == "none":
        raise ValueError("time migration requires remaining or constant mode")
    saved_config = checkpoint.get("train_config", {})
    saved_mode = (saved_config.get("time_observation_mode", "none")
                  if isinstance(saved_config, dict)
                  else getattr(saved_config, "time_observation_mode", "none"))
    if saved_mode != "none":
        raise ValueError("time migration requires a source with time mode none")
    for key in ("model", "optimizer_state", "train_state_step", "next_update"):
        if key not in checkpoint:
            raise ValueError(f"time migration requires native checkpoint field {key}")

    model = _add_zero_time_parameters(checkpoint["model"])
    grown_shapes = {k: jnp.shape(v) for k, v in flatten_dict(model).items()}
    target_shapes = {k: jnp.shape(v) for k, v in flatten_dict(rebuilt_params).items()}
    if grown_shapes != target_shapes:
        raise ValueError("time migration may only add the two remaining-time embeddings")

    adam_count = 0

    def grow_adam(state):
        nonlocal adam_count
        if not isinstance(state, optax.ScaleByAdamState):
            return state
        adam_count += 1
        return state._replace(
            mu=_add_zero_time_parameters(state.mu),
            nu=_add_zero_time_parameters(state.nu),
        )

    optimizer = jax.tree.map(
        grow_adam, checkpoint["optimizer_state"],
        is_leaf=lambda state: isinstance(state, optax.ScaleByAdamState),
    )
    if adam_count != 1:
        raise ValueError(f"time migration requires exactly one Adam state, got {adam_count}")
    migrated_config = copy.copy(saved_config)
    if isinstance(migrated_config, dict):
        migrated_config["time_observation_mode"] = mode
    else:
        setattr(migrated_config, "time_observation_mode", mode)
    return {
        **checkpoint,
        "model": model,
        "optimizer_state": optimizer,
        "train_config": migrated_config,
        "remaining_time_migration": {
            "source_mode": "none", "target_mode": mode,
            "origin_update": int(checkpoint["next_update"]),
            "new_parameters": list(TIME_PARAMETER_NAMES),
        },
    }
