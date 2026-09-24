"""Native checkpoint migration adding the machine-work observation.

The observation is two more continuous agent-state features (indices 9-10):
each machine's executed-plan time and the team's fair share, over the job
time. They enter the first layer of the agent-state continuous MLP, so the
migration appends zero input rows to that kernel (the policy and value are
unchanged) and zero Adam slots.
"""

import copy

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict

KERNEL_PATH = ("params", "agent_state_net", "mlp_continuous", "layers_0", "kernel")
NEW_FEATURES = 2


def _add_zero_input_rows(tree):
    mutable = tree.unfreeze() if isinstance(tree, FrozenDict) else copy.deepcopy(tree)
    node = mutable
    for key in KERNEL_PATH[:-1]:
        node = node[key]
    kernel = jnp.asarray(node[KERNEL_PATH[-1]])
    node[KERNEL_PATH[-1]] = jnp.concatenate(
        (kernel, jnp.zeros((NEW_FEATURES, kernel.shape[1]), dtype=kernel.dtype)), axis=0
    )
    return FrozenDict(mutable) if isinstance(tree, FrozenDict) else mutable


def migrate_machine_work_observation_checkpoint(checkpoint, rebuilt_params):
    """Grow a checkpoint without the machine-work input to one with it."""
    saved_config = checkpoint.get("train_config", {})
    enabled = (
        saved_config.get("machine_work_observation", False)
        if isinstance(saved_config, dict)
        else getattr(saved_config, "machine_work_observation", False)
    )
    if enabled:
        raise ValueError("machine-work migration requires a source without the observation")
    for key in ("model", "optimizer_state", "train_state_step", "next_update"):
        if key not in checkpoint:
            raise ValueError(f"machine-work migration requires native checkpoint field {key}")
    model = _add_zero_input_rows(checkpoint["model"])
    grown = {k: jnp.shape(v) for k, v in flatten_dict(model).items()}
    target = {k: jnp.shape(v) for k, v in flatten_dict(rebuilt_params).items()}
    if grown != target:
        raise ValueError("machine-work migration may only add the machine-work input rows")

    adam_count = 0

    def grow_adam(state):
        nonlocal adam_count
        if not isinstance(state, optax.ScaleByAdamState):
            return state
        adam_count += 1
        return state._replace(
            mu=_add_zero_input_rows(state.mu), nu=_add_zero_input_rows(state.nu)
        )

    optimizer = jax.tree.map(
        grow_adam, checkpoint["optimizer_state"],
        is_leaf=lambda state: isinstance(state, optax.ScaleByAdamState),
    )
    if adam_count != 1:
        raise ValueError(f"machine-work migration requires exactly one Adam state, got {adam_count}")
    config = copy.copy(saved_config)
    if isinstance(config, dict):
        config["machine_work_observation"] = True
    else:
        setattr(config, "machine_work_observation", True)
    return {
        **checkpoint,
        "model": model,
        "optimizer_state": optimizer,
        "train_config": config,
        "machine_work_migration": {
            "origin_update": int(checkpoint["next_update"]),
            "parameter": "/".join(KERNEL_PATH),
            "features": "agent_states[..., 9:11] = executed-plan seconds, fair share / job seconds",
        },
    }
