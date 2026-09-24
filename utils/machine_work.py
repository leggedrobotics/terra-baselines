"""Native checkpoint migration adding the machine-work observation.

The observation is one more continuous agent-state feature (index 9): each
machine's executed-plan time over the job time. It enters the first layer of
the agent-state continuous MLP, so the migration appends one zero input row to
that kernel (the policy and value are unchanged) and zero Adam slots.
"""

import copy

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict

KERNEL_PATH = ("params", "agent_state_net", "mlp_continuous", "layers_0", "kernel")


def _add_zero_input_row(tree):
    mutable = tree.unfreeze() if isinstance(tree, FrozenDict) else copy.deepcopy(tree)
    node = mutable
    for key in KERNEL_PATH[:-1]:
        node = node[key]
    kernel = jnp.asarray(node[KERNEL_PATH[-1]])
    node[KERNEL_PATH[-1]] = jnp.concatenate(
        (kernel, jnp.zeros((1, kernel.shape[1]), dtype=kernel.dtype)), axis=0
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
    model = _add_zero_input_row(checkpoint["model"])
    grown = {k: jnp.shape(v) for k, v in flatten_dict(model).items()}
    target = {k: jnp.shape(v) for k, v in flatten_dict(rebuilt_params).items()}
    if grown != target:
        raise ValueError("machine-work migration may only add one agent-state input row")

    adam_count = 0

    def grow_adam(state):
        nonlocal adam_count
        if not isinstance(state, optax.ScaleByAdamState):
            return state
        adam_count += 1
        return state._replace(
            mu=_add_zero_input_row(state.mu), nu=_add_zero_input_row(state.nu)
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
            "feature": "agent_states[..., 9] = executed-plan seconds / job seconds",
        },
    }
