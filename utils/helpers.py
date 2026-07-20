import pickle
from pathlib import Path

import jax.numpy as jnp


def load_pkl_object(filename: str):
    """Helper to reload pickle objects."""
    import pickle

    with open(filename, "rb") as input:
        obj = pickle.load(input)
    print(f"Loaded data from {filename}.")
    return obj


def save_pkl_object(obj, filename):
    """Helper to store pickle objects."""
    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print(f"Stored data at {filename}.")


def replicate_checkpoint_env_config(env_config, n_envs: int):
    """Batch a scalar checkpoint EnvConfig without losing agent-type vectors.

    Current checkpoints store ordinary leaves as scalars and ``agent_types`` /
    ``action_types`` as vectors. Older checkpoints may still carry leading
    device/environment axes, so those axes are peeled before replication.
    """
    if n_envs <= 0:
        raise ValueError(f"n_envs must be positive, got {n_envs}")

    def _replicate(node, field_name=None):
        if node is None:
            return None
        if isinstance(node, tuple) and hasattr(node, "_fields"):
            return type(node)(
                *(
                    _replicate(getattr(node, child), child)
                    for child in node._fields
                )
            )

        if field_name in {"agent_types", "action_types"}:
            if isinstance(node, (tuple, list)):
                members = []
                for member in node:
                    member_array = jnp.asarray(member)
                    while member_array.ndim > 0:
                        member_array = member_array[0]
                    members.append(member_array)
                array = jnp.stack(members)
            else:
                array = jnp.asarray(node)
            while array.ndim > 1:
                array = array[0]
            if array.ndim == 0:
                array = array.reshape((1,))
            return jnp.broadcast_to(array, (n_envs,) + array.shape)

        array = jnp.asarray(node)
        while array.ndim > 0:
            array = array[0]
        return jnp.broadcast_to(array, (n_envs,))

    return _replicate(env_config)


def checkpoint_batch_config(train_config, action_type):
    """Rebuild the map curriculum used to create a training checkpoint."""
    from terra.config import BatchConfig, CurriculumGlobalConfig

    levels = getattr(train_config, "curriculum_levels_override", None)
    if not levels:
        return BatchConfig(action_type=action_type)

    increase_threshold = getattr(
        train_config,
        "curriculum_increase_level_threshold",
        None,
    )
    decrease_threshold = getattr(
        train_config,
        "curriculum_decrease_level_threshold",
        None,
    )
    last_level_type = getattr(train_config, "curriculum_last_level_type", None)
    checkpoint_levels = levels
    checkpoint_last_level_type = last_level_type

    class CheckpointCurriculumGlobalConfig(CurriculumGlobalConfig):
        levels = checkpoint_levels
        last_level_type = (
            checkpoint_last_level_type
            if checkpoint_last_level_type is not None
            else CurriculumGlobalConfig.last_level_type
        )

    curriculum = CheckpointCurriculumGlobalConfig(
        increase_level_threshold=(
            increase_threshold
            if increase_threshold is not None
            else CurriculumGlobalConfig.increase_level_threshold
        ),
        decrease_level_threshold=(
            decrease_threshold
            if decrease_threshold is not None
            else CurriculumGlobalConfig.decrease_level_threshold
        ),
    )

    return BatchConfig(
        action_type=action_type,
        curriculum_global=curriculum,
    )
