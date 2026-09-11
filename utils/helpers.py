import copy
import os
import pickle
import uuid
from pathlib import Path

import jax.numpy as jnp
import numpy as np


FOUNDATION_BEHAVIOR_DEFAULTS = {
    "executable_dig_observation": False,
    "lateral_dig_cost": 0.0,
    "base_travel_cost": 0.0,
    "base_turn_cost": 0.0,
}


def _config_field(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def validate_executable_dig_observation(config, *, env=None):
    """Executable volume reuses the existing width-12 admissible-dig input."""
    executable = bool(_config_field(config, "executable_dig_observation", False))
    if executable and not bool(
        _config_field(config, "admissible_dig_observation", False)
    ):
        raise ValueError(
            "executable_dig_observation requires admissible_dig_observation"
        )
    if env is not None and executable != bool(
        getattr(env, "executable_dig_observation", False)
    ):
        raise ValueError(
            "executable_dig_observation mismatch between train_config and "
            "TerraEnvBatch's static observation selector"
        )


def _foundation_scalar(value, name):
    array = np.asarray(value)
    if array.size == 0 or not np.all(array == array.flat[0]):
        raise ValueError(f"{name} must be uniform across checkpoint environments")
    scalar = array.flat[0]
    if name == "executable_dig_observation":
        if scalar not in (False, True):
            raise ValueError(f"{name} must be a boolean")
        return bool(scalar)
    scalar = float(scalar)
    if not np.isfinite(scalar) or scalar < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return scalar


def _foundation_values_match(left, right):
    # Saved EnvConfig leaves may be float32 while train_config keeps Python floats.
    return bool(np.isclose(left, right, rtol=1e-6, atol=0.0))


def checkpoint_foundation_behavior(checkpoint):
    """Resolve the trained behavior, rejecting conflicting saved metadata.

    Old dataclass pickles inherit newly added class defaults on unpickling.
    Only instance fields count as explicitly saved train_config metadata; a
    missing field can therefore be recovered from the saved environment.
    """
    config = checkpoint.get("train_config")
    if config is None:
        raise ValueError("checkpoint has no train_config")
    saved_config = config if isinstance(config, dict) else vars(config)
    env_config = checkpoint.get("env_config")
    missing = object()
    settings = {}
    for name, default in FOUNDATION_BEHAVIOR_DEFAULTS.items():
        trained = saved_config.get(name, missing)
        actual = _config_field(env_config, name, missing)
        if trained is not missing:
            trained = _foundation_scalar(trained, name)
        if actual is not missing:
            actual = _foundation_scalar(actual, name)
        if (
            trained is not missing
            and actual is not missing
            and not _foundation_values_match(trained, actual)
        ):
            raise ValueError(
                f"checkpoint {name} mismatch: train_config={trained}, "
                f"env_config={actual}"
            )
        settings[name] = (
            trained if trained is not missing else actual
            if actual is not missing else default
        )
    validate_executable_dig_observation(
        {**settings, "admissible_dig_observation": _config_field(
            config, "admissible_dig_observation", False
        )}
    )
    return settings


def checkpoint_evaluation_config(checkpoint):
    """Copy the recorded config, filling only the four behavior settings."""
    settings = checkpoint_foundation_behavior(checkpoint)
    config = copy.deepcopy(checkpoint["train_config"])
    if isinstance(config, dict):
        config.update(settings)
    else:
        for name, value in settings.items():
            setattr(config, name, value)
    return config


def overlay_foundation_behavior(env_config, settings):
    """Apply resolved settings without losing existing environment batch axes."""
    updates = {}
    for name, default in FOUNDATION_BEHAVIOR_DEFAULTS.items():
        value = _foundation_scalar(settings.get(name, default), name)
        if not hasattr(env_config, name):
            if value != default:
                raise ValueError(f"this Terra runtime has no EnvConfig.{name}")
            continue
        updates[name] = jnp.full(
            jnp.shape(getattr(env_config, name)), value,
            dtype=jnp.bool_ if isinstance(default, bool) else jnp.float32,
        )
    return env_config._replace(**updates) if updates else env_config


def validate_foundation_behavior_env(config, env_config, *, env=None):
    """Verify semantics before a rollout; observation values cannot prove them."""
    validate_executable_dig_observation(config, env=env)
    for name, default in FOUNDATION_BEHAVIOR_DEFAULTS.items():
        expected = _foundation_scalar(_config_field(config, name, default), name)
        actual = _foundation_scalar(_config_field(env_config, name, default), name)
        if not _foundation_values_match(expected, actual):
            raise ValueError(
                f"evaluation {name} mismatch: train_config={expected}, "
                f"env_config={actual}"
            )


def load_pkl_object(filename: str):
    """Helper to reload pickle objects."""
    import pickle

    with open(filename, "rb") as input:
        obj = pickle.load(input)
    print(f"Loaded data from {filename}.")
    return obj


def save_pkl_object(obj, filename):
    """Store a pickle without exposing a partially written checkpoint."""
    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    temporary = output_file.with_name(
        f".{output_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
        with os.fdopen(descriptor, "wb") as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_file)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Stored data at {filename}.")


def register_checkpoint_config_classes():
    """Alias the training config dataclasses into ``__main__`` for unpickling.

    ``train.py`` / ``train_mixed.py`` run as scripts, so their ``TrainConfig`` /
    ``MixedAgentTrainConfig`` dataclasses are pickled inside checkpoints under
    ``__main__.<name>``. When a checkpoint is later loaded from a different
    entry point (e.g. ``grow_checkpoint.py``, or a ``train_mixed`` run loading a
    ``train.py``-saved teacher), the current ``__main__`` does not define those
    names and unpickling fails. Alias both classes into the running ``__main__``
    so any checkpoint unpickles regardless of which script is executing.

    Both modules are always importable in this repository, so the imports are
    plain. They are performed lazily inside the function to avoid a circular
    import when ``utils.helpers`` is first loaded (``train``/``train_mixed``
    import ``utils.helpers`` at module load time).
    """
    import sys

    from train import TrainConfig
    from train_mixed import MixedAgentTrainConfig

    # Only fill in MISSING names. When the running __main__ already defines a
    # config class (train_mixed.py itself, or a derived per-run trainer),
    # overwriting it makes pickle refuse to SAVE new checkpoints with
    # "it's not the same object as __main__.MixedAgentTrainConfig".
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "TrainConfig"):
        main_module.TrainConfig = TrainConfig
    if not hasattr(main_module, "MixedAgentTrainConfig"):
        main_module.MixedAgentTrainConfig = MixedAgentTrainConfig


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
