"""Opt-in evidence of the actual state immediately before a first PPO rollout."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import jax
import numpy as np


def _host_leaf(leaf):
    """Expose key data for typed PRNG keys without losing their logical dtype."""
    dtype = getattr(leaf, "dtype", None)
    is_key = dtype is not None and jax.dtypes.issubdtype(
        dtype, jax.dtypes.prng_key
    )
    value = jax.random.key_data(leaf) if is_key else leaf
    array = np.asarray(jax.device_get(value))
    if array.dtype.hasobject or array.dtype.kind in {"U", "S"}:
        raise TypeError(f"Initialization receipt requires numeric leaves: {array.dtype}")
    return array, str(dtype if is_key else array.dtype), list(np.shape(leaf))


def tree_fingerprint(tree):
    """Hash paths, tree structure, logical dtypes/shapes and all numeric bytes.

    Fingerprints are intended for comparisons within a pinned JAX runtime. No
    array values are retained in the receipt. Device reads happen only here.
    """
    leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    digest = hashlib.sha256()
    structure_bytes = str(structure).encode("utf-8")
    digest.update(len(structure_bytes).to_bytes(8, "little"))
    digest.update(structure_bytes)
    nonfinite_count = 0
    byte_count = 0
    all_zero = True
    for path, leaf in leaves:
        array, dtype, logical_shape = _host_leaf(leaf)
        metadata = json.dumps(
            {
                "path": jax.tree_util.keystr(path),
                "dtype": dtype,
                "shape": logical_shape,
                "storage_dtype": str(array.dtype),
                "storage_shape": list(array.shape),
            },
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        raw = np.ascontiguousarray(array).tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        byte_count += len(raw)
        nonfinite_count += int(array.size - np.count_nonzero(np.isfinite(array)))
        all_zero = all_zero and bool(np.all(array == 0))
    return {
        "sha256": digest.hexdigest(),
        "leaf_count": len(leaves),
        "byte_count": byte_count,
        "nonfinite_count": nonfinite_count,
        "all_finite": nonfinite_count == 0,
        "all_zero": all_zero,
    }


def write_initialization_receipt(
    path, *, config, checkpoint_mode, checkpoint_path, optimizer_restored,
    next_update, train_state, timestep, rollout_rng, reset_rng, teacher_params,
    initial_history,
):
    """Write once after reset/restoration and before model-state replication.

    ``rollout_rng`` contains the actual per-device keys handed to PPO.
    ``fresh_optimizer`` describes restoration, with zero-step/moment evidence
    recorded separately. Nonfinite reset sentinels are reported, not rejected.
    """
    counters = []
    for key_path, leaf in jax.tree_util.tree_flatten_with_path(train_state.opt_state)[0]:
        if key_path and getattr(key_path[-1], "name", None) == "count":
            counters.append({
                "path": jax.tree_util.keystr(key_path),
                "value": int(np.asarray(jax.device_get(leaf)).reshape(())),
            })
    env_steps = np.asarray(jax.device_get(timestep.state.env_steps))
    payload = {
        "schema": "terra_initialization_v1",
        "phase": "after_initial_reset_before_first_rollout",
        "jax_version": jax.__version__,
        "run_name": config.name,
        "seed": int(config.seed),
        "num_devices": int(config.num_devices),
        "num_envs_per_device": int(config.num_envs_per_device),
        "checkpoint_mode": checkpoint_mode or "scratch",
        "checkpoint_path": checkpoint_path,
        "teacher_checkpoint": config.teacher_checkpoint,
        "optimizer_restored": bool(optimizer_restored),
        "fresh_optimizer": not optimizer_restored,
        "next_update": int(next_update),
        "train_state_step": int(np.asarray(jax.device_get(train_state.step)).reshape(())),
        "optimizer_counters": counters,
        "optimizer_counters_all_zero": all(item["value"] == 0 for item in counters),
        "initial_env_steps_min": int(env_steps.min()),
        "initial_env_steps_max": int(env_steps.max()),
        "initial_env_steps_all_zero": bool(np.all(env_steps == 0)),
        "model": tree_fingerprint(train_state.params),
        "optimizer": tree_fingerprint(train_state.opt_state),
        "timestep": tree_fingerprint(timestep),
        "rollout_rng": tree_fingerprint(rollout_rng),
        "reset_rng": tree_fingerprint(reset_rng),
        "initial_history": tree_fingerprint(initial_history),
        "teacher_model": (
            tree_fingerprint(teacher_params) if teacher_params is not None else None
        ),
    }
    if getattr(config, "trench_teacher_checkpoint", None) is not None:
        payload.update({field: getattr(config, field) for field in (
            "trench_teacher_checkpoint", "teacher_checkpoint_sha256",
            "trench_teacher_checkpoint_sha256", "task_teacher_family_ids",
        )})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        # Publish complete JSON atomically; unlike replace(), link() refuses to
        # overwrite a prior receipt even if another process races this writer.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return payload
