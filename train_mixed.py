#!/usr/bin/env python3
"""
Training script for mixed-agent environments using a unified network with agent-type conditioning.

================================================================================
CONFIGURATION SYSTEM
================================================================================

Configurations are defined in: configs/training_configs.yaml

This YAML file contains named presets that specify:
- agent_types: Which agents to use (0=excavator, 1=truck, 2=skidsteer)
- action_types: Movement type per agent (0=tracked, 1=wheeled)
- relocation_progress_mult: Agent-neutral signed relocation-progress multiplier
- maps: Which map datasets to train on
- capacity overrides: truck_capacity, skidsteer_capacity, truck_road_restricted

Available Presets (run `python configs/training_configs.py` for details):
---------------------------------------------------------------------------
  Solo:        solo_excavator, solo_skidsteer
  Two-agent:   excavator_skidsteer, excavator_truck, excavator_truck_roads, dual_excavator
  Three-agent: excavators_truck
  Trench:      trench_excavator
  Wheeled:     wheeled_excavator

Check configs/training_configs.yaml for more details.

================================================================================
QUICK START
================================================================================

# Use a preset configuration
python train_mixed.py --config excavator_truck

# Use a preset with custom name for wandb
python train_mixed.py --config excavator_skidsteer --name "my-experiment"

# Override the agent-neutral relocation progress multiplier
python train_mixed.py --config excavator_truck --relocation_progress_mult 2.0

# List all available presets
python configs/training_configs.py

================================================================================
MANUAL OVERRIDES (without using presets)
================================================================================

# Two agents: excavator + skidsteer (tracked)
python train_mixed.py --agent_types "(0,2)" --action_types "(0,0)"

# Four agents: 2 excavators + 2 skidsteers with mixed movement
python train_mixed.py --agent_types "(0,2,0,2)" --action_types "(0,1,0,1)"

================================================================================
ADDING NEW CONFIGURATIONS
================================================================================

Edit configs/training_configs.yaml to add new presets:

    my_new_config:
      description: My custom training setup
      agent_types: [0, 2, 2]
      action_types: [0, 0, 0]
      relocation_progress_mult: 1.5
      maps:
        - path: foundations_dumpzones_v3
          max_steps: 900

================================================================================
REFERENCE
================================================================================

Agent Types:
  0 = Excavator (digs and dumps)
  1 = Truck (transport, road-restricted optional)
  2 = Skidsteer (transport)

Action Types:
  0 = Tracked movement
  1 = Wheeled movement

Reward Multipliers:
  relocation_progress_mult - Agent-neutral multiplier for signed relocation progress
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from utils.models import MAP_ENCODER_ALIASES, canonical_map_encoder, get_model_ready
from terra.env import TerraEnvBatch
from terra.config import (
    EnvConfig,
    BatchConfig,
    Rewards,
    CurriculumGlobalConfig,
    RewardStage,
    RewardsType,
)
from flax.training.train_state import TrainState
import optax
import wandb
import eval_ppo
from datetime import datetime
from dataclasses import asdict, dataclass
import time
from tqdm import tqdm
from functools import partial
from flax.jax_utils import replicate, unreplicate
from flax import struct
import utils.helpers as helpers
from utils.utils_ppo import select_action_ppo, wrap_action, obs_to_model_input, policy
from utils.episode_aggregates import (
    aggregate_to_payload,
    assert_aggregate_integrity,
    empty_episode_aggregate,
    EpisodeStep,
    new_episode_accumulator,
    reduce_episode_aggregate,
    update_episode_aggregate,
)
from utils.accepted_bank import ARMS as ACCEPTED_BANK_ARMS
from utils.accepted_bank import AcceptedBank, load_accepted_bank
from utils.pooled_sampler import PooledConditionSampler, SamplerSettings
from utils.wandb_human import (
    CONDITION_COLUMNS,
    LOGGING_SCHEMA,
    TRAINING_SCALAR_KEYS,
    condition_rows,
    curriculum_metrics,
    episode_metrics,
    loss_metrics,
)
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Import the base training infrastructure
from train import calculate_gae, ppo_update_networks

jax.config.update("jax_threefry_partitionable", True)


def kickstart_coef_schedule(
    update_index: int, initial_coef: float, anneal_updates: float
) -> float:
    """Cosine-anneal a kickstart coefficient from ``initial_coef`` to 0.

    Returns ``initial_coef`` at update 0, ``0.5*initial_coef`` at the midpoint,
    and exactly 0 at (and after) ``anneal_updates`` (clamped past the window).
    """
    import math

    if anneal_updates <= 0:
        return 0.0
    if update_index >= anneal_updates:
        return 0.0
    fraction = update_index / anneal_updates
    return float(initial_coef * 0.5 * (1.0 + math.cos(math.pi * fraction)))


class Transition(struct.PyTreeNode):
    done: jax.Array
    task_done: jax.Array
    curriculum_level: jax.Array
    active_curriculum_level: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    terminal_reward: jax.Array
    dig_completion_edge: jax.Array
    dig_completion_inner: jax.Array
    dig_completion_total: jax.Array
    dig_completion_min_edge_inner: jax.Array
    dump_completion_action_map: jax.Array
    total_dig_dump_completion: jax.Array
    remaining_edge_dig_tiles: jax.Array
    remaining_inner_dig_tiles: jax.Array
    transition_mass_residual: jax.Array
    target_mutation: jax.Array
    obstacle_mutation: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    prev_actions: jax.Array
    prev_reward: jax.Array


_REQUIRED_FINITE_LOSS_KEYS = (
    "total_loss",
    "value_loss",
    "actor_loss",
    "entropy",
)
_OPTIONAL_FINITE_LOSS_KEYS = (
    "approx_kl",
    "clip_fraction",
    "kickstart/kl",
    "kickstart/value_mse",
    "diagnostics/grad_global_norm",
)
_FINITE_FRACTION_KEYS = (
    "diagnostics/grads_all_finite",
    "diagnostics/params_all_finite",
    "diagnostics/rollout_finite_fraction",
    "diagnostics/rollout_obs_finite_fraction",
    "diagnostics/rollout_value_finite_fraction",
    "diagnostics/rollout_reward_finite_fraction",
    "diagnostics/rollout_log_prob_finite_fraction",
    "diagnostics/model_obs_finite_fraction",
    "diagnostics/raw_advantages_finite_fraction",
    "diagnostics/raw_targets_finite_fraction",
    "diagnostics/advantages_finite_fraction",
    "diagnostics/targets_finite_fraction",
    "diagnostics/student_value_finite_fraction",
    "diagnostics/student_logits_finite_fraction",
    "diagnostics/log_prob_finite_fraction",
    "diagnostics/ratio_finite_fraction",
    "diagnostics/entropy_finite_fraction",
    "diagnostics/teacher_value_finite_fraction",
    "diagnostics/teacher_logits_finite_fraction",
)
_FINITE_CONTEXT_KEYS = (
    "diagnostics/student_value_abs_max",
    "diagnostics/student_logits_abs_max",
    "diagnostics/raw_targets_abs_max",
    "diagnostics/raw_advantages_abs_max",
    "diagnostics/targets_abs_max",
    "diagnostics/advantages_abs_max",
    "diagnostics/log_prob_delta_abs_max",
    "diagnostics/ratio_abs_max",
    "diagnostics/teacher_value_abs_max",
    "diagnostics/teacher_logits_abs_max",
)


def _assert_transition_integrity(integrity: dict) -> None:
    failures = []
    for field in (
        "maximum_mass_residual",
        "target_mutation_count",
        "obstacle_mutation_count",
    ):
        value = int(np.asarray(integrity[field]))
        if value:
            failures.append(f"{field}={value}")
    if failures:
        raise RuntimeError(
            "Terra rollout transition integrity failed: " + ", ".join(failures)
        )


def _nonfinite_count(value) -> int:
    arr = np.asarray(jax.device_get(value))
    if arr.dtype.kind not in {"f", "c"}:
        return 0
    return int(arr.size - np.isfinite(arr).sum())


def _assert_finite_loss_info(loss_info, update_index: int) -> None:
    failures: list[str] = []
    for key in _REQUIRED_FINITE_LOSS_KEYS:
        if key not in loss_info:
            failures.append(f"{key}=missing")
            continue
        count = _nonfinite_count(loss_info[key])
        if count:
            failures.append(f"{key}: {count} non-finite")

    for key in _OPTIONAL_FINITE_LOSS_KEYS:
        if key in loss_info:
            count = _nonfinite_count(loss_info[key])
            if count:
                failures.append(f"{key}: {count} non-finite")

    for key in _FINITE_FRACTION_KEYS:
        if key in loss_info:
            arr = np.asarray(jax.device_get(loss_info[key]), dtype=np.float32)
            if (not np.all(np.isfinite(arr))) or float(np.min(arr)) < 0.999:
                failures.append(f"{key}: min={float(np.nanmin(arr)):.6g}")

    if failures:
        context = []
        for key in _FINITE_CONTEXT_KEYS:
            if key in loss_info:
                arr = np.asarray(jax.device_get(loss_info[key]), dtype=np.float32)
                if arr.size:
                    context.append(f"{key}: max={float(np.nanmax(arr)):.6g}")
        raise FloatingPointError(
            f"Non-finite PPO update detected at update {update_index}: "
            + "; ".join(failures + context)
        )


def _format_pytree_path(path) -> str:
    try:
        return jax.tree_util.keystr(path)
    except Exception:
        return str(path)


def _assert_finite_tree(tree, label: str, max_items: int = 8) -> None:
    failures: list[str] = []
    total = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        arr = np.asarray(jax.device_get(leaf))
        if arr.dtype.kind not in {"f", "c"}:
            continue
        count = int(arr.size - np.isfinite(arr).sum())
        if count:
            total += count
            if len(failures) < max_items:
                failures.append(
                    f"{_format_pytree_path(path)} shape={arr.shape} "
                    f"dtype={arr.dtype} nonfinite={count}"
                )
    if total:
        extra = "" if len(failures) < max_items else " ..."
        raise FloatingPointError(
            f"{label} contains {total} non-finite scalar(s): "
            + "; ".join(failures)
            + extra
        )


def _strip_checkpoint_env_axis(env_config, num_envs_per_device: int):
    """Store/load EnvConfig without a leading vectorized-env axis when present."""
    del num_envs_per_device

    def _strip_agent_leaf(x):
        if isinstance(x, (tuple, list)):
            return jnp.stack([_strip_scalar_leaf(member) for member in x])
        arr = jnp.asarray(x)
        if arr.ndim == 0:
            return arr.reshape((1,))
        # Agent/action vectors are the final axis. Drop only explicit leading
        # device/environment axes and keep an already-scalar vector unchanged.
        while arr.ndim > 1:
            arr = arr[0]
        return arr

    def _strip_scalar_leaf(x):
        arr = jnp.asarray(x)
        while arr.ndim > 0:
            arr = arr[0]
        return arr

    def _strip_node(x, field_name: str | None = None):
        if isinstance(x, tuple) and hasattr(x, "_fields"):
            return type(x)(
                *(_strip_node(getattr(x, child), child) for child in x._fields)
            )
        try:
            jnp.asarray(x)
        except Exception:
            return x
        if field_name in {"agent_types", "action_types"}:
            return _strip_agent_leaf(x)
        return _strip_scalar_leaf(x)

    return _strip_node(env_config)


def _checkpoint_config_value(checkpoint, field_name: str, default):
    saved_config = checkpoint.get("train_config")
    if saved_config is None:
        return default
    if isinstance(saved_config, dict):
        return saved_config.get(field_name, default)
    return getattr(saved_config, field_name, default)


def _teacher_maps_edge_length(checkpoint):
    """Best-effort read of a teacher checkpoint's native map edge length (F15).

    Checkpoints store the reset-time env_config, whose ``maps.edge_length_px`` is
    set by ``TerraEnvBatch.update_env_cfgs`` from the loaded maps (e.g. 64).
    Returns an int, or None when the checkpoint carries no usable env_config so
    the caller can document the assumption instead of silently guessing.
    """
    env_config = checkpoint.get("env_config")
    if env_config is None:
        return None
    maps = getattr(env_config, "maps", None)
    if maps is None and isinstance(env_config, dict):
        maps = env_config.get("maps")
    edge = getattr(maps, "edge_length_px", None)
    if edge is None and isinstance(maps, dict):
        edge = maps.get("edge_length_px")
    if edge is None:
        return None
    try:
        edge = int(np.asarray(edge).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return edge if edge > 0 else None


def _teacher_model_env_from_checkpoint(checkpoint, student_env):
    """Build a minimal env-like object matching a teacher checkpoint's model shape.

    ``get_model_ready`` derives position-embedding width from
    ``env.batch_cfg.maps_dims.maps_edge_length``. For cross-resolution kickstart,
    the frozen teacher must therefore be instantiated with its own checkpoint
    edge length even though the student rollout env is larger.
    """

    teacher_edge = _teacher_maps_edge_length(checkpoint)
    if teacher_edge is None:
        return student_env

    batch_cfg = student_env.batch_cfg._replace(
        maps_dims=student_env.batch_cfg.maps_dims._replace(
            maps_edge_length=int(teacher_edge)
        )
    )
    return SimpleNamespace(batch_cfg=batch_cfg)


def _validate_checkpoint_architecture(checkpoint, config) -> None:
    """Fail before model initialization when checkpoint/model shapes cannot match."""
    defaults = {
        "map_encoder": "atari",
        "model_core": "mlp",
        "model_size": "base",
        "critic_hidden_dims": None,
        # F15: spatial-ResNet stage overrides. None (use the model_size preset)
        # vs an explicit tuple are different trunks, so they must match too.
        "resnet_stage_channels": None,
        "resnet_blocks_per_stage": None,
    }
    tuple_fields = (
        "critic_hidden_dims",
        "resnet_stage_channels",
        "resnet_blocks_per_stage",
    )
    mismatches = []
    for field_name, default in defaults.items():
        saved = _checkpoint_config_value(checkpoint, field_name, default)
        current = getattr(config, field_name, default)
        if field_name == "map_encoder":
            saved = canonical_map_encoder(saved)
            current = canonical_map_encoder(current)
        elif field_name in tuple_fields:
            # None (use the model_size preset) vs an explicit override are
            # different shapes; compare as tuples so (16, 32) and [16, 32] read
            # as equal, while None stays distinct from any tuple.
            saved = tuple(saved) if saved is not None else None
            current = tuple(current) if current is not None else None
        if saved != current:
            mismatches.append(
                f"{field_name}: checkpoint={saved!r}, current={current!r}"
            )
    if mismatches:
        raise ValueError(
            "Checkpoint architecture does not match the requested model: "
            + "; ".join(mismatches)
            + ". Pass matching --map_encoder, --model_core, --model_size, "
            "--critic_hidden_dims, --resnet_stage_channels, and "
            "--resnet_blocks_per_stage values."
        )


def _validate_checkpoint_history_width(checkpoint, config) -> None:
    saved = _checkpoint_config_value(checkpoint, "num_prev_actions", None)
    if saved is not None and int(saved) != int(config.num_prev_actions):
        raise ValueError(
            "Checkpoint action-history width does not match the selected environment: "
            f"checkpoint={int(saved)}, current={int(config.num_prev_actions)}"
        )


def _num_agents_from_env_params(env_params) -> int:
    agent_types = getattr(env_params, "agent_types", None)
    if isinstance(agent_types, (tuple, list)):
        return len(agent_types)
    if hasattr(agent_types, "shape"):
        return 1 if agent_types.ndim == 0 else int(agent_types.shape[-1])
    raise ValueError("environment config has no usable agent_types")


def _validate_resume_update(resume_update: int, num_updates: int) -> None:
    if not 0 <= resume_update < num_updates:
        raise ValueError(
            f"resume_update must be in [0, {num_updates}), got {resume_update}. "
            "Increase --total_timesteps when continuing a completed checkpoint."
        )


def _checkpoint_load_mode(config) -> str | None:
    if config.resume_from is not None and config.warm_start_from is not None:
        raise ValueError("--resume_from and --warm_start_from are mutually exclusive")
    if config.warm_start_from is not None:
        if config.resume_update is not None:
            raise ValueError("--resume_update is incompatible with --warm_start_from")
        return "warm_start"
    if config.resume_from is not None:
        return "resume"
    return None


def resolve_run_name(
    requested_name: str, machine: str, timestamp: str, exact_run_name: bool
) -> str:
    """Keep a resumed treatment name stable when explicitly requested."""
    if exact_run_name:
        return requested_name
    return f"{requested_name}-{machine}-{timestamp}"


def _backfill_terminal_rewards(
    reward_seq: jax.Array,
    terminal_reward_seq: jax.Array,
    done_seq: jax.Array,
    num_agents_per_env: jax.Array,
    max_agents: int = 4,
) -> jax.Array:
    """Share terminal credit with prior same-episode agent turns."""
    terminal_reward_seq = jnp.where(done_seq, terminal_reward_seq, 0.0)
    backfill = jnp.zeros_like(reward_seq)
    for k in range(1, max_agents):
        zeros = jnp.zeros_like(terminal_reward_seq[:k])
        shifted = jnp.concatenate([terminal_reward_seq[k:], zeros], axis=0)

        # A reward at t+k belongs to the same episode as t only when no step in
        # [t, t+k) terminated an episode.
        same_episode = jnp.ones_like(done_seq, dtype=jnp.bool_)
        for offset in range(k):
            done_ahead = jnp.concatenate(
                [done_seq[offset:], jnp.ones_like(done_seq[:offset])],
                axis=0,
            )
            same_episode = jnp.logical_and(same_episode, ~done_ahead)

        use_k = (num_agents_per_env > k).astype(reward_seq.dtype)
        backfill += jnp.where(same_episode, shifted, 0.0) * use_k
    return reward_seq + backfill


def assert_initial_env_steps_zero(timestep):
    """Fail the full-reset training path if any initial horizon is shortened."""

    def _check(env_steps):
        values = np.asarray(env_steps)
        if np.any(values != 0):
            raise RuntimeError(
                "Full-task reset must start with env_steps == 0; "
                f"observed range [{values.min()}, {values.max()}]."
            )

    jax.debug.callback(_check, timestep.state.env_steps)
    return timestep


def _write_episode_aggregate_receipt(
    config,
    payload: dict,
) -> Path:
    """Atomically save the bounded population receipt for one log window."""
    output_dir = Path(config.checkpoint_dir) / "episode_aggregates"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.name}_update_{int(payload['update']):06d}.json"
    if output_path.exists():
        raise FileExistsError(
            f"Episode aggregate receipt already exists: {output_path}"
        )
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_dir,
    )
    try:
        with os.fdopen(file_descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return output_path


def _sorted_map_indices(images_dir: Path) -> list[int]:
    indices = []
    for image_path in images_dir.glob("img_*.npy"):
        try:
            indices.append(int(image_path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(indices)


def _load_optional_array(path: Path):
    if path.exists():
        return np.load(path)
    return None


def _copy_or_fill_array(src: Path | None, dst: Path, fallback_array) -> None:
    if src is not None and src.exists():
        shutil.copy2(src, dst)
    else:
        np.save(dst, fallback_array)


def _maybe_load_single_map_metadata(map_path: Path):
    metadata_file = map_path / "metadata" / "map.json"
    if metadata_file.exists():
        return metadata_file
    flat_metadata_file = map_path / "metadata.json"
    if flat_metadata_file.exists():
        return flat_metadata_file
    return None


def _build_mixed_dataset_pool(
    curriculum_levels: list[dict],
    target_map_path: str,
    replay_map_count: int,
    target_map_repeat: int,
) -> tuple[list[dict], str, int]:
    dataset_root = os.getenv("DATASET_PATH", "")
    if not dataset_root:
        raise RuntimeError(
            "DATASET_PATH must be set to build a mixed target-map training pool."
        )

    if replay_map_count <= 0:
        raise ValueError(
            "replay_map_count must be > 0 when building a mixed dataset pool."
        )
    if target_map_repeat <= 0:
        raise ValueError(
            "target_map_repeat must be > 0 when building a mixed dataset pool."
        )
    if not curriculum_levels:
        raise ValueError(
            "curriculum_levels_override must be set when building a mixed dataset pool."
        )

    target_map_dir = Path(target_map_path).resolve()
    if not target_map_dir.exists():
        raise FileNotFoundError(f"Target map path does not exist: {target_map_dir}")

    target_image = _load_optional_array(target_map_dir / "images" / "img_1.npy")
    if target_image is None:
        target_image = _load_optional_array(target_map_dir / "image.npy")
    if target_image is None:
        raise FileNotFoundError(
            f"Could not find target-map image data under {target_map_dir}"
        )

    target_occupancy = _load_optional_array(target_map_dir / "occupancy" / "img_1.npy")
    if target_occupancy is None:
        target_occupancy = _load_optional_array(target_map_dir / "occupancy.npy")
    target_dumpability = _load_optional_array(
        target_map_dir / "dumpability" / "img_1.npy"
    )
    if target_dumpability is None:
        target_dumpability = _load_optional_array(target_map_dir / "dumpability.npy")
    target_distance = _load_optional_array(target_map_dir / "distance" / "img_1.npy")
    if target_distance is None:
        target_distance = _load_optional_array(target_map_dir / "distance.npy")
    if (
        target_occupancy is None
        or target_dumpability is None
        or target_distance is None
    ):
        raise FileNotFoundError(
            f"Target map at {target_map_dir} is missing occupancy, dumpability, or distance data."
        )
    target_actions = _load_optional_array(target_map_dir / "actions" / "img_1.npy")
    if target_actions is None:
        target_actions = _load_optional_array(target_map_dir / "actions.npy")
    if target_actions is None:
        target_actions = np.zeros_like(target_image)
    target_metadata = _maybe_load_single_map_metadata(target_map_dir)

    temp_root = Path(tempfile.mkdtemp(prefix="terra_mixed_pool_", dir="/tmp"))
    mixed_levels = []
    mixed_pool_size = replay_map_count + target_map_repeat

    for level_idx, level in enumerate(curriculum_levels):
        source_dir = Path(dataset_root) / level["maps_path"]
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Configured dataset path does not exist: {source_dir}"
            )

        indices = _sorted_map_indices(source_dir / "images")
        if not indices:
            raise RuntimeError(f"No dataset maps found under {source_dir / 'images'}")
        selected_indices = indices[-min(replay_map_count, len(indices)) :]
        if len(selected_indices) < replay_map_count:
            selected_indices.extend(
                [selected_indices[-1]] * (replay_map_count - len(selected_indices))
            )

        level_dir = temp_root / f"level_{level_idx}"
        for subdir in [
            "images",
            "occupancy",
            "dumpability",
            "distance",
            "actions",
            "metadata",
        ]:
            (level_dir / subdir).mkdir(parents=True, exist_ok=True)

        dataset_has_actions = (source_dir / "actions").exists()
        metadata_copied = False

        out_idx = 1
        for src_idx in selected_indices:
            image_path = source_dir / "images" / f"img_{src_idx}.npy"
            occupancy_path = source_dir / "occupancy" / f"img_{src_idx}.npy"
            dumpability_path = source_dir / "dumpability" / f"img_{src_idx}.npy"
            distance_path = source_dir / "distance" / f"img_{src_idx}.npy"
            if not all(
                path.exists()
                for path in [
                    image_path,
                    occupancy_path,
                    dumpability_path,
                    distance_path,
                ]
            ):
                raise FileNotFoundError(
                    f"Dataset map {src_idx} in {source_dir} is incomplete."
                )

            shutil.copy2(image_path, level_dir / "images" / f"img_{out_idx}.npy")
            shutil.copy2(occupancy_path, level_dir / "occupancy" / f"img_{out_idx}.npy")
            shutil.copy2(
                dumpability_path, level_dir / "dumpability" / f"img_{out_idx}.npy"
            )
            shutil.copy2(distance_path, level_dir / "distance" / f"img_{out_idx}.npy")

            if dataset_has_actions:
                _copy_or_fill_array(
                    source_dir / "actions" / f"img_{src_idx}.npy",
                    level_dir / "actions" / f"img_{out_idx}.npy",
                    np.zeros_like(target_image),
                )
            else:
                np.save(
                    level_dir / "actions" / f"img_{out_idx}.npy",
                    np.zeros_like(target_image),
                )

            dataset_metadata = source_dir / "metadata" / f"trench_{src_idx}.json"
            if dataset_metadata.exists():
                shutil.copy2(
                    dataset_metadata,
                    level_dir / "metadata" / f"trench_{out_idx}.json",
                )
                metadata_copied = True
            out_idx += 1

        for _ in range(target_map_repeat):
            np.save(level_dir / "images" / f"img_{out_idx}.npy", target_image)
            np.save(level_dir / "occupancy" / f"img_{out_idx}.npy", target_occupancy)
            np.save(
                level_dir / "dumpability" / f"img_{out_idx}.npy", target_dumpability
            )
            np.save(level_dir / "distance" / f"img_{out_idx}.npy", target_distance)
            np.save(level_dir / "actions" / f"img_{out_idx}.npy", target_actions)
            if target_metadata is not None:
                shutil.copy2(
                    target_metadata,
                    level_dir / "metadata" / f"trench_{out_idx}.json",
                )
                metadata_copied = True
            out_idx += 1

        if not metadata_copied:
            shutil.rmtree(level_dir / "metadata")

        mixed_level = dict(level)
        mixed_level["maps_path"] = f"level_{level_idx}"
        mixed_levels.append(mixed_level)

    return mixed_levels, str(temp_root), mixed_pool_size


PER_ENV_RATCHET_DISABLED_THRESHOLD = 1_000_000
REWARD_ANNEAL_SCHEMA = "terra_reward_anneal_v1"
REWARD_ANNEAL_DURATION_UPDATES = 5_000


def pooled_sampler_settings(config) -> SamplerSettings | None:
    raw = getattr(config, "pooled_sampler", None)
    if not raw or not raw.get("enabled", False):
        return None
    values = {
        field: raw[field]
        for field in SamplerSettings.__dataclass_fields__
        if field in raw
    }
    return SamplerSettings(**values)


def accepted_bank_sampler_labels(bank, settings: SamplerSettings) -> dict[str, dict]:
    """Build the one runtime label contract consumed by the host sampler."""
    if bank.curriculum_depths and len(bank.curriculum_depths) != len(bank.levels):
        raise ValueError("accepted-bank curriculum depths do not match its levels")
    return {
        level.condition_id: {
            "family": level.family,
            "branch_depth": level.branch_depth,
            **(
                {"curriculum_depth": bank.curriculum_depths[index]}
                if bank.curriculum_depths
                else {}
            ),
            **(
                {"sampling_weight": bank.sampling_probabilities[index]}
                if settings.rule == "fixed" and bank.sampling_probabilities
                else {}
            ),
        }
        for index, level in enumerate(bank.levels)
    }


def _restore_pooled_sampler_checkpoint(
    sampler: PooledConditionSampler | None,
    checkpoint: dict | None,
    checkpoint_mode: str | None,
) -> None:
    """Restore sampler history only for a true ``--resume_from`` load."""
    if checkpoint_mode != "resume":
        return

    saved_state = (
        checkpoint.get("pooled_sampler_state") if checkpoint is not None else None
    )
    if sampler is None:
        if saved_state is not None:
            raise ValueError(
                "resume checkpoint contains pooled sampler state, but the "
                "current run has no pooled sampler"
            )
        return
    if saved_state is None:
        raise ValueError(
            "resume with a pooled sampler requires checkpoint field "
            "'pooled_sampler_state'; use --warm_start_from for a fresh sampler"
        )
    sampler.restore_state_dict(saved_state)


def _new_reward_anneal_state() -> dict:
    return {
        "schema": REWARD_ANNEAL_SCHEMA,
        "started_update": None,
        "duration_updates": REWARD_ANNEAL_DURATION_UPDATES,
        "last_applied_mix": 0.0,
    }


def _restore_reward_anneal_checkpoint(
    reward_stage: str,
    checkpoint: dict | None,
    checkpoint_mode: str | None,
    resume_update: int = 0,
) -> dict | None:
    """Restore the one-way fade only for a true annealed-objective resume."""
    saved = checkpoint.get("reward_anneal_state") if checkpoint is not None else None
    if reward_stage != "annealed_objective":
        if checkpoint_mode == "resume" and saved is not None:
            raise ValueError(
                "resume checkpoint contains reward_anneal_state, but the current "
                f"reward_stage is {reward_stage!r}"
            )
        return None
    if checkpoint_mode != "resume":
        return _new_reward_anneal_state()
    if saved is None:
        raise ValueError(
            "annealed-objective resume requires checkpoint field "
            "'reward_anneal_state'; use --warm_start_from for a fresh fade"
        )
    if checkpoint.get("next_update") != int(resume_update):
        raise ValueError(
            "annealed-objective resume must use the checkpoint's exact next_update"
        )
    if set(saved) != {
        "schema",
        "started_update",
        "duration_updates",
        "last_applied_mix",
    }:
        raise ValueError("reward_anneal_state fields do not match the v1 schema")
    if saved["schema"] != REWARD_ANNEAL_SCHEMA:
        raise ValueError("reward_anneal_state schema is incompatible")
    if int(saved["duration_updates"]) != REWARD_ANNEAL_DURATION_UPDATES:
        raise ValueError("reward anneal duration changed across resume")
    started_update = saved["started_update"]
    if started_update is not None and (
        isinstance(started_update, bool)
        or not isinstance(started_update, (int, np.integer))
        or int(started_update) < 0
    ):
        raise ValueError("reward anneal started_update must be a non-negative integer")
    last_applied_mix = float(saved["last_applied_mix"])
    if not 0.0 <= last_applied_mix <= 1.0:
        raise ValueError("reward anneal last_applied_mix must be in [0, 1]")
    restored = {
        "schema": REWARD_ANNEAL_SCHEMA,
        "started_update": (None if started_update is None else int(started_update)),
        "duration_updates": REWARD_ANNEAL_DURATION_UPDATES,
        "last_applied_mix": last_applied_mix,
    }
    if restored["started_update"] is not None and restored["started_update"] >= int(
        resume_update
    ):
        raise ValueError("reward anneal starts at or after the resumed update")
    expected_last_mix = reward_anneal_mix(restored, max(0, int(resume_update) - 1))
    if not np.isclose(last_applied_mix, expected_last_mix, atol=1e-12, rtol=0.0):
        raise ValueError(
            "reward anneal last_applied_mix disagrees with checkpoint next_update"
        )
    saved_env_cfg = checkpoint.get("env_config")
    if saved_env_cfg is None or not hasattr(saved_env_cfg, "terminal_reward_mix"):
        raise ValueError("annealed-objective resume requires env_config reward mix")
    saved_env_mix = np.asarray(saved_env_cfg.terminal_reward_mix, dtype=np.float32)
    if not np.all(saved_env_mix == np.float32(expected_last_mix)):
        raise ValueError(
            "checkpoint env_config terminal_reward_mix disagrees with anneal state"
        )
    return restored


def maybe_start_reward_anneal(
    state: dict | None,
    sampler_receipt: dict,
    update_index: int,
) -> bool:
    """Latch the fade once both families have reached curriculum depth two."""
    if state is None or state["started_update"] is not None:
        return False
    active = sampler_receipt["mastery"]["family_active_depth"]
    # None means every depth in that family is mastered, i.e. it has progressed
    # beyond the depth-two trigger rather than falling short of it.
    if all(
        active[family] is None or active[family] >= 2
        for family in ("foundation", "trench")
    ):
        state["started_update"] = int(update_index)
        return True
    return False


def reward_anneal_mix(state: dict | None, update_index: int) -> float:
    if state is None or state["started_update"] is None:
        return 0.0
    elapsed = max(0, int(update_index) - int(state["started_update"]))
    return min(1.0, elapsed / float(state["duration_updates"]))


def assign_terminal_reward_mix(env_cfg, mix: float):
    if not 0.0 <= float(mix) <= 1.0:
        raise ValueError("terminal reward mix must be in [0, 1]")
    current = jnp.asarray(env_cfg.terminal_reward_mix)
    return env_cfg._replace(
        terminal_reward_mix=jnp.full_like(current, float(mix), dtype=current.dtype)
    )


def _assert_pooled_level_contract(
    curriculum_levels: list[dict],
    increase_threshold: int,
    decrease_threshold: int,
) -> None:
    """Keep condition selection from silently changing the environment task."""
    if not curriculum_levels:
        raise ValueError("pooled sampler needs at least one condition level")
    contracts = {
        (
            int(level["max_steps_in_episode"]),
            int(level["rewards_type"]),
            bool(level["apply_trench_rewards"]),
        )
        for level in curriculum_levels
    }
    if contracts != {(450, 0, False)}:
        raise ValueError(
            "accepted-bank condition levels must all use horizon=450, DENSE, "
            f"apply_trench_rewards=false; observed {sorted(contracts)}"
        )
    if min(int(increase_threshold), int(decrease_threshold)) < (
        PER_ENV_RATCHET_DISABLED_THRESHOLD
    ):
        raise ValueError(
            "the global sampler owns condition assignment; Terra's per-env "
            "ratchet must be disabled with thresholds >= "
            f"{PER_ENV_RATCHET_DISABLED_THRESHOLD}"
        )


def assign_curriculum_levels(env_cfg, levels: np.ndarray):
    current = jnp.asarray(env_cfg.curriculum.level)
    assigned = jnp.asarray(levels, dtype=current.dtype)
    if assigned.shape != current.shape:
        raise ValueError(
            f"condition assignment shape {assigned.shape} != env shape "
            f"{current.shape}"
        )
    return env_cfg._replace(curriculum=env_cfg.curriculum._replace(level=assigned))


def reset_exposure_histogram(
    done: jax.Array,
    curriculum_level: jax.Array,
    num_stages: int,
) -> jax.Array:
    """Count completed episodes by the level active on each transition."""
    if done.shape != curriculum_level.shape:
        raise ValueError(
            "done and curriculum_level must have identical [step, env] shapes"
        )
    return (
        jnp.zeros((num_stages,), dtype=jnp.int32)
        .at[curriculum_level.reshape(-1)]
        .add(done.reshape(-1).astype(jnp.int32))
    )


def transition_exposure_histogram(
    active_curriculum_level: jax.Array,
    num_stages: int,
) -> jax.Array:
    """Count policy transitions by the map condition that produced them."""
    return (
        jnp.zeros((num_stages,), dtype=jnp.int32)
        .at[active_curriculum_level.reshape(-1)]
        .add(1)
    )


@dataclass
class MixedAgentTrainConfig:
    """Configuration for training mixed agent environments

    Supports loading from named presets via --config <name>.
    See configs/training_configs.py for available presets.
    """

    name: str
    num_devices: int = 0
    project: str = "mixed-agents"
    group: str = "tracked-skidsteer"
    num_envs_per_device: int = 2048
    num_steps: int = 32
    update_epochs: int = 2
    num_minibatches: int = 16
    total_timesteps: int = 50_000_000_000
    lr: float = 3e-4
    clip_eps: float = 0.2
    gamma: float = 0.9984
    gae_lambda: float = 0.95
    ent_coef: float = 0.06
    vf_coef: float = 2.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 100
    seed: int = 42
    log_train_interval: int = 1  # Number of updates between logging train stats
    log_eval_interval: int = 100
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    keep_checkpoint_history: bool = False

    # Model settings optimized for mixed agents
    num_prev_actions: int = 10  # overridden to 5 * num_agents at runtime
    clip_action_maps: bool = True  # clips the action maps to [-1, 1]
    local_map_normalization_bounds: tuple[int, int] = (-16, 16)
    maps_net_normalization_bounds: tuple[int, int] = (-10, 10)
    loaded_max: int = 100
    local_map_area_scale: float = 1.0
    num_rollouts_eval: int = 200
    cache_clear_interval: int = 1000
    # Entropy scheduler (cosine decay)
    ent_schedule_start: float = 0.15
    ent_schedule_end: float = 0.005
    ent_schedule_steps: int = 9500

    # Removed agent-type curriculum; use override only
    # Optional override to specify an arbitrary list of agent types, e.g. (2,0,0,2)
    agent_types_override: tuple | None = None
    # Optional override to specify action types for each agent, e.g. (0,1,0,1) for tracked/wheeled
    action_types_override: tuple | None = None
    # Debug assertions and one-time validations
    debug: bool = False
    # Fail fast when PPO produces non-finite core losses/grad diagnostics.
    # Params and optimizer state are additionally checked before checkpoint/final
    # writes. Default off preserves historical runs; E9+ smoke/prod jobs enable it.
    fail_on_nonfinite: bool = False
    finite_check_interval: int = 0
    # Checkpoint loading
    resume_from: str | None = None  # Path to a checkpoint .pkl to resume from
    # Load only model parameters. Optimizer, update counter, environment,
    # curriculum state, RNG, and action history are always fresh.
    warm_start_from: str | None = None
    load_env_from_checkpoint: bool = True  # If true, use env_config from checkpoint
    resume_update: int | None = None  # Optional override for old param-only checkpoints

    # Named configuration preset (loads from configs/training_configs.py)
    config_name: str | None = None  # e.g., "excavator_truck", "solo_excavator"

    # Agent-neutral relocation reward (can be set via config preset or CLI).
    relocation_progress_mult: float | None = None
    # One global reward objective. Map-level rewards_type remains frozen.
    reward_stage: str = "dense_skill"

    # Capacity overrides
    truck_capacity: int | None = None
    skidsteer_capacity: int | None = None
    truck_road_restricted: bool | None = None
    enforce_foundation_border_alignment: bool | None = None

    # Curriculum/maps override (from YAML config)
    # Format: list of dicts with keys: maps_path, max_steps_in_episode, rewards_type, apply_trench_rewards
    curriculum_levels_override: list | None = None
    curriculum_increase_level_threshold: int | None = None
    curriculum_decrease_level_threshold: int | None = None
    curriculum_last_level_type: str | None = None
    pooled_sampler: dict | None = None
    accepted_bank: AcceptedBank | None = None
    # Optional single-map training path. When set, map loading uses this path directly
    # and does not rely on DATASET_PATH / DATASET_SIZE.
    single_map_path: str | None = None
    # Mixed fine-tuning mode: sample recent config maps and oversample the target map.
    replay_map_count: int = 0
    target_map_repeat: int = 0
    model_size: str = "base"
    model_core: str = "mlp"
    map_encoder: str = "atari"
    # Encoder mixed precision: "float32" (default) or "bfloat16". bf16 is only
    # valid for the spatial ResNet encoders (validated in get_model_ready).
    encoder_compute_dtype: str = "float32"
    # F16: attention precision override for v4/v5 only. "encoder" preserves
    # current behavior; "float32" keeps attention logits/softmax/projections in
    # f32 while the spatial-ResNet conv trunk can still run in bf16.
    attention_compute_dtype: str = "encoder"
    # F16: optional small nonzero init for v5 token-mixer residual projections.
    # Default 0.0 preserves the exact identity-at-init mixer.
    token_mixer_residual_init_scale: float = 0.0
    # Optional critic-head width override; None keeps the model_size preset.
    critic_hidden_dims: tuple | None = None
    # F5: PPO value-clipping toggle. Default True preserves current behavior;
    # --no_value_clip flips to a plain 0.5*MSE value loss.
    use_value_clip: bool = True
    # F6: flatten [seq, env] into a single sample axis before per-minibatch
    # permutation. Default False keeps the env-only shuffle (blocked layout).
    flat_minibatch_shuffle: bool = False
    # F7: kickstart distillation. teacher_checkpoint=None disables the feature
    # entirely; everything below is inert while it is None.
    teacher_checkpoint: str | None = None
    kickstart_kl_coef: float = 1.0
    kickstart_kl_anneal_updates: int = 1500
    kickstart_value_coef: float = 0.5
    kickstart_value_anneal_updates: int = 500
    kickstart_lr_warmup_updates: int = 100

    # F15: 128x128 resolution scaling. All default None/1 = no change (the
    # default env/model/teacher paths stay bit-identical). Tile-denominated
    # agent geometry (move_tiles, dig_radius_tiles) and the tile-count capacity /
    # reward-normalizer quantities are applied via env_config._replace at
    # env-config creation; loaded_max_override rescales the model-side loaded
    # normalizer. agent.width/height are NOT exposed: the terra env recomputes
    # them in update_env_cfgs from ExcavatorDims meters / tile_size on every
    # reset, so they auto-scale with maps_edge_length and any config override
    # would be silently clobbered (see F15 report).
    agent_move_tiles: int | None = None
    dig_radius_tiles: int | None = None
    reward_normalizer: float | None = None
    loaded_max_override: int | None = None
    # Spatial ResNet stage overrides (F15). Comma lists parsed to tuples on the
    # CLI; None keeps the model_size preset (or the module default for "base").
    resnet_stage_channels: tuple | None = None
    resnet_blocks_per_stage: tuple | None = None
    # Teacher obs-downsampling for cross-resolution kickstart (F15). 1 = off;
    # 2 subsamples the teacher's obs to its native (half-resolution) world.
    teacher_obs_downsample: int = 1

    def __post_init__(self):
        _checkpoint_load_mode(self)
        if self.reward_stage not in (
            "dense_skill",
            "annealed_objective",
            "terminal_objective",
        ):
            raise ValueError(
                "reward_stage must be 'dense_skill', 'annealed_objective', or "
                f"'terminal_objective', got {self.reward_stage!r}"
            )
        if self.reward_stage == "annealed_objective":
            sampler = self.pooled_sampler or {}
            if sampler.get("rule") != "continuous_banded_v1":
                raise ValueError(
                    "annealed_objective requires the continuous_banded_v1 sampler"
                )
        self.map_encoder = canonical_map_encoder(self.map_encoder)
        if self.attention_compute_dtype not in ("encoder", "float32", "bfloat16"):
            raise ValueError(
                "attention_compute_dtype must be one of 'encoder', 'float32', "
                f"or 'bfloat16', got {self.attention_compute_dtype!r}"
            )
        attention_encoders = (
            "resnet_spatial_8x8_se_xattn",
            "resnet_spatial_8x8_se_sa_xattn",
        )
        if (
            self.attention_compute_dtype != "encoder"
            and self.map_encoder not in attention_encoders
        ):
            raise ValueError(
                "--attention_compute_dtype only applies to the v4/v5 attention "
                f"encoders, not map_encoder={self.map_encoder!r}."
            )
        self.token_mixer_residual_init_scale = float(
            self.token_mixer_residual_init_scale
        )
        if self.token_mixer_residual_init_scale < 0.0:
            raise ValueError(
                "--token_mixer_residual_init_scale must be >= 0, "
                f"got {self.token_mixer_residual_init_scale}."
            )
        if (
            self.token_mixer_residual_init_scale != 0.0
            and self.map_encoder != "resnet_spatial_8x8_se_sa_xattn"
        ):
            raise ValueError(
                "--token_mixer_residual_init_scale only applies to "
                "map_encoder=resnet_spatial_8x8_se_sa_xattn "
                f"(resnet_spatial_v5), not map_encoder={self.map_encoder!r}."
            )
        # F15: loaded_max_override rescales the model-side loaded normalizer
        # (tile-count quantity, ~x4 at 128). Fold it into loaded_max so
        # get_model_ready and the teacher preprocessing check read one value.
        if self.loaded_max_override is not None:
            self.loaded_max = int(self.loaded_max_override)
        self.local_map_area_scale = float(self.local_map_area_scale)
        if self.local_map_area_scale <= 0.0:
            raise ValueError(
                f"local_map_area_scale must be > 0, got {self.local_map_area_scale}."
            )
        # F15: teacher obs-downsampling is only meaningful with a teacher.
        if self.teacher_obs_downsample < 1:
            raise ValueError(
                f"teacher_obs_downsample must be >= 1, got {self.teacher_obs_downsample}"
            )
        if self.teacher_obs_downsample != 1 and self.teacher_checkpoint is None:
            raise ValueError(
                "--teacher_obs_downsample is only valid together with "
                "--teacher_checkpoint (cross-resolution kickstart)."
            )
        if self.finite_check_interval < 0:
            raise ValueError(
                f"finite_check_interval must be >= 0, got {self.finite_check_interval}"
            )
        if self.fail_on_nonfinite and self.finite_check_interval == 0:
            self.finite_check_interval = 1
        self.num_devices = (
            jax.local_device_count() if self.num_devices == 0 else self.num_devices
        )
        if not 1 <= self.num_devices <= jax.local_device_count():
            raise ValueError(
                f"num_devices must be in [1, {jax.local_device_count()}], "
                f"got {self.num_devices}"
            )
        if (
            self.num_envs_per_device <= 0
            or self.num_steps <= 0
            or self.update_epochs <= 0
            or self.num_minibatches <= 0
        ):
            raise ValueError(
                "num_envs_per_device, num_steps, update_epochs, and "
                "num_minibatches must be positive"
            )
        if self.num_envs_per_device % self.num_minibatches != 0:
            raise ValueError("num_envs_per_device must be divisible by num_minibatches")
        if (
            self.agent_types_override is not None
            and self.action_types_override is not None
            and len(self.agent_types_override) != len(self.action_types_override)
        ):
            raise ValueError(
                "agent_types_override and action_types_override must have equal length"
            )
        self.num_envs = self.num_envs_per_device * self.num_devices
        self.total_timesteps_per_device = self.total_timesteps // self.num_devices
        self.eval_episodes_per_device = self.eval_episodes // self.num_devices
        assert (
            self.num_envs % self.num_devices == 0
        ), "Number of environments must be divisible by the number of devices."
        self.env_steps_per_update = self.num_steps * self.num_envs
        self.num_updates = self.total_timesteps // self.env_steps_per_update
        if self.num_updates <= 0:
            raise ValueError("total_timesteps must cover at least one PPO update")
        self.actual_total_timesteps = self.num_updates * self.env_steps_per_update

        print(f"Devices: {jax.devices()}")
        print(
            "Mixed Agent Training - "
            f"Devices: {self.num_devices}, Updates: {self.num_updates}, "
            f"Env steps/update: {self.env_steps_per_update}"
        )
        print(f"Using overridden agent types: {self.agent_types_override}")

    # make object subscriptable - required for compatibility with existing code
    def __getitem__(self, key):
        return getattr(self, key)


def _reward_stage_value(reward_stage: str) -> RewardStage:
    values = {
        "dense_skill": RewardStage.DENSE_SKILL,
        "annealed_objective": RewardStage.ANNEALED_OBJECTIVE,
        "terminal_objective": RewardStage.TERMINAL_OBJECTIVE,
    }
    try:
        return values[reward_stage]
    except KeyError as exc:
        raise ValueError(f"unsupported reward_stage {reward_stage!r}") from exc


def create_mixed_agent_env_config(
    agent_types=(0, 2),
    action_types=(0, 0),
    relocation_progress_mult=None,
    reward_stage="dense_skill",
    # Optional capacity overrides
    truck_capacity=None,
    skidsteer_capacity=None,
    truck_road_restricted=None,
    enforce_foundation_border_alignment=None,
    # F15: resolution-scaling overrides (all None = no change)
    agent_move_tiles=None,
    dig_radius_tiles=None,
    reward_normalizer=None,
):
    """Create environment configuration optimized for mixed agent training

    Args:
        agent_types: Tuple of agent type IDs (0=excavator, 1=truck, 2=skidsteer)
        action_types: Tuple of action type IDs (0=tracked, 1=wheeled)
        relocation_progress_mult: Agent-neutral signed relocation progress multiplier
        truck_capacity: Override for truck capacity
        skidsteer_capacity: Override for skidsteer capacity
        truck_road_restricted: Whether trucks are restricted to roads
        enforce_foundation_border_alignment: Whether foundation border alignment is enforced
        agent_move_tiles: F15 override for agent.move_tiles (tiles/move action)
        dig_radius_tiles: F15 override for agent.dig_radius_tiles (workspace/cone reach)
        reward_normalizer: F15 override for rewards.normalizer (per-tile reward scaling)
    """

    # Use the existing dense rewards from config
    env_config = EnvConfig()._replace(
        reward_stage=_reward_stage_value(reward_stage),
        terminal_reward_mix=0.0,
    )

    # Set the agent types from the training configuration
    env_config = env_config._replace(agent_types=agent_types)

    # Set the action types from the training configuration
    env_config = env_config._replace(action_types=action_types)

    if relocation_progress_mult is not None:
        env_config = env_config._replace(
            relocation_progress_mult=relocation_progress_mult
        )

    # Apply capacity overrides if provided
    if truck_capacity is not None:
        env_config = env_config._replace(truck_capacity=truck_capacity)
    if skidsteer_capacity is not None:
        env_config = env_config._replace(skidsteer_capacity=skidsteer_capacity)
    if truck_road_restricted is not None:
        env_config = env_config._replace(truck_road_restricted=truck_road_restricted)
    if enforce_foundation_border_alignment is not None:
        env_config = env_config._replace(
            enforce_foundation_border_alignment=enforce_foundation_border_alignment
        )

    # F15: tile-denominated agent geometry. These survive update_env_cfgs (which
    # only recomputes width/height/tile_size), so the _replace here is the
    # authoritative override. agent.width/height are intentionally NOT settable
    # here: update_env_cfgs derives them from ExcavatorDims meters / tile_size on
    # every reset, so they already double when maps_edge_length doubles.
    if agent_move_tiles is not None or dig_radius_tiles is not None:
        agent_cfg = env_config.agent
        if agent_move_tiles is not None:
            agent_cfg = agent_cfg._replace(move_tiles=int(agent_move_tiles))
        if dig_radius_tiles is not None:
            agent_cfg = agent_cfg._replace(dig_radius_tiles=int(dig_radius_tiles))
        env_config = env_config._replace(agent=agent_cfg)

    # F15: per-tile reward normalizer (env_config.rewards is a NamedTuple).
    if reward_normalizer is not None:
        env_config = env_config._replace(
            rewards=env_config.rewards._replace(normalizer=float(reward_normalizer))
        )

    return env_config


def _print_resolution_scaling_table(config) -> None:
    """Print an old->new table of every F15 resolution-scaling override.

    Only emitted when at least one override is set; the default (all None) path
    prints nothing and is bit-identical to pre-F15 behavior. Reports the
    effective scaled fields so a 128x128 run's equivalence assumptions are
    auditable from the logs.
    """
    env_defaults = EnvConfig()
    candidates = []
    if config.agent_move_tiles is not None:
        candidates.append(
            ("agent.move_tiles", env_defaults.agent.move_tiles, config.agent_move_tiles)
        )
    if config.dig_radius_tiles is not None:
        candidates.append(
            (
                "agent.dig_radius_tiles",
                env_defaults.agent.dig_radius_tiles,
                config.dig_radius_tiles,
            )
        )
    if config.truck_capacity is not None:
        candidates.append(
            ("truck_capacity", env_defaults.truck_capacity, config.truck_capacity)
        )
    if config.skidsteer_capacity is not None:
        candidates.append(
            (
                "skidsteer_capacity",
                env_defaults.skidsteer_capacity,
                config.skidsteer_capacity,
            )
        )
    if config.loaded_max_override is not None:
        candidates.append(
            ("loaded_max", MixedAgentTrainConfig.loaded_max, config.loaded_max)
        )
    if getattr(config, "local_map_area_scale", 1.0) != 1.0:
        candidates.append(("local_map_area_scale", 1.0, config.local_map_area_scale))
    if config.reward_normalizer is not None:
        candidates.append(
            (
                "rewards.normalizer",
                env_defaults.rewards.normalizer,
                config.reward_normalizer,
            )
        )
    # Only genuine changes are a "scaling"; a truck preset with the default
    # capacity (52 -> 52) is not.
    rows = [(name, old, new) for name, old, new in candidates if old != new]
    if not rows:
        return
    print("\n📐 F15 resolution scaling (old -> new):", flush=True)
    print(f"   {'field':24s} {'old':>12s} -> {'new':>12s}", flush=True)
    for name, old, new in rows:
        print(f"   {name:24s} {str(old):>12s} -> {str(new):>12s}", flush=True)
    print(
        "   note: agent.width/height auto-scale in the env (update_env_cfgs "
        "derives them from tile_size); not overridable here.",
        flush=True,
    )


class ConfigurableAgentManager:
    """Simplified: agent types come only from override or defaults."""

    def __init__(self, config: MixedAgentTrainConfig):
        self.config = config

    def get_current_agent_types(self, *_, **__) -> tuple[int, int]:
        if self.config.agent_types_override is not None:
            ats = tuple(self.config.agent_types_override)
        else:
            ats = EnvConfig().agent_types
        # Ensure we always return a 2-tuple for prints; extra types still supported elsewhere
        if len(ats) >= 2:
            return (int(ats[0]), int(ats[1]))
        if len(ats) == 1:
            return (int(ats[0]), int(ats[0]))
        return (0, 2)

    def get_current_action_types(self, *_, **__) -> tuple[int, int]:
        if self.config.action_types_override is not None:
            ats = tuple(self.config.action_types_override)
        else:
            # Default to tracked actions (0) for all agents
            ats = (0, 0)
        # Ensure we always return a 2-tuple for prints; extra types still supported elsewhere
        if len(ats) >= 2:
            return (int(ats[0]), int(ats[1]))
        if len(ats) == 1:
            return (int(ats[0]), int(ats[0]))
        return (0, 0)


def make_mixed_agent_states(
    config: MixedAgentTrainConfig,
    env_params: EnvConfig = None,
    env_params_override: EnvConfig = None,
):
    """Initialize states for mixed agent training - compatible with make_states interface"""
    curriculum_levels = config.curriculum_levels_override
    single_map_path = config.single_map_path

    if (
        single_map_path is not None
        and config.replay_map_count > 0
        and config.target_map_repeat > 0
    ):
        curriculum_levels, mixed_dataset_root, mixed_pool_size = (
            _build_mixed_dataset_pool(
                curriculum_levels=curriculum_levels,
                target_map_path=single_map_path,
                replay_map_count=config.replay_map_count,
                target_map_repeat=config.target_map_repeat,
            )
        )
        os.environ["DATASET_PATH"] = mixed_dataset_root
        os.environ["DATASET_SIZE"] = str(mixed_pool_size)
        single_map_path = None
        print(
            "📍 Using mixed target-map pool: "
            f"{config.replay_map_count} recent maps + "
            f"{config.target_map_repeat} target-map repeats per curriculum level"
        )
        print(f"📍 Mixed dataset root: {mixed_dataset_root}")

    # Create batch config - override curriculum levels if provided
    if curriculum_levels is not None and len(curriculum_levels) > 0:
        increase_th = (
            config.curriculum_increase_level_threshold
            if config.curriculum_increase_level_threshold is not None
            else CurriculumGlobalConfig.increase_level_threshold
        )
        decrease_th = (
            config.curriculum_decrease_level_threshold
            if config.curriculum_decrease_level_threshold is not None
            else CurriculumGlobalConfig.decrease_level_threshold
        )
        last_level = (
            config.curriculum_last_level_type
            if config.curriculum_last_level_type is not None
            else CurriculumGlobalConfig.last_level_type
        )
        sampler_settings = pooled_sampler_settings(config)
        if sampler_settings is not None:
            if single_map_path is not None:
                raise ValueError(
                    "the pooled condition sampler cannot be combined with " "--map_path"
                )
            _assert_pooled_level_contract(
                curriculum_levels,
                increase_th,
                decrease_th,
            )

        class CustomCurriculumGlobalConfig(CurriculumGlobalConfig):
            levels = curriculum_levels
            increase_level_threshold = increase_th
            decrease_level_threshold = decrease_th
            last_level_type = last_level

        batch_cfg = BatchConfig(curriculum_global=CustomCurriculumGlobalConfig())
        level_paths = [level["maps_path"] for level in curriculum_levels]
        if len(level_paths) > 8:
            print(
                f"📍 Using {len(level_paths)} condition levels: "
                f"{level_paths[0]} .. {level_paths[-1]}"
            )
        else:
            print(f"📍 Using maps from config: {level_paths}")
        if sampler_settings is None:
            print(
                f"📍 Curriculum: promote after {increase_th} task success(es), "
                f"demote after {decrease_th} failure(s), "
                f"last_level_type={last_level!r}"
            )
        else:
            print(
                "📍 Global condition sampler owns levels; per-env ratchet "
                f"disabled at {increase_th}/{decrease_th}"
            )
    else:
        batch_cfg = BatchConfig()
        print("📍 Using default maps from config.py")

    # Initialize environment with configurable agents
    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        shuffle_maps=False,
        single_map_path=single_map_path,
    )
    if single_map_path is not None:
        print(f"📍 Using single map path: {single_map_path}")

    # Get environment parameters with agent types from config
    if env_params is None:
        if env_params_override is not None:
            # Use environment config from checkpoint
            env_params = env_params_override
            print("Using environment config from checkpoint")
        else:
            # Use override if provided; else default EnvConfig agent_types
            if config.agent_types_override is not None:
                agent_types = tuple(config.agent_types_override)
            else:
                agent_types = EnvConfig().agent_types

            # Use action types override if provided, otherwise use default (0,0)
            action_types = (
                config.action_types_override
                if config.action_types_override is not None
                else (0, 0)
            )
            env_params = create_mixed_agent_env_config(
                agent_types=agent_types,
                action_types=action_types,
                relocation_progress_mult=config.relocation_progress_mult,
                reward_stage=config.reward_stage,
                # Pass capacity overrides
                truck_capacity=config.truck_capacity,
                skidsteer_capacity=config.skidsteer_capacity,
                truck_road_restricted=config.truck_road_restricted,
                enforce_foundation_border_alignment=config.enforce_foundation_border_alignment,
                # F15 resolution-scaling overrides
                agent_move_tiles=config.agent_move_tiles,
                dig_radius_tiles=config.dig_radius_tiles,
                reward_normalizer=config.reward_normalizer,
            )
            _print_resolution_scaling_table(config)
            # Verbose training configuration summary
            type_names = {0: "Excavator", 1: "Truck", 2: "SkidSteer"}
            print("🧩 Agent Types (effective):", agent_types)
            print(
                "🧩 Agent Types (names):",
                " + ".join(type_names.get(t, f"Unknown({t})") for t in agent_types),
            )
            if config.agent_types_override is not None:
                print("✅ Using --agent_types override")

            # Print action types information
            action_type_names = {0: "Tracked", 1: "Wheeled"}
            print("🚗 Action Types (effective):", action_types)
            print(
                "🚗 Action Types (names):",
                " + ".join(
                    action_type_names.get(t, f"Unknown({t})") for t in action_types
                ),
            )
            if config.action_types_override is not None:
                print("✅ Using --action_types override")
            else:
                print("🚗 Using default action types (all tracked)")

            if config.relocation_progress_mult is not None:
                print(
                    "📊 Relocation progress multiplier: "
                    f"{config.relocation_progress_mult}"
                )

    # The command-line treatment wins over a checkpoint's reward selector. The
    # mix itself is assigned from explicit host-side anneal state every update.
    env_params = env_params._replace(
        reward_stage=_reward_stage_value(config.reward_stage),
        terminal_reward_mix=0.0,
    )

    # Report the effective value after preset, CLI, and checkpoint precedence.
    print(
        "🪣 Relocation progress multiplier (effective): "
        f"{float(env_params.relocation_progress_mult)}"
    )
    print(f"🎯 Reward stage (effective): {config.reward_stage}")

    num_devices = config.num_devices
    num_envs_per_device = config.num_envs_per_device

    print("⏱️  Batching env_params...", flush=True)
    t_env_params = time.time()
    env_params = jax.tree_map(
        lambda x: jnp.array(x)[None, None]
        .repeat(num_devices, 0)
        .repeat(num_envs_per_device, 1),
        env_params,
    )
    print(
        f"⏱️  Batching env_params done in {time.time() - t_env_params:.2f}s",
        flush=True,
    )

    print(
        f"Mixed Agent Environment - Tile size shape: {env_params.tile_size.shape}",
        flush=True,
    )

    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)

    # Infer num_prev_actions as 5 per agent without triggering a reset/pmap
    try:
        MAX_AGENTS = 4
        # The actual batched environment is authoritative, including when it
        # came from a checkpoint and the CLI still has its default override.
        try:
            na = _num_agents_from_env_params(env_params)
        except ValueError:
            if config.agent_types_override is not None:
                na = len(tuple(config.agent_types_override))
            elif hasattr(env.batch_cfg, "agent_types") and isinstance(
                env.batch_cfg.agent_types, (tuple, list)
            ):
                na = len(env.batch_cfg.agent_types)
            else:
                na = MAX_AGENTS
        na = max(1, min(MAX_AGENTS, int(na)))
        config.num_prev_actions = int(5 * na)
        print(
            f"Setting num_prev_actions to {config.num_prev_actions} (5 per agent × {na} agents)",
            flush=True,
        )
    except Exception as e:
        print(
            f"Warning: failed to infer num_agents for num_prev_actions ({e}); keeping {config.num_prev_actions}",
            flush=True,
        )

    # Create the unified network with agent type features (now that num_prev_actions is set)
    print(f"🧠 Model size preset: {getattr(config, 'model_size', 'base')}", flush=True)
    print(f"🧠 Model core: {getattr(config, 'model_core', 'mlp')}", flush=True)
    print(f"🧠 Map encoder: {getattr(config, 'map_encoder', 'atari')}", flush=True)
    print(
        f"🧠 Encoder dtype: {getattr(config, 'encoder_compute_dtype', 'float32')}, "
        f"attention dtype: {getattr(config, 'attention_compute_dtype', 'encoder')}",
        flush=True,
    )
    print("⏱️  Initializing model...", flush=True)
    t_model_init = time.time()
    network, network_params = get_model_ready(_rng, config, env)
    print(
        f"⏱️  Model init done in {time.time() - t_model_init:.2f}s",
        flush=True,
    )
    # Print architecture summary for easy debugging/comparison in logs.
    model_core = getattr(config, "model_core", "mlp")
    print("🏗️ Architecture:", flush=True)
    print(f"   core: {model_core}", flush=True)
    print(f"   model_size: {getattr(config, 'model_size', 'base')}", flush=True)
    print(f"   map_encoder: {getattr(config, 'map_encoder', 'atari')}", flush=True)
    print(
        f"   encoder_compute_dtype: {getattr(config, 'encoder_compute_dtype', 'float32')}",
        flush=True,
    )
    print(
        f"   attention_compute_dtype: "
        f"{getattr(config, 'attention_compute_dtype', 'encoder')}",
        flush=True,
    )
    print(
        f"   token_mixer_residual_init_scale: "
        f"{getattr(config, 'token_mixer_residual_init_scale', 0.0)}",
        flush=True,
    )
    if model_core == "transformer":
        max_agents = 4
        token_count = max_agents + 3  # agent tokens + actions/local/maps tokens
        print("   transformer details:", flush=True)
        print(f"     tokens_total: {token_count}", flush=True)
        print(f"     tokens_agent: {max_agents}", flush=True)
        print(
            "     tokens_global: 3 (prev_actions, local_map, global_maps)", flush=True
        )
        print(f"     layers: {network.transformer_num_layers}", flush=True)
        print(f"     heads: {network.transformer_num_heads}", flush=True)
        print(f"     model_dim: {network.transformer_model_dim}", flush=True)
        print(f"     ffn_dim: {network.transformer_ffn_dim}", flush=True)
    else:
        print("   mlp details:", flush=True)
        print(
            "     fusion: concat(agent_state, prev_actions, local_map, cnn_maps)",
            flush=True,
        )
        print(f"     intermediate_mlp_dim: {network.intermediate_mlp_dim}", flush=True)
    # Debug: print number of actions for current action type (kept as requested)
    try:
        num_actions_debug = env.batch_cfg.action_type.get_num_actions()
        print(f"🛠️ Debug: Number of actions = {num_actions_debug}", flush=True)
    except Exception as e:
        print(f"🛠️ Debug: Failed to read number of actions: {e}", flush=True)

    # Optimizer with mixed agent considerations.
    # F7: when a kickstart teacher is set, warm the LR up linearly from lr/3 to
    # lr over kickstart_lr_warmup_updates PPO updates, then hold constant. Each
    # PPO update runs update_epochs*num_minibatches optax steps, so convert the
    # warmup window from updates to optimizer steps. Without a teacher the LR is
    # the plain constant, keeping default runs bit-identical.
    if getattr(config, "teacher_checkpoint", None) is not None:
        warmup_updates = int(getattr(config, "kickstart_lr_warmup_updates", 0))
        grad_steps_per_update = config.update_epochs * config.num_minibatches
        warmup_steps = max(1, warmup_updates * grad_steps_per_update)
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=config.lr / 3.0,
                    end_value=config.lr,
                    transition_steps=warmup_steps,
                ),
                optax.constant_schedule(config.lr),
            ],
            boundaries=[warmup_steps],
        )
        adam_learning_rate = lr_schedule
        print(
            "🔥 Kickstart LR warmup: "
            f"{config.lr / 3.0:.2e} -> {config.lr:.2e} over "
            f"{warmup_updates} updates ({warmup_steps} optax steps)",
            flush=True,
        )
    else:
        adam_learning_rate = config.lr
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=adam_learning_rate, eps=1e-5),
    )

    train_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx
    )

    print(f"Network: Unified with agent type conditioning", flush=True)

    return rng, env, env_params, train_state


def _wandb_tags_for_config(config: MixedAgentTrainConfig) -> list[str]:
    def _tag_value(value) -> str:
        return str(value).replace(" ", "_").replace("/", "-")

    env_defaults = EnvConfig()
    agent_type_names = {0: "excavator", 1: "truck", 2: "skidsteer"}
    action_type_names = {0: "tracked", 1: "wheeled"}
    agent_types = tuple(config.agent_types_override or env_defaults.agent_types)
    action_types = tuple(config.action_types_override or ((0,) * len(agent_types)))
    dump_min_free_fraction = env_defaults.foundation_dump_min_free_fraction
    edge_align_enabled = (
        config.enforce_foundation_border_alignment
        if config.enforce_foundation_border_alignment is not None
        else env_defaults.enforce_foundation_border_alignment
    )

    model_size = config.model_size if hasattr(config, "model_size") else "unknown"
    map_encoder = getattr(config, "map_encoder", "atari")
    encoder_compute_dtype = getattr(config, "encoder_compute_dtype", "float32")
    attention_compute_dtype = getattr(config, "attention_compute_dtype", "encoder")
    token_mixer_residual_init_scale = getattr(
        config, "token_mixer_residual_init_scale", 0.0
    )
    critic_hidden_dims = getattr(config, "critic_hidden_dims", None)
    critic_hidden_tag = (
        "-".join(str(int(x)) for x in critic_hidden_dims)
        if critic_hidden_dims
        else "preset"
    )

    tags = [
        "mixed-agents",
        "unified-network",
        f"config:{_tag_value(config.config_name or 'manual')}",
        f"agents:{'-'.join(agent_type_names.get(int(t), str(t)) for t in agent_types)}",
        f"actions:{'-'.join(action_type_names.get(int(t), str(t)) for t in action_types)}",
        f"model-size:{_tag_value(model_size)}",
        f"map-encoder:{_tag_value(map_encoder)}",
        f"encoder-dtype:{_tag_value(encoder_compute_dtype)}",
        f"attention-dtype:{_tag_value(attention_compute_dtype)}",
        f"token-mixer-init:{_tag_value(token_mixer_residual_init_scale)}",
        f"critic-hidden:{_tag_value(critic_hidden_tag)}",
        f"local-map-area-scale:{_tag_value(getattr(config, 'local_map_area_scale', 1.0))}",
        f"dump-min-free-fraction:{_tag_value(dump_min_free_fraction)}",
        f"move-tiles:{_tag_value(env_defaults.agent.move_tiles)}",
        f"dig-radius-tiles:{_tag_value(env_defaults.agent.dig_radius_tiles)}",
        "edge-align:on" if edge_align_enabled else "edge-align:off",
        "terminal:digdump60-inner20-edge20",
        "terminal-fallback:digdump60-dig40",
    ]

    slurm_job_id = os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOBID")
    if slurm_job_id:
        tags.append(f"job:{_tag_value(slurm_job_id)}")

    slurm_gpu_count = os.getenv("SLURM_GPUS_ON_NODE") or os.getenv("SLURM_GPUS")
    if slurm_gpu_count:
        tags.append(f"gpus:{_tag_value(slurm_gpu_count)}")
    else:
        tags.append(f"gpus:{_tag_value(config.num_devices)}")

    sampler_config = config.pooled_sampler or {}
    if sampler_config.get("enabled", False):
        tags.append(f"sampler:{_tag_value(sampler_config.get('rule', 'uniform'))}")
    if config.accepted_bank is not None:
        if config.warm_start_from is not None:
            initialization = "params-only-warm"
        elif config.resume_from is not None:
            initialization = "resume"
        else:
            initialization = "scratch"
        tags.extend(
            (
                f"accepted-arm:{_tag_value(config.accepted_bank.arm)}",
                (
                    "terra-revision:"
                    f"{_tag_value(config.accepted_bank.terra_revision[:12])}"
                ),
                "bank:terra-curriculum-loader-v1",
                f"init:{initialization}",
            )
        )
        sampler_profile = getattr(config.accepted_bank, "sampler_profile", None)
        curriculum_stage = getattr(config.accepted_bank, "curriculum_stage", None)
        if sampler_profile == "continuous_banded_v1":
            tags.append("support:all47-continuous")
        elif curriculum_stage is not None:
            tags.append("curriculum-stage:" f"{_tag_value(curriculum_stage)}")
        if sampler_profile is not None:
            tags.append(f"sampler-profile:{_tag_value(sampler_profile)}")

    if config.curriculum_levels_override:
        if len(config.curriculum_levels_override) > 8:
            tags.append(f"map:per-condition-x{len(config.curriculum_levels_override)}")
        else:
            for level in config.curriculum_levels_override:
                tags.append(f"map:{_tag_value(level['maps_path'])}")
    else:
        tags.append("map:default")

    if config.single_map_path is not None:
        tags.append(f"single-map:{_tag_value(Path(config.single_map_path).stem)}")

    return list(dict.fromkeys(tags))


def train_mixed_agents(config: MixedAgentTrainConfig):
    """Main training function for mixed agents - with full feature parity to original train.py"""

    wandb_config = asdict(config)
    wandb_config["logging_schema"] = LOGGING_SCHEMA
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=wandb_config,
        save_code=True,
        tags=_wandb_tags_for_config(config),
    )
    wandb.define_metric("train/update")
    for metric in sorted(TRAINING_SCALAR_KEYS - {"train/update"}):
        wandb.define_metric(metric, step_metric="train/update")
    for metric in (
        "online_eval/completed_episode_success_rate",
        "online_eval/success_within_horizon_rate",
        "online_eval/termination_within_horizon_rate",
    ):
        wandb.define_metric(metric, step_metric="train/update")

    # Log source files - same as original train.py
    train_py_path = os.path.abspath(__file__)
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "terra", "terra", "config.py"
    )
    models_path = os.path.join(os.path.dirname(__file__), "utils", "models.py")

    code_artifact = wandb.Artifact(name="mixed_agent_source_code", type="code")

    for file_path, name in [
        (train_py_path, "train_mixed_agents.py"),
        (config_path, "config.py"),
        (models_path, "models.py"),
    ]:
        if os.path.exists(file_path):
            code_artifact.add_file(file_path, name=name)

    if code_artifact.files:
        run.log_artifact(code_artifact)

    # Optionally load checkpoint before creating states
    checkpoint = None
    env_params_override = None
    resume_update = 0
    checkpoint_mode = _checkpoint_load_mode(config)
    checkpoint_path = (
        config.warm_start_from
        if checkpoint_mode == "warm_start"
        else config.resume_from
    )
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        try:
            checkpoint = helpers.load_pkl_object(checkpoint_path)
            if "model" not in checkpoint:
                raise KeyError("checkpoint has no 'model' parameters")
            _validate_checkpoint_architecture(checkpoint, config)
            if (
                checkpoint_mode == "resume"
                and config.load_env_from_checkpoint
                and "env_config" in checkpoint
            ):
                env_params_override = _strip_checkpoint_env_axis(
                    checkpoint["env_config"],
                    config.num_envs_per_device,
                )
            if checkpoint_mode == "resume":
                if "next_update" in checkpoint:
                    resume_update = int(checkpoint["next_update"])
                elif "update" in checkpoint:
                    resume_update = int(checkpoint["update"]) + 1
                if config.resume_update is not None:
                    resume_update = int(config.resume_update)
                print(f"Loaded resume checkpoint from {checkpoint_path}")
            else:
                print(
                    "Loaded parameters-only warm start from "
                    f"{checkpoint_path}; optimizer, update counter, "
                    "environment, curriculum, RNG, and histories are fresh."
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}"
            ) from e
    # Initialize training components (optionally with env override)
    rng, env, env_params, train_state = make_mixed_agent_states(
        config, env_params_override=env_params_override
    )
    if checkpoint is not None:
        _validate_checkpoint_history_width(checkpoint, config)

    # If checkpoint has model params, overwrite initialized params
    if checkpoint is not None and "model" in checkpoint:
        try:
            train_state = train_state.replace(params=checkpoint["model"])
            print("Replaced model parameters from checkpoint.")
            if checkpoint_mode == "resume" and "optimizer_state" in checkpoint:
                train_state = train_state.replace(
                    opt_state=checkpoint["optimizer_state"],
                    step=checkpoint.get("train_state_step", train_state.step),
                )
                print(
                    "Restored optimizer state from checkpoint "
                    f"(next_update={resume_update})."
                )
                print(
                    "Environment, RNG, and action-history state restart on resume; "
                    "the continuation is not bit-exact."
                )
            elif checkpoint_mode == "resume":
                if config.resume_update is None:
                    resume_update = 0
                print(
                    "Checkpoint has no optimizer_state/update metadata; "
                    "warm-starting params only with a fresh optimizer/schedule."
                )
                if config.resume_update is not None:
                    print(
                        "Using manual resume_update="
                        f"{resume_update} for logging/entropy schedule."
                    )
            else:
                if int(jax.device_get(train_state.step)) != 0:
                    raise RuntimeError(
                        "parameters-only warm start did not keep a fresh optimizer step"
                    )
                print(
                    "Verified parameters-only warm start: train_state.step=0 "
                    "and PPO update range starts at 0."
                )
        except Exception as e:
            raise RuntimeError("Failed to restore checkpoint training state") from e
    _validate_resume_update(resume_update, config.num_updates)

    # F7: kickstart distillation teacher setup. Built after
    # make_mixed_agent_states so config.num_prev_actions is finalized. The
    # teacher architecture comes from ITS OWN train_config; the student keeps
    # the current run config. Frozen teacher params are closure-captured by the
    # pmapped update below (~1M params replicated per device).
    teacher_apply_fn = None
    teacher_params = None
    if config.teacher_checkpoint is not None:
        if not os.path.exists(config.teacher_checkpoint):
            raise FileNotFoundError(
                f"Teacher checkpoint does not exist: {config.teacher_checkpoint}"
            )
        # F6: teachers may have been saved by train.py (config pickled as
        # __main__.TrainConfig); alias those classes so the pkl unpickles here.
        helpers.register_checkpoint_config_classes()
        teacher_ckpt = helpers.load_pkl_object(config.teacher_checkpoint)
        if "model" not in teacher_ckpt:
            raise KeyError("teacher checkpoint has no 'model' parameters")
        teacher_train_config = teacher_ckpt.get("train_config")
        if teacher_train_config is None:
            raise KeyError("teacher checkpoint has no 'train_config'")

        teacher_num_prev_actions = _checkpoint_config_value(
            teacher_ckpt, "num_prev_actions", None
        )
        if teacher_num_prev_actions is None or int(teacher_num_prev_actions) != int(
            config.num_prev_actions
        ):
            raise ValueError(
                "Teacher/student action-history width mismatch: teacher "
                f"num_prev_actions={teacher_num_prev_actions}, student "
                f"num_prev_actions={config.num_prev_actions}. The teacher must "
                "share the student's environment interface."
            )

        # F3: the teacher forward runs on obs preprocessed with the STUDENT
        # config (obs_to_model_input in ppo_update_networks), then optionally
        # transformed to the teacher's native resolution. Fields that change raw
        # map/local-map preprocessing must match. loaded_max is deliberately
        # handled below: it is model-side normalization of raw loaded tile count,
        # and cross-resolution kickstart expects teacher/student values to differ
        # by the area scale.
        # Missing teacher fields fall back to the class defaults.
        def _normalize_bound(value):
            return tuple(value) if isinstance(value, (list, tuple)) else value

        preprocessing_fields = (
            ("clip_action_maps", MixedAgentTrainConfig.clip_action_maps),
            (
                "maps_net_normalization_bounds",
                MixedAgentTrainConfig.maps_net_normalization_bounds,
            ),
            (
                "local_map_normalization_bounds",
                MixedAgentTrainConfig.local_map_normalization_bounds,
            ),
        )
        preprocessing_mismatches = []
        for field_name, field_default in preprocessing_fields:
            teacher_value = _normalize_bound(
                _checkpoint_config_value(teacher_ckpt, field_name, field_default)
            )
            student_value = _normalize_bound(getattr(config, field_name, field_default))
            if teacher_value != student_value:
                preprocessing_mismatches.append(
                    f"{field_name}: teacher={teacher_value!r}, student={student_value!r}"
                )
        if preprocessing_mismatches:
            raise ValueError(
                "Teacher/student observation-preprocessing mismatch: "
                + "; ".join(preprocessing_mismatches)
                + ". The teacher is evaluated on obs preprocessed with the "
                "student config, so these fields must match."
            )

        # F15: cross-resolution kickstart. The teacher model is built from ITS
        # own train_config (its native world), but its FORWARD runs on the
        # student's obs subsampled by teacher_obs_downsample. Validate that the
        # teacher's native edge length times the factor equals the student env's
        # edge length, so the downsampled obs land exactly in-distribution. This
        # also rejects downsample != 1 when the teacher already matches the
        # student resolution (teacher_edge * N would then exceed student_edge).
        downsample = int(config.teacher_obs_downsample)
        student_edge = int(env.batch_cfg.maps_dims.maps_edge_length)
        teacher_edge = _teacher_maps_edge_length(teacher_ckpt)
        teacher_loaded_max = int(
            _checkpoint_config_value(
                teacher_ckpt, "loaded_max", MixedAgentTrainConfig.loaded_max
            )
        )
        student_loaded_max = int(config.loaded_max)
        if downsample != 1:
            if teacher_edge is None:
                raise ValueError(
                    "--teacher_obs_downsample requires the teacher checkpoint's "
                    "map edge length, but its env_config has no usable "
                    "maps.edge_length_px. Re-save the teacher with its env_config."
                )
            if teacher_edge * downsample != student_edge:
                raise ValueError(
                    "Teacher obs-downsampling mismatch: teacher maps_edge_length="
                    f"{teacher_edge} * downsample={downsample} != student "
                    f"maps_edge_length={student_edge}. The subsampled teacher obs "
                    "would be out of distribution."
                )
            expected_loaded_max = teacher_loaded_max * downsample * downsample
            if student_loaded_max != expected_loaded_max:
                raise ValueError(
                    "Cross-resolution kickstart loaded_max mismatch: teacher "
                    f"loaded_max={teacher_loaded_max} and downsample={downsample} "
                    f"imply student loaded_max={expected_loaded_max}, but the "
                    f"student has loaded_max={student_loaded_max}. Pass "
                    f"--loaded_max_override {expected_loaded_max} so the student "
                    "normalizes loaded tile-counts at the 128 resolution while "
                    "the teacher obs transform divides loaded by downsample**2."
                )
            print(
                f"🔻 Teacher obs-downsampling x{downsample}: student edge "
                f"{student_edge} -> teacher edge {teacher_edge}.",
                flush=True,
            )
        elif teacher_edge is not None and teacher_edge != student_edge:
            # Same-resolution kickstart (downsample==1) but the teacher was
            # trained at a different edge length: the teacher forward would see
            # OOD obs. Point at --teacher_obs_downsample instead of failing late.
            raise ValueError(
                f"Teacher maps_edge_length={teacher_edge} != student "
                f"maps_edge_length={student_edge} with --teacher_obs_downsample=1. "
                "Set --teacher_obs_downsample to student_edge/teacher_edge for "
                "cross-resolution kickstart."
            )
        elif student_loaded_max != teacher_loaded_max:
            raise ValueError(
                "Teacher/student loaded_max mismatch for same-resolution "
                f"kickstart: teacher={teacher_loaded_max}, "
                f"student={student_loaded_max}."
            )
        rng, rng_teacher = jax.random.split(rng)
        teacher_model_env = _teacher_model_env_from_checkpoint(teacher_ckpt, env)
        teacher_model, _ = get_model_ready(
            rng_teacher, teacher_train_config, teacher_model_env
        )
        teacher_apply_fn = teacher_model.apply
        teacher_params = teacher_ckpt["model"]
        print(
            "🎓 Kickstart teacher loaded from "
            f"{config.teacher_checkpoint} "
            f"(map_encoder={_checkpoint_config_value(teacher_ckpt, 'map_encoder', 'atari')}, "
            f"model_size={_checkpoint_config_value(teacher_ckpt, 'model_size', 'base')})",
            flush=True,
        )
        if float(config.ent_schedule_start) > 0.05:
            print(
                "⚠️  Kickstart teacher is set but ent_schedule_start="
                f"{config.ent_schedule_start} > 0.05. A low entropy schedule "
                "(e.g. 0.02 -> 0.005) is recommended so distillation is not "
                "swamped by exploration entropy.",
                flush=True,
            )

    def make_mixed_agent_train(
        env, env_params, config, teacher_apply_fn=None, teacher_params=None
    ):
        family_names = tuple(env.maps_buffer.family_names)
        primary_cell_names = tuple(env.maps_buffer.primary_cell_names)
        stage_names = tuple(
            level["maps_path"] for level in env.batch_cfg.curriculum_global.levels
        )
        include_trench_reward = any(
            bool(level["apply_trench_rewards"])
            for level in env.batch_cfg.curriculum_global.levels
        )
        num_stages = len(stage_names)
        num_families = len(family_names)
        num_primary_cells = len(primary_cell_names)
        aggregate_group_count = num_stages * num_families * num_primary_cells * 4
        sampler_settings = pooled_sampler_settings(config)
        pooled_sampler = None
        if sampler_settings is not None:
            if config.accepted_bank is None:
                raise ValueError(
                    "pooled condition sampling is supported only through "
                    "--accepted-bank-root"
                )
            bank = config.accepted_bank
            level_paths = tuple(
                level["maps_path"] for level in env.batch_cfg.curriculum_global.levels
            )
            expected_paths = tuple(level.maps_path for level in bank.levels)
            if level_paths != expected_paths:
                raise ValueError(
                    "accepted-bank level order changed between validation and "
                    "environment construction"
                )
            labels = accepted_bank_sampler_labels(bank, sampler_settings)
            pooled_sampler = PooledConditionSampler(
                [level.condition_id for level in bank.levels],
                sampler_settings,
                maps_per_condition=[level.map_count for level in bank.levels],
                labels=labels,
            )
            print(
                "📊 Condition sampler: "
                f"{sampler_settings.rule}, {len(bank.levels)} conditions, "
                f"{bank.map_count_per_condition} maps/condition",
                flush=True,
            )
        if config.reward_stage == "annealed_objective" and pooled_sampler is None:
            raise ValueError(
                "annealed_objective requires an enabled continuous_banded_v1 sampler"
            )
        _restore_pooled_sampler_checkpoint(
            pooled_sampler,
            checkpoint,
            checkpoint_mode,
        )
        if pooled_sampler is not None and checkpoint_mode == "resume":
            print("📊 Restored pooled condition sampler state.", flush=True)
        reward_anneal_state = _restore_reward_anneal_checkpoint(
            config.reward_stage,
            checkpoint,
            checkpoint_mode,
            resume_update,
        )
        if reward_anneal_state is not None and checkpoint_mode == "resume":
            print("🎯 Restored reward anneal state.", flush=True)

        def train(rng: jax.Array, train_state: TrainState):
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(
                _rng, config.num_envs_per_device * config.num_devices
            )
            reset_rng = reset_rng.reshape(
                (config.num_devices, config.num_envs_per_device, -1)
            )

            reset_env_params = env_params
            if pooled_sampler is not None:
                initial_levels = pooled_sampler.sample_levels(
                    (config.num_devices, config.num_envs_per_device)
                )
                pooled_sampler.observe_reset_exposures(
                    np.bincount(
                        initial_levels.reshape(-1),
                        minlength=len(pooled_sampler.names),
                    )
                )
                reset_env_params = assign_curriculum_levels(
                    env_params,
                    initial_levels,
                )

            # TERRA: Reset envs
            (
                env_params_reset,
                target_maps,
                padding_masks,
                trench_axes,
                trench_type,
                foundation_border_axes,
                foundation_border_type,
                dumpability_mask_init,
                action_maps,
                distance_maps,
            ) = env.prepare_reset(reset_env_params, reset_rng)
            reset_fn_p = jax.pmap(env.reset_prepared, axis_name="devices")
            timestep = reset_fn_p(
                env_params_reset,
                reset_rng,
                target_maps,
                padding_masks,
                trench_axes,
                trench_type,
                foundation_border_axes,
                foundation_border_type,
                dumpability_mask_init,
                action_maps,
                distance_maps,
            )
            timestep = assert_initial_env_steps_zero(timestep)
            # Removed one-time debug sanity prints

            # Initialize reward_components in timestep.info to maintain consistent pytree structure
            # This prevents JAX scan errors when reward_components is added later
            if hasattr(timestep, "info") and isinstance(timestep.info, dict):
                # Add empty reward_components to match the structure produced in env.step/state._get_reward
                # Shapes follow timestep.reward's batch shape; agent vectors add a MAX_AGENTS axis (4)
                batch_shape = timestep.reward.shape
                MAX_AGENTS = 4
                dummy_components = {
                    "agent_rewards": jnp.zeros(
                        batch_shape + (MAX_AGENTS,), dtype=jnp.float32
                    ),
                    "agent_active": jnp.zeros(
                        batch_shape + (MAX_AGENTS,), dtype=jnp.int32
                    ),
                    "num_agents": jnp.zeros(batch_shape, dtype=jnp.int32),
                    "terminal": jnp.zeros_like(timestep.reward),
                    "trench": jnp.zeros_like(timestep.reward),
                    "existence": jnp.zeros_like(timestep.reward),
                    "dig_completion_edge": jnp.zeros_like(timestep.reward),
                    "dig_completion_inner": jnp.zeros_like(timestep.reward),
                    "dig_completion_total": jnp.zeros_like(timestep.reward),
                    "dig_completion_min_edge_inner": jnp.zeros_like(timestep.reward),
                    "dump_completion_action_map": jnp.zeros_like(timestep.reward),
                    "total_dig_dump_completion": jnp.zeros_like(timestep.reward),
                    "absolute_completion": jnp.zeros_like(timestep.reward),
                    "unloaded_completion": jnp.zeros_like(timestep.reward),
                    "task_present": jnp.zeros_like(timestep.reward),
                    "dump_mask_integrity": jnp.zeros_like(timestep.reward),
                    "accepted_dump_volume": jnp.zeros_like(timestep.reward),
                    "illegal_dump_volume": jnp.zeros_like(timestep.reward),
                    "remaining_edge_dig_tiles": jnp.zeros_like(timestep.reward),
                    "remaining_inner_dig_tiles": jnp.zeros_like(timestep.reward),
                    "workspace_efficiency": jnp.zeros_like(timestep.reward),
                    "step_efficiency": jnp.zeros_like(timestep.reward),
                }
                # Create new timestep with reward_components added to info
                timestep = timestep._replace(
                    info={**timestep.info, "reward_components": dummy_components}
                )
            prev_actions = jnp.zeros(
                (
                    config.num_devices,
                    config.num_envs_per_device,
                    config.num_prev_actions,
                ),
                dtype=jnp.int32,
            )
            prev_reward = jnp.zeros((config.num_devices, config.num_envs_per_device))
            provenance_fn = jax.vmap(jax.vmap(env.maps_buffer.get_map_provenance))
            (
                _,
                initial_family_id,
                initial_primary_cell_id,
                _,
            ) = provenance_fn(reset_rng, env_params_reset)
            episode_accumulator = new_episode_accumulator(
                initial_family_id,
                initial_primary_cell_id,
                env_params_reset.curriculum.level,
            )
            pending_aggregate_single = empty_episode_aggregate(aggregate_group_count)
            pending_aggregate = jtu.tree_map(
                lambda value: jnp.broadcast_to(
                    value,
                    (config.num_devices,) + value.shape,
                ),
                pending_aggregate_single,
            )

            # TRAIN LOOP
            @partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
            def _update_step(
                runner_state,
                ent_coef_current,
                kickstart_kl_coef_current,
                kickstart_value_coef_current,
                flush_episode_aggregate,
            ):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, step_idx):
                    (
                        rng,
                        train_state,
                        prev_timestep,
                        prev_actions,
                        prev_reward,
                        episode_accumulator,
                        pending_aggregate,
                    ) = runner_state

                    # SELECT ACTION
                    rng, _rng_model, _rng_env = jax.random.split(rng, 3)
                    action, log_prob, value, _ = select_action_ppo(
                        train_state,
                        prev_timestep.observation,
                        prev_actions,
                        _rng_model,
                        config,
                    )

                    # STEP ENV
                    _rng_env = jax.random.split(_rng_env, config.num_envs_per_device)
                    action_env = wrap_action(action, env.batch_cfg.action_type)
                    timestep = env.step(prev_timestep, action_env, _rng_env)
                    reward_components = timestep.info["reward_components"]
                    (
                        _,
                        next_family_id,
                        next_primary_cell_id,
                        _,
                    ) = jax.vmap(
                        env.maps_buffer.get_map_provenance
                    )(_rng_env, timestep.env_cfg)
                    episode_step = EpisodeStep(
                        done=timestep.done,
                        task_done=timestep.info["task_done"],
                        timeout=timestep.info["timeout"],
                        reward=timestep.reward,
                        agent_rewards=reward_components["agent_rewards"],
                        terminal_reward=reward_components["terminal"],
                        trench_reward=reward_components["trench"],
                        existence_reward=reward_components["existence"],
                        reward_normalizer=(prev_timestep.env_cfg.rewards.normalizer),
                        action=action,
                        action_had_effect=timestep.info["action_had_effect"],
                        productive_workspace_cycle=timestep.info[
                            "productive_workspace_cycle"
                        ],
                        transition_mass_residual=timestep.info[
                            "transition_mass_residual"
                        ],
                        target_mutation=timestep.info["target_mutation"],
                        obstacle_mutation=timestep.info["obstacle_mutation"],
                        dig_completion=reward_components["dig_completion_total"],
                        dump_purity=reward_components["dump_completion_action_map"],
                        dump_volume_completion=reward_components[
                            "total_dig_dump_completion"
                        ],
                        combined_completion=reward_components["absolute_completion"],
                        unloaded_completion=reward_components["unloaded_completion"],
                        accepted_dump_volume=reward_components["accepted_dump_volume"],
                        illegal_dump_volume=reward_components["illegal_dump_volume"],
                    )
                    active_curriculum_level = episode_accumulator.stage_id
                    episode_accumulator, pending_aggregate = update_episode_aggregate(
                        episode_accumulator,
                        pending_aggregate,
                        episode_step,
                        next_family_id=next_family_id,
                        next_primary_cell_id=next_primary_cell_id,
                        next_stage_id=(timestep.env_cfg.curriculum.level),
                        num_stages=num_stages,
                        num_families=num_families,
                        num_primary_cells=num_primary_cells,
                    )

                    # Removed SWAP debug prints
                    transition = Transition(
                        done=timestep.done,
                        task_done=timestep.info["task_done"],
                        curriculum_level=timestep.env_cfg.curriculum.level,
                        active_curriculum_level=active_curriculum_level,
                        action=action,
                        value=value,
                        reward=timestep.reward,
                        terminal_reward=reward_components["terminal"],
                        dig_completion_edge=reward_components["dig_completion_edge"],
                        dig_completion_inner=reward_components["dig_completion_inner"],
                        dig_completion_total=reward_components["dig_completion_total"],
                        dig_completion_min_edge_inner=reward_components[
                            "dig_completion_min_edge_inner"
                        ],
                        dump_completion_action_map=reward_components[
                            "dump_completion_action_map"
                        ],
                        total_dig_dump_completion=reward_components[
                            "total_dig_dump_completion"
                        ],
                        remaining_edge_dig_tiles=reward_components[
                            "remaining_edge_dig_tiles"
                        ],
                        remaining_inner_dig_tiles=reward_components[
                            "remaining_inner_dig_tiles"
                        ],
                        transition_mass_residual=timestep.info[
                            "transition_mass_residual"
                        ],
                        target_mutation=timestep.info["target_mutation"],
                        obstacle_mutation=timestep.info["obstacle_mutation"],
                        log_prob=log_prob,
                        obs=prev_timestep.observation,
                        prev_actions=prev_actions,
                        prev_reward=prev_reward,
                    )

                    # UPDATE PREVIOUS ACTIONS
                    prev_actions = jnp.roll(prev_actions, shift=1, axis=-1)
                    prev_actions = prev_actions.at[..., 0].set(action)
                    prev_actions = jnp.where(
                        timestep.done[..., None],
                        jnp.zeros_like(prev_actions),
                        prev_actions,
                    )

                    runner_state = (
                        rng,
                        train_state,
                        timestep,
                        prev_actions,
                        timestep.reward,
                        episode_accumulator,
                        pending_aggregate,
                    )
                    return runner_state, transition

                # transitions: [seq_len, batch_size, ...]
                runner_state, transitions = jax.lax.scan(
                    _env_step, runner_state, None, config.num_steps
                )
                transition_integrity = {
                    "maximum_mass_residual": jax.lax.pmax(
                        jnp.max(transitions.transition_mass_residual),
                        "devices",
                    ),
                    "target_mutation_count": jax.lax.psum(
                        jnp.sum(transitions.target_mutation.astype(jnp.int32)),
                        "devices",
                    ),
                    "obstacle_mutation_count": jax.lax.psum(
                        jnp.sum(transitions.obstacle_mutation.astype(jnp.int32)),
                        "devices",
                    ),
                }
                reset_exposure_count = jax.lax.psum(
                    reset_exposure_histogram(
                        transitions.done,
                        transitions.curriculum_level,
                        num_stages,
                    ),
                    "devices",
                )
                transition_exposure_count = jax.lax.psum(
                    transition_exposure_histogram(
                        transitions.active_curriculum_level,
                        num_stages,
                    ),
                    "devices",
                )

                # Share terminal credit with preceding same-episode agent turns.
                done_seq = transitions.done  # [seq, batch]
                reward_seq = transitions.reward  # [seq, batch]

                # Get num_agents per env (assumed constant across sequence); shape [batch]
                # transitions.obs stores prev_timestep.observation
                num_agents_per_env = transitions.obs["num_agents"][0]  # [batch]
                # Clip to supported window 1..MAX_AGENTS
                MAX_AGENTS = 4
                num_agents_per_env = jnp.clip(
                    num_agents_per_env.astype(jnp.int32), 1, MAX_AGENTS
                )

                augmented_reward = _backfill_terminal_rewards(
                    reward_seq,
                    transitions.terminal_reward,
                    done_seq,
                    num_agents_per_env,
                    max_agents=MAX_AGENTS,
                )
                transitions = transitions.replace(reward=augmented_reward)

                # CALCULATE ADVANTAGE
                (
                    rng,
                    train_state,
                    timestep,
                    prev_actions,
                    prev_reward,
                    episode_accumulator,
                    pending_aggregate,
                ) = runner_state
                rng, _rng = jax.random.split(rng)
                _, _, last_val, _ = select_action_ppo(
                    train_state, timestep.observation, prev_actions, _rng, config
                )
                advantages, targets = calculate_gae(
                    transitions, last_val, config.gamma, config.gae_lambda
                )

                # UPDATE NETWORK
                def _update_epoch(update_state, _):
                    def _update_minbatch(train_state, batch_info):
                        transitions, advantages, targets = batch_info
                        new_train_state, update_info = ppo_update_networks(
                            train_state=train_state,
                            transitions=transitions,
                            advantages=advantages,
                            targets=targets,
                            config=config,
                            ent_coef_override=ent_coef_current,
                            teacher_apply_fn=teacher_apply_fn,
                            teacher_params=teacher_params,
                            kickstart_kl_coef=kickstart_kl_coef_current,
                            kickstart_value_coef=kickstart_value_coef_current,
                        )
                        return new_train_state, update_info

                    rng, train_state, transitions, advantages, targets = update_state

                    # MINIBATCHES PREPARATION
                    rng, _rng = jax.random.split(rng)
                    if getattr(config, "flat_minibatch_shuffle", False):
                        # F6: collapse [seq_len, batch_size] into a single sample
                        # axis, permute over ALL samples, then reshape into
                        # minibatches. GAE was computed above and is unaffected;
                        # ppo_update_networks skips the [mb, seq] reshape for the
                        # resulting flat layout.
                        # [seq_len, batch_size, ...]
                        batch = (transitions, advantages, targets)
                        n_samples = config.num_steps * config.num_envs_per_device
                        flat_batch = jtu.tree_map(
                            lambda x: jnp.reshape(x, (n_samples,) + x.shape[2:]),
                            batch,
                        )
                        permutation = jax.random.permutation(_rng, n_samples)
                        shuffled_batch = jtu.tree_map(
                            lambda x: jnp.take(x, permutation, axis=0), flat_batch
                        )
                        # [num_minibatches, minibatch_size, ...] (no seq axis)
                        minibatches = jtu.tree_map(
                            lambda x: jnp.reshape(
                                x, (config.num_minibatches, -1) + x.shape[1:]
                            ),
                            shuffled_batch,
                        )
                    else:
                        permutation = jax.random.permutation(
                            _rng, config.num_envs_per_device
                        )
                        # [seq_len, batch_size, ...]
                        batch = (transitions, advantages, targets)
                        # [batch_size, seq_len, ...], as our model assumes
                        batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                        shuffled_batch = jtu.tree_map(
                            lambda x: jnp.take(x, permutation, axis=0), batch
                        )
                        # [num_minibatches, minibatch_size, seq_len, ...]
                        minibatches = jtu.tree_map(
                            lambda x: jnp.reshape(
                                x, (config.num_minibatches, -1) + x.shape[1:]
                            ),
                            shuffled_batch,
                        )
                    train_state, update_info = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )

                    update_state = (rng, train_state, transitions, advantages, targets)
                    return update_state, update_info

                # [seq_len, batch_size, num_layers, hidden_dim]
                update_state = (rng, train_state, transitions, advantages, targets)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config.update_epochs
                )

                # averaging over minibatches then over epochs
                loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

                # Explained variance between value predictions and returns
                # Use transitions and targets from current update_state (first device in pmap)
                _, _, transitions_ev, _, targets_ev = update_state
                vpred = transitions_ev.value
                vtrue = targets_ev
                vpred_flat = vpred.reshape(-1)
                vtrue_flat = vtrue.reshape(-1)
                var_y = jnp.var(vtrue_flat)
                explained_var = 1 - jnp.var(vtrue_flat - vpred_flat) / (var_y + 1e-8)
                # Attach to loss_info for logging
                loss_info = dict(loss_info)
                loss_info["explained_variance"] = explained_var

                rng, train_state = update_state[:2]
                # EVALUATE AGENT
                rng, _rng = jax.random.split(rng)

                aggregate_snapshot = reduce_episode_aggregate(
                    pending_aggregate,
                    "devices",
                )
                pending_aggregate = jax.lax.cond(
                    flush_episode_aggregate,
                    lambda _: empty_episode_aggregate(aggregate_group_count),
                    lambda aggregate: aggregate,
                    pending_aggregate,
                )
                runner_state = (
                    rng,
                    train_state,
                    timestep,
                    prev_actions,
                    prev_reward,
                    episode_accumulator,
                    pending_aggregate,
                )
                return (
                    runner_state,
                    loss_info,
                    aggregate_snapshot,
                    transition_integrity,
                    reset_exposure_count,
                    transition_exposure_count,
                )

            # Setup runner state for multiple devices
            rng, rng_rollout = jax.random.split(rng)
            rng = jax.random.split(rng, num=config.num_devices)
            train_state = replicate(
                train_state, jax.local_devices()[: config.num_devices]
            )
            runner_state = (
                rng,
                train_state,
                timestep,
                prev_actions,
                prev_reward,
                episode_accumulator,
                pending_aggregate,
            )

            # Entropy scheduler: cosine decay using config variables
            ent_start = float(config.ent_schedule_start)
            ent_end = float(config.ent_schedule_end)
            ent_T = float(config.ent_schedule_steps)

            for i in tqdm(range(resume_update, config.num_updates), desc="Training"):
                need_train_log = (
                    config.log_train_interval > 0 and i % config.log_train_interval == 0
                )
                need_checkpoint = (
                    config.checkpoint_interval > 0
                    and (i + 1) % config.checkpoint_interval == 0
                )
                need_eval = (
                    config.log_eval_interval > 0
                    and i > 0
                    and i % config.log_eval_interval == 0
                )
                need_final_state = i == config.num_updates - 1
                need_finite_check = (
                    config.fail_on_nonfinite
                    and config.finite_check_interval > 0
                    and i % config.finite_check_interval == 0
                )
                need_episode_flush = (
                    need_train_log
                    or need_checkpoint
                    or need_final_state
                    or pooled_sampler is not None
                )
                need_host_state = need_episode_flush or need_eval or need_finite_check
                f = min(1.0, i / ent_T) if ent_T > 0 else 1.0
                # Cosine decay: starts at ent_start when f=0, ends at ent_end when f=1
                ent_coef_current = ent_end + 0.5 * (ent_start - ent_end) * (
                    1.0 + jnp.cos(jnp.pi * f)
                )
                # Linear decay from ent_start to ent_end over ent_T updates
                # ent_coef_current = ent_start + (ent_end - ent_start) * f
                # Broadcast scalar to devices for pmap input
                ent_broadcast = jnp.array([ent_coef_current] * config.num_devices)
                # F7: host-computed cosine-annealed kickstart coefficients,
                # broadcast to devices like the entropy coefficient. Zero (and
                # inert) whenever no teacher is configured.
                if teacher_apply_fn is not None:
                    kickstart_kl_coef_current = kickstart_coef_schedule(
                        i, config.kickstart_kl_coef, config.kickstart_kl_anneal_updates
                    )
                    kickstart_value_coef_current = kickstart_coef_schedule(
                        i,
                        config.kickstart_value_coef,
                        config.kickstart_value_anneal_updates,
                    )
                else:
                    kickstart_kl_coef_current = 0.0
                    kickstart_value_coef_current = 0.0
                kickstart_kl_broadcast = jnp.array(
                    [kickstart_kl_coef_current] * config.num_devices
                )
                kickstart_value_broadcast = jnp.array(
                    [kickstart_value_coef_current] * config.num_devices
                )
                flush_episode_broadcast = jnp.array(
                    [need_episode_flush] * config.num_devices,
                    dtype=jnp.bool_,
                )
                sampler_refreshed = False
                if pooled_sampler is not None:
                    pooled_sampler.start(i)
                    if pooled_sampler.due(i):
                        pooled_sampler.refresh(i)
                        sampler_refreshed = True
                    if maybe_start_reward_anneal(
                        reward_anneal_state,
                        pooled_sampler.receipt(),
                        i,
                    ):
                        print(
                            "🎯 Reward fade started at update "
                            f"{i + 1}; dense -> terminal over "
                            f"{REWARD_ANNEAL_DURATION_UPDATES} updates.",
                            flush=True,
                        )
                    next_env_cfg = assign_curriculum_levels(
                        runner_state[2].env_cfg,
                        pooled_sampler.sample_levels(
                            (config.num_devices, config.num_envs_per_device)
                        ),
                    )
                    runner_state = (
                        runner_state[0],
                        runner_state[1],
                        runner_state[2]._replace(env_cfg=next_env_cfg),
                        *runner_state[3:],
                    )
                current_reward_mix = (
                    1.0
                    if config.reward_stage == "terminal_objective"
                    else reward_anneal_mix(reward_anneal_state, i)
                )
                if reward_anneal_state is not None:
                    reward_anneal_state["last_applied_mix"] = current_reward_mix
                    mixed_env_cfg = assign_terminal_reward_mix(
                        runner_state[2].env_cfg,
                        current_reward_mix,
                    )
                    runner_state = (
                        runner_state[0],
                        runner_state[1],
                        runner_state[2]._replace(env_cfg=mixed_env_cfg),
                        *runner_state[3:],
                    )
                start_time = time.time()
                (
                    runner_state,
                    loss_info,
                    episode_aggregate_snapshot,
                    transition_integrity,
                    reset_exposure_count,
                    transition_exposure_count,
                ) = jax.block_until_ready(
                    _update_step(
                        runner_state,
                        ent_broadcast,
                        kickstart_kl_broadcast,
                        kickstart_value_broadcast,
                        flush_episode_broadcast,
                    )
                )
                transition_integrity_single = unreplicate(transition_integrity)
                _assert_transition_integrity(transition_integrity_single)
                reset_exposure_single = None
                transition_exposure_single = None
                if pooled_sampler is not None:
                    reset_exposure_single = np.asarray(
                        unreplicate(reset_exposure_count)
                    )
                    pooled_sampler.observe_reset_exposures(reset_exposure_single)
                    transition_exposure_single = np.asarray(
                        unreplicate(transition_exposure_count)
                    )
                    pooled_sampler.observe_transition_exposures(
                        transition_exposure_single
                    )
                end_time = time.time()

                iteration_duration = end_time - start_time
                iterations_per_second = 1 / iteration_duration
                steps_per_second = iterations_per_second * config.env_steps_per_update

                tqdm.write(f"Steps/s: {steps_per_second:.2f}")

                if need_host_state:
                    loss_info_single = unreplicate(loss_info)
                    runner_state_single = unreplicate(runner_state)
                    _, _, timestep, prev_actions = runner_state_single[:4]
                    env_params_single = timestep.env_cfg
                if need_episode_flush:
                    aggregate_single = unreplicate(episode_aggregate_snapshot)
                    episode_payload = aggregate_to_payload(
                        aggregate_single,
                        family_names=family_names,
                        primary_cell_names=primary_cell_names,
                        stage_names=stage_names,
                        update=i + 1,
                        run_name=config.name,
                    )
                    assert_aggregate_integrity(episode_payload)
                    if need_train_log or need_checkpoint or need_final_state:
                        _write_episode_aggregate_receipt(
                            config,
                            episode_payload,
                        )
                    if pooled_sampler is not None:
                        pooled_sampler.observe_episode_payload(episode_payload)

                if config.fail_on_nonfinite and (
                    need_finite_check or need_checkpoint or need_final_state
                ):
                    _assert_finite_loss_info(loss_info_single, i)

                need_condition_snapshot = pooled_sampler is not None and (
                    sampler_refreshed or need_checkpoint or need_final_state
                )
                if need_train_log or need_condition_snapshot:
                    log_dict = {}
                    active_levels = None
                    if pooled_sampler is not None:
                        # Unlike replicated losses, each device owns different
                        # environment assignments. Preserve that device axis.
                        active_levels = np.asarray(
                            jax.device_get(runner_state[2].env_cfg.curriculum.level)
                        )
                    if need_train_log:
                        log_dict.update(
                            {
                                "train/update": i + 1,
                                "system/steps_per_second": steps_per_second,
                                "system/environment_steps": (i + 1)
                                * config.env_steps_per_update,
                                **episode_metrics(
                                    episode_payload,
                                    include_trench_reward=include_trench_reward,
                                ),
                                **loss_metrics(
                                    loss_info_single,
                                    entropy_coef=float(ent_coef_current),
                                    teacher_enabled=teacher_apply_fn is not None,
                                    kickstart_kl_coef=kickstart_kl_coef_current,
                                    kickstart_value_coef=kickstart_value_coef_current,
                                ),
                                "reward/terminal_objective_mix": current_reward_mix,
                            }
                        )
                        if pooled_sampler is not None:
                            log_dict.update(
                                curriculum_metrics(
                                    active_levels,
                                    names=pooled_sampler.names,
                                    labels=pooled_sampler.labels,
                                    probabilities=pooled_sampler.probabilities,
                                    refreshes=pooled_sampler.refreshes,
                                )
                            )
                    if need_condition_snapshot:
                        log_dict["curriculum/conditions"] = wandb.Table(
                            columns=list(CONDITION_COLUMNS),
                            data=condition_rows(
                                active_levels,
                                reset_exposure_single,
                                transition_exposure_single,
                                episode_payload,
                                names=pooled_sampler.names,
                                labels=pooled_sampler.labels,
                                probabilities=pooled_sampler.probabilities,
                            ),
                        )

                    scalar_keys = set(log_dict) - {"curriculum/conditions"}
                    unexpected = scalar_keys - TRAINING_SCALAR_KEYS
                    if unexpected or len(scalar_keys) > len(TRAINING_SCALAR_KEYS):
                        raise RuntimeError(
                            "human W&B scalar contract violated: "
                            f"unexpected={sorted(unexpected)}, "
                            f"count={len(scalar_keys)}"
                        )
                    wandb.log(log_dict)

                if need_checkpoint:
                    if config.fail_on_nonfinite:
                        _assert_finite_tree(
                            runner_state_single[1].params,
                            f"model params before checkpoint update {i}",
                        )
                        _assert_finite_tree(
                            runner_state_single[1].opt_state,
                            f"optimizer state before checkpoint update {i}",
                        )
                    env_config_checkpoint = _strip_checkpoint_env_axis(
                        env_params_single,
                        config.num_envs_per_device,
                    )
                    checkpoint = {
                        "checkpoint_version": 2,
                        "train_config": config,
                        "env_config": env_config_checkpoint,
                        "model": runner_state_single[1].params,
                        "optimizer_state": runner_state_single[1].opt_state,
                        "train_state_step": runner_state_single[1].step,
                        "update": i,
                        "next_update": i + 1,
                        "loss_info": loss_info_single,
                        "transition_integrity": {
                            key: int(np.asarray(value))
                            for key, value in (transition_integrity_single.items())
                        },
                    }
                    if pooled_sampler is not None:
                        checkpoint["pooled_sampler_state"] = pooled_sampler.state_dict()
                    if reward_anneal_state is not None:
                        checkpoint["reward_anneal_state"] = dict(reward_anneal_state)
                    checkpoint_name = f"{config.name}.pkl"
                    if config.keep_checkpoint_history:
                        checkpoint_name = f"{config.name}_update_{i + 1:06d}.pkl"
                    helpers.save_pkl_object(
                        checkpoint,
                        str(Path(config.checkpoint_dir) / checkpoint_name),
                    )
                    if pooled_sampler is not None:
                        receipt = pooled_sampler.receipt()
                        receipt_contract = {
                            "update": i + 1,
                            "run_name": config.name,
                            "accepted_bank_arm": config.accepted_bank.arm,
                            "terra_revision": (config.accepted_bank.terra_revision),
                            "accepted_bank_root": str(config.accepted_bank.root),
                            "environment_protocol_sha256": (
                                config.accepted_bank.environment_protocol_sha256
                            ),
                            "sampler_profile": (config.accepted_bank.sampler_profile),
                        }
                        if config.accepted_bank.sampler_profile == (
                            "continuous_banded_v1"
                        ):
                            receipt_contract.update(
                                {
                                    "support_scope": "all47_continuous",
                                    "curriculum_graph_sha256": (
                                        config.accepted_bank.curriculum_graph_sha256
                                    ),
                                }
                            )
                        else:
                            receipt_contract["curriculum_stage"] = (
                                config.accepted_bank.curriculum_stage
                            )
                        receipt.update(receipt_contract)
                        receipt_dir = Path(config.checkpoint_dir) / "pooled_sampler"
                        receipt_dir.mkdir(parents=True, exist_ok=True)
                        receipt_path = receipt_dir / (
                            f"{config.name}_update_{i + 1:06d}.json"
                        )
                        receipt_path.write_text(
                            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
                        )

                if need_eval:
                    # Reuse the training reset shape regime and keep only the
                    # rollout loop outside XLA. This avoids the separate
                    # env.reset compile that can crash on RTX 4090 eval.
                    print(
                        f"🧪 Starting pmapped step-wise eval at update {i}", flush=True
                    )
                    rng_eval_base = jax.random.fold_in(rng_rollout, i)
                    rng_eval = jax.random.split(rng_eval_base, config.num_devices)
                    reset_rng_eval = jax.random.split(
                        jax.random.fold_in(rng_eval_base, 1),
                        config.num_devices * config.num_envs_per_device,
                    ).reshape((config.num_devices, config.num_envs_per_device, -1))
                    (
                        eval_env_params_reset,
                        eval_target_maps,
                        eval_padding_masks,
                        eval_trench_axes,
                        eval_trench_type,
                        eval_foundation_border_axes,
                        eval_foundation_border_type,
                        eval_dumpability_mask_init,
                        eval_action_maps,
                        eval_distance_maps,
                    ) = env.prepare_reset(runner_state[2].env_cfg, reset_rng_eval)
                    reset_fn_p = jax.pmap(env.reset_prepared, axis_name="devices")
                    eval_timestep = reset_fn_p(
                        eval_env_params_reset,
                        reset_rng_eval,
                        eval_target_maps,
                        eval_padding_masks,
                        eval_trench_axes,
                        eval_trench_type,
                        eval_foundation_border_axes,
                        eval_foundation_border_type,
                        eval_dumpability_mask_init,
                        eval_action_maps,
                        eval_distance_maps,
                    )
                    eval_stats = eval_ppo.rollout_from_timestep(
                        rng_eval,
                        env,
                        eval_timestep,
                        runner_state[1],
                        config,
                    )
                    eval_stats = eval_ppo.aggregate_device_stats(eval_stats)
                    print(
                        f"🧪 Finished pmapped step-wise eval at update {i}", flush=True
                    )

                    total_eval_envs = config.num_devices * config.num_envs_per_device
                    wandb.log(
                        {
                            "train/update": i + 1,
                            "online_eval/completed_episode_success_rate": (
                                eval_ppo.episode_success_rate(
                                    eval_stats.positive_terminations,
                                    eval_stats.terminations,
                                )
                            ),
                            "online_eval/success_within_horizon_rate": (
                                eval_stats.initial_episode_successes / total_eval_envs
                            ),
                            "online_eval/termination_within_horizon_rate": (
                                eval_stats.initial_episode_terminations
                                / total_eval_envs
                            ),
                        }
                    )

                # Clear JAX caches and run garbage collection to stabilize memory use
                if (
                    config.cache_clear_interval > 0
                    and (i + 1) % config.cache_clear_interval == 0
                ):
                    jax.clear_caches()
                    import gc

                    gc.collect()

            return {
                "runner_state": runner_state_single,
                "loss_info": loss_info_single,
                "transition_integrity": {
                    key: int(np.asarray(value))
                    for key, value in transition_integrity_single.items()
                },
                "pooled_sampler_state": (
                    pooled_sampler.state_dict() if pooled_sampler is not None else None
                ),
                "reward_anneal_state": (
                    dict(reward_anneal_state)
                    if reward_anneal_state is not None
                    else None
                ),
            }

        return train

    train_fn = make_mixed_agent_train(
        env, env_params, config, teacher_apply_fn, teacher_params
    )

    def train_with_monitoring(rng, train_state):
        return train_fn(rng, train_state)

    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   - Environments per device: {config.num_envs_per_device}")
    print(f"   - Total environments: {config.num_envs}")
    print(f"   - Training steps: {config.num_steps}")
    print(f"   - Total timesteps: {config.total_timesteps:,}")
    print(f"   - Learning rate: {config.lr}")
    print(f"   - log_train_interval: {config.log_train_interval}")
    print(f"   - log_eval_interval: {config.log_eval_interval}")
    print(f"   - checkpoint_interval: {config.checkpoint_interval}")
    print(f"   - checkpoint_dir: {config.checkpoint_dir}")
    print(f"   - keep_checkpoint_history: {config.keep_checkpoint_history}")
    enforce_border_alignment = bool(
        jnp.ravel(env_params.enforce_foundation_border_alignment)[0]
    )
    enable_reachability_obs = bool(jnp.ravel(env_params.enable_reachability_obs)[0])
    foundation_dump_min_free_fraction = float(
        jnp.ravel(
            getattr(env_params, "foundation_dump_min_free_fraction", jnp.array(0.0))
        )[0]
    )
    print(f"   - enforce_foundation_border_alignment: {enforce_border_alignment}")
    print(
        f"   - foundation_dump_min_free_fraction: {foundation_dump_min_free_fraction}"
    )
    print(f"   - enable_reachability_obs: {enable_reachability_obs}")

    print("=" * 60)
    print("🚀 Starting Mixed Agent Training...")
    print(
        "⚙️  JAX is now compiling the control-flow graph. This is normal and taking a few minutes...",
        flush=True,
    )

    try:
        t = time.time()
        train_info = jax.block_until_ready(train_with_monitoring(rng, train_state))
        elapsed_time = time.time() - t
        print(f"✅ Mixed agent training completed in {elapsed_time:.2f}s")

        # Save final checkpoint with special naming - enhanced metadata
        try:
            at_final = train_info["runner_state"][2].env_cfg.agent_types
            if hasattr(at_final, "shape") and len(at_final.shape) > 1:
                a1 = int(jnp.mean(at_final[0, :, 0]))
                a2 = int(jnp.mean(at_final[0, :, 1]))
            else:
                a1 = int(at_final[0])
                a2 = int(at_final[1])
            type_names = {0: "Excavator", 1: "Truck", 2: "SkidSteer"}
            agent_types_str = (
                f"{type_names.get(a1, 'unknown')}_{type_names.get(a2, 'unknown')}"
            )
        except Exception:
            agent_types_str = "unknown_unknown"

        final_env_config = _strip_checkpoint_env_axis(
            train_info["runner_state"][2].env_cfg,
            config.num_envs_per_device,
        )
        final_train_state = train_info["runner_state"][1]
        if config.fail_on_nonfinite:
            _assert_finite_loss_info(train_info["loss_info"], config.num_updates - 1)
            _assert_finite_tree(final_train_state.params, "final model params")
            _assert_finite_tree(final_train_state.opt_state, "final optimizer state")
        final_checkpoint = {
            "checkpoint_version": 2,
            "train_config": config,
            "env_config": final_env_config,
            "model": final_train_state.params,
            "optimizer_state": final_train_state.opt_state,
            "train_state_step": final_train_state.step,
            "update": config.num_updates - 1,
            "next_update": config.num_updates,
            "loss_info": train_info["loss_info"],
            "transition_integrity": train_info["transition_integrity"],
            "agent_types": agent_types_str,
            "network_type": "unified_with_agent_type_conditioning",
            "training_duration": elapsed_time,
            "final_reward": train_info.get("final_reward", None),
        }
        if train_info["pooled_sampler_state"] is not None:
            final_checkpoint["pooled_sampler_state"] = train_info[
                "pooled_sampler_state"
            ]
        if train_info["reward_anneal_state"] is not None:
            final_checkpoint["reward_anneal_state"] = train_info["reward_anneal_state"]
        final_path = Path(config.checkpoint_dir) / f"{config.name}_FINAL.pkl"
        helpers.save_pkl_object(final_checkpoint, str(final_path))
        print(f"💾 Final mixed agent model saved to {final_path}")

    except KeyboardInterrupt:
        print("⏹️ Training interrupted. Finalizing...")
    finally:
        run.finish()
        print("📈 Wandb session finished.")


if __name__ == "__main__":
    DT = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    import argparse

    parser = argparse.ArgumentParser(
        description="Train mixed agent policies (Tracked + Skid Steer)"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="mixed-agents-skidsteer-skidsteer",
        help="Experiment name",
    )
    parser.add_argument(
        "--exact_run_name",
        action="store_true",
        help=(
            "Use --name verbatim. Intended for a true resume whose fixed-bank "
            "treatment identity must match its source checkpoint."
        ),
    )
    parser.add_argument(
        "-m", "--machine", type=str, default="local", help="Machine identifier"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training initialization and rollout seed.",
    )
    parser.add_argument(
        "-d",
        "--num_devices",
        type=int,
        default=0,
        help="Number of devices to use. If 0, uses all available devices.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--num_envs_per_device",
        type=int,
        default=1024,
        help="Number of parallel envs per device",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=50_000_000_000,
        help="Total environment timesteps across all devices",
    )
    parser.add_argument(
        "--num_steps", type=int, default=32, help="Rollout length per PPO update"
    )
    parser.add_argument(
        "--update_epochs", type=int, default=2, help="Number of PPO epochs per rollout"
    )
    parser.add_argument(
        "--num_minibatches",
        type=int,
        default=16,
        help="Number of minibatches per PPO epoch",
    )
    parser.add_argument(
        "--log_train_interval",
        type=int,
        default=1,
        help="Training metric logging interval in PPO updates.",
    )
    parser.add_argument(
        "--log_eval_interval",
        type=int,
        default=100,
        help="Eval logging interval in PPO updates. Set 0 to disable inline eval.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Checkpoint save interval in PPO updates.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for periodic and final checkpoints.",
    )
    parser.add_argument(
        "--keep_checkpoint_history",
        action="store_true",
        help="Write update-numbered periodic checkpoints instead of overwriting one file.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=100,
        help="Requested evaluation episode count (evaluation is currently step-limited).",
    )
    parser.add_argument(
        "--cache_clear_interval",
        type=int,
        default=1000,
        help="JAX cache-clear interval in updates; set 0 to disable.",
    )
    parser.add_argument(
        "--ent_schedule_start",
        type=float,
        default=0.15,
        help="Initial entropy coefficient for the cosine schedule.",
    )
    parser.add_argument(
        "--ent_schedule_end",
        type=float,
        default=0.005,
        help="Final entropy coefficient for the cosine schedule.",
    )
    parser.add_argument(
        "--ent_schedule_steps",
        type=int,
        default=9500,
        help="Number of PPO updates over which to cosine-anneal entropy.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=["base", "medium", "large"],
        help="Model capacity preset. 'medium' and 'large' progressively widen CNN and policy/value heads.",
    )
    parser.add_argument(
        "--model_core",
        type=str,
        default="mlp",
        choices=["mlp", "transformer"],
        help="Core policy architecture. 'mlp' keeps current behavior; 'transformer' uses a lightweight token-mixer core.",
    )
    parser.add_argument(
        "--map_encoder",
        type=str,
        default="atari",
        choices=sorted(MAP_ENCODER_ALIASES),
        help=(
            "Global-map encoder. Use 'resnet_global_pool' for the PR #15 "
            "topology or 'resnet_spatial_8x8' for the scaled spatial readout. "
            "'resnet_spatial_8x8_se' (alias 'resnet_spatial_v3') adds derived "
            "channels, coordinates, and SE gates. The old names remain accepted "
            "as compatibility aliases."
        ),
    )
    parser.add_argument(
        "--encoder_compute_dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help=(
            "Compute dtype for the spatial ResNet encoder. 'bfloat16' halves "
            "encoder memory bandwidth while keeping float32 params/loss math. "
            "Only valid with the spatial ResNet encoders."
        ),
    )
    parser.add_argument(
        "--attention_compute_dtype",
        type=str,
        default="encoder",
        choices=["encoder", "float32", "bfloat16"],
        help=(
            "Compute dtype for v4/v5 attention submodules. 'encoder' preserves "
            "current behavior; 'float32' keeps attention logits/softmax/projections "
            "in f32 while the conv trunk can run with --encoder_compute_dtype bfloat16."
        ),
    )
    parser.add_argument(
        "--token_mixer_residual_init_scale",
        type=float,
        default=0.0,
        help=(
            "Optional small nonzero kernel-init scale for v5 token-mixer residual "
            "projections. 0.0 preserves exact identity-at-init behavior."
        ),
    )
    parser.add_argument(
        "--critic_hidden_dims",
        type=str,
        default=None,
        help=(
            "Comma-separated critic-head widths, e.g. '512,256'. Overrides the "
            "model_size preset's value head. Omit to keep the preset."
        ),
    )
    # F5: value-clipping toggle.
    parser.add_argument(
        "--no_value_clip",
        dest="use_value_clip",
        action="store_false",
        help=(
            "Disable PPO value clipping and use a plain 0.5*MSE value loss. "
            "Default keeps the clipped-value objective."
        ),
    )
    parser.set_defaults(use_value_clip=True)
    # F6: flat env x time minibatch shuffling.
    parser.add_argument(
        "--flat_minibatch_shuffle",
        action="store_true",
        help=(
            "Flatten [seq, env] into a single sample axis and permute over all "
            "samples per minibatch, instead of the default env-only shuffle."
        ),
    )
    # F7: kickstart distillation.
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a teacher checkpoint .pkl for kickstart distillation. When "
            "unset the feature is disabled entirely."
        ),
    )
    parser.add_argument(
        "--kickstart_kl_coef",
        type=float,
        default=1.0,
        help="Initial KL(teacher||student) coefficient for kickstart distillation.",
    )
    parser.add_argument(
        "--kickstart_kl_anneal_updates",
        type=int,
        default=1500,
        help="Cosine-anneal the kickstart KL coefficient to 0 over this many updates.",
    )
    parser.add_argument(
        "--kickstart_value_coef",
        type=float,
        default=0.5,
        help="Initial value-distillation coefficient for kickstart distillation.",
    )
    parser.add_argument(
        "--kickstart_value_anneal_updates",
        type=int,
        default=500,
        help="Cosine-anneal the kickstart value coefficient to 0 over this many updates.",
    )
    parser.add_argument(
        "--kickstart_lr_warmup_updates",
        type=int,
        default=100,
        help=(
            "Linear LR warmup from lr/3 to lr over this many updates, applied "
            "only when --teacher_checkpoint is set."
        ),
    )
    # F15: 128x128 resolution scaling. All default None/1 = no change.
    parser.add_argument(
        "--agent_move_tiles",
        type=int,
        default=None,
        help="Override agent.move_tiles (tiles of progress per move; x2 at 128).",
    )
    parser.add_argument(
        "--dig_radius_tiles",
        type=int,
        default=None,
        help="Override agent.dig_radius_tiles (workspace/cone reach; x2 at 128).",
    )
    parser.add_argument(
        "--truck_capacity",
        type=int,
        default=None,
        help="Override truck_capacity (tile-count volume; x4 at 128). Overrides --config.",
    )
    parser.add_argument(
        "--skidsteer_capacity",
        type=int,
        default=None,
        help="Override skidsteer_capacity (tile-count volume; x4 at 128). Overrides --config.",
    )
    parser.add_argument(
        "--loaded_max_override",
        type=int,
        default=None,
        help="Override loaded_max, the model-side loaded normalizer (x4 at 128).",
    )
    parser.add_argument(
        "--local_map_area_scale",
        type=float,
        default=1.0,
        help=(
            "Divide local workspace-sum observations by this factor before model "
            "preprocessing. Use 4.0 for 128x128 runs that double tile resolution."
        ),
    )
    parser.add_argument(
        "--reward_normalizer",
        type=float,
        default=None,
        help="Override rewards.normalizer (per-tile reward scaling; 70->280 at 128).",
    )
    parser.add_argument(
        "--resnet_stage_channels",
        type=str,
        default=None,
        help=(
            "Comma-separated spatial-ResNet stage channels, e.g. '16,32,48,64,64'. "
            "Overrides the model_size preset. A 5th stride-2 stage keeps 128 inputs "
            "at an 8x8 readout. Omit to keep the preset."
        ),
    )
    parser.add_argument(
        "--resnet_blocks_per_stage",
        type=str,
        default=None,
        help=(
            "Comma-separated spatial-ResNet blocks per stage, e.g. '1,1,2,2,2'. "
            "Overrides the model_size preset. Omit to keep the preset."
        ),
    )
    parser.add_argument(
        "--teacher_obs_downsample",
        type=int,
        default=1,
        help=(
            "Downsample factor for the teacher forward path only (cross-resolution "
            "kickstart). 1 = off. 2 subsamples global maps and halves agent "
            "positions/dims so a 64-world teacher sees a 128-world student's obs. "
            "Only valid with --teacher_checkpoint."
        ),
    )
    parser.add_argument(
        "--agent_types",
        type=str,
        default=None,  # 0=excavator, 1=truck, 2=skidsteer
        help="Override agent types with a Python tuple, e.g. '(2,0,2,0)'. Overrides --config.",
    )
    parser.add_argument(
        "--action_types",
        type=str,
        default=None,
        help="Override action types with a Python tuple, e.g. '(1,)' for wheeled. Overrides --config.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable one-time sanity assertions/prints for agent ordering and masks",
    )
    parser.add_argument(
        "--fail_on_nonfinite",
        action="store_true",
        help=(
            "Fail the run when core PPO/kickstart losses, gradient diagnostics, "
            "params, or optimizer state become non-finite."
        ),
    )
    parser.add_argument(
        "--finite_check_interval",
        type=int,
        default=0,
        help=(
            "Host-side finite-check interval in PPO updates. 0 disables unless "
            "--fail_on_nonfinite is set, which checks every update."
        ),
    )
    # Named configuration preset
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Load a named training config preset (e.g., 'excavator_truck', 'solo_excavator'). "
        "Run 'python configs/training_configs.py' to see available presets.",
    )
    parser.add_argument(
        "--accepted-bank-root",
        type=Path,
        default=None,
        help=(
            "Required by F-ANCHOR, F-SPECIALIST, T-ANCHOR, T-SPECIALIST, "
            "G-UNIFORM and G-ADAPTIVE. "
            "Must contain a frozen terra_curriculum_loader_bank_v1 dataset.json."
        ),
    )
    parser.add_argument(
        "--terra-revision",
        type=str,
        default=None,
        help=(
            "Exact immutable Terra revision bound into the accepted bank. "
            "Required by accepted-bank configs because source archives do not "
            "contain .git metadata."
        ),
    )
    accepted_scope = parser.add_mutually_exclusive_group()
    accepted_scope.add_argument(
        "--accepted-bank-stage",
        choices=("capability", "nearby", "full"),
        default=None,
        help=(
            "Legacy hard-stage bank selector. New continuous runs use "
            "--accepted-bank-scope full."
        ),
    )
    accepted_scope.add_argument(
        "--accepted-bank-scope",
        dest="accepted_bank_stage",
        choices=("full",),
        help="Select all 47 V8 conditions for one continuous full-support run.",
    )
    parser.add_argument(
        "--accepted-bank-sampler-profile",
        choices=(
            "bank_v4",
            "bounded_replay25_v1",
            "banded_preview15_v1",
            "continuous_banded_v1",
        ),
        default=None,
        help=(
            "Named V8 population contract. continuous_banded_v1 retains "
            "positive support on all 47 conditions while shifting mass by "
            "family depth."
        ),
    )
    parser.add_argument(
        "--pooled-sampler-interval",
        type=int,
        default=None,
        help="Override the adaptive sampler refresh interval for a smoke only.",
    )
    parser.add_argument(
        "--pooled-sampler-min-episodes",
        type=int,
        default=None,
        help=(
            "Override the minimum completed episodes per condition/window for "
            "a smoke only."
        ),
    )
    parser.add_argument(
        "--map_path",
        type=str,
        default=None,
        help=(
            "Optional single-map folder/file path for training. "
            "By default this enables pure single-map training. "
            "If --replay_map_count and --target_map_repeat are also set, "
            "the map is mixed with recent config-dataset maps instead."
        ),
    )
    parser.add_argument(
        "--replay_map_count",
        type=int,
        default=0,
        help=(
            "When used with --map_path, keep the last N maps from each config dataset "
            "and mix them with repeated copies of the target map."
        ),
    )
    parser.add_argument(
        "--target_map_repeat",
        type=int,
        default=0,
        help=(
            "When used with --map_path and --replay_map_count, add the target map this many "
            "times to the mixed training pool."
        ),
    )

    parser.add_argument(
        "--relocation_progress_mult",
        type=float,
        default=None,
        help=(
            "Agent-neutral multiplier for signed relocation progress "
            "(overrides config preset)"
        ),
    )
    parser.add_argument(
        "--reward_stage",
        choices=("dense_skill", "annealed_objective", "terminal_objective"),
        default="dense_skill",
        help=(
            "Global reward objective. annealed_objective starts a one-way "
            "5000-update dense-to-terminal fade after both continuous-sampler "
            "families reach curriculum depth two."
        ),
    )
    # Checkpoint loading arguments
    parser.add_argument(
        "-r",
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint .pkl to resume training from.",
    )
    parser.add_argument(
        "--warm_start_from",
        type=str,
        default=None,
        help=(
            "Load model parameters only. Optimizer, update counter, "
            "environment, curriculum, RNG, and histories start fresh."
        ),
    )
    parser.add_argument(
        "--resume_update",
        type=int,
        default=None,
        help=(
            "Manual next update index for old checkpoints that only contain "
            "model params. New checkpoints store this automatically."
        ),
    )
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument(
        "--load_env_from_checkpoint",
        dest="load_env_from_checkpoint",
        action="store_true",
        help="Load env_config from the checkpoint (default).",
    )
    env_group.add_argument(
        "--no-load-env-from-checkpoint",
        dest="load_env_from_checkpoint",
        action="store_false",
        help="Do not load env_config from checkpoint; use default/current EnvConfig().",
    )

    args, unknown = parser.parse_known_args()

    # Common mistake: `--preset_name` instead of `--config preset_name`
    if args.config is None and unknown:
        try:
            from configs.training_configs import list_configs

            known_configs = set(list_configs())
            for token in list(unknown):
                candidate = token.lstrip("-")
                if candidate in known_configs:
                    print(
                        f"⚠️  Treating {token!r} as --config {candidate} "
                        "(use --config <name> for presets)"
                    )
                    args.config = candidate
                    unknown.remove(token)
        except ImportError:
            pass
    if unknown:
        raise SystemExit(
            f"Unrecognized arguments: {unknown}. "
            "Load a YAML preset with --config <name> (e.g. --config solo_excavator_rectangles_2stage)."
        )

    # default to True unless explicitly disabled
    if args.load_env_from_checkpoint is None:
        args.load_env_from_checkpoint = True

    # Initialize config values from preset if --config is provided
    agent_types_override = None
    action_types_override = None
    relocation_progress_mult = None
    truck_capacity = None
    skidsteer_capacity = None
    truck_road_restricted = None
    enforce_foundation_border_alignment = None
    curriculum_levels_override = None
    curriculum_increase_level_threshold = None
    curriculum_decrease_level_threshold = None
    curriculum_last_level_type = None
    pooled_sampler_override = None
    accepted_bank_arm = None
    accepted_bank = None

    if args.config is not None:
        try:
            from configs.training_configs import get_config, list_configs

            preset = get_config(args.config)
            print(f"\n📦 Loading config preset: '{args.config}'")
            print(f"   Description: {preset.description}")

            # Apply preset values
            agent_types_override = preset.agent_types
            action_types_override = preset.action_types

            relocation_progress_mult = preset.relocation_progress_mult
            accepted_bank_arm = preset.accepted_bank_arm
            sampler = preset.pooled_sampler
            pooled_sampler_override = {
                field: getattr(sampler, field) for field in sampler.__dataclass_fields__
            }
            curriculum_increase_level_threshold = (
                preset.curriculum.increase_level_threshold
            )
            curriculum_decrease_level_threshold = (
                preset.curriculum.decrease_level_threshold
            )
            curriculum_last_level_type = preset.curriculum.last_level_type

            # Apply capacity overrides from preset
            truck_capacity = preset.truck_capacity
            skidsteer_capacity = preset.skidsteer_capacity
            truck_road_restricted = preset.truck_road_restricted
            enforce_foundation_border_alignment = (
                preset.enforce_foundation_border_alignment
            )

            # Apply maps/curriculum from preset (convert MapLevel objects to dict format)
            if preset.maps and len(preset.maps) > 0:
                from terra.config import RewardsType

                curriculum_levels_override = []
                for map_level in preset.maps:
                    # Convert rewards_type string to enum
                    rewards_type = (
                        RewardsType.DENSE
                        if map_level.rewards_type == "DENSE"
                        else RewardsType.SPARSE
                    )
                    curriculum_levels_override.append(
                        {
                            "maps_path": map_level.maps_path,
                            "max_steps_in_episode": map_level.max_steps_in_episode,
                            "rewards_type": rewards_type,
                            "apply_trench_rewards": map_level.apply_trench_rewards,
                        }
                    )
        except ImportError as e:
            print(f"⚠️  Failed to import training configs: {e}")
            print("   Make sure configs/training_configs.py exists")
        except ValueError as e:
            print(f"⚠️  {e}")
            print(
                "   Run 'python configs/training_configs.py' to see available presets"
            )

    # Override with explicit CLI arguments (these take precedence over preset)
    if args.agent_types is not None:
        try:
            import ast

            parsed = ast.literal_eval(args.agent_types)
            # Normalize to a tuple of ints; accept tuple, list, or single int
            if isinstance(parsed, tuple):
                agent_types_override = tuple(int(x) for x in parsed)
            elif isinstance(parsed, list):
                agent_types_override = tuple(int(x) for x in parsed)
            elif isinstance(parsed, (int,)):
                agent_types_override = (int(parsed),)
            else:
                raise ValueError(
                    "--agent_types must be a tuple/list like (2,0,0,2) or a single int like (0)"
                )
            print(f"➡️  CLI override agent types: {agent_types_override}")
        except Exception as e:
            print(f"⚠️  Failed to parse --agent_types '{args.agent_types}': {e}")

    if args.action_types is not None:
        try:
            import ast

            parsed = ast.literal_eval(args.action_types)
            # Normalize to a tuple of ints; accept tuple, list, or single int
            if isinstance(parsed, tuple):
                action_types_override = tuple(int(x) for x in parsed)
            elif isinstance(parsed, list):
                action_types_override = tuple(int(x) for x in parsed)
            elif isinstance(parsed, (int,)):
                action_types_override = (int(parsed),)
            else:
                raise ValueError(
                    "--action_types must be a tuple/list like (0,1,0,1) or a single int like (0)"
                )
            print(f"➡️  CLI override action types: {action_types_override}")
        except Exception as e:
            print(f"⚠️  Failed to parse --action_types '{args.action_types}': {e}")

    if (
        args.replay_map_count > 0 or args.target_map_repeat > 0
    ) and args.map_path is None:
        raise ValueError("Mixed target-map replay requires --map_path.")
    if (
        args.replay_map_count > 0 or args.target_map_repeat > 0
    ) and args.config is None:
        raise ValueError(
            "Mixed target-map replay requires --config so the dataset source is defined."
        )
    if (args.replay_map_count > 0) != (args.target_map_repeat > 0):
        raise ValueError(
            "Set both --replay_map_count and --target_map_repeat to use mixed target-map replay."
        )

    if accepted_bank_arm is None:
        if args.accepted_bank_root is not None:
            raise ValueError(
                "--accepted-bank-root requires one of the accepted-bank configs: "
                + ", ".join(ACCEPTED_BANK_ARMS)
            )
        if (
            args.pooled_sampler_interval is not None
            or args.pooled_sampler_min_episodes is not None
        ):
            raise ValueError(
                "--pooled-sampler-* overrides require an accepted-bank config"
            )
        if args.terra_revision is not None:
            raise ValueError("--terra-revision requires an accepted-bank config")
        if args.accepted_bank_stage is not None:
            raise ValueError("--accepted-bank-stage requires an accepted-bank config")
        if args.accepted_bank_sampler_profile is not None:
            raise ValueError(
                "--accepted-bank-sampler-profile requires an accepted-bank config"
            )
    else:
        if accepted_bank_arm not in ACCEPTED_BANK_ARMS:
            raise ValueError(
                f"config {args.config!r} names unknown accepted-bank arm "
                f"{accepted_bank_arm!r}"
            )
        if args.accepted_bank_root is None:
            raise ValueError(f"--config {args.config} requires --accepted-bank-root")
        if args.terra_revision is None:
            raise ValueError(f"--config {args.config} requires --terra-revision")
        if curriculum_levels_override:
            raise ValueError("accepted-bank configs must not hard-code map paths")
        if args.map_path is not None:
            raise ValueError("accepted-bank configs cannot be combined with --map_path")
        # Checkpoint mode is an explicit treatment input. Parameters-only warm
        # starts keep a fresh optimizer and sampler. Resume restores compatible
        # optimizer and pooled-sampler state but cannot restore the exact JAX
        # environment trajectory, RNG, or action history.
        # Those invariants are validated by the common checkpoint-loading path.
        accepted_bank = load_accepted_bank(
            args.accepted_bank_root,
            accepted_bank_arm,
            args.terra_revision,
            curriculum_stage=args.accepted_bank_stage,
            sampler_profile=args.accepted_bank_sampler_profile,
        )
        os.environ["DATASET_PATH"] = str(accepted_bank.root)
        os.environ["DATASET_SIZE"] = str(accepted_bank.map_count_per_condition)
        from terra.config import RewardsType

        curriculum_levels_override = [
            {
                "maps_path": level.maps_path,
                "max_steps_in_episode": 450,
                "rewards_type": RewardsType.DENSE,
                "apply_trench_rewards": False,
            }
            for level in accepted_bank.levels
        ]
        if not pooled_sampler_override or not pooled_sampler_override.get(
            "enabled", False
        ):
            raise ValueError(
                f"accepted-bank config {args.config} must enable the pooled sampler"
            )
        if accepted_bank.sampler_profile == "continuous_banded_v1":
            expected_rule = "continuous_banded_v1"
        elif accepted_bank.sampling_probabilities:
            expected_rule = "fixed"
        else:
            expected_rule = (
                "adaptive" if accepted_bank_arm == "G-ADAPTIVE" else "uniform"
            )
        if pooled_sampler_override.get("rule") != expected_rule:
            raise ValueError(
                f"{accepted_bank_arm} requires sampler rule {expected_rule!r}"
            )
        if expected_rule != "adaptive" and (
            args.pooled_sampler_interval is not None
            or args.pooled_sampler_min_episodes is not None
        ):
            raise ValueError("sampler refresh overrides apply only to an adaptive arm")
        if args.pooled_sampler_interval is not None:
            pooled_sampler_override["update_interval"] = args.pooled_sampler_interval
        if args.pooled_sampler_min_episodes is not None:
            pooled_sampler_override["min_episodes"] = args.pooled_sampler_min_episodes
        pooled_sampler_override["seed"] = int(args.seed)
        print(
            f"📦 Accepted bank: {accepted_bank.root} | "
            f"{accepted_bank_arm} | {len(accepted_bank.levels)} conditions x "
            f"{accepted_bank.map_count_per_condition} maps | "
            f"sampler={accepted_bank.sampler_profile or 'release-default'}",
            flush=True,
        )

    if args.relocation_progress_mult is not None:
        relocation_progress_mult = args.relocation_progress_mult
    if accepted_bank_arm is not None and relocation_progress_mult != 1.5:
        raise ValueError(
            "accepted-bank map comparisons freeze relocation_progress_mult=1.5"
        )

    # F15: CLI capacity overrides take precedence over the preset values.
    if args.truck_capacity is not None:
        truck_capacity = args.truck_capacity
    if args.skidsteer_capacity is not None:
        skidsteer_capacity = args.skidsteer_capacity

    # Use default agent types if nothing was set
    if agent_types_override is None:
        agent_types_override = (0,)  # Default to single excavator
    if action_types_override is None:
        action_types_override = (0,)  # Default to tracked

    # Validate: skidsteers (agent_type=2) cannot use wheeled movement (action_type=1)
    for i in range(min(len(agent_types_override), len(action_types_override))):
        if agent_types_override[i] == 2 and action_types_override[i] == 1:
            raise ValueError(
                f"Agent {i}: Skidsteer (agent_type=2) does not support wheeled movement "
                f"(action_type=1). Skidsteers require tracked movement (action_type=0) "
                f"for auto-load, push-mode, and reverse-dump mechanics."
            )

    # Parse critic-head width override from a comma-separated string.
    def _parse_int_tuple(raw, flag_name, example):
        if raw is None:
            return None
        try:
            parsed = tuple(
                int(token.strip()) for token in raw.split(",") if token.strip()
            )
            if not parsed:
                raise ValueError("no values parsed")
            return parsed
        except ValueError as e:
            raise ValueError(
                f"Failed to parse {flag_name} '{raw}': {e}. "
                f"Use a comma-separated list like '{example}'."
            )

    critic_hidden_dims = _parse_int_tuple(
        args.critic_hidden_dims, "--critic_hidden_dims", "512,256"
    )
    # F15: spatial-ResNet stage overrides (comma lists -> tuples).
    resnet_stage_channels = _parse_int_tuple(
        args.resnet_stage_channels, "--resnet_stage_channels", "16,32,48,64,64"
    )
    resnet_blocks_per_stage = _parse_int_tuple(
        args.resnet_blocks_per_stage, "--resnet_blocks_per_stage", "1,1,2,2,2"
    )
    if (resnet_stage_channels is None) != (resnet_blocks_per_stage is None):
        raise ValueError(
            "Set both --resnet_stage_channels and --resnet_blocks_per_stage, or neither."
        )
    if resnet_stage_channels is not None and len(resnet_stage_channels) != len(
        resnet_blocks_per_stage
    ):
        raise ValueError(
            "--resnet_stage_channels and --resnet_blocks_per_stage must have equal "
            f"length (got {len(resnet_stage_channels)} vs {len(resnet_blocks_per_stage)})."
        )

    name = resolve_run_name(args.name, args.machine, DT, args.exact_run_name)

    config = MixedAgentTrainConfig(
        name=name,
        seed=args.seed,
        num_devices=args.num_devices,
        lr=args.lr,
        num_envs_per_device=args.num_envs_per_device,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        log_train_interval=args.log_train_interval,
        log_eval_interval=args.log_eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        keep_checkpoint_history=args.keep_checkpoint_history,
        cache_clear_interval=args.cache_clear_interval,
        ent_schedule_start=args.ent_schedule_start,
        ent_schedule_end=args.ent_schedule_end,
        ent_schedule_steps=args.ent_schedule_steps,
        resume_from=args.resume_from,
        warm_start_from=args.warm_start_from,
        resume_update=args.resume_update,
        load_env_from_checkpoint=args.load_env_from_checkpoint,
        agent_types_override=agent_types_override,
        action_types_override=action_types_override,
        debug=args.debug,
        fail_on_nonfinite=args.fail_on_nonfinite,
        finite_check_interval=args.finite_check_interval,
        config_name=args.config,
        relocation_progress_mult=relocation_progress_mult,
        reward_stage=args.reward_stage,
        truck_capacity=truck_capacity,
        skidsteer_capacity=skidsteer_capacity,
        truck_road_restricted=truck_road_restricted,
        enforce_foundation_border_alignment=enforce_foundation_border_alignment,
        curriculum_levels_override=curriculum_levels_override,
        curriculum_increase_level_threshold=curriculum_increase_level_threshold,
        curriculum_decrease_level_threshold=curriculum_decrease_level_threshold,
        curriculum_last_level_type=curriculum_last_level_type,
        pooled_sampler=pooled_sampler_override,
        accepted_bank=accepted_bank,
        single_map_path=args.map_path,
        replay_map_count=args.replay_map_count,
        target_map_repeat=args.target_map_repeat,
        model_size=args.model_size,
        model_core=args.model_core,
        map_encoder=args.map_encoder,
        encoder_compute_dtype=args.encoder_compute_dtype,
        attention_compute_dtype=args.attention_compute_dtype,
        token_mixer_residual_init_scale=args.token_mixer_residual_init_scale,
        critic_hidden_dims=critic_hidden_dims,
        use_value_clip=args.use_value_clip,
        flat_minibatch_shuffle=args.flat_minibatch_shuffle,
        teacher_checkpoint=args.teacher_checkpoint,
        kickstart_kl_coef=args.kickstart_kl_coef,
        kickstart_kl_anneal_updates=args.kickstart_kl_anneal_updates,
        kickstart_value_coef=args.kickstart_value_coef,
        kickstart_value_anneal_updates=args.kickstart_value_anneal_updates,
        kickstart_lr_warmup_updates=args.kickstart_lr_warmup_updates,
        # F15 resolution scaling (truck_capacity/skidsteer_capacity already
        # passed above; CLI precedence applied when parsing the preset).
        agent_move_tiles=args.agent_move_tiles,
        dig_radius_tiles=args.dig_radius_tiles,
        loaded_max_override=args.loaded_max_override,
        local_map_area_scale=args.local_map_area_scale,
        reward_normalizer=args.reward_normalizer,
        resnet_stage_channels=resnet_stage_channels,
        resnet_blocks_per_stage=resnet_blocks_per_stage,
        teacher_obs_downsample=args.teacher_obs_downsample,
    )

    train_mixed_agents(config)
