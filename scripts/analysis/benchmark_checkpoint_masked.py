#!/usr/bin/env python3
"""Reproducible local rollout benchmark for masked vs unmasked Terra checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from terra.actions import TrackedAction, WheeledAction
from terra.config import BatchConfig, CurriculumGlobalConfig, EnvConfig, RewardsType
from terra.env import TerraEnvBatch
from train import TrainConfig  # noqa: F401 - needed for unpickling old checkpoints
from train_mixed import MixedAgentTrainConfig
from utils.action_masking import apply_action_mask
from utils.helpers import load_pkl_object
from utils.models import infer_edge_features_dim_from_model_params, load_neural_network
from utils.utils_ppo import obs_to_model_input, wrap_action

sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig


def _take0(value):
    try:
        arr = jnp.asarray(value)
    except Exception:
        return value
    while arr.ndim >= 1 and arr.shape[0] > 1:
        arr = arr[0]
    while arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _unbatch_env_config(env_config: EnvConfig) -> EnvConfig:
    return jax.tree_util.tree_map(_take0, env_config)


def _as_int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(np.asarray(v).reshape(-1)[0]) for v in value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return (int(arr.item()),)
    return tuple(int(v) for v in arr.reshape(-1).tolist())


def _replicate_value(value, n_envs: int):
    if value is None:
        return None
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return jnp.asarray(value)[None, ...].repeat(n_envs, 0)
    if isinstance(value, (int, float, bool)):
        return jnp.asarray([value] * n_envs)
    arr = jnp.asarray(value)
    return arr[None, ...].repeat(n_envs, 0)


def _replicate_namedtuple(nt, n_envs: int):
    updates = {}
    for field in getattr(nt, "_fields", ()):
        value = getattr(nt, field)
        if hasattr(value, "_fields"):
            updates[field] = _replicate_namedtuple(value, n_envs)
        else:
            updates[field] = _replicate_value(value, n_envs)
    return nt._replace(**updates)


def _build_env_config(
    env_config: EnvConfig,
    config_name: str | None,
    n_envs: int,
    enforce_foundation_border_alignment: bool,
):
    action_types = _as_int_tuple(env_config.action_types)
    batch_cfg = BatchConfig(
        action_type=WheeledAction
        if action_types and all(t == 1 for t in action_types)
        else TrackedAction
    )

    if config_name is None:
        env_config = env_config._replace(
            enforce_foundation_border_alignment=enforce_foundation_border_alignment
        )
        return batch_cfg, _replicate_namedtuple(env_config, n_envs)

    from configs.training_configs import get_config

    preset = get_config(config_name)
    curriculum_levels = []
    for level in preset.maps:
        curriculum_levels.append(
            {
                "maps_path": level.maps_path,
                "max_steps_in_episode": level.max_steps_in_episode,
                "rewards_type": (
                    RewardsType.DENSE
                    if level.rewards_type == "DENSE"
                    else RewardsType.SPARSE
                ),
                "apply_trench_rewards": level.apply_trench_rewards,
            }
        )

    class CustomCurriculumGlobalConfig(CurriculumGlobalConfig):
        levels = curriculum_levels

    action_types = tuple(int(v) for v in preset.action_types)
    batch_cfg = BatchConfig(
        curriculum_global=CustomCurriculumGlobalConfig(),
        action_type=WheeledAction
        if action_types and all(t == 1 for t in action_types)
        else TrackedAction,
    )
    first_level = curriculum_levels[0]
    env_config = env_config._replace(
        agent_types=tuple(int(v) for v in preset.agent_types),
        action_types=action_types,
        max_steps_in_episode=int(first_level["max_steps_in_episode"]),
        apply_trench_rewards=bool(first_level["apply_trench_rewards"]),
        dump_bonus_mult=preset.reward_multipliers.dump_bonus_mult,
        excavator_relocate_dumped_mult=(
            preset.reward_multipliers.excavator_relocate_dumped_mult
        ),
        excavator_relocate_dug_dirt_mult=(
            preset.reward_multipliers.excavator_relocate_dug_dirt_mult
        ),
        transport_relocate_mult=preset.reward_multipliers.transport_relocate_mult,
        enforce_foundation_border_alignment=enforce_foundation_border_alignment,
    )
    return batch_cfg, _replicate_namedtuple(env_config, n_envs)


def _select_action(model, params, obs, prev_actions, config, use_mask: bool, rng):
    obs_model = obs_to_model_input(obs, prev_actions, config)
    _, logits = model.apply(params, obs_model)
    action_mask = obs_model[22]
    logits_for_action = apply_action_mask(logits, action_mask) if use_mask else logits
    if config.benchmark_stochastic:
        action = tfp.distributions.Categorical(logits=logits_for_action).sample(seed=rng)
    else:
        action = jnp.argmax(logits_for_action, axis=-1)
    return action.astype(jnp.int32), logits, action_mask


def _terrain_scores(obs, target_maps_init):
    reduce_axes = tuple(range(1, target_maps_init.ndim))
    dig_req = target_maps_init < 0
    undug = dig_req & (obs["action_map"] >= 0)
    total_dig_req = dig_req.sum(reduce_axes)
    dig_coverage = jnp.where(
        total_dig_req > 0,
        1.0 - undug.sum(reduce_axes) / jnp.maximum(total_dig_req, 1),
        1.0,
    )

    dump_req = target_maps_init > 0
    dumped_on_target = jnp.where(dump_req & (obs["action_map"] > 0), obs["action_map"], 0)
    dumped_total = jnp.where(obs["action_map"] > 0, obs["action_map"], 0).sum(reduce_axes)
    dump_coverage = jnp.where(
        dumped_total > 0,
        dumped_on_target.sum(reduce_axes) / jnp.maximum(dumped_total, 1),
        0.0,
    )
    return {
        "dig_coverage_mean": float(np.asarray(dig_coverage.mean())),
        "dig_coverage_std": float(np.asarray(dig_coverage.std())),
        "dump_coverage_mean": float(np.asarray(dump_coverage.mean())),
        "dump_coverage_std": float(np.asarray(dump_coverage.std())),
    }


def run_rollout(env, env_cfgs, model, params, config, args, use_mask: bool):
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    timestep = env.reset(env_cfgs, reset_keys)
    target_maps_init = timestep.observation["target_map"].copy()
    prev_actions = jnp.zeros((args.num_envs, config.num_prev_actions), dtype=jnp.int32)

    done_once = jnp.zeros((args.num_envs,), dtype=jnp.bool_)
    task_done_once = jnp.zeros((args.num_envs,), dtype=jnp.bool_)
    first_task_done_step = jnp.full((args.num_envs,), -1, dtype=jnp.int32)
    returns = jnp.zeros((args.num_envs,), dtype=jnp.float32)
    lengths = jnp.zeros((args.num_envs,), dtype=jnp.int32)
    invalid_selected = 0
    action_counts = jnp.zeros((env.batch_cfg.action_type.get_num_actions(),), dtype=jnp.int32)

    for step in range(args.num_steps):
        rng, rng_action, rng_env = jax.random.split(rng, 3)
        alive_before = ~done_once
        action, _, action_mask = _select_action(
            model, params, timestep.observation, prev_actions, config, use_mask, rng_action
        )
        selected_allowed = jnp.take_along_axis(action_mask, action[:, None], axis=-1)[:, 0]
        invalid_selected += int(np.asarray((~selected_allowed & alive_before).sum()))
        action_counts = action_counts + jnp.bincount(
            action, length=action_counts.shape[0]
        ).astype(jnp.int32)

        step_keys = jax.random.split(rng_env, args.num_envs)
        timestep = env.step(
            timestep,
            wrap_action(action, env.batch_cfg.action_type),
            step_keys,
        )

        returns = returns + timestep.reward * alive_before.astype(jnp.float32)
        lengths = lengths + alive_before.astype(jnp.int32)
        task_done = timestep.info["task_done"].astype(jnp.bool_)
        newly_task_done = task_done & ~task_done_once
        first_task_done_step = jnp.where(newly_task_done, step + 1, first_task_done_step)
        task_done_once = task_done_once | task_done
        done_once = done_once | timestep.done.astype(jnp.bool_)
        prev_actions = jnp.concatenate([action[:, None], prev_actions[:, :-1]], axis=-1)

        if bool(jnp.all(done_once)):
            break

    action_counts_np = np.asarray(action_counts)
    total_actions = max(1, int(action_counts_np.sum()))
    result = {
        "mode": "masked" if use_mask else "unmasked",
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "stochastic": args.stochastic,
        "executed_steps": int(np.asarray(lengths.max())),
        "success_count": int(np.asarray(task_done_once.sum())),
        "success_rate": float(np.asarray(task_done_once.mean())),
        "done_count": int(np.asarray(done_once.sum())),
        "done_rate": float(np.asarray(done_once.mean())),
        "avg_return": float(np.asarray(returns.mean())),
        "max_return": float(np.asarray(returns.max())),
        "avg_length": float(np.asarray(lengths.mean())),
        "avg_success_step": float(
            np.asarray(
                jnp.where(
                    task_done_once.any(),
                    first_task_done_step[first_task_done_step >= 0].mean(),
                    jnp.nan,
                )
            )
        ),
        "invalid_selected": invalid_selected,
        "invalid_selected_rate": invalid_selected / total_actions,
        "action_counts": action_counts_np.tolist(),
        "action_percent": (action_counts_np / total_actions).tolist(),
    }
    result.update(_terrain_scores(timestep.observation, target_maps_init))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", default=None)
    parser.add_argument("--map-path", default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("unmasked", "masked"),
        default=("unmasked", "masked"),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--disable-foundation-border-alignment",
        action="store_true",
        help="Use only for maps without foundation-border metadata.",
    )
    args = parser.parse_args()

    log = load_pkl_object(str(args.checkpoint))
    config = log["train_config"]
    config.edge_features_dim = infer_edge_features_dim_from_model_params(log["model"])
    config.use_action_mask = False
    config.num_test_rollouts = args.num_envs
    config.num_devices = 1
    config.benchmark_stochastic = args.stochastic

    env_config = _unbatch_env_config(log["env_config"])
    batch_cfg, env_cfgs = _build_env_config(
        env_config,
        args.config,
        args.num_envs,
        enforce_foundation_border_alignment=not args.disable_foundation_border_alignment,
    )
    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        rendering=False,
        shuffle_maps=args.map_path is None,
        single_map_path=args.map_path,
    )
    model = load_neural_network(config, env)
    params = log["model"]

    results = [
        run_rollout(env, env_cfgs, model, params, config, args, mode == "masked")
        for mode in args.modes
    ]
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
