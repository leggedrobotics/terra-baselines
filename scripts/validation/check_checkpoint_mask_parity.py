#!/usr/bin/env python3
"""Check whether action masking changes deterministic checkpoint rollouts."""

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
from utils.models import (
    load_neural_network,
    restore_checkpoint_model_config,
)
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


def _batch_cfg_and_env_cfg(
    env_config: EnvConfig,
    config_name: str | None,
    n_envs: int,
    enforce_foundation_border_alignment: bool,
):
    action_types = _as_int_tuple(env_config.action_types)
    batch_cfg = BatchConfig(
        action_type=WheeledAction if action_types and all(t == 1 for t in action_types) else TrackedAction
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
        action_type=WheeledAction if action_types and all(t == 1 for t in action_types) else TrackedAction,
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
    logits_for_action = apply_action_mask(logits, obs_model[22]) if use_mask else logits
    if config.parity_stochastic:
        dist = tfp.distributions.Categorical(logits=logits_for_action)
        action = dist.sample(seed=rng)
    else:
        action = jnp.argmax(logits_for_action, axis=-1)
    return action.astype(jnp.int32), logits, obs_model[22]


def check_checkpoint(path: Path, args: argparse.Namespace) -> dict:
    log = load_pkl_object(str(path))
    config = log["train_config"]
    restore_checkpoint_model_config(config, log["model"])
    config.num_test_rollouts = args.num_envs
    config.num_devices = 1
    config.parity_stochastic = args.stochastic

    env_config = _unbatch_env_config(log["env_config"])
    batch_cfg, env_cfgs = _batch_cfg_and_env_cfg(
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

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    timestep_mask = env.reset(env_cfgs, reset_keys)
    timestep_nomask = timestep_mask
    prev_mask = jnp.zeros((args.num_envs, config.num_prev_actions), dtype=jnp.int32)
    prev_nomask = jnp.zeros_like(prev_mask)

    invalid_unmasked_argmax = 0
    max_reward_abs_diff = 0.0
    max_action_masked_prob = None

    for step in range(args.num_steps):
        rng, rng_action, rng_env = jax.random.split(rng, 3)
        action_nomask, logits_nomask, action_mask = _select_action(
            model,
            params,
            timestep_nomask.observation,
            prev_nomask,
            config,
            use_mask=False,
            rng=rng_action,
        )
        action_masked, _, _ = _select_action(
            model,
            params,
            timestep_mask.observation,
            prev_mask,
            config,
            use_mask=True,
            rng=rng_action,
        )

        chosen_allowed = np.asarray(
            jnp.take_along_axis(action_mask, action_nomask[:, None], axis=-1)[:, 0]
        )
        invalid_unmasked_argmax += int((~chosen_allowed).sum())
        if max_action_masked_prob is None:
            probs = tfp.distributions.Categorical(logits=logits_nomask).probs_parameter()
            max_action_masked_prob = float(np.asarray(jnp.max(jnp.where(action_mask, 0.0, probs))))

        if not np.array_equal(np.asarray(action_nomask), np.asarray(action_masked)):
            return {
                "checkpoint": str(path),
                "status": "diverged",
                "first_divergent_step": step,
                "nomask_actions": np.asarray(action_nomask).tolist(),
                "mask_actions": np.asarray(action_masked).tolist(),
                "nomask_actions_allowed": chosen_allowed.tolist(),
                "invalid_unmasked_argmax": invalid_unmasked_argmax,
                "max_invalid_action_probability_step0": max_action_masked_prob,
            }

        step_keys = jax.random.split(rng_env, args.num_envs)
        env_action = wrap_action(action_nomask, env.batch_cfg.action_type)
        timestep_nomask = env.step(timestep_nomask, env_action, step_keys)
        timestep_mask = env.step(timestep_mask, env_action, step_keys)

        reward_diff = float(
            np.max(np.abs(np.asarray(timestep_nomask.reward - timestep_mask.reward)))
        )
        max_reward_abs_diff = max(max_reward_abs_diff, reward_diff)
        if reward_diff > args.tolerance:
            return {
                "checkpoint": str(path),
                "status": "reward_diverged",
                "first_divergent_step": step,
                "max_reward_abs_diff": max_reward_abs_diff,
                "invalid_unmasked_argmax": invalid_unmasked_argmax,
            }

        prev_nomask = jnp.concatenate([action_nomask[:, None], prev_nomask[:, :-1]], axis=-1)
        prev_mask = jnp.concatenate([action_masked[:, None], prev_mask[:, :-1]], axis=-1)

    return {
        "checkpoint": str(path),
        "status": "parity",
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "config": args.config,
        "map_path": args.map_path,
        "stochastic": args.stochastic,
        "invalid_unmasked_argmax": invalid_unmasked_argmax,
        "max_reward_abs_diff": max_reward_abs_diff,
        "max_invalid_action_probability_step0": max_action_masked_prob,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional preset override. By default the checkpoint's saved "
            "env_config is used."
        ),
    )
    parser.add_argument("--map-path", default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--disable-foundation-border-alignment",
        action="store_true",
        help="Use only for local maps that do not contain foundation-border metadata.",
    )
    args = parser.parse_args()

    results = [check_checkpoint(path, args) for path in args.checkpoints]
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")
    failed = [r for r in results if r["status"] != "parity"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
