#!/usr/bin/env python3
"""Trace actions selected by raw logits that the Terra action mask rejects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.benchmark_checkpoint_masked import (  # noqa: E402
    _build_env_config,
    _select_action,
    _unbatch_env_config,
)
from terra.env import TerraEnvBatch  # noqa: E402
from train import TrainConfig  # noqa: F401,E402 - needed for unpickling
from train_mixed import MixedAgentTrainConfig  # noqa: E402
from utils.action_masking import apply_action_mask  # noqa: E402
from utils.helpers import load_pkl_object  # noqa: E402
from utils.models import load_neural_network, restore_checkpoint_model_config  # noqa: E402
from utils.utils_ppo import obs_to_model_input, wrap_action  # noqa: E402

sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig

ACTION_NAMES = {
    0: "FORWARD",
    1: "BACKWARD",
    2: "CLOCK",
    3: "ANTICLOCK",
    4: "CABIN_CLOCK",
    5: "CABIN_ANTICLOCK",
    6: "DO",
    7: "DO_NOTHING",
}


def _tolist(x):
    return np.asarray(x).tolist()


def _edge_feature_dict(edge_features):
    values = np.asarray(edge_features).reshape(-1).tolist()
    names = [
        "do_valid",
        "mask_fraction",
        "remaining_edge_count_norm",
        "remaining_core_count_norm",
        "raw_edge_in_cone_frac",
        "legal_edge_in_cone_frac",
        "blocked_edge_in_cone_frac",
        "legal_remaining_in_cone_frac",
        "remaining_edge_share",
        "completion",
    ]
    return {name: values[i] for i, name in enumerate(names[: len(values)])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", default="solo_excavator")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    log = load_pkl_object(str(args.checkpoint))
    config = log["train_config"]
    restore_checkpoint_model_config(config, log["model"])
    config.num_test_rollouts = args.num_envs
    config.num_devices = 1
    config.benchmark_stochastic = False

    env_config = _unbatch_env_config(log["env_config"])
    batch_cfg, env_cfgs = _build_env_config(
        env_config,
        args.config,
        args.num_envs,
        enforce_foundation_border_alignment=True,
    )
    env = TerraEnvBatch(batch_cfg=batch_cfg, rendering=False, shuffle_maps=True)
    model = load_neural_network(config, env)
    params = log["model"]

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    timestep = env.reset(env_cfgs, reset_keys)
    prev_actions = jnp.zeros((args.num_envs, config.num_prev_actions), dtype=jnp.int32)
    events = []

    for step in range(args.num_steps):
        rng, rng_action, rng_env = jax.random.split(rng, 3)
        obs_model = obs_to_model_input(timestep.observation, prev_actions, config)
        _, logits = model.apply(params, obs_model)
        action, _, action_mask = _select_action(
            model,
            params,
            timestep.observation,
            prev_actions,
            config,
            use_mask=False,
            rng=rng_action,
        )
        masked_action = jnp.argmax(apply_action_mask(logits, action_mask), axis=-1)
        selected_allowed = jnp.take_along_axis(action_mask, action[:, None], axis=-1)[:, 0]

        before_action_map = timestep.observation["action_map"].copy()
        before_agent = timestep.observation["agent_states"][:, 0, :].copy()
        step_keys = jax.random.split(rng_env, args.num_envs)
        next_timestep = env.step(
            timestep,
            wrap_action(action, env.batch_cfg.action_type),
            step_keys,
        )

        invalid_envs = np.where(~np.asarray(selected_allowed))[0].tolist()
        probs = jax.nn.softmax(logits, axis=-1)
        for env_idx in invalid_envs:
            before_map = before_action_map[env_idx]
            after_map = next_timestep.observation["action_map"][env_idx]
            delta_map = after_map - before_map
            before_loaded = before_agent[env_idx, 5]
            after_loaded = next_timestep.observation["agent_states"][env_idx, 0, 5]
            event = {
                "step": step,
                "env": int(env_idx),
                "action": int(np.asarray(action[env_idx])),
                "action_name": ACTION_NAMES.get(int(np.asarray(action[env_idx])), "UNKNOWN"),
                "masked_action": int(np.asarray(masked_action[env_idx])),
                "masked_action_name": ACTION_NAMES.get(
                    int(np.asarray(masked_action[env_idx])), "UNKNOWN"
                ),
                "mask": _tolist(action_mask[env_idx]),
                "agent_state_before": _tolist(before_agent[env_idx]),
                "edge_features": _edge_feature_dict(
                    timestep.observation["edge_features"][env_idx]
                ),
                "selected_probability": float(np.asarray(probs[env_idx, action[env_idx]])),
                "reward": float(np.asarray(next_timestep.reward[env_idx])),
                "task_done": bool(np.asarray(next_timestep.info["task_done"][env_idx])),
                "done": bool(np.asarray(next_timestep.done[env_idx])),
                "loaded_before": int(np.asarray(before_loaded)),
                "loaded_after": int(np.asarray(after_loaded)),
                "loaded_delta": int(np.asarray(after_loaded - before_loaded)),
                "action_map_delta_abs": float(
                    np.asarray(jnp.abs(delta_map).sum())
                ),
                "action_map_delta_sum": float(np.asarray(delta_map.sum())),
                "action_map_positive_delta": float(
                    np.asarray(jnp.clip(delta_map, a_min=0).sum())
                ),
                "action_map_negative_delta": float(
                    np.asarray(jnp.clip(delta_map, a_max=0).sum())
                ),
                "action_map_changed_cells": int(np.asarray((delta_map != 0).sum())),
                "action_map_sum_before": float(np.asarray(before_map.sum())),
                "action_map_sum_after": float(np.asarray(after_map.sum())),
            }
            events.append(event)
            if len(events) >= args.max_events:
                break

        timestep = next_timestep
        prev_actions = jnp.concatenate([action[:, None], prev_actions[:, :-1]], axis=-1)
        if len(events) >= args.max_events:
            break

    result = {
        "checkpoint": str(args.checkpoint),
        "config": args.config,
        "num_envs": args.num_envs,
        "num_steps_requested": args.num_steps,
        "seed": args.seed,
        "events_recorded": len(events),
        "events": events,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
