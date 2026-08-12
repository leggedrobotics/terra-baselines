#!/usr/bin/env python3
"""Prove that --model_size actually widens a given map encoder.

The July encoder review found a size-wiring gap (``resnet_global_pool`` is
instantiated with no arguments, so ``model_size`` cannot reach it). This is the
pre-launch gate for the M1 screen: it prints total and encoder-only parameter
counts for every requested size and fails if two sizes share an encoder count.

    DATASET_PATH=<root> DATASET_SIZE=8 python scripts/check_model_size_wiring.py \
        --map_encoder resnet_spatial_8x8_se --maps_path train/L0
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from train_mixed import MixedAgentTrainConfig, make_mixed_agent_states  # noqa: E402


def count(tree) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(tree)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_encoder", default="resnet_spatial_8x8_se")
    parser.add_argument("--model_core", default="mlp")
    parser.add_argument("--maps_path", default="train/L0")
    parser.add_argument("--sizes", nargs="+", default=["base", "medium"])
    parser.add_argument("--encoder_compute_dtype", default="bfloat16")
    parser.add_argument("--critic_hidden_dims", default="512,256")
    args = parser.parse_args()

    results = {}
    for size in args.sizes:
        config = MixedAgentTrainConfig(
            name=f"param-count-{size}",
            num_devices=1,
            num_envs_per_device=16,
            num_minibatches=16,
            num_steps=2,
            total_timesteps=16 * 2,
            eval_episodes=0,
            log_eval_interval=0,
            checkpoint_interval=0,
            cache_clear_interval=0,
            model_size=size,
            model_core=args.model_core,
            map_encoder=args.map_encoder,
            agent_types_override=(0,),
            action_types_override=(0,),
            encoder_compute_dtype=args.encoder_compute_dtype,
            critic_hidden_dims=tuple(int(v) for v in args.critic_hidden_dims.split(",")),
            curriculum_levels_override=[
                {
                    "maps_path": args.maps_path,
                    "max_steps_in_episode": 450,
                    "rewards_type": 0,
                    "apply_trench_rewards": False,
                }
            ],
            curriculum_increase_level_threshold=3,
            curriculum_decrease_level_threshold=3,
            curriculum_last_level_type="none",
        )
        _, _, _, train_state = make_mixed_agent_states(config)
        params = train_state.params
        total = count(params)
        collection = params["params"] if "params" in params else params
        encoder = {
            key: value
            for key, value in collection.items()
            if "MapsNet" in key or "ResNet" in key or "maps_net" in key.lower()
        }
        results[size] = {
            "total": total,
            "encoder": count(encoder),
            "encoder_modules": sorted(encoder),
            "top_level": {k: count(v) for k, v in collection.items()},
        }

    print("\n================ model_size wiring check ================")
    print(f"map_encoder = {args.map_encoder}  model_core = {args.model_core}")
    for size, row in results.items():
        print(f"\n  {size}:")
        print(f"    total params   : {row['total']:,}")
        print(f"    encoder params : {row['encoder']:,}  {row['encoder_modules']}")
        for module, value in sorted(row["top_level"].items()):
            print(f"      {module:<28} {value:,}")

    sizes = list(results)
    ok = True
    for a, b in zip(sizes, sizes[1:]):
        if results[a]["encoder"] == results[b]["encoder"]:
            print(f"\nFAIL: {a} and {b} share encoder param count {results[a]['encoder']:,}")
            ok = False
        if results[a]["total"] == results[b]["total"]:
            print(f"\nFAIL: {a} and {b} share total param count {results[a]['total']:,}")
            ok = False
    if not ok:
        raise SystemExit(1)
    print("\nPASS: every requested size changes both encoder and total parameter count")


if __name__ == "__main__":
    jnp.zeros(1)
    main()
