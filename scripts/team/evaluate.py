#!/usr/bin/env python3
"""First-episode outcomes of a single agent or a team on the same reset maps.

The reset key selects the map (and agent 0's spawn), so runs with different
team sizes but the same --seed and --envs face identical tasks. A single-agent
checkpoint evaluated with --agents > 1 is migrated first: every agent plays the
single-agent policy (zero-shot team baseline).

  python scripts/team/evaluate.py --checkpoint CK.pkl --bank BANK --agents 2 \
      --envs 256 --output team_zero_shot.json
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def build(checkpoint_path, bank, dataset_size, agents, envs, render_grid=None, types=None,
          maps_path=None):
    """Environment, batched env params, config and TrainState for ``agents``.

    ``types`` are the agent types (default: ``agents`` excavators), all
    tracked. A single-agent checkpoint used with ``agents > 1`` is migrated
    first. ``render_grid`` (n) enables the pygame renderer for an n x n env
    grid. ``maps_path`` replaces the bank subdirectory of the checkpoint's
    curriculum (e.g. a held-out split).
    """
    types = tuple(types) if types else (0,) * agents
    agents = len(types)
    os.environ["DATASET_PATH"] = str(Path(bank).resolve())
    os.environ["DATASET_SIZE"] = str(dataset_size)
    import copy

    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    from terra.config import BatchConfig, CurriculumGlobalConfig
    from terra.env import TerraEnvBatch

    from utils.helpers import load_pkl_object, register_checkpoint_config_classes
    from utils.models import get_model_ready
    from utils.team_migration import team_params_from_single_agent

    register_checkpoint_config_classes()
    checkpoint = load_pkl_object(str(checkpoint_path))
    config = copy.copy(checkpoint["train_config"])
    config.agent_types_override = types
    config.action_types_override = (0,) * agents
    config.num_prev_actions = 5

    class Curriculum(CurriculumGlobalConfig):
        pass

    Curriculum.levels = config.curriculum_levels_override
    if maps_path:
        Curriculum.levels = [dict(level, maps_path=maps_path) for level in Curriculum.levels]
    render = {} if render_grid is None else dict(
        rendering=True, n_envs_x_rendering=render_grid, n_envs_y_rendering=render_grid,
        display=False,
    )
    env = TerraEnvBatch(
        batch_cfg=BatchConfig(curriculum_global=Curriculum()),
        shuffle_maps=False,
        executable_dig_observation=bool(config.executable_dig_observation),
        distance_protocol_id=config.distance_protocol_id,
        **render,
    )
    env_cfg = checkpoint["env_config"]._replace(
        agent_types=types, action_types=(0,) * agents
    )
    env_params = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x)[None].repeat(envs, 0), env_cfg
    )
    model, params = get_model_ready(jax.random.PRNGKey(0), config, env)
    saved = checkpoint["model"]
    if agents > 1 and "intent_decoder" not in saved["params"]:
        saved = team_params_from_single_agent(saved, params)
    state = TrainState.create(apply_fn=model.apply, params=saved, tx=optax.identity())
    return env, env_cfg, env_params, config, state


def policy_step(env, config, agents, envs, greedy, idle_teammates=False, reset=True):
    """Jitted (state, timestep, prev, rng) -> (timestep, prev, rng, moved, completion, action)."""
    import jax
    import jax.numpy as jnp
    from utils.utils_ppo import (
        select_action_ppo, select_joint_action, update_prev_actions,
        wrap_action, wrap_joint_action,
    )

    team = agents > 1
    stepper = env.step if reset else env.step_no_reset

    # Parameters are a traced argument: closing over them would turn three
    # million weights into compile-time constants.
    @jax.jit
    def step(state, timestep, prev, rng, frozen):
        rng, act_key, env_key = jax.random.split(rng, 3)
        if team:
            action, order, _, _, _ = select_joint_action(
                state, timestep.observation, prev, act_key, config, greedy=greedy
            )
            if idle_teammates:
                action = action.at[:, 1:].set(7)
            action = jnp.where(frozen[:, None], 7, action)
            env_action = wrap_joint_action(action, env.batch_cfg.action_type)
        else:
            _, _, _, pi = select_action_ppo(state, timestep.observation, prev, act_key, config)
            action = pi.mode() if greedy else pi.sample(seed=act_key)
            action = jnp.where(frozen, 7, action)
            order = None
            env_action = wrap_action(action, env.batch_cfg.action_type)
        before = timestep.state.agent.agent_states[:agents]
        timestep = stepper(timestep, env_action, jax.random.split(env_key, envs), order)
        after = timestep.state.agent.agent_states[:agents]
        # Per agent: its own pose or load changed (auto-reset rows are masked by the caller).
        moved = jnp.stack([
            jnp.any(a.pos_base != b.pos_base, -1) | jnp.any(a.angle_base != b.angle_base, -1)
            | jnp.any(a.angle_cabin != b.angle_cabin, -1) | jnp.any(a.loaded != b.loaded, -1)
            for a, b in zip(before, after)
        ], axis=-1)
        prev = update_prev_actions(prev, action, timestep.done)
        final_completion = timestep.info["reward_components"]["absolute_completion"]
        # Work log for the executed-plan time model: base pose and load of each
        # agent before its action, and its load after.
        work = (
            jnp.stack([a.pos_base for a in before], axis=1),
            jnp.stack([a.angle_base[..., 0] for a in before], axis=1),
            jnp.stack([a.loaded[..., 0] for a in before], axis=1),
            jnp.stack([a.loaded[..., 0] for a in after], axis=1),
        )
        return timestep, prev, rng, moved, final_completion, action, work

    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, default=3840)
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument("--types", help="comma-separated agent types (default: --agents excavators)")
    parser.add_argument("--maps-path", help="bank subdirectory (default: the checkpoint's)")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--idle-teammates", action="store_true",
                        help="diagnostic: agents other than slot 0 always do nothing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    types = tuple(int(t) for t in args.types.split(",")) if args.types else (0,) * args.agents
    agents = len(types)
    env, env_cfg, env_params, config, state = build(
        args.checkpoint, args.bank, args.dataset_size, agents, args.envs, types=types,
        maps_path=args.maps_path,
    )
    horizon = int(np.max(np.asarray(env_cfg.max_steps_in_episode)))
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    timestep = env.reset(env_params, jax.random.split(reset_key, args.envs))
    team = agents > 1
    prev = jnp.zeros((args.envs,) + ((agents,) if team else ()) + (5,), jnp.int32)

    # Reset geometry: how much of each chassis covers dig or dump targets.
    @jax.jit
    def reset_geometry(state):
        def one(env_state):
            target = env_state.world.target_map.map
            rows = []
            for slot in range(agents):
                view = env_state._replace(
                    agent=env_state.agent._replace(current_agent=jnp.int32(slot))
                )
                footprint = view._current_base_footprint_mask()
                rows.append(jnp.stack((
                    jnp.sum(footprint & (target < 0)), jnp.sum(footprint & (target > 0)),
                )))
            return jnp.stack(rows)
        return jax.vmap(one)(state)

    geometry = np.asarray(reset_geometry(timestep.state))  # [envs, agents, (dig, dump)]

    step = policy_step(env, config, agents, args.envs, args.greedy, args.idle_teammates)
    frozen = jnp.zeros(args.envs, bool)

    # Executed plan per machine: the base poses of work events (own load
    # changed: excavator digs, dumps and relifts on DO; skid-steer pickups on
    # FORWARD and dumps on DO). The robot's nav stack drives between them.
    tile_size = float(np.ravel(np.asarray(env_cfg.tile_size))[0])
    angles_base = int(np.ravel(np.asarray(env_cfg.agent.angles_base))[0])
    start = timestep.state.agent.agent_states[:agents]
    last_pose = np.concatenate([
        np.stack([np.asarray(a.pos_base) for a in start], axis=1),
        np.stack([np.asarray(a.angle_base)[..., 0] for a in start], axis=1)[..., None],
    ], axis=-1).astype(np.int64)  # [envs, agents, (x, y, heading bin)]
    travel_m = np.zeros((args.envs, agents))
    heading_rad = np.zeros((args.envs, agents))
    scooped_units = np.zeros((args.envs, agents))
    setups = np.zeros((args.envs, agents), np.int64)
    required_units = np.asarray(
        jax.vmap(lambda t: jnp.sum(jnp.clip(-t, 0, None)))(timestep.state.world.target_map.map)
    ).astype(float)

    ended = np.zeros(args.envs, bool)
    effective = np.zeros((args.envs, agents), np.int32)
    acted = np.zeros(args.envs, np.int32)
    # Per agent and action index: times chosen and times it changed pose/load.
    chosen = np.zeros((agents, 8), np.int64)
    worked = np.zeros((agents, 8), np.int64)
    success = np.zeros(args.envs, bool)
    steps = np.full(args.envs, horizon, np.int32)
    completion = np.zeros(args.envs, np.float32)
    start = time.time()
    for t in range(horizon + 1):
        timestep, prev, rng, moved, final, taken, work = step(state, timestep, prev, rng, frozen)
        done = np.asarray(timestep.done)
        # Work events count until the step that ends the first episode.
        counting = ~ended
        pos, heading, load_before, load_after = (np.asarray(x) for x in work)
        pose = np.concatenate([pos, heading[..., None]], axis=-1).astype(np.int64)
        work_event = counting[:, None] & (load_before != load_after)
        new_pose = work_event & np.any(pose != last_pose, axis=-1)
        travel_m += new_pose * np.linalg.norm(pose[..., :2] - last_pose[..., :2], axis=-1) * tile_size
        turn = np.abs((pose[..., 2] - last_pose[..., 2] + angles_base / 2) % angles_base - angles_base / 2)
        heading_rad += new_pose * turn * (2 * np.pi / angles_base)
        setups += new_pose
        scooped_units += work_event * np.clip(load_after - load_before, 0, None)
        last_pose = np.where(work_event[..., None], pose, last_pose)
        live = ~ended & ~done
        moved = np.asarray(moved)
        taken = np.asarray(taken).reshape(args.envs, agents)
        effective[live] += moved[live]
        acted[live] += 1
        for slot in range(agents):
            np.add.at(chosen[slot], taken[live, slot], 1)
            np.add.at(worked[slot], taken[live, slot], moved[live, slot].astype(np.int64))
        new = done & ~ended
        success[new] = np.asarray(timestep.info["task_done"])[new]
        steps[new] = t + 1
        completion[new] = np.asarray(final)[new]
        ended |= done
        if ended.all():
            break
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "agents": agents,
        "agent_types": list(types),
        "envs": args.envs,
        "seed": args.seed,
        "greedy": bool(args.greedy),
        "horizon": horizon,
        "success_rate": float(success.mean()),
        "mean_steps_success": float(steps[success].mean()) if success.any() else None,
        "median_steps_success": float(np.median(steps[success])) if success.any() else None,
        "mean_final_completion": float(completion.mean()),
        "idle_teammates": bool(args.idle_teammates),
        # Fraction of each agent's actions that changed its own pose or load.
        "effective_action_fraction": (
            effective.sum(0) / max(int(acted.sum()), 1)
        ).round(4).tolist(),
        "tile_size_m": tile_size,
        "action_counts": chosen.tolist(),
        "action_effective_counts": worked.tolist(),
        "per_env": {
            "success": success.tolist(),
            "steps": steps.tolist(),
            "final_completion": completion.round(4).tolist(),
            # [agent][dig cells, dump cells] under each chassis at reset.
            "reset_target_overlap": geometry.tolist(),
            # Executed plan per machine (first episode): straight-line travel
            # between distinct work-event base poses (from the start pose),
            # heading change, material scooped (load increases: digs, relifts,
            # pickups), and the number of distinct work poses. Required
            # excavation in units.
            "travel_m": travel_m.round(3).tolist(),
            "heading_change_rad": heading_rad.round(4).tolist(),
            "scooped_units": scooped_units.tolist(),
            "work_setups": setups.tolist(),
            "required_units": required_units.tolist(),
        },
        "wall_seconds": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_env"}, indent=2))


if __name__ == "__main__":
    main()
