#!/usr/bin/env python3
"""Render first episodes of a single agent or a team as one GIF (n x n envs).

Same reset maps as evaluate.py for the same --seed and env count. Finished
environments stay frozen on their last frame; a team is drawn with every
agent and the union of their workspaces.

  python scripts/team/render.py --checkpoint CK.pkl --bank BANK --agents 2 \
      --grid 2 --output team.gif
"""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.team.evaluate import build, policy_step  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, default=3840)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--types", help="comma-separated agent types (default: --agents excavators)")
    parser.add_argument("--maps-path", help="bank subdirectory (default: the checkpoint's)")
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    envs = args.grid * args.grid
    types = tuple(int(t) for t in args.types.split(",")) if args.types else (0,) * args.agents
    args.agents = len(types)
    env, env_cfg, env_params, config, state = build(
        args.checkpoint, args.bank, args.dataset_size, args.agents, envs,
        render_grid=args.grid, types=types, maps_path=args.maps_path,
    )
    horizon = int(np.max(np.asarray(env_cfg.max_steps_in_episode)))
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    timestep = env.reset(env_params, jax.random.split(reset_key, envs))
    team = args.agents > 1
    prev = jnp.zeros((envs,) + ((args.agents,) if team else ()) + (5,), jnp.int32)
    step = policy_step(env, config, args.agents, envs, args.greedy, reset=False)

    def draw(observation):
        observation = jax.device_get(observation)
        if team:
            observation = dict(
                observation,
                agent_states=observation["agent_states"][:, 0],
                agent_active=observation["agent_active"][:, 0],
                interaction_mask=observation["interaction_mask"].max(axis=1),
            )
        env.terra_env.render_obs_pygame(observation, generate_gif=True)

    frozen = np.zeros(envs, bool)
    draw(timestep.observation)
    for t in range(horizon):
        timestep, prev, rng, _, _, _, _ = step(state, timestep, prev, rng, jnp.asarray(frozen))
        frozen |= np.asarray(timestep.done)
        if t % args.frame_stride == 0 or frozen.all():
            draw(timestep.observation)
        if frozen.all():
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env.terra_env.rendering_engine.create_gif(str(args.output))
    print(f"rounds {t + 1}, success {np.asarray(timestep.info['task_done']).tolist()}, gif {args.output}")


if __name__ == "__main__":
    main()
