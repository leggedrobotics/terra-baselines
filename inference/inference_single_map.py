"""
Run single-map inference for a mixed-agent checkpoint and save a GIF.

Similar to `visualize_mixed.py`, but forces a single environment and supports
loading one explicit map via `--map_path`.
"""

from datetime import datetime
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

# Add the parent directory to the path so we can import utils
sys.path.append(str(Path(__file__).parent.parent))

from terra.config import (
    BatchConfig,
    CurriculumConfig,
    CurriculumGlobalConfig,
    EnvConfig,
    RewardsType,
)
from terra.env import TerraEnvBatch
from train import TrainConfig  # needed for unpickling checkpoints
from train_mixed import MixedAgentTrainConfig
from utils.helpers import load_pkl_object
from utils.models import load_neural_network, restore_checkpoint_model_config
from utils.utils_ppo import obs_to_model_input, wrap_action

sys.modules["__main__"].MixedAgentTrainConfig = MixedAgentTrainConfig


def _take0(v):
    try:
        arr = jnp.asarray(v)
    except Exception:
        return v
    while arr.ndim >= 1 and arr.shape[0] > 1:
        arr = arr[0]
    while arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _unbatch_namedtuple(nt):
    return jax.tree_util.tree_map(_take0, nt)


def _replicate_value(x, n_envs):
    if x is None:
        return None
    if isinstance(x, tuple) and not hasattr(x, "_fields"):
        return jnp.asarray(x)[None, ...].repeat(n_envs, 0)
    if isinstance(x, (int, float, bool)):
        return jnp.asarray([x] * n_envs)
    x_arr = jnp.asarray(x)
    if x_arr.ndim >= 1 and x_arr.shape[0] == 1:
        return x_arr[0][None, ...].repeat(n_envs, 0)
    return x_arr[None, ...].repeat(n_envs, 0)


def _replicate_namedtuple(nt, n_envs):
    updates = {}
    for f in getattr(nt, "_fields", ()):
        v = getattr(nt, f)
        if hasattr(v, "_fields"):
            updates[f] = _replicate_namedtuple(v, n_envs)
        else:
            updates[f] = _replicate_value(v, n_envs)
    return nt._replace(**updates)


def _safe_scalar_float(v, default):
    try:
        arr = jnp.asarray(v)
        if arr.size == 0:
            return jnp.float32(default)
        return jnp.asarray(arr.reshape(-1)[0], dtype=jnp.float32)
    except Exception:
        return jnp.float32(default)


def rollout_and_render_episode(
    env, model, model_params, env_cfgs, rl_config, max_frames, seed, deterministic
):
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    timestep = env.reset(env_cfgs, jax.random.split(_rng, 1))

    prev_actions = jnp.zeros((1, rl_config.num_prev_actions), dtype=jnp.int32)
    rewards = []

    # Render initial frame immediately to avoid storing full trajectories in memory.
    initial_obs = dict(timestep.observation)
    initial_obs["action_map"] = timestep.state.world.action_map.map
    if "interaction_mask" in initial_obs:
        initial_obs["interaction_mask"] = jnp.zeros_like(initial_obs["interaction_mask"])
    env.terra_env.render_obs_pygame(initial_obs, generate_gif=True)

    steps = 0
    pbar = tqdm(total=max_frames, desc="Rollout+Rendering")
    while True:
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        obs_in = obs_to_model_input(timestep.observation, prev_actions, rl_config)
        _, logits_pi = model.apply(model_params, obs_in)
        if deterministic:
            action = jnp.argmax(logits_pi, axis=-1)
        else:
            pi = tfp.distributions.Categorical(logits=logits_pi)
            action = pi.sample(seed=rng_act)

        prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        prev_actions = prev_actions.at[:, 0].set(action)

        timestep = env.step(
            timestep,
            wrap_action(action, env.batch_cfg.action_type),
            jax.random.split(rng_step, 1),
        )

        steps += 1
        pbar.update(1)

        # Render each step immediately (memory-light path).
        modified_obs = dict(timestep.observation)
        modified_obs["action_map"] = timestep.state.world.action_map.map
        if "interaction_mask" in modified_obs:
            modified_obs["interaction_mask"] = jnp.zeros_like(modified_obs["interaction_mask"])
        env.terra_env.render_obs_pygame(modified_obs, generate_gif=True)

        rewards.append(float(np.asarray(timestep.reward)[0]))

        if bool(np.asarray(timestep.done)[0]) or steps >= max_frames:
            info = timestep.info
            task_done = None
            if isinstance(info, dict) and "task_done" in info:
                task_done = bool(np.asarray(info["task_done"])[0])
            break

    pbar.close()
    return rewards, steps, task_done


if __name__ == "__main__":
    import argparse

    default_maps_dir = Path(__file__).resolve().parent / "maps"

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Checkpoint path (.pkl)")
    parser.add_argument(
        "--map_name",
        type=str,
        default="map",
        help="Map folder/file name inside inference/maps (default: map)",
    )
    parser.add_argument(
        "--map_path",
        type=str,
        default=None,
        help="Optional explicit map folder/file path (overrides --map_name)",
    )
    parser.add_argument("--n_steps", type=int, default=350, help="Max rollout steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--deterministic", type=int, default=0, help="1=argmax actions, 0=sample")
    parser.add_argument("--out_path", type=str, default=None, help="Output gif path")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Named config preset to match visualize_mixed behavior",
    )
    args, _ = parser.parse_known_args()

    log = load_pkl_object(args.policy)
    config = log["train_config"]
    restore_checkpoint_model_config(config, log["model"])
    config.num_test_rollouts = 1
    config.num_devices = 1
    config.num_embeddings_agent_min = 60

    env_cfgs_ckpt = log.get("env_config", None)
    if env_cfgs_ckpt is not None and hasattr(env_cfgs_ckpt, "_fields"):
        env_cfg = _unbatch_namedtuple(env_cfgs_ckpt)
        # Keep agent_types/action_types as Python tuples (not JAX arrays), matching visualize_mixed.
        try:
            env_cfg = env_cfg._replace(
                agent_types=tuple(int(x) for x in jnp.asarray(env_cfg.agent_types).tolist()),
                action_types=tuple(int(x) for x in jnp.asarray(env_cfg.action_types).tolist()),
            )
        except Exception:
            pass
    else:
        env_cfg = EnvConfig()

    # Keep compatibility with checkpoints created before new fields existed.
    try:
        if not hasattr(env_cfg.curriculum, "_fields"):
            env_cfg = env_cfg._replace(curriculum=CurriculumConfig())
    except Exception:
        env_cfg = EnvConfig()

    env_cfg = env_cfg._replace(
        cabin_alignment_coefficient=_safe_scalar_float(
            getattr(env_cfg, "cabin_alignment_coefficient", -0.04), -0.04
        )
    )

    if hasattr(env_cfg, "_fields"):
        env_cfgs = _replicate_namedtuple(env_cfg, n_envs=1)
    else:
        env_cfgs = jax.tree_util.tree_map(
            lambda x: _replicate_value(x, 1),
            env_cfg,
            is_leaf=lambda x: isinstance(x, tuple) and not hasattr(x, "_fields"),
        )

    # Keep exactly the same default map/curriculum bootstrap behavior as visualize_mixed.
    batch_cfg = None
    if args.config is not None:
        try:
            from configs.training_configs import get_config

            preset = get_config(args.config)
            print(f"\nLoading config preset: '{args.config}'")

            if preset.maps and len(preset.maps) > 0:
                curriculum_levels = []
                for map_level in preset.maps:
                    rewards_type = (
                        RewardsType.DENSE
                        if map_level.rewards_type == "DENSE"
                        else RewardsType.SPARSE
                    )
                    curriculum_levels.append(
                        {
                            "maps_path": map_level.maps_path,
                            "max_steps_in_episode": map_level.max_steps_in_episode,
                            "rewards_type": rewards_type,
                            "apply_trench_rewards": map_level.apply_trench_rewards,
                        }
                    )

                class CustomCurriculumGlobalConfig(CurriculumGlobalConfig):
                    levels = curriculum_levels

                batch_cfg = BatchConfig(curriculum_global=CustomCurriculumGlobalConfig())

                # Same runtime env overrides used in visualize_mixed.
                first_level = curriculum_levels[0]
                env_cfgs = env_cfgs._replace(
                    agent_types=jnp.asarray(tuple(preset.agent_types), dtype=jnp.int32)[
                        None, ...
                    ].repeat(1, 0),
                    action_types=jnp.asarray(tuple(preset.action_types), dtype=jnp.int32)[
                        None, ...
                    ].repeat(1, 0),
                    max_steps_in_episode=jnp.full(
                        (1,), int(first_level["max_steps_in_episode"]), dtype=jnp.int32
                    ),
                    apply_trench_rewards=jnp.full(
                        (1,), bool(first_level["apply_trench_rewards"]), dtype=jnp.bool_
                    ),
                )
        except Exception as e:
            print(f"Failed to load --config preset '{args.config}': {e}")

    if batch_cfg is None:
        batch_cfg = BatchConfig()

    if args.map_path:
        map_path = args.map_path
    else:
        map_path = str(default_maps_dir / args.map_name)

    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        rendering=True,
        n_envs_x_rendering=1,
        n_envs_y_rendering=1,
        display=False,
        shuffle_maps=False,
        single_map_path=map_path,
    )

    # Match visualize_mixed behavior exactly: freeze curriculum/config updates
    # so checkpoint env_cfgs are used as-is during reset/step.
    class _NoopCurriculumManager:
        def reset_cfgs(self, env_cfgs):
            return env_cfgs

        def update_cfgs(self, timesteps, rng):
            return timesteps

    env.curriculum_manager = _NoopCurriculumManager()

    model = load_neural_network(config, env)
    model_params = log["model"]

    deterministic = bool(args.deterministic)
    rewards, steps, task_done = rollout_and_render_episode(
        env=env,
        model=model,
        model_params=model_params,
        env_cfgs=env_cfgs,
        rl_config=config,
        max_frames=args.n_steps,
        seed=args.seed,
        deterministic=deterministic,
    )

    if args.out_path:
        out_path = args.out_path
    else:
        ckpt_stem = Path(args.policy).stem
        map_stem = Path(map_path).stem
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_path = f"./{ckpt_stem}-singlemap-{map_stem}-{ts}.gif"

    env.terra_env.rendering_engine.create_gif(out_path)

    total_return = float(np.sum(rewards)) if len(rewards) > 0 else 0.0
    mean_step_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    print(f"GIF saved to {out_path}")
    print(f"steps={steps} return={total_return:.4f} mean_step_reward={mean_step_reward:.4f} task_done={task_done}")
