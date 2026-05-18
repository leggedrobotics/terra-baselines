"""
Visualization script for mixed agent training (Tracked Excavator + Skid Steer).
Functionality is identical to visualize.py, but defaults and documentation are for mixed agent checkpoints.

Usage:
    python visualize_mixed.py -run checkpoint.pkl --config excavator_skidsteer
    python visualize_mixed.py -run checkpoint.pkl --config solo_excavator -nx 4 -ny 4
"""

import numpy as np
import jax
from tqdm import tqdm
from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.utils_ppo import action_type_from_policy_action, obs_to_model_input, policy, wrap_action
from terra.state import State
import matplotlib.animation as animation
from train import TrainConfig  # needed for unpickling checkpoints
from terra.config import EnvConfig, BatchConfig, CurriculumGlobalConfig, RewardsType, CurriculumConfig
import sys
from train_mixed import MixedAgentTrainConfig
sys.modules['__main__'].MixedAgentTrainConfig = MixedAgentTrainConfig

def rollout_episode(
    env: TerraEnvBatch, model, model_params, env_cfgs, rl_config, max_frames, seed
):
    print(f"Using {seed=}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)
    prev_actions = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    t_counter = 0
    reward_seq = []
    obs_seq = []
    state_seq = []  # Also collect states
    
    # Add initial observation and state (after reset)
    obs_seq.append(timestep.observation)
    state_seq.append(timestep.state)
    
    while True:
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            obs = obs_to_model_input(timestep.observation, prev_actions, rl_config)
            v, pi = policy(model.apply, model_params, obs)
            action = pi.sample(seed=rng_act)
            action_type_sample = action_type_from_policy_action(action)
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action_type_sample)
        else:
            raise RuntimeError("Model is None!")
        rng_step = jax.random.split(rng_step, rl_config.num_test_rollouts)
        timestep = env.step(
            timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
        )
        
        t_counter += 1
        
        # COLLECT OBSERVATION AFTER STEP (includes soil mechanics changes)
        obs_seq.append(timestep.observation)
        state_seq.append(timestep.state)
        
        reward_seq.append(timestep.reward)
        print(t_counter, timestep.reward, action, timestep.done)
        print(10 * "=")
        
        if jnp.all(timestep.done).item() or t_counter == max_frames:
            break
    print(f"Terra - Steps: {t_counter}, Return: {np.sum(reward_seq)}")
    return obs_seq, np.cumsum(reward_seq), state_seq


def update_render(seq, env: TerraEnvBatch, frame):
    obs = {k: v[:, frame] for k, v in seq.items()}
    return env.terra_env.render_obs(obs, mode="gif")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        default="mixed_agents_checkpoint.pkl",
        help="Path to mixed agent trained checkpoint.",
    )
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="Terra",
        help="Environment name.",
    )
    parser.add_argument(
        "-nx",
        "--n_envs_x",
        type=int,
        default=4,
        help="Number of environments on x.",
    )
    parser.add_argument(
        "-ny",
        "--n_envs_y",
        type=int,
        default=4,
        help="Number of environments on y.",
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=350,
        help="Number of steps.",
    )
    parser.add_argument(
        "-o",
        "--out_path",  
        type=str,
        default="./solo-excavator-6-2026-05-15-17-59-29.pkl.gif",
        #default="./visualize_mixed_skid_exec___foundations_dumpzones_harder_nodump_test_2x2_env_2.gif",
        help="Output path.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Named config preset to load maps from (e.g., 'solo_excavator', 'excavator_skidsteer'). See configs/training_configs.yaml",
    )
    args, _ = parser.parse_known_args()
    n_envs = args.n_envs_x * args.n_envs_y

    log = load_pkl_object(f"{args.run_name}")
    config = log["train_config"]
    config.num_test_rollouts = n_envs
    config.num_devices = 1

    # Checkpoints often store a *batched* env_config (tree-mapped to arrays for pmap/vmap),
    # which breaks attribute access like `env_cfg.curriculum.level` inside TerraEnvBatch reset/map loading.
    # For visualization we *unbatch* the checkpoint env_config back into a structured EnvConfig
    # (take the first element along leading batch axes) and then replicate to `n_envs`.
    env_cfgs_ckpt = log.get("env_config", None)

    def _take0(v):
        """Take first element along leading axes until it looks unbatched."""
        try:
            arr = jnp.asarray(v)
        except Exception:
            return v
        # Peel leading batch axes if present.
        while arr.ndim >= 1 and arr.shape[0] > 1:
            arr = arr[0]
        # If leading axis is size-1, also peel it (common checkpoint batching).
        while arr.ndim >= 1 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def _unbatch_namedtuple(nt):
        updates = {}
        for f in getattr(nt, "_fields", ()):
            v = getattr(nt, f)
            if hasattr(v, "_fields"):
                updates[f] = _unbatch_namedtuple(v)
            else:
                updates[f] = _take0(v)
        return nt._replace(**updates)

    if env_cfgs_ckpt is not None and hasattr(env_cfgs_ckpt, "_fields"):
        env_cfgs = _unbatch_namedtuple(env_cfgs_ckpt)
        # Ensure agent_types/action_types are Python tuples (not JAX arrays) for downstream code.
        try:
            env_cfgs = env_cfgs._replace(
                agent_types=tuple(int(x) for x in jnp.asarray(env_cfgs.agent_types).tolist()),
                action_types=tuple(int(x) for x in jnp.asarray(env_cfgs.action_types).tolist()),
            )
        except Exception:
            pass
    else:
        env_cfgs = EnvConfig()

    # Ensure curriculum is a proper CurriculumConfig (some checkpoints store it as a batched array).
    # Visualization only needs a valid integer level for map selection.
    try:
        if not hasattr(env_cfgs.curriculum, "_fields"):
            env_cfgs = env_cfgs._replace(curriculum=CurriculumConfig())
    except Exception:
        env_cfgs = EnvConfig()

    # Robustly sanitize new scalar fields that can be malformed when loading older checkpoints
    # with changed EnvConfig layout.
    def _safe_scalar_float(v, default):
        try:
            arr = jnp.asarray(v)
            if arr.size == 0:
                return jnp.float32(default)
            return jnp.asarray(arr.reshape(-1)[0], dtype=jnp.float32)
        except Exception:
            return jnp.float32(default)

    env_cfgs = env_cfgs._replace(
        cabin_alignment_coefficient=_safe_scalar_float(
            getattr(env_cfgs, "cabin_alignment_coefficient", -0.04), -0.04
        )
    )
    
    # Replicate env_cfg leaves to `n_envs` without converting nested NamedTuples
    # (EnvConfig.curriculum must remain a CurriculumConfig so `.level` exists under vmap/jit).
    def _replicate_value(x):
        if x is None:
            return None
        # Plain tuples (e.g., agent_types/action_types): make [n_envs, ...]
        if isinstance(x, tuple) and not hasattr(x, "_fields"):
            return jnp.asarray(x)[None, ...].repeat(n_envs, 0)
        # Python scalars -> [n_envs]
        if isinstance(x, (int, float, bool)):
            return jnp.asarray([x] * n_envs)
        # Arrays / array-likes
        x_arr = jnp.asarray(x)
        if x_arr.ndim >= 1 and x_arr.shape[0] == 1:
            return x_arr[0][None, ...].repeat(n_envs, 0)
        return x_arr[None, ...].repeat(n_envs, 0)

    def _replicate_namedtuple(nt):
        updates = {}
        for f in getattr(nt, "_fields", ()):
            v = getattr(nt, f)
            if hasattr(v, "_fields"):  # nested NamedTuple
                updates[f] = _replicate_namedtuple(v)
            else:
                updates[f] = _replicate_value(v)
        return nt._replace(**updates)

    if hasattr(env_cfgs, "_fields"):
        env_cfgs = _replicate_namedtuple(env_cfgs)
    else:
        # Fallback: treat as pytree, but keep plain tuples as leaves.
        env_cfgs = jax.tree_util.tree_map(
            _replicate_value,
            env_cfgs,
            is_leaf=lambda x: isinstance(x, tuple) and not hasattr(x, "_fields"),
        )
    
    # Load maps from YAML config if --config is specified
    batch_cfg = None
    if args.config is not None:
        try:
            from configs.training_configs import get_config
            preset = get_config(args.config)
            print(f"\n📦 Loading config preset: '{args.config}'")
            print(f"   Description: {preset.description}")
            
            # Convert MapLevel objects to curriculum levels format
            if preset.maps and len(preset.maps) > 0:
                curriculum_levels = []
                for map_level in preset.maps:
                    rewards_type = RewardsType.DENSE if map_level.rewards_type == "DENSE" else RewardsType.SPARSE
                    curriculum_levels.append({
                        "maps_path": map_level.maps_path,
                        "max_steps_in_episode": map_level.max_steps_in_episode,
                        "rewards_type": rewards_type,
                        "apply_trench_rewards": map_level.apply_trench_rewards,
                    })
                
                # Create custom CurriculumGlobalConfig with overridden levels
                class CustomCurriculumGlobalConfig(CurriculumGlobalConfig):
                    levels = curriculum_levels
                
                batch_cfg = BatchConfig(curriculum_global=CustomCurriculumGlobalConfig())
                print(f"📍 Using maps from config: {[lvl['maps_path'] for lvl in curriculum_levels]}")

                # Since visualization uses a no-op curriculum manager, explicitly apply
                # the first level's runtime env settings to avoid stale/misaligned values
                # from checkpoint env_config (e.g. wrong max_steps causing early termination).
                first_level = curriculum_levels[0]
                env_cfgs = env_cfgs._replace(
                    # Keep leading batch dimension for vmap over env_cfgs
                    agent_types=jnp.asarray(tuple(preset.agent_types), dtype=jnp.int32)[None, ...].repeat(n_envs, 0),
                    action_types=jnp.asarray(tuple(preset.action_types), dtype=jnp.int32)[None, ...].repeat(n_envs, 0),
                    max_steps_in_episode=jnp.full((n_envs,), int(first_level["max_steps_in_episode"]), dtype=jnp.int32),
                    apply_trench_rewards=jnp.full((n_envs,), bool(first_level["apply_trench_rewards"]), dtype=jnp.bool_),
                )
        except ImportError as e:
            print(f"⚠️  Failed to import training configs: {e}")
        except ValueError as e:
            print(f"⚠️  {e}")
    
    if batch_cfg is None:
        print("📍 Using default maps from config.py")
        batch_cfg = BatchConfig()
    
    shuffle_maps = True
    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        rendering=True,
        n_envs_x_rendering=args.n_envs_x,
        n_envs_y_rendering=args.n_envs_y,
        display=False,
        shuffle_maps=shuffle_maps,
    )

    # For visualization, freeze curriculum: use the checkpoint env_config as-is
    # and avoid changing levels or rewards over time. This also sidesteps
    # curriculum NamedTuple tracing issues under vmap.
    class _NoopCurriculumManager:
        def reset_cfgs(self, env_cfgs):
            return env_cfgs

        def update_cfgs(self, timesteps, rng):
            return timesteps

    env.curriculum_manager = _NoopCurriculumManager()
    config.num_embeddings_agent_min = 60

    print(f"🧠 Model size preset: {getattr(config, 'model_size', 'base')}")
    model = load_neural_network(config, env)
    model_params = log["model"]
    obs_seq, cum_rewards, state_seq = rollout_episode(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        seed=args.seed,
    )

    
    # Render each frame with dirt gradient (hide interaction map in visualization)
    for i, o in enumerate(tqdm(obs_seq, desc="Rendering")):
        # Try using state action_map instead of observation action_map
        if i < len(state_seq):
            # Create modified observation with raw state action_map
            modified_obs = dict(o)
            modified_obs['action_map'] = state_seq[i].world.action_map.map
            # Hide interaction cones by zeroing the correct key used by renderer
            if 'interaction_mask' in modified_obs:
                modified_obs['interaction_mask'] = jnp.zeros_like(modified_obs['interaction_mask'])
            env.terra_env.render_obs_pygame(modified_obs, generate_gif=True)
        else:
            # Hide interaction cones by zeroing the correct key used by renderer
            obs_no_interact = dict(o)
            if 'interaction_mask' in obs_no_interact:
                obs_no_interact['interaction_mask'] = jnp.zeros_like(obs_no_interact['interaction_mask'])
            env.terra_env.render_obs_pygame(obs_no_interact, generate_gif=True)

    env.terra_env.rendering_engine.create_gif(args.out_path)
    print(f"GIF saved to {args.out_path}") 
