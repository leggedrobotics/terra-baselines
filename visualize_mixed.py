"""
Visualization script for mixed agent training (Tracked Excavator + Skid Steer).
Functionality is identical to visualize.py, but defaults and documentation are for mixed agent checkpoints.
"""

import numpy as np
import jax
from tqdm import tqdm
from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action
from terra.state import State
import matplotlib.animation as animation
from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints
from terra.config import EnvConfig
import sys
from train_mixed_agents import MixedAgentTrainConfig
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
            v, logits_pi = model.apply(model_params, obs)
            pi = tfp.distributions.Categorical(logits=logits_pi)
            action = pi.sample(seed=rng_act)
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action)
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
        default=200,
        help="Number of steps.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./visualize_mixed_skid_exec_13.gif",
        help="Output path.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    args, _ = parser.parse_known_args()
    n_envs = args.n_envs_x * args.n_envs_y

    log = load_pkl_object(f"{args.run_name}")
    config = log["train_config"]
    config.num_test_rollouts = n_envs
    config.num_devices = 1

    env_cfgs = log["env_config"]
    
    # Custom handling for different field types
    def replicate_field(x):
        if x is None:
            return None
        # Handle agent_types tuple specially
        if isinstance(x, tuple) and len(x) == 2:
            # Convert tuple to array and replicate for each environment
            return jnp.array(x)[None, ...].repeat(n_envs, 0)
        # Handle scalars (int, float, bool) - just replicate the value
        elif isinstance(x, (int, float, bool)):
            return jnp.array([x] * n_envs)
        # Handle arrays - take first element and replicate
        else:
            return x[0][None, ...].repeat(n_envs, 0)
    
    env_cfgs = jax.tree_map(replicate_field, env_cfgs)
    suffle_maps = True
    env = TerraEnvBatch(
        rendering=True,
        n_envs_x_rendering=args.n_envs_x,
        n_envs_y_rendering=args.n_envs_y,
        display=False,
        shuffle_maps=suffle_maps,
    )
    config.num_embeddings_agent_min = 60

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

    
    # Render each frame with dirt gradient
    for i, o in enumerate(tqdm(obs_seq, desc="Rendering")):
        # Try using state action_map instead of observation action_map
        if i < len(state_seq):
            # Create modified observation with raw state action_map
            modified_obs = dict(o)
            modified_obs['action_map'] = state_seq[i].world.action_map.map
            env.terra_env.render_obs_pygame(modified_obs, generate_gif=True)
        else:
            env.terra_env.render_obs_pygame(o, generate_gif=True)

    env.terra_env.rendering_engine.create_gif(args.out_path)
    print(f"GIF saved to {args.out_path}") 