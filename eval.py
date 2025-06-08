import jax
import jax.numpy as jnp
from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)
from utils.utils_ppo import obs_to_model_input, wrap_action

from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints


def _append_to_obs(o, obs_log):
    if obs_log == {}:
        obs_log = {k: v[:, None] for k, v in o.items()}
    else:
        obs_log = {
            k: jnp.concatenate((v, o[k][:, None]), axis=1) for k, v in obs_log.items()
        }
    return obs_log


def rollout_episode(
    env: TerraEnvBatch,
    model,
    model_params,
    env_cfgs,
    rl_config,
    max_frames,
    deterministic,
    seed,
):
    """
    Rollout episodes with 2-agent centralized policy for visualization.
    """
    print(f"Using {seed=}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)
    
    # Initialize previous actions for both agents
    prev_actions_1 = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )
    prev_actions_2 = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    cum_reward = jnp.zeros(rl_config.num_test_rollouts)
    cum_reward_history = jnp.array([]).reshape(rl_config.num_test_rollouts, 0)
    obs_log = {}
    frames = 0
    done_envs = jnp.zeros(rl_config.num_test_rollouts, dtype=bool)

    print(f"{timestep.observation['agent_state_1'].shape=}")
    print(f"{timestep.observation['agent_state_2'].shape=}")

    while True:
        rng_model_1, rng_model_2, rng_step_base, rng_next_loop = jax.random.split(rng, 4)
        rng = rng_next_loop # Update rng for next iteration loop

        obs_current = timestep.observation
        
        # --- Action for Agent 1 ---
        obs_model_input_a1 = obs_to_model_input(obs_current, prev_actions_1, prev_actions_2, rl_config)
        _, logits_a1 = model.apply(model_params, obs_model_input_a1)
        dist_a1 = tfp.distributions.Categorical(logits=logits_a1)
        if deterministic:
            action1_raw = jnp.argmax(logits_a1, axis=-1)
        else:
            action1_raw = dist_a1.sample(seed=rng_model_1)
        action_env_1 = wrap_action(action1_raw, env.batch_cfg.action_type)

        # --- Action for Agent 2 ---
        obs_for_agent2_persp = obs_current.copy()
        obs_for_agent2_persp["agent_state_1"] = obs_current["agent_state_2"]
        obs_for_agent2_persp["agent_state_2"] = obs_current["agent_state_1"]
        local_map_keys_agent1 = [
            "local_map_action_neg", "local_map_action_pos", "local_map_target_neg", 
            "local_map_target_pos", "local_map_dumpability", "local_map_obstacles"
        ]
        local_map_keys_agent2 = [key + "_2" for key in local_map_keys_agent1]
        for key1, key2 in zip(local_map_keys_agent1, local_map_keys_agent2):
            obs_for_agent2_persp[key1] = obs_current[key2]
            obs_for_agent2_persp[key2] = obs_current[key1]

        obs_model_input_a2 = obs_to_model_input(obs_for_agent2_persp, prev_actions_2, prev_actions_1, rl_config)
        _, logits_a2 = model.apply(model_params, obs_model_input_a2)
        dist_a2 = tfp.distributions.Categorical(logits=logits_a2)
        if deterministic:
            action2_raw = jnp.argmax(logits_a2, axis=-1)
        else:
            action2_raw = dist_a2.sample(seed=rng_model_2)
        action_env_2 = wrap_action(action2_raw, env.batch_cfg.action_type)
        
        _rng_step_split = jax.random.split(rng_step_base, rl_config.num_test_rollouts)
        timestep = env.step(timestep, action_env_1, action_env_2, _rng_step_split)

        # Log observations for visualization
        obs_log = _append_to_obs(timestep.observation, obs_log)

        # Update cumulative rewards
        cum_reward += timestep.reward
        cum_reward_history = jnp.concatenate(
            [cum_reward_history, cum_reward[:, None]], axis=1
        )

        # Update done environments
        done_envs = done_envs | timestep.done

        # UPDATE PREVIOUS ACTIONS for both agents
        prev_actions_1 = jnp.roll(prev_actions_1, shift=1, axis=-1)
        prev_actions_1 = prev_actions_1.at[..., 0].set(action1_raw)
        
        prev_actions_2 = jnp.roll(prev_actions_2, shift=1, axis=-1)
        prev_actions_2 = prev_actions_2.at[..., 0].set(action2_raw)

        frames += 1

        # Check termination conditions
        if jnp.all(done_envs) or frames >= max_frames:
            break

    print(f"Rollout completed after {frames} frames")
    print(f"Final rewards: {cum_reward}")
    
    return {
        "observations": obs_log,
        "cumulative_rewards": cum_reward_history,
        "final_rewards": cum_reward,
        "total_frames": frames,
        "done_envs": done_envs,
    }


def print_stats(stats):
    print(f"Total frames: {stats['total_frames']}")
    print(f"Final rewards: {stats['final_rewards']}")
    print(f"Average reward: {jnp.mean(stats['final_rewards']):.2f}")
    print(f"Max reward: {jnp.max(stats['final_rewards']):.2f}")
    print(f"Min reward: {jnp.min(stats['final_rewards']):.2f}")
    print(f"Environments completed: {jnp.sum(stats['done_envs'])}/{len(stats['done_envs'])}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained Terra agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--max_frames", type=int, default=1000, help="Maximum frames per episode")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = load_pkl_object(args.checkpoint)
    train_config = checkpoint["train_config"]
    env_config = checkpoint["env_config"]
    model_params = checkpoint["model"]
    
    # Setup environment
    env = TerraEnvBatch()
    model = load_neural_network(train_config, env)
    
    # Create evaluation config
    eval_config = type('EvalConfig', (), {
        'num_test_rollouts': args.num_episodes,
        'num_prev_actions': train_config.num_prev_actions,
        'clip_action_maps': train_config.clip_action_maps,
    })()
    
    # Run evaluation
    stats = rollout_episode(
        env=env,
        model=model,
        model_params=model_params,
        env_cfgs=env_config,
        rl_config=eval_config,
        max_frames=args.max_frames,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    
    print_stats(stats)
    
    # Optional: Render episodes
    if args.render:
        print("Rendering episodes...")
        for i in range(min(5, args.num_episodes)):  # Render first 5 episodes
            obs = {k: v[i] for k, v in stats["observations"].items()}
            env.render_obs_pygame(obs, generate_gif=True)
            print(f"Rendered episode {i+1}")
