import jax.numpy as jnp
import numpy as np
import tensorflow_probability as tfp
from terra.env import TerraEnvBatch
import jax
from utils.utils_ppo import obs_to_model_input, wrap_action

def get_best_action(
    env_mcts: TerraEnvBatch,
    model,
    model_params,
    timestep_mcts,
    rng, 
    rl_config,
    epsilon=0.9,  # Probability of choosing a random action
    num_rollouts=10,  # Number of rollouts to perform for each action decision
    n_steps=4,  # Number of future steps to simulate
    n_envs=8
    ):
    obs = timestep_mcts.observation
    best_action = np.full(n_envs, -1, dtype=np.int32)  # Initialize with -1 (invalid action)
    best_total_reward = -np.inf * np.ones(n_envs)

    def simulate_action_sequence(current_timestep, rng, depth):
        if depth == n_steps:  # This is the first step
            rng, rng_act, rng_step = jax.random.split(rng, 3)
            obs_model = obs_to_model_input(current_timestep.observation, rl_config)
            _, logits_pi = model.apply(model_params, obs_model)
            if np.random.rand() < epsilon:
                # Explore: choose a random action for each environment
                action = np.random.randint(logits_pi.shape[-1], size=n_envs)
            else:
                # Exploit: choose the best action based on model prediction
                action = np.argmax(logits_pi, axis=-1)
            first_action = action
        else:
            first_action = None  # We only care about the first action
            action = np.argmax(model.apply(model_params, obs_to_model_input(current_timestep.observation, rl_config))[1], axis=-1)

        if depth == 0:
            return 0, first_action  # No more rewards beyond this depth

        # Simulate the action
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        rng_step = jax.random.split(rng_step, rl_config.num_test_rollouts)
        next_timestep = env_mcts.step(
            current_timestep, wrap_action(action, env_mcts.batch_cfg.action_type), rng_step
        )
        immediate_reward = next_timestep.reward
        future_reward, _ = simulate_action_sequence(next_timestep, rng, depth - 1)

        return immediate_reward + future_reward, first_action

    for _ in range(num_rollouts):
        rng, rng_sim = jax.random.split(rng)
        total_reward, action = simulate_action_sequence(timestep_mcts, rng_sim, n_steps)
        # Update best action and reward for each environment
        for i in range(n_envs):
            if total_reward[i] > best_total_reward[i]:
                best_total_reward[i] = total_reward[i]
                best_action[i] = action[i]

    return best_action
