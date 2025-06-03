import numpy as np
import jax
import argparse
import pickle
from pathlib import Path
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
from terra.actions import TrackedAction, WheeledAction, TrackedActionType, WheeledActionType
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action
from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints


def load_neural_network(config, env):
    """Load neural network model based on config and environment."""
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env)
    return model


def extract_plan(env, model, model_params, env_cfgs, rl_config, max_frames, seed):
    """Extract plan by capturing action_map and robot state on DO actions."""
    print(f"Using seed={seed}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, 1)  # Just one environment
    timestep = env.reset(env_cfgs, rng_reset)
    prev_actions = jnp.zeros(
        (1, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    # Determine action type and DO action
    action_type = env.batch_cfg.action_type
    if action_type == TrackedAction:
        do_action = TrackedActionType.DO
    elif action_type == WheeledAction:
        do_action = WheeledActionType.DO
    else:
        raise ValueError(f"Unknown action type: {action_type}")

    print(f"Action type: {action_type.__name__}, DO action value: {do_action}")

    # Plan storage
    plan = []
    t_counter = 0

    while True:
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # Get action from policy
        obs_model = obs_to_model_input(timestep.observation, prev_actions, rl_config)
        v, logits_pi = model.apply(model_params, obs_model)
        pi = tfp.distributions.Categorical(logits=logits_pi)
        action = pi.sample(seed=rng_act)

        # Update previous actions
        prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        prev_actions = prev_actions.at[:, 0].set(action)

        # Take step in environment
        rng_step = jax.random.split(rng_step, 1)
        timestep = env.step(
            timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
        )

        # Check if DO action and record state if it is
        if action[0] == do_action:
            print(f"DO action at step {t_counter}")
            plan_entry = {
                'step': t_counter,
                'action_map': timestep.observation["action_map"][0].copy(),
                'agent_state': {
                    'pos_base': timestep.observation["agent_state"][0].copy(),
                    'angle_base': timestep.observation["agent_state"][1].copy(),
                    'wheel_angle': timestep.observation["agent_state"][3].copy(),
                    'loaded': timestep.observation["agent_state"][4].copy()
                }
            }
            plan.append(plan_entry)

        t_counter += 1
        print(f"Step {t_counter}, Action: {action[0]}")

        # Check if done
        if jnp.all(timestep.info["task_done"]).item() or t_counter == max_frames:
            break

    return plan


def main():
    parser = argparse.ArgumentParser(description="Extract plan from policy")
    parser.add_argument(
        "-policy",
        "--policy_path",
        type=str,
        required=True,
        help="Path to the policy .pkl file"
    )
    parser.add_argument(
        "-map",
        "--map_path",
        type=str,
        required=True,
        help="Path to the map file"
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=500,
        help="Maximum number of steps"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="plan.pkl",
        help="Output path for the plan"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )

    args = parser.parse_args()

    # Load policy
    log = load_pkl_object(args.policy_path)
    config = log["train_config"]
    config.num_test_rollouts = 1  # Only one environment
    config.num_devices = 1

    # Create environment
    env_cfgs = log["env_config"]
    env_cfgs = jax.tree_map(lambda x: x[0][None, ...], env_cfgs)
    env = TerraEnvBatch(rendering=False, shuffle_maps=False, single_map_path=args.map_path)

    # Load neural network
    model = load_neural_network(config, env)
    model_params = log["model"]

    # Extract plan
    plan = extract_plan(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        seed=args.seed
    )

    # Save plan
    output_path = Path(args.output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(plan, f)

    print(f"Plan extracted and saved to {output_path}")
    print(f"Total DO actions: {len(plan)}")


if __name__ == "__main__":
    main()
