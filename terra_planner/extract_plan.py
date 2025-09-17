import numpy as np
import jax
import argparse
import pickle
import sys
from pathlib import Path

# Add the parent directory to the path so we can import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.models import load_neural_network
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
from terra.actions import TrackedAction, WheeledAction, TrackedActionType, WheeledActionType
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action
from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints


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

        # Check if DO action and record state BEFORE executing the action
        if action[0] == do_action:
            print(f"DO action at step {t_counter}")
            action_map_before = jnp.squeeze(timestep.observation["action_map"]).copy()
            traversability_mask = jnp.squeeze(timestep.observation["traversability_mask"]).copy()
            agent_state_before = jnp.squeeze(timestep.observation["agent_state"]).copy()
            loaded_before = jnp.bool_(agent_state_before[5])

            # Update previous actions
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action)

            # Take step in environment
            rng_step = jax.random.split(rng_step, 1)
            timestep = env.step(
                timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
            )

            # Get state AFTER executing the action
            action_map_after = jnp.squeeze(timestep.observation["action_map"]).copy()
            agent_state_after = jnp.squeeze(timestep.observation["agent_state"]).copy()
            loaded_after = jnp.bool_(agent_state_after[5])

            changed_tiles = action_map_before != action_map_after
            terrain_modification_mask = changed_tiles.astype(jnp.bool_)

            if loaded_before != loaded_after:
                if not loaded_before and loaded_after:
                    print(f"  Digging detected: {jnp.sum(changed_tiles)} tiles modified")
                elif loaded_before and not loaded_after:
                    print(f"  Dumping detected: {jnp.sum(changed_tiles)} tiles modified")
            else:
                # Case 3: loaded state did not change, but still check for modifications
                print(f"  Loaded state unchanged ({loaded_before}), {jnp.sum(changed_tiles)} tiles modified")

            plan_entry = {
                'step': t_counter,
                'traversability_mask': traversability_mask,
                'agent_state': {
                    'pos_base': (agent_state_before[0], agent_state_before[1]),
                    'angle_base': agent_state_before[2],
                    'wheel_angle': agent_state_before[4],
                },
                'terrain_modification_mask': terrain_modification_mask,
                'loaded_state_change': {
                    'before': loaded_before,
                    'after': loaded_after,
                }
            }
            plan.append(plan_entry)
        else:
            # Update previous actions
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action)

            # Take step in environment
            rng_step = jax.random.split(rng_step, 1)
            timestep = env.step(
                timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
            )

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

    # Disable action map clipping to see full terrain state
    print(f"Original clip_action_maps setting: {config.clip_action_maps}")
    config.clip_action_maps = False
    # Add the missing attribute that the model expects when clipping is disabled
    config.maps_net_normalization_bounds = [-100, 100]  # Reasonable range for terrain heights
    print(f"Modified clip_action_maps setting: {config.clip_action_maps}")
    print(f"Added maps_net_normalization_bounds: {config.maps_net_normalization_bounds}")

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
