import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from terra.env import TerraEnvBatch
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
from utils.utils_ppo import obs_to_model_input, wrap_action, policy
import mctx
from functools import partial
import time

from train import TrainConfig  # needed for unpickling checkpoints
from tensorflow_probability.substrates import jax as tfp


def fix_env_cfg_dtypes(env_cfgs):
    """
    Fix the dtypes in env_cfgs to prevent JAX type promotion issues.
    Python ints in the config cause int32 promotion during JAX operations.
    We need to ensure config values that are used in JAX ops have int8 dtype.
    """
    # Fix agent config integer fields that participate in JAX operations
    fixed_agent = env_cfgs.agent._replace(
        angles_base=jnp.int8(env_cfgs.agent.angles_base),
        angles_cabin=jnp.int8(env_cfgs.agent.angles_cabin),
        max_wheel_angle=jnp.int8(env_cfgs.agent.max_wheel_angle),
        move_tiles=jnp.int8(env_cfgs.agent.move_tiles),
        dig_depth=jnp.int8(env_cfgs.agent.dig_depth),
        height=jnp.int8(env_cfgs.agent.height),
        width=jnp.int8(env_cfgs.agent.width),
    )
    return env_cfgs._replace(agent=fixed_agent)


def load_neural_network(config, env):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env)
    return model

def root_fn(apply_fn, params, timestep, prev_actions, config):
    obs = timestep.observation
    inp = obs_to_model_input(obs, prev_actions, config)
    value, dist = apply_fn(params, inp)
    return mctx.RootFnOutput(
        prior_logits=dist.logits,  # unnormalized action logits
        value=value[:, 0],         # value of the root state
        embedding=(timestep, prev_actions),  # embedding = (timestep, prev_actions)
    )

def make_recurrent_fn(env, apply_fn, config):
    def recurrent_fn(params, rng, actions, embedding):
        # embedding is (timestep, prev_actions)
        timestep, prev_actions = embedding
        rng, rng_env = jrandom.split(rng)
        rng_envs = jrandom.split(rng_env, config.num_test_rollouts)

        actions = actions.astype(jnp.int32)
        terra_actions = wrap_action(actions, env.batch_cfg.action_type)
        next_timestep = env.step(timestep, terra_actions, rng_envs)
        next_obs = next_timestep.observation

        # Update prev_actions for the simulation
        next_prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        next_prev_actions = next_prev_actions.at[:, 0].set(actions)

        inp = obs_to_model_input(next_obs, next_prev_actions, config)
        value, dist = apply_fn(params, inp)

        reward = next_timestep.reward
        done = next_timestep.done
        discount = (1.0 - done) * config.gamma

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=dist.logits,
            value=value[:,0],
        ), (next_timestep, next_prev_actions)
    return recurrent_fn


def make_mcts_step_fn(model, env, config, use_mcts=True):
    """Create a JIT-compiled MCTS step function for maximum GPU utilization."""
    
    def apply_model(params, inp):
        val, logits_pi = model.apply(params, inp)
        pi = tfp.distributions.Categorical(logits=logits_pi)
        return val, pi
    
    recurrent_fn = make_recurrent_fn(env, apply_model, config)
    
    @jax.jit
    def mcts_step(params, rng, timestep, prev_actions):
        """Single MCTS step - fully JIT compiled."""
        # Compute root
        obs = timestep.observation
        inp = obs_to_model_input(obs, prev_actions, config)
        value, dist = apply_model(params, inp)
        
        # Get greedy PPO action for comparison
        ppo_action = jnp.argmax(dist.logits, axis=-1)
        
        root = mctx.RootFnOutput(
            prior_logits=dist.logits,
            value=value[:, 0],
            embedding=(timestep, prev_actions),
        )
        
        # Run MCTS
        rng, rng_mcts = jrandom.split(rng)
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_mcts,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
        )
        
        mcts_action = policy_output.action.astype(jnp.int32)
        
        # Choose which action to use
        actions = mcts_action if use_mcts else ppo_action
        
        # Step environment
        rng, rng_step = jrandom.split(rng)
        rng_steps = jrandom.split(rng_step, config.num_test_rollouts)
        action_type = env.batch_cfg.action_type
        next_timestep = env.step(timestep, wrap_action(actions, action_type), rng_steps)
        
        # Update prev_actions
        next_prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        next_prev_actions = next_prev_actions.at[:, 0].set(actions)
        
        # Return both actions for debugging
        # Return task_done (successful completion) not just done (could be timeout)
        task_done = next_timestep.info["task_done"]
        return rng, next_timestep, next_prev_actions, actions, next_timestep.reward, task_done, ppo_action, mcts_action, value[:, 0]
    
    return mcts_step

def _append_to_obs(o, obs_log):
    if obs_log == {}:
        return {k: v[:, None] for k, v in o.items()}
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
    use_mcts=True,
    debug=False,
):
    rng = jrandom.PRNGKey(seed)
    rng, _rng = jrandom.split(rng)
    rng_reset = jrandom.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)

    # Store these as JAX arrays to avoid repeated .item() calls
    tile_size = env_cfgs.tile_size[0]
    move_tiles = env_cfgs.agent.move_tiles[0]
    action_type = env.batch_cfg.action_type

    # Determine action types based on agent type
    from terra.actions import (
        WheeledAction,
        TrackedAction,
        WheeledActionType,
        TrackedActionType,
    )
    if action_type == TrackedAction:
        move_actions = (TrackedActionType.FORWARD, TrackedActionType.BACKWARD)
        l_actions = ()
        do_action = TrackedActionType.DO
        action_names = ['FWD', 'BWD', 'CLK', 'ACLK', 'CAB_CLK', 'CAB_ACLK', 'DO']
    elif action_type == WheeledAction:
        move_actions = (WheeledActionType.FORWARD, WheeledActionType.BACKWARD)
        l_actions = (WheeledActionType.WHEELS_LEFT, WheeledActionType.WHEELS_RIGHT)
        do_action = WheeledActionType.DO
        action_names = ['FWD', 'BWD', 'WHL_L', 'WHL_R', 'CAB_CLK', 'CAB_ACLK', 'DO']
    else:
        raise (ValueError(f"{action_type=}"))

    obs = timestep.observation
    tile_size_float = float(tile_size)
    move_tiles_float = float(move_tiles)
    areas = (obs["target_map"] == -1).sum(
        tuple([i for i in range(len(obs["target_map"].shape))][1:])
    ) * (tile_size_float**2)
    target_maps_init = obs["target_map"].copy()
    dig_tiles_per_target_map_init = (target_maps_init == -1).sum(
        tuple([i for i in range(len(target_maps_init.shape))][1:])
    )

    # Create JIT-compiled MCTS step function
    mcts_step = make_mcts_step_fn(model, env, rl_config, use_mcts=use_mcts)
    
    # Initialize prev_actions
    prev_actions = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    # Warmup JIT compilation (first call compiles, subsequent calls are fast)
    print("Warming up JIT compilation...")
    start_warmup = time.time()
    rng, timestep, prev_actions, actions, reward, done, ppo_act, mcts_act, values = mcts_step(
        model_params, rng, timestep, prev_actions
    )
    # Block until warmup is complete
    jax.block_until_ready(actions)
    print(f"JIT warmup complete in {time.time() - start_warmup:.2f}s")
    
    # Reset for actual run
    rng = jrandom.PRNGKey(seed)
    rng, _rng = jrandom.split(rng)
    rng_reset = jrandom.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)
    prev_actions = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    t_counter = 0
    reward_seq = []
    episode_done_once = None
    episode_length = None
    move_cumsum = None
    do_cumsum = None
    obs_seq = {}
    
    # Action tracking for debugging
    action_counts = {i: 0 for i in range(7)}
    mcts_ppo_diff_count = 0
    
    mode_str = "MCTS" if use_mcts else "PPO (greedy)"
    print(f"\nStarting rollout for {max_frames} steps with {rl_config.num_test_rollouts} envs using {mode_str}...")
    start_time = time.time()

    while True:
        # Run JIT-compiled MCTS step (all GPU work happens here)
        rng, timestep, prev_actions, actions, reward, done, ppo_act, mcts_act, values = mcts_step(
            model_params, rng, timestep, prev_actions
        )

        reward_seq.append(reward)
        t_counter += 1
        
        # Debug: compare MCTS vs PPO actions
        if debug and t_counter <= 20:
            # Print first 20 steps in detail
            ppo_np = np.array(ppo_act)
            mcts_np = np.array(mcts_act)
            actions_np = np.array(actions)
            values_np = np.array(values)
            rewards_np = np.array(reward)
            
            diff_count = (ppo_np != mcts_np).sum()
            print(f"\n--- Step {t_counter} ---")
            print(f"PPO actions:  {ppo_np[:8]}  (showing first 8 envs)")
            print(f"MCTS actions: {mcts_np[:8]}")
            print(f"Used actions: {actions_np[:8]}")
            print(f"Values:       {values_np[:8]}")
            print(f"Rewards:      {rewards_np[:8]}")
            print(f"MCTS != PPO:  {diff_count}/{len(ppo_np)}")
        
        # Track action distribution
        for a in np.array(actions):
            action_counts[int(a)] += 1
        mcts_ppo_diff_count += int((np.array(ppo_act) != np.array(mcts_act)).sum())
        
        # Only check termination every N steps to reduce sync overhead
        if t_counter >= max_frames:
            break
        
        # Check if all done (this forces a sync, but only once per step)
        all_done = jnp.all(done)
        if all_done:
            break

        # Log stats (keep on GPU, no sync needed)
        if episode_done_once is None:
            episode_done_once = done
            episode_length = jnp.zeros_like(done, dtype=jnp.int32)
            move_cumsum = jnp.zeros_like(done, dtype=jnp.int32)
            do_cumsum = jnp.zeros_like(done, dtype=jnp.int32)

        episode_done_once = episode_done_once | done
        episode_length = episode_length + (~episode_done_once).astype(jnp.int32)

        move_cumsum_tmp = jnp.zeros_like(done, dtype=jnp.int32)
        for move_action in move_actions:
            move_mask = (actions == move_action) & (~episode_done_once)
            move_cumsum_tmp = move_cumsum_tmp + (move_tiles_float * tile_size_float * move_mask).astype(jnp.int32)
        for la in l_actions:
            l_mask = (actions == la) & (~episode_done_once)
            move_cumsum_tmp = move_cumsum_tmp + (2 * move_tiles_float * tile_size_float * l_mask).astype(jnp.int32)
        move_cumsum = move_cumsum + move_cumsum_tmp

        do_cumsum = do_cumsum + ((actions == do_action) & (~episode_done_once)).astype(jnp.int32)
        
        # Print progress every 50 steps (reduced from every step)
        if t_counter % 50 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = t_counter / elapsed
            print(f"Step {t_counter}/{max_frames} | {steps_per_sec:.1f} steps/sec | "
                  f"Done: {int(episode_done_once.sum())}/{rl_config.num_test_rollouts}")

    elapsed = time.time() - start_time
    print(f"\nRollout complete: {t_counter} steps in {elapsed:.2f}s ({t_counter/elapsed:.1f} steps/sec)")
    
    # Print action distribution
    total_actions = sum(action_counts.values())
    print(f"\nAction distribution ({mode_str}):")
    for i, name in enumerate(action_names):
        pct = 100 * action_counts[i] / total_actions if total_actions > 0 else 0
        print(f"  {name} ({i}): {action_counts[i]:6d} ({pct:5.1f}%)")
    
    total_comparisons = t_counter * rl_config.num_test_rollouts
    print(f"\nMCTS != PPO: {mcts_ppo_diff_count}/{total_comparisons} ({100*mcts_ppo_diff_count/total_comparisons:.1f}%)")

    # Final stats computation (sync to CPU only at the end)
    obs = timestep.observation
    if episode_done_once is None:
        episode_done_once = done
        episode_length = jnp.zeros_like(done, dtype=jnp.int32)
        move_cumsum = jnp.zeros_like(done, dtype=jnp.int32)
        do_cumsum = jnp.zeros_like(done, dtype=jnp.int32)
    
    move_cumsum = move_cumsum * episode_done_once
    path_efficiency = (move_cumsum / jnp.sqrt(areas))[episode_done_once]
    path_efficiency_std = float(path_efficiency.std())
    path_efficiency_mean = float(path_efficiency.mean())

    reference_workspace_area = 0.5 * np.pi * (8**2)
    n_dig_actions = do_cumsum // 2
    workspaces_efficiency = (
        reference_workspace_area
        * ((n_dig_actions * episode_done_once) / areas)[episode_done_once]
    )
    workspaces_efficiency_mean = float(workspaces_efficiency.mean())
    workspaces_efficiency_std = float(workspaces_efficiency.std())

    dug_tiles_per_action_map = (obs["action_map"] == -1).sum(
        tuple([i for i in range(len(obs["action_map"].shape))][1:])
    )
    coverage_ratios = dug_tiles_per_action_map / dig_tiles_per_target_map_init
    coverage_scores = episode_done_once + (~episode_done_once) * coverage_ratios
    coverage_score_mean = float(coverage_scores.mean())
    coverage_score_std = float(coverage_scores.std())

    stats = {
        "episode_done_once": episode_done_once,
        "episode_length": episode_length,
        "path_efficiency": {
            "mean": path_efficiency_mean,
            "std": path_efficiency_std,
        },
        "workspaces_efficiency": {
            "mean": workspaces_efficiency_mean,
            "std": workspaces_efficiency_std,
        },
        "coverage": {
            "mean": coverage_score_mean,
            "std": coverage_score_std,
        },
    }
    
    return np.cumsum(np.array(reward_seq)), stats, obs_seq

def print_stats(stats):
    episode_done_once = stats["episode_done_once"]
    path_efficiency = stats["path_efficiency"]
    workspaces_efficiency = stats["workspaces_efficiency"]
    coverage = stats["coverage"]

    completion_rate = 100 * episode_done_once.sum() / len(episode_done_once)
    print("\nStats:\n")
    print(f"Completion: {completion_rate:.2f}%")
    print(
        f"Path efficiency: {path_efficiency['mean']:.2f} ({path_efficiency['std']:.2f})"
    )
    print(
        f"Workspaces efficiency: {workspaces_efficiency['mean']:.2f} ({workspaces_efficiency['std']:.2f})"
    )
    print(f"Coverage: {coverage['mean']:.2f} ({coverage['std']:.2f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        default="checkpoints/tracked-dense.pkl",
        help="Path to the checkpoint with the trained model.",
    )
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="Terra",
        help="Environment name.",
    )
    parser.add_argument(
        "-n",
        "--n_envs",
        type=int,
        default=32,
        help="Number of environments.",
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=305,
        help="Number of steps to run.",
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        type=int,
        default=1,
        help="Deterministic. 0 for stochastic (not directly relevant since MCTS picks argmax?), 1 for deterministic.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for the environment.",
    )
    parser.add_argument(
        "--no-mcts",
        action="store_true",
        help="Use greedy PPO instead of MCTS (for comparison).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug info for first 20 steps.",
    )
    parser.add_argument(
        "-sim",
        "--num_simulations",
        type=int,
        default=32,
        help="Number of MCTS simulations per step.",
    )
    args, _ = parser.parse_known_args()
    n_envs = args.n_envs
    use_mcts = not args.no_mcts

    log = load_pkl_object(f"{args.run_name}")
    config = log["train_config"]
    config.num_test_rollouts = n_envs
    config.num_devices = 1
    # Set MCTS parameters
    config.num_simulations = args.num_simulations
    if not hasattr(config, 'gamma'):
        config.gamma = 0.99

    env_cfgs = log["env_config"]
    env_cfgs = jax.tree_map(
        lambda x: x[0][None, ...].repeat(n_envs, 0), env_cfgs
    )  # replicate for n_envs
    
    # Fix config dtypes to prevent JAX type promotion issues during MCTS
    env_cfgs = fix_env_cfg_dtypes(env_cfgs)
    
    shuffle_maps = True  # Match eval.py behavior
    env = TerraEnvBatch(rendering=False, shuffle_maps=shuffle_maps)
    config.num_embeddings_agent_min = 60

    model = load_neural_network(config, env)
    model_params = log["model"]
    deterministic = bool(args.deterministic)
    
    mode_str = "MCTS" if use_mcts else "PPO (greedy)"
    print(f"\nMode: {mode_str}")
    print(f"MCTS simulations: {config.num_simulations}")
    print(f"Gamma: {config.gamma}")
    print(f"Debug: {args.debug}\n")

    cum_rewards, stats, _ = rollout_episode(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        deterministic=deterministic,
        seed=args.seed,
        use_mcts=use_mcts,
        debug=args.debug,
    )

    print_stats(stats)