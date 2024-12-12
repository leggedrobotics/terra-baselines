import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from terra.env import TerraEnvBatch
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
from utils.utils_ppo import obs_to_model_input, wrap_action, policy
import mctx

from train import TrainConfig  # needed for unpickling checkpoints
from tensorflow_probability.substrates import jax as tfp

def load_neural_network(config, env):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env)
    return model

def root_fn(apply_fn, params, timestep, config):
    obs = timestep.observation
    inp = obs_to_model_input(obs, config)
    value, dist = apply_fn(params, inp)
    return mctx.RootFnOutput(
        prior_logits=dist.logits,  # unnormalized action logits
        value=value[:, 0],         # value of the root state
        embedding=timestep,        # embedding = current timestep
    )

def make_recurrent_fn(env, apply_fn, config):
    def recurrent_fn(params, rng, actions, embedding):
        # embedding is the current timestep
        timestep = embedding
        rng, rng_env = jrandom.split(rng)
        rng_envs = jrandom.split(rng_env, config.num_test_rollouts)  # adapt if needed

        actions = actions.astype(jnp.int32)
        terra_actions = wrap_action(actions, env.batch_cfg.action_type)
        next_timestep = env.step(timestep, terra_actions, rng_envs)
        next_obs = next_timestep.observation

        inp = obs_to_model_input(next_obs, config)
        value, dist = apply_fn(params, inp)

        reward = next_timestep.reward
        done = next_timestep.done
        discount = (1.0 - done) * config.gamma

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=dist.logits,
            value=value[:,0],
        ), next_timestep
    return recurrent_fn

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
):
    rng = jrandom.PRNGKey(seed)
    rng, _rng = jrandom.split(rng)
    rng_reset = jrandom.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)

    tile_size = env_cfgs.tile_size[0].item()
    move_tiles = env_cfgs.agent.move_tiles[0].item()
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
    elif action_type == WheeledAction:
        move_actions = (WheeledActionType.FORWARD, WheeledActionType.BACKWARD)
        l_actions = (
            WheeledActionType.CLOCK_FORWARD,
            WheeledActionType.CLOCK_BACKWARD,
            WheeledActionType.ANTICLOCK_FORWARD,
            WheeledActionType.ANTICLOCK_BACKWARD,
        )
        do_action = WheeledActionType.DO
    else:
        raise (ValueError(f"{action_type=}"))

    obs = timestep.observation
    areas = (obs["target_map"] == -1).sum(
        tuple([i for i in range(len(obs["target_map"].shape))][1:])
    ) * (tile_size**2)
    target_maps_init = obs["target_map"].copy()
    dig_tiles_per_target_map_init = (target_maps_init == -1).sum(
        tuple([i for i in range(len(target_maps_init.shape))][1:])
    )

    # Setup MCTS functions
    # model.apply returns (value, dist), exactly as PPO model does
    def apply_model(params, inp):
        val, logits_pi = model.apply(params, inp)
        pi = tfp.distributions.Categorical(logits=logits_pi)
        return val, pi

    recurrent = make_recurrent_fn(env, apply_model, rl_config)

    t_counter = 0
    reward_seq = []
    episode_done_once = None
    episode_length = None
    move_cumsum = None
    do_cumsum = None
    obs_seq = {}

    while True:
        obs_seq = _append_to_obs(obs, obs_seq)
        # Run MCTS from the current state
        root = root_fn(apply_model, model_params, timestep, rl_config)

        rng, rng_mcts = jrandom.split(rng)
        policy_output = mctx.gumbel_muzero_policy(
            params=model_params,
            rng_key=rng_mcts,
            root=root,
            recurrent_fn=recurrent,
            num_simulations=rl_config.num_simulations,
        )

        actions = policy_output.action.astype(jnp.int32)
        # print ppo action without mcts
        obs_model = obs_to_model_input(timestep.observation, rl_config)
        v, logits_pi = model.apply(model_params, obs_model)
        action_ppo = np.argmax(logits_pi, axis=-1)
        print("--------------------------------")
        print("action_mct:", actions)
        print("action_ppo:", action_ppo)

        rng, rng_step = jrandom.split(rng)
        rng_step = jrandom.split(rng_step, rl_config.num_test_rollouts)
        timestep = env.step(timestep, wrap_action(actions, action_type), rng_step)

        reward = timestep.reward
        next_obs = timestep.observation
        done = timestep.done

        reward_seq.append(reward)

        t_counter += 1
        if jnp.all(done).item() or t_counter == max_frames:
            break
        obs = next_obs

        # Log stats
        if episode_done_once is None:
            episode_done_once = done
        if episode_length is None:
            episode_length = jnp.zeros_like(done, dtype=jnp.int32)
        if move_cumsum is None:
            move_cumsum = jnp.zeros_like(done, dtype=jnp.int32)
        if do_cumsum is None:
            do_cumsum = jnp.zeros_like(done, dtype=jnp.int32)

        episode_done_once = episode_done_once | done
        episode_length += ~episode_done_once

        move_cumsum_tmp = jnp.zeros_like(done, dtype=jnp.int32)
        for move_action in move_actions:
            move_mask = (actions == move_action) & (~episode_done_once)
            move_cumsum_tmp += move_tiles * tile_size * move_mask.astype(jnp.int32)
        for la in l_actions:
            l_mask = (actions == la) & (~episode_done_once)
            move_cumsum_tmp += 2 * move_tiles * tile_size * l_mask.astype(jnp.int32)
        move_cumsum += move_cumsum_tmp

        do_cumsum += (actions == do_action) & (~episode_done_once)
        print("t_counter:", t_counter)
        print("done:", episode_done_once)
        print("reward:", reward)

    # Compute final stats
    move_cumsum *= episode_done_once
    path_efficiency = (move_cumsum / jnp.sqrt(areas))[episode_done_once]
    path_efficiency_std = path_efficiency.std()
    path_efficiency_mean = path_efficiency.mean()

    reference_workspace_area = 0.5 * np.pi * (8**2)
    n_dig_actions = do_cumsum // 2
    workspaces_efficiency = (
        reference_workspace_area
        * ((n_dig_actions * episode_done_once) / areas)[episode_done_once]
    )
    workspaces_efficiency_mean = workspaces_efficiency.mean()
    workspaces_efficiency_std = workspaces_efficiency.std()

    dug_tiles_per_action_map = (obs["action_map"] == -1).sum(
        tuple([i for i in range(len(obs["action_map"].shape))][1:])
    )
    coverage_ratios = dug_tiles_per_action_map / dig_tiles_per_target_map_init
    coverage_scores = episode_done_once + (~episode_done_once) * coverage_ratios
    coverage_score_mean = coverage_scores.mean()
    coverage_score_std = coverage_scores.std()

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
        default=80,
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
    args, _ = parser.parse_known_args()
    n_envs = args.n_envs

    log = load_pkl_object(f"{args.run_name}")
    config = log["train_config"]
    config.num_test_rollouts = n_envs
    config.num_devices = 1
    # The MCTS code uses config.num_simulations etc. Ensure these are set or default:
    if not hasattr(config, 'num_simulations'):
        config.num_simulations = 32

    env_cfgs = log["env_config"]
    env_cfgs = jax.tree_map(
        lambda x: x[0][None, ...].repeat(n_envs, 0), env_cfgs
    )  # replicate for n_envs
    shuffle_maps = False
    env = TerraEnvBatch(rendering=False, shuffle_maps=shuffle_maps)
    config.num_embeddings_agent_min = 60

    model = load_neural_network(config, env)
    model_params = log["model"]
    deterministic = bool(args.deterministic)
    print(f"\nDeterministic = {deterministic}\n")

    cum_rewards, stats, _ = rollout_episode(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        deterministic=deterministic,
        seed=args.seed,
    )

    print_stats(stats)
