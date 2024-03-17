# utilities for PPO training and evaluation
import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
# from xminigrid.environment import Environment, EnvParams
from tensorflow_probability.substrates import jax as tfp
from typing import NamedTuple
import numpy as np
from terra.config import EnvConfig, MapType, RewardsType
from pathlib import Path
import pickle

# Training stuff
class Transition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    # for rnn policy
    prev_action: jax.Array
    prev_reward: jax.Array

def wrap_action(action, action_type):
    action = action_type.new(action[:, None])
    return action

def get_cfgs_init(train_cfg):
    num_devices = train_cfg.num_devices
    n_envs = train_cfg.num_envs
    num_rollouts = train_cfg.num_rollouts

    env_cfgs = jax.vmap(EnvConfig.parametrized)(
        np.array([60] * n_envs),
        np.array([60] * n_envs),
        np.array([num_rollouts] * n_envs),
        np.array([0] * n_envs),
        np.array([MapType.FOUNDATIONS] * n_envs),
        np.array([RewardsType.DENSE] * n_envs),
        np.array([False] * n_envs),
        )
    env_cfgs = jax.tree_map(
        lambda x: jax.numpy.reshape(x, (num_devices, x.shape[0] // num_devices, *x.shape[1:])), env_cfgs
    )
    return env_cfgs

def policy(
    apply_fn,
    params,
    obs,
):
    value, logits_pi = apply_fn(params, obs)
    pi = tfp.distributions.Categorical(logits=logits_pi)
    return value, pi

def clip_action_maps_in_obs(obs):
    """Clip action maps to [-1, 1] on the intuition that a binary map is enough for the agent to take decisions."""
    obs["action_map"] = jnp.clip(obs["action_map"], a_min=-1, a_max=1)
    obs["do_preview"] = jnp.clip(obs["do_preview"], a_min=-1, a_max=1)
    obs["dig_map"] = jnp.clip(obs["dig_map"], a_min=-1, a_max=1)
    return obs

def cut_local_map_layers(obs):
    """Only keep the first layer of the local map (makes sense especially if the arm extension action is blocked)"""
    obs["local_map_action_neg"] = obs["local_map_action_neg"][..., [0]]
    obs["local_map_action_pos"] = obs["local_map_action_pos"][..., [0]]
    obs["local_map_target_neg"] = obs["local_map_target_neg"][..., [0]]
    obs["local_map_target_pos"] = obs["local_map_target_pos"][..., [0]]
    obs["local_map_dumpability"] = obs["local_map_dumpability"][..., [0]]
    obs["local_map_obstacles"] = obs["local_map_obstacles"][..., [0]]
    return obs

def obs_to_model_input(obs):
    """
    Need to convert Dict to List to make it usable by JAX.
    """
    obs = jax.tree_map(lambda x: x.copy(), obs)  # TODO copy is a hack, find a proper solution, it crashes without it, but this makes it slow
    obs = clip_action_maps_in_obs(obs)
    # TODO only use the following function if mask_out_arm_extension is True
    obs = cut_local_map_layers(obs)

    obs = [
        obs["agent_state"],
        obs["local_map_action_neg"],
        obs["local_map_action_pos"],
        obs["local_map_target_neg"],
        obs["local_map_target_pos"],
        obs["local_map_dumpability"],
        obs["local_map_obstacles"],
        obs["action_map"],
        obs["target_map"],
        obs["traversability_mask"],
        obs["do_preview"],
        obs["dig_map"],
        obs["dumpability_mask"],
    ]
    return obs

def select_action_ppo(
    train_state: TrainState,
    obs: jnp.ndarray,
    rng: jax.random.PRNGKey,
):
    # Prepare policy input from Terra State
    obs = obs_to_model_input(obs)

    value, pi = policy(train_state.apply_fn, train_state.params, obs)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob, value[:, 0], pi

def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # Terra: Reshape
        # [minibatch_size, seq_len, ...] -> [minibatch_size * seq_len, ...]
        print(f"ppo_update_networks {transitions.obs['agent_state'].shape=}")
        transitions_obs_reshaped = jax.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), transitions.obs
        )

        # TODO make a function
        # NOTE: can't use select_action_ppo here because it doesn't decouple params from train_state
        print(f"ppo_update_networks {transitions_obs_reshaped['agent_state'].shape=}")
        obs = obs_to_model_input(transitions_obs_reshaped)
        value, dist = policy(train_state.apply_fn, params, obs)
        value = value[:, 0]
        # action = dist.sample(seed=rng_model)
        transitions_actions_reshaped = jnp.reshape(transitions.action, (-1, *transitions.action.shape[2:]))
        log_prob = dist.log_prob(transitions_actions_reshaped)

        # Terra: Reshape
        value = jnp.reshape(value, transitions.value.shape)
        log_prob = jnp.reshape(log_prob, transitions.log_prob.shape)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(-clip_eps, clip_eps)
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
        # TODO: ablate this!
        # value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean((loss, vloss, aloss, entropy, grads), axis_name="devices")
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


# for evaluation (evaluate for N consecutive episodes, sum rewards)
# N=1 single task, N>1 for meta-RL
class RolloutStats(NamedTuple):
    max_reward: jax.Array = jnp.asarray(-100)
    min_reward: jax.Array = jnp.asarray(100)
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    # episodes: jax.Array = jnp.asarray(0)

    action_0: jax.Array = jnp.asarray(0)
    action_1: jax.Array = jnp.asarray(0)
    action_2: jax.Array = jnp.asarray(0)
    action_3: jax.Array = jnp.asarray(0)
    action_4: jax.Array = jnp.asarray(0)
    action_5: jax.Array = jnp.asarray(0)
    action_6: jax.Array = jnp.asarray(0)
    action_7: jax.Array = jnp.asarray(0)
    action_8: jax.Array = jnp.asarray(0)


def rollout(
    rng: jax.Array,
    env,
    env_params,
    train_state: TrainState,
    num_envs: int,
    num_rollouts: int,
) -> RolloutStats:
    # def _cond_fn(carry):
    #     rng, stats, timestep = carry
    #     return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep = carry

        rng, _rng_step, _rng_model = jax.random.split(rng, 3)

        action, _, _, _ = select_action_ppo(train_state, timestep.observation, _rng_model)
        _rng_step = jax.random.split(_rng_step, num_envs)
        action_env = wrap_action(action, env.batch_cfg.action_type)
        timestep = env.step(env_params, timestep, action_env, _rng_step) # vmapped inside

        # stats = stats.replace(
        #     reward=stats.reward + timestep.reward,
        #     length=stats.length + 1,
        #     episodes=stats.episodes + timestep.last(),
        # )
        stats = RolloutStats(
            max_reward=jnp.maximum(stats.max_reward, timestep.reward.max()),
            min_reward=jnp.minimum(stats.min_reward, timestep.reward.min()),
            reward=stats.reward + timestep.reward.sum(),
            length=stats.length + 1,
            # episodes=stats.episodes + timestep.done.any(),

            action_0=stats.action_0 + (action == 0).sum(),
            action_1=stats.action_1 + (action == 1).sum(),
            action_2=stats.action_2 + (action == 2).sum(),
            action_3=stats.action_3 + (action == 3).sum(),
            action_4=stats.action_4 + (action == 4).sum(),
            action_5=stats.action_5 + (action == 5).sum(),
            action_6=stats.action_6 + (action == 6).sum(),
            action_7=stats.action_7 + (action == 7).sum(),
            action_8=stats.action_8 + (action == 8).sum(),
        )
        carry = (rng, stats, timestep)
        return carry

    rng, _rng_reset = jax.random.split(rng)
    _rng_reset = jax.random.split(_rng_reset, num_envs)
    timestep = env.reset(env_params, _rng_reset)
    init_carry = (rng, RolloutStats(), timestep)

    # final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    final_carry = jax.lax.fori_loop(0, num_rollouts, lambda i, carry: _body_fn(carry), init_carry)
    return final_carry[1]

def save_pkl_object(obj, filename):
    """Helper to store pickle objects."""
    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print(f"Stored data at {filename}.")
