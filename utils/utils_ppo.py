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

def get_cfgs_init():      
    # TODO from config
    n_envs = 4096
    # n_devices = 1

    env_cfgs = jax.vmap(EnvConfig.parametrized)(
        np.array([60] * n_envs),
        np.array([60] * n_envs),
        np.array([200] * n_envs),
        np.array([0] * n_envs),
        np.array([MapType.FOUNDATIONS] * n_envs),
        np.array([RewardsType.DENSE] * n_envs),
        np.array([False] * n_envs),
        )
    # env_cfgs = jax.tree_map(
    #     lambda x: jax.numpy.reshape(x, (n_devices, x.shape[0] // n_devices, *x.shape[1:])), env_cfgs
    # )
    return env_cfgs

def policy(
    apply_fn,
    params,
    obs,
):
    value, logits_pi = apply_fn(params, obs)
    pi = tfp.distributions.Categorical(logits=logits_pi)
    return value, pi

def obs_to_model_input(obs):
    """
    Need to convert Dict to List to make it usable by JAX.
    """
    return [
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
    rng_model: jax.random.PRNGKey,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # RERUN NETWORK
        # dist, value, _ = train_state.apply_fn(
        #     params,
        #     {
        #         # [batch_size, seq_len, ...]
        #         "observation": transitions.obs,
        #         "prev_action": transitions.prev_action,
        #         "prev_reward": transitions.prev_reward,
        #     },
        #     init_hstate,
        # )
        # log_prob = dist.log_prob(transitions.action)

        # Terra: Reshape
        transitions_obs_reshaped = jax.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), transitions.obs
        )

        _, log_prob, value, dist = select_action_ppo(train_state, transitions_obs_reshaped, rng_model)

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
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    episodes: jax.Array = jnp.asarray(0)


def rollout(
    rng: jax.Array,
    env,
    env_params,
    train_state: TrainState,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, timestep = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep = carry

        rng, _rng_step, _rng_model = jax.random.split(rng, 3)
        # dist, _, hstate = train_state.apply_fn(
        #     train_state.params,
        #     {
        #         "observation": timestep.observation[None, None, ...],
        #         "prev_action": prev_action[None, None, ...],
        #         "prev_reward": prev_reward[None, None, ...],
        #     },
        #     hstate,
        # )
        # action = dist.sample(seed=_rng).squeeze()
        action, _, _, _ = select_action_ppo(train_state, timestep.observation, _rng_model)
        num_envs_per_device = 4096  # TODO: from config
        _rng_step = jax.random.split(_rng_step, num_envs_per_device)
        action_env = wrap_action(action, env.batch_cfg.action_type)
        timestep = env.step(env_params, timestep, action_env, _rng_step) # vmapped inside

        # stats = stats.replace(
        #     reward=stats.reward + timestep.reward,
        #     length=stats.length + 1,
        #     episodes=stats.episodes + timestep.last(),
        # )
        print(f"{timestep.done.any()=}")
        stats = RolloutStats(
            reward=stats.reward + timestep.reward.mean(),
            length=stats.length + 1,
            episodes=stats.episodes + timestep.done.any(),
        )
        carry = (rng, stats, timestep)
        return carry

    num_envs_per_device = 4096  # TODO: from config
    rng, _rng_reset = jax.random.split(rng)
    _rng_reset = jax.random.split(_rng_reset, num_envs_per_device)
    timestep = env.reset(env_params, _rng_reset)
    init_carry = (rng, RolloutStats(), timestep)

    print("rollout start")
    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    print("rollout end")
    return final_carry[1]
