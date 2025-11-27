# train_alphaZero_jitted.py
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, vmap, lax
from functools import partial
import optax
import wandb
import random
import mctx
from dataclasses import dataclass, asdict

from flax import struct
from flax.training.train_state import TrainState

from terra.env import TerraEnvBatch
from terra.config import EnvConfig
import eval_ppo
import utils.helpers as helpers
from utils.utils_ppo import obs_to_model_input, wrap_action
from utils.models import get_model_ready


@dataclass(frozen=True)
class TrainConfig:
    name: str = "Local-trials-one"
    project: str = "terra-alphazero"
    group: str = "default"

    num_devices: int = 1
    num_envs_per_device: int = 2
    num_envs: int = num_devices * num_envs_per_device
    max_episode_steps: int = 300
    episodes_per_iteration: int = 10
    num_iterations: int = 10000

    gamma: float = 0.99
    num_simulations: int = 16
    value_target: str = "maxq"

    batch_size: int = 128
    policy_lr: float = 7e-5
    value_lr: float = 7e-5
    max_grad_norm: float = 0.1

    seed: int = 42

    log_interval: int = 1
    eval_interval: int = 2
    checkpoint_interval: int = 5

    total_timesteps: int = 30_000_000_000
    clip_action_maps = True
    mask_out_arm_extension = True
    local_map_normalization_bounds = [-16, 16]
    loaded_max = 100
    num_rollouts_eval = 300

    def __getitem__(self, key):
        return getattr(self, key)

@struct.dataclass
class SelfPlayTransition:
    """Stores number of enviroments states for each time step:
       (obs, pi_mcts, final_return).
    """
    obs: jnp.ndarray
    pi_mcts: jnp.ndarray
    final_return: jnp.ndarray


def make_recurrent_fn(env: TerraEnvBatch, apply_fn, gamma, config: TrainConfig):
    """Build root & recurrent functions for mctx using the single network."""

    def env_step_fn(env_states, actions, rng):
        bsize = env_states.done.shape[0]
        rngs = jrandom.split(rng, bsize)
        wrapped_acts = wrap_action(actions.astype(jnp.int32), env.batch_cfg.action_type)
        return env.step(env_states, wrapped_acts, rngs)

    def root_fn(params, env_states):
        obs_inp = obs_to_model_input(env_states.observation, config)
        # the network returns (value, logits)
        v, logits = apply_fn(params, obs_inp)
        return mctx.RootFnOutput(
            prior_logits=logits,
            value=v[:, 0],
            embedding=env_states
        )

    def recurrent_fn(params, rng_key, actions, embedding):
        next_env_states = env_step_fn(embedding, actions, rng_key)
        obs_inp = obs_to_model_input(next_env_states.observation, config)
        v, logits = apply_fn(params, obs_inp)
        reward  = next_env_states.reward
        done    = next_env_states.done
        discount= (1.0 - done) * gamma
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=v[:, 0]
        ), next_env_states

    return root_fn, recurrent_fn


@partial(jit, static_argnames=("num_simulations", "env", "root_fn", "recurrent_fn", "config"))
def one_mcts_step(rng, env, states, done_mask, params,
                  root_fn, recurrent_fn,
                  num_simulations,
                  config):
    """
    A single environment step for *all* envs in 'states', skipping where done_mask=1.

    1) MCTS => action, pi_mcts
    2) Env step => next state
    3) Return (next_states, actions, pi_mcts).
    """
    # invalid actions mask
    B = states.done.shape[0]
    invalid_mask = jnp.zeros((B, 9), dtype=jnp.bool_)
    invalid_mask = invalid_mask.at[:,-2].set(True).at[:,-3].set(True)

    rng, rng_mcts, rng_step = jrandom.split(rng, 3)
    root = root_fn(params, states)
    out = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_mcts,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=invalid_mask,   # <--- pass here
        gumbel_scale=2.0,
    )
    actions = out.action            # shape [B]
    pi_mcts = out.action_weights    # shape [B, num_actions]

    rngs_step = jrandom.split(rng_step, states.done.shape[0])
    wrapped = wrap_action(actions.astype(jnp.int32), env.batch_cfg.action_type)
    next_states = env.step(states, wrapped, rngs_step)

    return rng, next_states, actions, pi_mcts


def build_obs_buffer_template(obs_dict, max_steps):
    """
    For each key in obs_dict, which has shape [B, ...],
    create a buffer of shape [max_steps, B, ...].
    """
    buf_dict = {}
    for k, arr in obs_dict.items():
        # arr.shape might be (B, *some_dims)
        buf_shape = (max_steps,) + arr.shape
        buf_dict[k] = jnp.zeros(buf_shape, dtype=arr.dtype)
    return buf_dict

@jit
def discount_cumsum(rewards, dones, gamma):
    """
    Given:
      rewards: float32[time, batch]
      dones:   bool[time, batch]
      gamma:   scalar discount factor
    Return:
      returns: float32[time, batch]
        Where returns[t, b] = rewards[t, b] + gamma * returns[t+1, b], if not done[t].
        If done[t], then returns[t] = rewards[t] only (no future accumulation).
    """
    T, B = rewards.shape

    def scan_fun(carry, t):
        # t goes from T-1 down to 0
        future_return = carry  # shape [B]
        r_t = rewards[t]       # shape [B]
        done_t = dones[t]      # shape [B]
        # If done at step t, we do not add gamma * future_return
        ret_t = r_t + gamma * future_return * (1.0 - done_t.astype(jnp.float32))
        return ret_t, ret_t

    init = jnp.zeros((B,), dtype=jnp.float32)
    # We'll scan backwards: range(T-1, ..., 0).
    # 'lax.scan' runs forward on the given sequence, so we reverse indices.
    indices = jnp.arange(T - 1, -1, -1)
    final, all_returns_reversed = lax.scan(scan_fun, init, indices)

    # all_returns_reversed has shape [T, B] in reversed time order
    # we flip back to [0..T-1].
    all_returns = jnp.flip(all_returns_reversed, axis=0)
    return all_returns

@partial(jax.jit, static_argnames=('env', 'root_fn', 'recurrent_fn', 'config'))
def collect_episodes_jitted(
    rng,
    env,             # TerraEnvBatch
    env_params,      # EnvConfig, repeated for B envs
    params,          # model params
    root_fn,         # MCTS root function
    recurrent_fn,    # MCTS recurrent function
    config,
    states
):
    B = config.num_envs
    max_steps = config.max_episode_steps

    # 1) Reset environment
    # rng_keys = jax.random.split(rng, B)
    # states   = env.reset(env_params, rng_keys)  # shape [B]

    # 2) Prepare buffers
    obs_buf = build_obs_buffer_template(states.observation, max_steps)
    num_actions = 9
    pi_buf = jnp.zeros((max_steps, B, num_actions), dtype=jnp.float32)

    # (NEW) We'll also store the raw reward and done flag at each step
    reward_buf = jnp.zeros((max_steps, B), dtype=jnp.float32)
    done_buf   = jnp.zeros((max_steps, B), dtype=jnp.bool_)

    step0 = jnp.array(0, dtype=jnp.int32)
    init_carry = (states, step0, rng, obs_buf, pi_buf, reward_buf, done_buf)

    def cond_fun(carry):
        (states, step, rng, obs_buf, pi_buf, reward_buf, done_buf) = carry
        all_done = jnp.all(states.done)
        return step < max_steps

    def body_fun(carry):
        (states, step, rng, obs_buf, pi_buf, reward_buf, done_buf) = carry
        rng, next_states, actions, pi_mcts = one_mcts_step(
            rng, env, states, states.done, params,
            root_fn, recurrent_fn, config.num_simulations, config
        )

        # store current obs in obs_buf
        obs_buf = jax.tree_map(
            lambda buf, val: buf.at[step].set(val),
            obs_buf,
            states.observation
        )
        # store pi_mcts
        pi_buf = pi_buf.at[step].set(pi_mcts)

        # (NEW) store the reward & done from NEXT state
        reward_buf = reward_buf.at[step].set(next_states.reward)
        done_buf   = done_buf.at[step].set(next_states.done)

        return (next_states, step + 1, rng, obs_buf, pi_buf, reward_buf, done_buf)

    states_final, step_final, rng_final, obs_buf_final, pi_buf_final, reward_buf_final, done_buf_final = \
        lax.while_loop(cond_fun, body_fun, init_carry)

    ########################################
    # (NEW) Post-processing: compute discounted returns for each step
    ########################################
    # step_final tells us how many time steps were actually used (<= max_steps).
    # We'll slice the valid portion.
    # reward_buf_used = reward_buf_final[:step_final]  # shape [step_final, B]
    # done_buf_used   = done_buf_final[:step_final]    # shape [step_final, B]

    # returns_buf has shape [step_final, B].
    returns_buf = discount_cumsum(reward_buf_final, done_buf_final, config.gamma)

    ########################################
    # Return everything we need
    ########################################
    return (
        obs_buf_final,      # dict-of-arrays [max_steps, B, ...]
        pi_buf_final,       # [max_steps, B, num_actions]
        returns_buf,        # (NEW) discounted returns, shape [step_final, B]
        step_final,
        rng_final,
        states_final
    )


@partial(jit, static_argnames=("apply_fn",))
def alpha_zero_loss(apply_fn, params, obs_batch, pi_batch, returns_batch):
    """
    Cross-entropy( pi_batch, pi_pred ) + MSE(value, returns).
    obs_batch: a batched dict of arrays => pass to obs_to_model_input(...) before call!
    """
    v, logits = apply_fn(params, obs_batch)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    # for i in range(pi_batch.shape[0]):
    #     # jax.debug.print("pi_batch   {}", pi_batch[i].round(3))
    #     # jax.debug.print("probs      {}", probs[i].round(3))
    #     # jax.debug.print("sum probs  {}", probs[i].sum())

    #     pi_max_idx = jnp.argmax(pi_batch[i])
    #     probs_max_idx = jnp.argmax(probs[i])
    #     match = pi_max_idx == probs_max_idx
    #     jax.debug.print("Match? {}", match)

    weight_decay = 0.0
    if weight_decay > 0:
        def param_l2(p):
            return jnp.sum(p**2)
        l2_sum = jax.tree_util.tree_reduce(
            lambda acc, x: acc + jnp.sum(x**2),
            params, 
            initializer=jnp.float32(0.0)
        )
        reg_loss = weight_decay * l2_sum
    else:
        reg_loss = 0.0

    pol_loss = optax.softmax_cross_entropy(logits=logits, labels=pi_batch).mean()
    val_loss = jnp.mean((v[:, 0] - returns_batch)**2)

    total_loss = pol_loss + val_loss + reg_loss

    return total_loss, (pol_loss, val_loss, reg_loss)


@partial(jit, static_argnames=("apply_fn",))
def train_step(train_state: TrainState,
               batch_obs,
               batch_pi,
               batch_returns,
               apply_fn):
    def loss_fn(params):
        loss_val, (p_loss, v_loss, reg_loss) = alpha_zero_loss(
            apply_fn, params,
            batch_obs, batch_pi, batch_returns
        )
        return loss_val, (p_loss, v_loss, reg_loss)

    (loss_val, (p_loss, v_loss, reg_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params
    )

    train_state = train_state.apply_gradients(grads=grads)
    return train_state, (loss_val, p_loss, v_loss, reg_loss)


def train_alphazero(config: TrainConfig):
    # wandb
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True
    )

    # Setup
    rng = jrandom.PRNGKey(config.seed)
    env = TerraEnvBatch()
    env_params = EnvConfig()
    env_params = jax.tree_map(lambda x: jnp.array(x).repeat(config.num_envs, axis=0), env_params)

    # Build network using get_model_ready (a single net for both policy & value)
    network, network_params = get_model_ready(rng, config, env)
    # Preload pretrained weights from checkpoint
    # log = helpers.load_pkl_object("checkpoints/medium-64sim-with-epoch.pkl")
    # network_params = log["model_params"]

    tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.policy_lr))

    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    # Create recurrent functions using the single network
    root_fn, rec_fn = make_recurrent_fn(env, network.apply, config.gamma, config)


    # #evalutate at start
    # eval_stats = eval_ppo.rollout(
    #         rng, env, env_params,
    #         train_state,
    #         config
    # )
    # wandb.log({
    #     "eval/reward": float(eval_stats.reward) / config.num_envs,
    #     "eval/max_reward": float(eval_stats.max_reward),
    #     "eval/min_reward": float(eval_stats.min_reward),
    #     "eval/episodes": float(eval_stats.episodes),
    #     "eval/terminations": float(eval_stats.terminations),
    #     "eval/positive_terminations": float(eval_stats.positive_terminations),
    # }, step=0)  
    # print("Eval stats: ", {
    #     "eval/reward": float(eval_stats.reward) / config.num_envs,
    #     "eval/max_reward": float(eval_stats.max_reward),
    #     "eval/min_reward": float(eval_stats.min_reward),
    #     "eval/episodes": float(eval_stats.episodes),
    #     "eval/terminations": float(eval_stats.terminations),
    #     "eval/positive_terminations": float(eval_stats.positive_terminations),
    # })

    B = config.num_envs

    rng_keys = jax.random.split(rng, B)
    states   = env.reset(env_params, rng_keys)  # shape [B]

    for iteration in range(config.num_iterations):
        print("start collect episodes")
        # A) Collect data via MCTS self-play
        (
            obs_buf,       # dict-of-arrays: [max_steps, B, ...]
            pi_buf,        # [max_steps, B, num_actions]
            returns_buf,   # (NEW) [step_final, B]
            step_final,
            rng,
            states_final
        ) = collect_episodes_jitted(
            rng, env, env_params,
            train_state.params,
            root_fn, rec_fn, config,
            states
        )

        states = states_final
        

        # B) Flatten or slice time+batch dimension for training
        # 'step_final' is how many steps we actually used.
        # We slice obs_buf and pi_buf accordingly.
        T = step_final  # integer <= max_steps

        # slice each obs_dict from [max_steps, B, ...] -> [T, B, ...]
        obs_dict_slice = jax.tree_map(lambda arr: arr[:T], obs_buf)
        pi_buf_slice   = pi_buf[:T]  # shape [T, B, num_actions]

        # flatten [T, B] -> [T*B]
        obs_dict_flat = jax.tree_map(
            lambda arr: arr.reshape((T * B,) + arr.shape[2:]),
            obs_dict_slice
        )
        pi_buf_flat = pi_buf_slice.reshape((T * B, pi_buf.shape[-1]))
        returns_buf_flat = returns_buf.reshape((T * B,))

        # C) Train in mini-batches

        for epoch in range(config.episodes_per_iteration):
            total_samples = T * B
            idx = 0
            batch_size = config.batch_size
            losses = []

            while idx < total_samples:
                slice_end = idx + batch_size
                obs_dict_batch = jax.tree_map(lambda x: x[idx:slice_end], obs_dict_flat)
                pi_batch       = pi_buf_flat[idx:slice_end]
                ret_batch      = returns_buf_flat[idx:slice_end]
                idx = slice_end

                # convert to model input
                inp = obs_to_model_input(obs_dict_batch, config)

                # single step of gradient descent
                train_state, (loss_val, pol_l, val_l, reg_l) = train_step(
                    train_state, inp, pi_batch, ret_batch, network.apply
                )
                losses.append((loss_val, pol_l, val_l, reg_l))

            if losses:
                arr_loss = jnp.array([l[0] for l in losses])
                arr_pl   = jnp.array([l[1] for l in losses])
                arr_vl   = jnp.array([l[2] for l in losses])
                arr_rl   = jnp.array([l[3] for l in losses])

                mean_loss = float(arr_loss.mean())
                mean_pl   = float(arr_pl.mean())
                mean_vl   = float(arr_vl.mean())
                mean_rl   = float(arr_rl.mean())
            else:
                mean_loss = 0
                mean_pl   = 0
                mean_vl   = 0
                mean_rl = 0

            wandb.log({
                "iteration": iteration,
                "train/total_loss": mean_loss,
                "train/policy_loss": mean_pl,
                "train/value_loss":  mean_vl,
                "train/regularization_loss": mean_rl
            }, step=iteration)

        # Evaluate raw policy
        if (iteration+1) % config.eval_interval == 0:
            eval_stats = eval_ppo.rollout(
                rng, env, env_params,
                train_state,
                config
            )
            wandb.log({
                "eval/reward": float(eval_stats.reward) / config.num_envs,
                "eval/max_reward": float(eval_stats.max_reward),
                "eval/min_reward": float(eval_stats.min_reward),
                "eval/episodes": float(eval_stats.episodes),
                "eval/terminations": float(eval_stats.terminations),
                "eval/positive_terminations": float(eval_stats.positive_terminations),
            }, step=iteration)

            # log action counts
            wandb.log({
                "action_counts/action_0": float(eval_stats.action_0) / config.num_envs,
                "action_counts/action_1": float(eval_stats.action_1) / config.num_envs,
                "action_counts/action_2": float(eval_stats.action_2) / config.num_envs,
                "action_counts/action_3": float(eval_stats.action_3) / config.num_envs,
                "action_counts/action_4": float(eval_stats.action_4) / config.num_envs,
                "action_counts/action_5": float(eval_stats.action_5) / config.num_envs,
                "action_counts/action_6": float(eval_stats.action_6) / config.num_envs,
                "action_counts/action_7": float(eval_stats.action_7) / config.num_envs,
                "action_counts/action_8": float(eval_stats.action_8) / config.num_envs,
            }, step=iteration)

        # checkpoint
        if (iteration+1) % config.checkpoint_interval == 0:
            ckpt = {
                "iteration": iteration,
                "model_params": train_state.params,
                "train_config": config,
                "env_config": env_params,
            }
            helpers.save_pkl_object(ckpt, f"checkpoints/{config.name}.pkl")

    run.finish()
    return train_state


if __name__ == "__main__":
    cfg = TrainConfig()
    final_model = train_alphazero(cfg)
    print("Done training!")
