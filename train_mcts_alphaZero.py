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


@dataclass(frozen=True)
class TrainConfig:
    name: str = "alphazero-fixed-v1"
    project: str = "terra-alphazero"
    group: str = "default"

    num_devices: int = 1
    num_envs_per_device: int = 256
    num_envs: int = num_devices * num_envs_per_device
    max_episode_steps: int = 300
    episodes_per_iteration: int = 1
    num_iterations: int = 10000

    gamma: float = 0.99
    num_simulations: int = 64
    value_target: str = "maxq"

    batch_size: int = 256
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    max_grad_norm: float = 0.5

    seed: int = 42

    log_interval: int = 1
    eval_interval: int = 5
    checkpoint_interval: int = 10

    total_timesteps: int = 30_000_000_000
    clip_action_maps: bool = True
    mask_out_arm_extension: bool = True
    local_map_normalization_bounds: tuple = (-16, 16)
    maps_net_normalization_bounds: tuple = (-16, 16)
    loaded_max: int = 100
    num_rollouts_eval: int = 300
    
    # Required by model
    num_prev_actions: int = 5
    num_test_rollouts: int = 32  # For evaluation

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
    """Build root & recurrent functions for mctx using the single network.
    
    Key fixes:
    1. Properly tracks prev_actions through MCTS simulations
    2. Handles terminal states correctly - zeros value/discount when done
       to prevent MCTS from simulating across episode boundaries
    """

    def env_step_fn(env_states, actions, rng):
        bsize = env_states.done.shape[0]
        rngs = jrandom.split(rng, bsize)
        wrapped_acts = wrap_action(actions.astype(jnp.int32), env.batch_cfg.action_type)
        return env.step(env_states, wrapped_acts, rngs)

    def root_fn(params, env_states, prev_actions):
        """Create root node for MCTS. Embedding = (timestep, prev_actions)."""
        obs_inp = obs_to_model_input(env_states.observation, prev_actions, config)
        v, logits = apply_fn(params, obs_inp)
        return mctx.RootFnOutput(
            prior_logits=logits,
            value=v[:, 0],
            embedding=(env_states, prev_actions)
        )

    def recurrent_fn(params, rng_key, actions, embedding):
        """MCTS transition function with proper terminal state handling."""
        env_states, prev_actions = embedding
        
        # Check if already done BEFORE stepping (to avoid simulating from reset states)
        was_done = env_states.done
        
        # Step environment
        next_env_states = env_step_fn(env_states, actions, rng_key)
        
        # Update prev_actions for the simulation
        next_prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
        next_prev_actions = next_prev_actions.at[:, 0].set(actions)
        
        # Compute network outputs
        obs_inp = obs_to_model_input(next_env_states.observation, next_prev_actions, config)
        v, logits = apply_fn(params, obs_inp)
        
        reward = next_env_states.reward
        done = next_env_states.done
        
        # KEY FIX: Handle terminal states properly
        # If was_done: we're simulating from a reset state (invalid), zero everything
        # If done: episode just ended, zero future value/discount
        terminal_mask = done | was_done
        
        # Zero reward for already-done states (those were invalid transitions)
        safe_reward = jnp.where(was_done, 0.0, reward)
        
        # Zero discount for terminal states (no future value)
        discount = jnp.where(terminal_mask, 0.0, gamma)
        
        # Zero value for terminal states
        safe_value = jnp.where(terminal_mask, 0.0, v[:, 0])
        
        # Mask logits for terminal states to prevent further expansion
        # (set to very negative so softmax gives ~uniform, but discount=0 prevents use)
        masked_logits = jnp.where(
            terminal_mask[:, None],
            jnp.full_like(logits, -1e9),
            logits
        )
        
        return mctx.RecurrentFnOutput(
            reward=safe_reward,
            discount=discount,
            prior_logits=masked_logits,
            value=safe_value
        ), (next_env_states, next_prev_actions)

    return root_fn, recurrent_fn


@partial(jit, static_argnames=("num_simulations", "env", "root_fn", "recurrent_fn", "config"))
def one_mcts_step(rng, env, states, prev_actions, params,
                  root_fn, recurrent_fn,
                  num_simulations,
                  config):
    """
    A single MCTS-guided environment step for all envs in 'states'.

    1) MCTS => action, pi_mcts
    2) Env step => next state
    3) Update prev_actions
    4) Return (next_states, next_prev_actions, actions, pi_mcts).
    """
    B = states.done.shape[0]
    num_actions = 7  # Terra has 7 actions for tracked agent
    
    # Invalid actions mask (if needed - currently none masked)
    invalid_mask = jnp.zeros((B, num_actions), dtype=jnp.bool_)

    rng, rng_mcts, rng_step = jrandom.split(rng, 3)
    
    # Create root with prev_actions
    root = root_fn(params, states, prev_actions)
    
    # Run MCTS
    out = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_mcts,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=invalid_mask,
        gumbel_scale=1.0,
    )
    actions = out.action            # shape [B]
    pi_mcts = out.action_weights    # shape [B, num_actions]

    # Step environment
    rngs_step = jrandom.split(rng_step, B)
    wrapped = wrap_action(actions.astype(jnp.int32), env.batch_cfg.action_type)
    next_states = env.step(states, wrapped, rngs_step)
    
    # Update prev_actions
    next_prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
    next_prev_actions = next_prev_actions.at[:, 0].set(actions)

    return rng, next_states, next_prev_actions, actions, pi_mcts


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
    states,
    prev_actions,    # [B, num_prev_actions]
):
    """Collect self-play data using MCTS, keeping everything on GPU."""
    B = config.num_envs
    max_steps = config.max_episode_steps
    num_actions = 7  # Terra tracked agent

    # Prepare buffers for observations, policies, prev_actions, rewards, dones
    obs_buf = build_obs_buffer_template(states.observation, max_steps)
    pi_buf = jnp.zeros((max_steps, B, num_actions), dtype=jnp.float32)
    prev_actions_buf = jnp.zeros((max_steps, B, config.num_prev_actions), dtype=jnp.int32)
    reward_buf = jnp.zeros((max_steps, B), dtype=jnp.float32)
    done_buf = jnp.zeros((max_steps, B), dtype=jnp.bool_)

    step0 = jnp.array(0, dtype=jnp.int32)
    init_carry = (states, prev_actions, step0, rng, obs_buf, pi_buf, prev_actions_buf, reward_buf, done_buf)

    def cond_fun(carry):
        (states, prev_actions, step, rng, obs_buf, pi_buf, prev_actions_buf, reward_buf, done_buf) = carry
        return step < max_steps

    def body_fun(carry):
        (states, prev_actions, step, rng, obs_buf, pi_buf, prev_actions_buf, reward_buf, done_buf) = carry
        
        # Run MCTS step
        rng, next_states, next_prev_actions, actions, pi_mcts = one_mcts_step(
            rng, env, states, prev_actions, params,
            root_fn, recurrent_fn, config.num_simulations, config
        )

        # Store current obs
        obs_buf = jax.tree_map(
            lambda buf, val: buf.at[step].set(val),
            obs_buf,
            states.observation
        )
        # Store MCTS policy
        pi_buf = pi_buf.at[step].set(pi_mcts)
        # Store prev_actions (needed for training)
        prev_actions_buf = prev_actions_buf.at[step].set(prev_actions)
        # Store reward & done from NEXT state
        reward_buf = reward_buf.at[step].set(next_states.reward)
        done_buf = done_buf.at[step].set(next_states.done)

        return (next_states, next_prev_actions, step + 1, rng, obs_buf, pi_buf, prev_actions_buf, reward_buf, done_buf)

    final_carry = lax.while_loop(cond_fun, body_fun, init_carry)
    (states_final, prev_actions_final, step_final, rng_final, 
     obs_buf_final, pi_buf_final, prev_actions_buf_final, reward_buf_final, done_buf_final) = final_carry

    # Compute discounted returns
    returns_buf = discount_cumsum(reward_buf_final, done_buf_final, config.gamma)

    return (
        obs_buf_final,          # dict-of-arrays [max_steps, B, ...]
        pi_buf_final,           # [max_steps, B, num_actions]
        prev_actions_buf_final, # [max_steps, B, num_prev_actions]
        returns_buf,            # [max_steps, B]
        step_final,
        rng_final,
        states_final,
        prev_actions_final,
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
    """AlphaZero-style training with MCTS self-play."""
    import time
    
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
    env_params = jax.tree.map(lambda x: jnp.array(x).repeat(config.num_envs, axis=0), env_params)
    
    # Fix dtypes to prevent JAX type promotion issues during MCTS
    env_params = fix_env_cfg_dtypes(env_params)

    # Build network
    network, network_params = get_model_ready(rng, config, env)
    
    # Optionally load pretrained weights
    # log = helpers.load_pkl_object("checkpoints/your-checkpoint.pkl")
    # network_params = log["model_params"]

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), 
        optax.adam(config.policy_lr)
    )

    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    # Create recurrent functions for MCTS
    root_fn, rec_fn = make_recurrent_fn(env, network.apply, config.gamma, config)

    B = config.num_envs

    # Initialize environment and prev_actions
    rng, rng_reset = jrandom.split(rng)
    rng_keys = jrandom.split(rng_reset, B)
    states = env.reset(env_params, rng_keys)
    prev_actions = jnp.zeros((B, config.num_prev_actions), dtype=jnp.int32)

    print(f"Starting AlphaZero training with {B} environments, {config.num_simulations} MCTS sims")
    
    for iteration in range(config.num_iterations):
        start_time = time.time()
        
        # A) Collect data via MCTS self-play
        (
            obs_buf,            # dict-of-arrays: [max_steps, B, ...]
            pi_buf,             # [max_steps, B, num_actions]
            prev_actions_buf,   # [max_steps, B, num_prev_actions]
            returns_buf,        # [max_steps, B]
            step_final,
            rng,
            states_final,
            prev_actions_final,
        ) = collect_episodes_jitted(
            rng, env, env_params,
            train_state.params,
            root_fn, rec_fn, config,
            states, prev_actions
        )

        # Continue from where we left off
        states = states_final
        prev_actions = prev_actions_final
        
        collect_time = time.time() - start_time

        # B) Prepare training data
        T = config.max_episode_steps  # Use full buffer since we run for max_steps

        # Flatten [T, B] -> [T*B]
        obs_dict_flat = jax.tree.map(
            lambda arr: arr.reshape((T * B,) + arr.shape[2:]),
            obs_buf
        )
        pi_buf_flat = pi_buf.reshape((T * B, pi_buf.shape[-1]))
        prev_actions_flat = prev_actions_buf.reshape((T * B, config.num_prev_actions))
        returns_buf_flat = returns_buf.reshape((T * B,))

        # C) Shuffle data for better training
        rng, rng_shuffle = jrandom.split(rng)
        total_samples = T * B
        perm = jrandom.permutation(rng_shuffle, total_samples)
        
        obs_dict_flat = jax.tree.map(lambda x: x[perm], obs_dict_flat)
        pi_buf_flat = pi_buf_flat[perm]
        prev_actions_flat = prev_actions_flat[perm]
        returns_buf_flat = returns_buf_flat[perm]

        # D) Train in mini-batches
        train_start = time.time()
        batch_size = config.batch_size
        num_batches = total_samples // batch_size
        losses = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            obs_dict_batch = jax.tree.map(lambda x: x[start_idx:end_idx], obs_dict_flat)
            pi_batch = pi_buf_flat[start_idx:end_idx]
            prev_actions_batch = prev_actions_flat[start_idx:end_idx]
            ret_batch = returns_buf_flat[start_idx:end_idx]

            # Convert to model input (with prev_actions)
            inp = obs_to_model_input(obs_dict_batch, prev_actions_batch, config)

            # Gradient step
            train_state, (loss_val, pol_l, val_l, reg_l) = train_step(
                train_state, inp, pi_batch, ret_batch, network.apply
            )
            losses.append((loss_val, pol_l, val_l, reg_l))

        train_time = time.time() - train_start

        # Compute mean losses
        if losses:
            mean_loss = float(jnp.mean(jnp.array([l[0] for l in losses])))
            mean_pl = float(jnp.mean(jnp.array([l[1] for l in losses])))
            mean_vl = float(jnp.mean(jnp.array([l[2] for l in losses])))
            mean_rl = float(jnp.mean(jnp.array([l[3] for l in losses])))
        else:
            mean_loss = mean_pl = mean_vl = mean_rl = 0

        # Log to wandb
        wandb.log({
            "iteration": iteration,
            "train/total_loss": mean_loss,
            "train/policy_loss": mean_pl,
            "train/value_loss": mean_vl,
            "train/regularization_loss": mean_rl,
            "timing/collect_time": collect_time,
            "timing/train_time": train_time,
            "timing/samples_per_sec": total_samples / (collect_time + train_time),
        }, step=iteration)

        print(f"Iter {iteration}: loss={mean_loss:.4f} (pol={mean_pl:.4f}, val={mean_vl:.4f}) | "
              f"collect={collect_time:.1f}s, train={train_time:.1f}s")

        # Evaluate
        if (iteration + 1) % config.eval_interval == 0:
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
            print(f"  Eval: reward={float(eval_stats.reward)/config.num_envs:.2f}, "
                  f"terminations={float(eval_stats.terminations)}")

        # Checkpoint
        if (iteration + 1) % config.checkpoint_interval == 0:
            ckpt = {
                "iteration": iteration,
                "model_params": train_state.params,
                "model": train_state.params,  # For compatibility with eval.py
                "train_config": config,
                "env_config": env_params,
            }
            helpers.save_pkl_object(ckpt, f"checkpoints/{config.name}.pkl")
            print(f"  Saved checkpoint to checkpoints/{config.name}.pkl")

    run.finish()
    return train_state


if __name__ == "__main__":
    cfg = TrainConfig()
    final_model = train_alphazero(cfg)
    print("Done training!")
