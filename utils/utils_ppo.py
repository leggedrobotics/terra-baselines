import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

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
    # TODO: check performance 
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

def policy(
    apply_fn,
    params,
    obs,
):
    value, logits_pi = apply_fn(params, obs)
    pi = tfp.distributions.Categorical(logits=logits_pi)
    return value, pi


def select_action_ppo(
    train_state,
    obs: jnp.ndarray,
    rng: jax.random.PRNGKey,
):
    # Prepare policy input from Terra State
    obs = obs_to_model_input(obs)

    value, pi = policy(train_state.apply_fn, train_state.params, obs)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob, value[:, 0], pi


def wrap_action(action, action_type):
    action = action_type.new(action[:, None])
    return action