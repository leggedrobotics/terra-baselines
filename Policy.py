from utils.utils_ppo import obs_to_model_input, wrap_action
import jax
import jax.numpy as jnp

def predict_action_values_and_policy(model, model_params, observation, rl_config):
    obs_model = obs_to_model_input(observation, rl_config)
    v, logits_pi = model.apply(model_params, obs_model)
    return v, logits_pi 

v, logits_pi = predict_action_values_and_policy(rng, model, model_params, observation, rl_config)