"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

import numpy as np
import jax
from utils.models import get_model_ready
from utils.helpers import load_pkl_object, load_config
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.ppo import obs_to_model_input, wrap_action
from terra.state import State
import matplotlib.animation as animation

def load_neural_network(config, agent_path, env):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env)

    params = load_pkl_object(agent_path)["network"]
    return model, params


def rollout_episode(env: TerraEnvBatch, model, model_params, max_frames=200):
    state_seq = []
    rng = jax.random.PRNGKey(0)

    rng, rng_reset = jax.random.split(rng)
    reset_seeds = jnp.array([rng_reset[0]])
    env_state, obs = env.reset(reset_seeds)

    # if model is not None:
    #     if model.model_name == "LSTM":
    #         hidden = model.initialize_carry()

    t_counter = 0
    reward_seq = []
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            # if model.model_name == "LSTM":
            #     hidden, action = model.apply(model_params, obs, hidden, rng_act)
            # else:
            #     if model.model_name.startswith("separate"):
            #         obs = obs_to_model_input(obs)
            #         v, pi = model.apply(model_params, obs, rng_act)
            #         action = pi.sample(seed=rng_act)
            #     else:
            #         action = model.apply(model_params, obs, rng_act)
            obs = obs_to_model_input(obs)
            v, pi = model.apply(model_params, obs)
            action = pi.sample(seed=rng_act)
        else:
            action = 0  # env.action_space(env_params).sample(rng_act)
        next_env_state, (next_obs, reward, done, info) = env.step(env_state, wrap_action(action, env.batch_cfg.action_type))
        reward_seq.append(reward)
        print(t_counter, reward, action, done)
        print(10 * "=")
        t_counter += 1
        if done or t_counter == max_frames:
            break
        else:
            env_state = next_env_state
            obs = next_obs
    print(f"Terra - Steps: {t_counter}, Return: {np.sum(reward_seq)}")
    return state_seq, np.cumsum(reward_seq)


def update_render(seq, env: TerraEnvBatch, frame):
    state: State = seq[frame]
    # Remove batch dimension
    state = state._replace(
        agent=state.agent._replace(
            agent_state=state.agent.agent_state._replace(
                pos_base=state.agent.agent_state.pos_base.squeeze(0),
                angle_base=state.agent.agent_state.angle_base.squeeze(0),
                angle_cabin=state.agent.agent_state.angle_cabin.squeeze(0),
                arm_extension=state.agent.agent_state.arm_extension.squeeze(0),
                loaded=state.agent.agent_state.loaded.squeeze(0),
            )
        ),
        world=state.world._replace(
            action_map=state.world.action_map._replace(
                map=state.world.action_map.map.squeeze(0)
            ),
            target_map=state.world.target_map._replace(
                map=state.world.target_map.map.squeeze(0)
            ),
            traversability_mask=state.world.traversability_mask._replace(
                map=state.world.traversability_mask.map.squeeze(0)
            ),
            local_map=state.world.local_map._replace(
                map=state.world.local_map.map.squeeze(0)
            ),
        )
    )
    return env.terra_env.render(state, mode="gif")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train",
        "--train_type",
        type=str,
        default="ppo",
        help="es/ppo trained agent.",
    )
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        default="ppo_2023_05_09_10_01_23",
        help="es/ppo trained agent.",
    )
    parser.add_argument(
        "-random",
        "--random",
        action="store_true",
        default=False,
        help="Random rollout.",
    )
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="Terra",
        help="Environment name.",
    )
    args, _ = parser.parse_known_args()

    configs = load_config(f"agents/{args.env_name}/{args.train_type}" + ".yaml")
    
    env = TerraEnvBatch(rendering=True)

    if not args.random:
        model, model_params = load_neural_network(
            configs.train_config, f"agents/{args.env_name}/{args.run_name}" + ".pkl", env
        )
    else:
        model, model_params = None, None
    state_seq, cum_rewards = rollout_episode(
        env, model, model_params
    )

    # end_frame = 200
    # state_seq = state_seq[:end_frame]  # TODO remove
    fig = env.terra_env.window.get_fig()
    update_partial = lambda x: update_render(seq=state_seq, env=env, frame=x)
    ani = animation.FuncAnimation(
            fig,
            update_partial,
            frames=len(state_seq),
            blit=False,
        )
    # Save the animation to a gif
    ani.save(f"docs/{args.env_name}.gif")
