import numpy as np
import jax
from utils.models import get_model_ready
from utils.helpers import load_pkl_object, load_config
from terra.env import TerraEnvBatch
import jax.numpy as jnp
from utils.ppo import obs_to_model_input, wrap_action
from terra.state import State
import matplotlib.animation as animation


def update_render(seq, env: TerraEnvBatch, frame):
    obs = {k: v[:, frame] for k, v in seq.items()}
    return env.terra_env.render_obs(obs, mode="gif")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pkl",
        "--pkl_file",
        type=str,
        default="agents/Terra/eval/eval_17.pkl",
    )
    parser.add_argument(
        "-steps",
        "--max_steps",
        type=int,
        default=1000,
        help="Environment name.",
    )
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="Terra",
        help="Environment name.",
    )
    args, _ = parser.parse_known_args()

    # configs = load_config(f"agents/{args.env_name}/{args.train_type}" + ".yaml")

    obs_seq = load_pkl_object(args.pkl_file)

    # Batch dim first
    if len(obs_seq["local_map_action"].shape) == 4:
        obs_seq = {k: v.swapaxes(0, 1) for k, v in obs_seq.items()}
    
    batch_size = obs_seq["local_map_action"].shape[0]
    seq_len = min(obs_seq["local_map_action"].shape[1], args.max_steps)

    # end_frame = 200
    # state_seq = state_seq[:end_frame]  # TODO remove
    env = TerraEnvBatch(rendering=True, n_imgs_row=int(np.sqrt(batch_size)))
    fig = env.terra_env.window.get_fig()
    update_partial = lambda x: update_render(seq=obs_seq, env=env, frame=x)
    ani = animation.FuncAnimation(
            fig,
            update_partial,
            frames=seq_len,
            blit=False,
        )
    # Save the animation to a gif
    ani.save(f"docs/{args.env_name}.gif")
