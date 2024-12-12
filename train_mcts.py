import jax
import jax.numpy as jnp
import jax.random as jrandom
from dataclasses import dataclass
from terra.env import TerraEnvBatch
from terra.config import EnvConfig
from utils.models import get_model_ready
from utils.utils_ppo import obs_to_model_input, policy, wrap_action
import mctx
import optax

@dataclass
class TrainConfig:
    name: str
    num_devices: int = 0
    total_iterations: int = 100
    num_simulations: int = 16
    project: str = "terra-baseline"
    group: str = "default"
    num_envs_per_device: int = 32
    num_steps: int = 32
    update_epochs: int = 5
    num_minibatches: int = 32
    total_timesteps: int = 30_000_000_000
    lr: float = 3e-4
    clip_eps: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    vf_coef: float = 5.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 100
    seed: int = 42
    log_train_interval: int = 1
    log_eval_interval: int = 10
    checkpoint_interval: int = 50
    # model settings
    clip_action_maps: bool = True
    mask_out_arm_extension: bool = True
    local_map_normalization_bounds = [-16, 16]
    loaded_max = 100
    num_rollouts_eval = 300
    maps_net_normalization_bounds = [-16, 16]

    def __post_init__(self):
        self.num_devices = (
            jax.local_device_count() if self.num_devices == 0 else self.num_devices
        )
        self.num_envs = self.num_envs_per_device * self.num_devices
        self.total_timesteps_per_device = self.total_timesteps // self.num_devices
        self.eval_episodes_per_device = self.eval_episodes // self.num_devices
        assert (
            self.num_envs % self.num_devices == 0
        ), "Number of environments must be divisible by the number of devices."
        self.num_updates = (
            self.total_timesteps // (self.num_steps * self.num_envs)
        ) // self.num_devices
        print(f"Num devices: {self.num_devices}, Num updates: {self.num_updates}")

    def __getitem__(self, key):
        return getattr(self, key)


def make_envs(config: TrainConfig):
    env = TerraEnvBatch()
    env_params = EnvConfig()
    # agent = env_params.agent._replace(
    #     angles_base=jnp.int32(env_params.agent.angles_base),
    #     angles_cabin=jnp.int32(env_params.agent.angles_cabin),
    #     max_arm_extension=jnp.int32(env_params.agent.max_arm_extension),
    #     move_tiles=jnp.int32(env_params.agent.move_tiles),
    #     dig_depth=jnp.int32(env_params.agent.dig_depth),
    # )
    # env_params = env_params._replace(agent=agent)
    env_params = jax.tree_map(
        lambda x: jnp.array(x)[None].repeat(config.num_envs, axis=0),
        env_params
    )
    return env, env_params

def initialize_model(rng, env, config: TrainConfig):
    network, params = get_model_ready(rng, config, env)
    return network, params

def root_fn(apply_fn, params, timestep, config: TrainConfig):
    obs = timestep.observation
    inp = obs_to_model_input(obs, config)  
    value, dist = policy(apply_fn, params, inp)
    return mctx.RootFnOutput(
        prior_logits=dist.logits,
        value=value[:,0],
        embedding=timestep,
    )

def make_recurrent_fn(env, apply_fn, config: TrainConfig):
    def wrapped_recurrent_fn(params, rng, actions, embedding):
        timestep = embedding
        rng, rng_env = jrandom.split(rng)
        rng_envs = jrandom.split(rng_env, config.num_envs)

        # Ensure actions are int32
        actions = actions.astype(jnp.int32)
        terra_actions = wrap_action(actions, env.batch_cfg.action_type)
        next_timestep = env.step(timestep, terra_actions, rng_envs)
        next_obs = next_timestep.observation

        inp = obs_to_model_input(next_obs, config)
        value, dist = policy(apply_fn, params, inp)

        reward = next_timestep.reward
        done = next_timestep.done
        discount = (1.0 - done) * config.gamma

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=dist.logits,
            value=value[:,0],
        ), next_timestep

    return wrapped_recurrent_fn

def main():
    config = TrainConfig("mcts")
    rng = jrandom.PRNGKey(config.seed)

    env, env_params = make_envs(config)
    rng, rng_envs = jrandom.split(rng)
    rng_envs = jrandom.split(rng_envs, config.num_envs)
    timestep = env.reset(env_params, rng_envs)

    rng, rng_model = jrandom.split(rng)
    network, params = initialize_model(rng_model, env, config)

    def wrapped_root_fn(params, timestep):
        return root_fn(network.apply, params, timestep, config)

    recurrent = make_recurrent_fn(env, network.apply, config)

    for iteration in range(config.total_iterations):
        root = wrapped_root_fn(params, timestep)
        print("type of root:", type(root))
        print("root prior logits:", root.prior_logits.shape)
        print("root value:", len(root.value))

        rng, rng_mcts = jrandom.split(rng)
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_mcts,
            root=root,
            recurrent_fn=recurrent,
            num_simulations=config.num_simulations,
        )

        actions = policy_output.action.astype(jnp.int32)
        terra_actions = wrap_action(actions, env.batch_cfg.action_type)

        rng, rng_step = jrandom.split(rng)
        rng_step = jrandom.split(rng_step, config.num_envs)
        timestep = env.step(timestep, terra_actions, rng_step)

        avg_reward = jnp.mean(timestep.reward)
        print(f"Iteration {iteration}, avg reward: {avg_reward}")


if __name__ == "__main__":
    main()
