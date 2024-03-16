import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
from functools import reduce
from typing import Sequence, Union
from terra.actions import TrackedAction, WheeledAction
from terra.env import TerraEnvBatch
from typing import Optional, Callable
from functools import partial


def get_model_ready(rng, config, env: TerraEnvBatch, speed=False):
    """Instantiate a model according to obs shape of environment."""
    num_embeddings_agent = jnp.max(jnp.array(
        [
            config["num_embeddings_agent_min"],
            env.batch_cfg.maps.max_height,
            env.batch_cfg.maps.max_width,
            env.batch_cfg.agent.angles_cabin,
            env.batch_cfg.agent.angles_base,
            ], dtype=jnp.int16)
        ).item()
    jax.debug.print("num_embeddings_agent = {x}", x=num_embeddings_agent)
    map_min_max = tuple(config["maps_net_normalization_bounds"]) if not config["clip_action_maps"] else (-1, 1)
    jax.debug.print("map normalization min max = {x}", x=map_min_max)
    model = SimplifiedCoupledCategoricalNet(
        mask_out_arm_extension=config["mask_out_arm_extension"],
        num_embeddings_agent=num_embeddings_agent,
        map_min_max=map_min_max,
        local_map_min_max=tuple(config["local_map_normalization_bounds"]),
        loaded_max=config["loaded_max"],
    )

    map_width = env.batch_cfg.maps.max_width
    map_height = env.batch_cfg.maps.max_height
    
    # TODO get this to work
    # n_local_maps_layers = env.batch_cfg.agent.max_arm_extension + 1 if not config["mask_out_arm_extension"] else 1
    n_local_maps_layers = 2

    obs = [
        jnp.zeros((config["num_envs"], 6,)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin, n_local_maps_layers)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin, n_local_maps_layers)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin, n_local_maps_layers)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin, n_local_maps_layers)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin, n_local_maps_layers)),
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin, n_local_maps_layers)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
    ]
    params = model.init(rng, obs)

    print(f"Model: {sum(x.size for x in jax.tree_leaves(params)):,} parameters")
    return model, params

def normalize(x: Array, x_min: Array, x_max: Array) -> Array:
    """
    Normalizes to [-1, 1]
    """
    return 2. * (x - x_min) / (x_max - x_min) - 1.

class MLP(nn.Module):
    """
    MLP without activation function at the last layer.
    """
    hidden_dim_layers: Sequence[int]
    use_layer_norm: bool
    last_layer_init_scaling: float = 1.0

    def setup(self) -> None:
        layer_init = nn.initializers.lecun_normal
        last_layer_init = lambda a,b,c: self.last_layer_init_scaling * layer_init()(a,b,c)
        self.activation = nn.relu

        if self.use_layer_norm:
            self.layers = [
                nn.Sequential([nn.Dense(self.hidden_dim_layers[i], kernel_init=layer_init()), nn.LayerNorm()])
                for i in range(len(self.hidden_dim_layers) - 1)
                ]
            self.layers += (nn.Dense(self.hidden_dim_layers[-1], kernel_init=last_layer_init),)
        else:
            self.layers = []
            for i, f in enumerate(self.hidden_dim_layers):
                if i < len(self.hidden_dim_layers) - 1:
                    self.layers += (nn.Dense(f, kernel_init=layer_init()),)
                else:
                    self.layers += (nn.Dense(f, kernel_init=last_layer_init),)
    
    def __call__(self, x):
        if self.use_layer_norm:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if ~(i % 2) and i != len(self.layers) - 1:
                    x = self.activation(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i != len(self.layers) - 1:
                    x = self.activation(x)
        return x


class AgentStateNet(nn.Module):
    """
    Pre-process the agent state features.
    """
    num_embeddings: int
    loaded_max: int
    mlp_use_layernorm: bool
    num_embedding_features: int = 8
    hidden_dim_layers_mlp_one_hot: Sequence[int] = (16, 32)
    hidden_dim_layers_mlp_continuous: Sequence[int] = (16, 32)
    
    def setup(self) -> None:
        self.embedding = nn.Embed(num_embeddings=self.num_embeddings, features=self.num_embedding_features)
        self.mlp_one_hot = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp_one_hot, use_layer_norm=self.mlp_use_layernorm)
        self.mlp_continuous = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp_continuous, use_layer_norm=self.mlp_use_layernorm)
    
    def __call__(self, obs: dict[str, Array]):
        x_one_hot = obs[0][..., :-1].astype(dtype=jnp.int32)
        x_loaded = obs[0][..., [-1]].astype(dtype=jnp.int32)

        x_one_hot = self.embedding(x_one_hot)
        x_one_hot = self.mlp_one_hot(x_one_hot.reshape(*x_one_hot.shape[:-2], -1))

        x_loaded = normalize(x_loaded, 0, self.loaded_max)
        x_continuous = self.mlp_continuous(x_loaded)

        return jnp.concatenate([x_one_hot, x_continuous], axis=-1)


class LocalMapNet(nn.Module):
    """
    Pre-process one or multiple maps.
    """
    map_min_max: Sequence[int]
    mlp_use_layernorm: bool
    hidden_dim_layers_mlp: Sequence[int] = (256, 32)

    def setup(self) -> None:
        self.mlp = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp, use_layer_norm=self.mlp_use_layernorm)

    def __call__(self, obs: dict[str, Array]):
        """
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
        """
        x_action_neg = normalize(obs[1], self.map_min_max[0], self.map_min_max[1])
        x_action_pos = normalize(obs[2], self.map_min_max[0], self.map_min_max[1])
        x_target_neg = normalize(obs[3], self.map_min_max[0], self.map_min_max[1])
        x_target_pos = normalize(obs[4], self.map_min_max[0], self.map_min_max[1])
        x_dumpability = obs[5]
        x_obstacles = obs[6]
        x = jnp.concatenate((
            x_action_neg[..., None],
            x_action_pos[..., None],
            x_target_neg[..., None],
            x_target_pos[..., None],
            x_dumpability[..., None],
            x_obstacles[..., None],
            ),
            -1,
        )

        x = self.mlp(x.reshape(*x.shape[:-3], -1))
        return x
    
class AtariCNN(nn.Module):
    """From https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/networks.py"""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=8, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        return x

@jax.jit
def min_pool(x):
    pool_fn = partial(nn.max_pool,
                      window_shape=(3, 3),
                      strides=(2, 2),
                      padding=((1, 1), (1, 1)))
    return -pool_fn(-x)

@jax.jit
def max_pool(x):
    pool_fn = partial(nn.max_pool,
                      window_shape=(3, 3),
                      strides=(2, 2),
                      padding=((1, 1), (1, 1)))
    return pool_fn(x)

@jax.jit
def zero_pool(x):
    """
    Given an input x with neg to pos values,
    zero_pool pools zeros with priority, then neg, then pos values.
    """
    x_pool = min_pool(x)
    mask_pool = max_pool(x == 0)

    return jnp.where(
        mask_pool,
        0,
        x_pool,
    )

class MapsNet(nn.Module):
    """
    Pre-process one or multiple maps.
    """
    map_min_max: Sequence[int]

    def setup(self) -> None:
        self.cnn = AtariCNN()

    @staticmethod
    def _generate_delta_map_negative(target_map: Array, action_map: Array):
        tm_clip = jnp.clip(target_map, a_max=0)
        am_clip = jnp.clip(action_map, a_max=0)
        return am_clip - tm_clip
    
    @staticmethod
    def _generate_delta_map_positive(target_map: Array, action_map: Array):
        tm_clip = jnp.clip(target_map, a_min=0)
        am_clip = jnp.clip(action_map, a_min=0)
        return am_clip - tm_clip

    def __call__(self, obs: dict[str, Array]):
        """
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
        """
        target_map = obs[8]
        traversability_map = obs[9]
        do_prediction = obs[10]
        dig_map = obs[11]
        dumpability_mask = obs[12]

        x = jnp.concatenate(
            (
                traversability_map[..., None],
                target_map[..., None],
                do_prediction[..., None],
                dig_map[..., None],
                dumpability_mask[..., None],
            ),
            axis=-1,
            )

        x = self.cnn(x)
        return x

    
class SimplifiedCoupledCategoricalNet(nn.Module):
    """
    The full net.

    The obs List follows the following order:
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
    """
    mask_out_arm_extension: bool
    num_embeddings_agent: int
    map_min_max: Sequence[int]
    local_map_min_max: Sequence[int]
    loaded_max: int
    action_type: Union[TrackedAction, WheeledAction] = TrackedAction
    hidden_dim_pi: Sequence[int] = (128, 32)
    hidden_dim_v: Sequence[int] = (128, 32, 1)
    mlp_use_layernorm: bool = False

    def setup(self) -> None:
        num_actions = self.action_type.get_num_actions()

        self.mlp_v = MLP(hidden_dim_layers=self.hidden_dim_v, use_layer_norm=self.mlp_use_layernorm, last_layer_init_scaling=0.01)
        self.mlp_pi = MLP(hidden_dim_layers=self.hidden_dim_pi + (num_actions,), use_layer_norm=self.mlp_use_layernorm, last_layer_init_scaling=0.01)

        self.local_map_net = LocalMapNet(map_min_max=self.local_map_min_max, mlp_use_layernorm=self.mlp_use_layernorm)

        self.agent_state_net = AgentStateNet(num_embeddings=self.num_embeddings_agent, loaded_max=self.loaded_max, mlp_use_layernorm=self.mlp_use_layernorm)

        self.maps_net = MapsNet(self.map_min_max)

        self.activation = nn.relu

    def __call__(self, obs: Array) -> Array:
        x_agent_state = self.agent_state_net(obs)
        
        x_maps = self.maps_net(obs)

        x_local_map = self.local_map_net(obs)

        x = jnp.concatenate((x_agent_state, x_maps, x_local_map), axis=-1)
        x = self.activation(x)

        v = self.mlp_v(x)

        xpi = self.mlp_pi(x)

        # INVALID ACTION MASKING
        # if self.use_action_masking:
        #     action_mask = action_mask.astype(jnp.bool_)
        #     # OPTION 1
        #     xpi = xpi * action_mask - 1e8 * (~action_mask)
        #     # OPTION 2
        #     # xpi = jnp.where(
        #     #     action_mask,
        #     #     xpi,
        #     #     -1e8
        #     # )

        if self.mask_out_arm_extension:
            xpi = xpi.at[..., -2].set(-1e8)
            xpi = xpi.at[..., -3].set(-1e8)

        return v, xpi
