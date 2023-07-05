"""
Partially from https://github.com/RobertTLange/gymnax-blines
"""

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp
from functools import reduce
from typing import Any, Sequence, Union
from terra.actions import TrackedAction, WheeledAction
from terra.env import TerraEnvBatch
import jax_resnet
from jax_resnet import ResNet as ResNetBase
from jax_resnet import ModuleDef
from jax_resnet import ConvBlock



def get_model_ready(rng, config, env: TerraEnvBatch, speed=False):
    """Instantiate a model according to obs shape of environment."""
    if config["train_type"] == "PPO":
        if config["network_name"] == "SimplifiedDecoupledCategoricalNet":
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
            jax.debug.print("config.use_action_masking={x}", x=config["use_action_masking"])
            model = SimplifiedDecoupledCategoricalNet(
                use_action_masking=config["use_action_masking"],
                mask_out_arm_extension=config["mask_out_arm_extension"],
                num_embeddings_agent=num_embeddings_agent
            )

    if config["network_name"] == "Categorical-MLP":
        obs_shape_cumsum = sum([reduce(lambda x, y: x*y, value) for value in env.observation_shapes.values()])
        params = model.init(rng, jnp.zeros((obs_shape_cumsum,)), rng=rng)
    elif config["network_name"] in ("CategoricalNet", "SimplifiedCategoricalNet", "SimplifiedDecoupledCategoricalNet"):
        map_width = env.batch_cfg.maps.max_width
        map_height = env.batch_cfg.maps.max_height
        obs = [
            jnp.zeros((config["num_train_envs"], 6,)),
            jnp.zeros((config["num_train_envs"], env.batch_cfg.agent.angles_cabin, env.batch_cfg.agent.max_arm_extension + 1)),
            jnp.zeros((config["num_train_envs"], env.batch_cfg.agent.angles_cabin, env.batch_cfg.agent.max_arm_extension + 1)),
            jnp.zeros((config["num_train_envs"], map_width, map_height)),
            jnp.zeros((config["num_train_envs"], map_width, map_height)),
            jnp.zeros((config["num_train_envs"], map_width, map_height)),
        ]
        action_mask = jnp.ones((5, env.batch_cfg.action_type.get_num_actions(),), dtype=jnp.bool_)
        # {
        #     "agent_state": jnp.zeros((5, 4,)),
        #     "local_map": jnp.zeros((5, env.env_cfg.agent.angles_cabin, env.env_cfg.agent.max_arm_extension + 1)),
        #     "traversability_mask": jnp.zeros((5, map_width, map_height)),
        #     "action_map": jnp.zeros((5, map_width, map_height)),
        #     "target_map": jnp.zeros((5, map_width, map_height)),
        # }
        params = model.init(rng, obs, action_mask)

    print(f"{config['network_name']}: {sum(x.size for x in jax.tree_leaves(params)):,} parameters")
    return model, params


def default_mlp_init(scale=0.05):
    return nn.initializers.uniform(scale)

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
    use_layer_norm: bool = True

    def setup(self) -> None:
        if self.use_layer_norm:
            self.layers = [
                nn.Sequential([nn.Dense(self.hidden_dim_layers[i]), nn.LayerNorm()])
                for i in range(len(self.hidden_dim_layers) - 1)
                ]
            self.layers += (nn.Dense(self.hidden_dim_layers[-1]),)
        else:
            self.layers = [nn.Dense(f) for f in self.hidden_dim_layers]
        self.activation = nn.relu
    
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
    # hidden_dim: int = 8
    # one_hot_num_classes: int
    num_embeddings: int
    num_embedding_features: int = 8
    hidden_dim_layers_mlp_one_hot: Sequence[int] = (16, 32)
    # hidden_dim_layers_mlp_continuous: Sequence[int] = (16, 16)
    
    def setup(self) -> None:
        self.embedding = nn.Embed(num_embeddings=self.num_embeddings, features=self.num_embedding_features)
        self.mlp_one_hot = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp_one_hot)
        # self.mlp_continuous = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp_continuous)
    
    def __call__(self, obs: dict[str, Array]):
        # NOTE: this implementation uses the pos of the agent explicitly

        x_one_hot = obs[0].astype(dtype=jnp.int32)
        # x_loaded = obs[0][..., [-1]]

        x_one_hot = self.embedding(x_one_hot)
        x_one_hot = self.mlp_one_hot(x_one_hot.reshape(*x_one_hot.shape[:-2], -1))

        # x_loaded = normalize(x_loaded, 0, self.loaded_max)
        # x_pos = normalize(x_pos, 0, self.pos_max)
        # x_continuous = self.mlp_continuous(jnp.concatenate((x_pos, x_loaded), axis=-1))

        # return jnp.concatenate([x_one_hot, x_continuous], axis=-1)
        return x_one_hot


class LocalMapNet(nn.Module):
    """
    Pre-process one or multiple maps.
    """
    map_min_max: Sequence[int] = (-16, 16)
    hidden_dim_layers_mlp: Sequence[int] = (128, 16)

    def setup(self) -> None:
        # self.conv1 = nn.Conv(3, kernel_size=(1, 1))
        # self.resnet = ResNetBase(
        #                 block_cls=jax_resnet.ResNetBlock,
        #                 stage_sizes=jax_resnet.STAGE_SIZES[18],
        #                 stem_cls=jax_resnet.ResNetStem,
        #                 n_classes=16,
        #                 # norm_cls=nn.LayerNorm,
        #             )
        self.mlp = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp)

    def __call__(self, obs: dict[str, Array]):
        x_action = normalize(obs[1], self.map_min_max[0], self.map_min_max[1])
        x_target = normalize(obs[2], self.map_min_max[0], self.map_min_max[1])
        x = jnp.concatenate((x_action[..., None], x_target[..., None]), -1)

        # x = self.conv1(x)
        # x = x[..., None]
        # x = self.resnet(x)
        x = self.mlp(x.reshape(*x.shape[:-3], -1))
        return x
    

class Id(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x

class ResNetStemIdentity(nn.Module):
    # need to name it conv_block_cls for compatibility with ResNet library
    conv_block_cls: ModuleDef = Id

    @nn.compact
    def __call__(self, x):
        return self.conv_block_cls()(x)

class MapsNet(nn.Module):
    """
    Pre-process one or multiple maps.
    """
    map_min_max: Sequence[int] = (-1, 4)  # TODO from config

    def setup(self) -> None:
        # self.conv1 = nn.Conv(3, kernel_size=(1, 1))
        self.resnet = ResNetBase(
                        block_cls=jax_resnet.ResNetBlock,
                        stage_sizes=[4, 4],
                        hidden_sizes=(8, 16),
                        stem_cls=ResNetStemIdentity,
                        n_classes=32,
                        # norm_cls=nn.LayerNorm,
                    )
        # self.mlp = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp)

    def __call__(self, obs: dict[str, Array]):
        action_map = normalize(obs[3], self.map_min_max[0], self.map_min_max[1])

        # delta_target_map = jnp.clip(obs[3] - obs[2], a_max=0)
        # delta_target_map = normalize(delta_target_map, self.map_min_max[0], 0)

        target_map = normalize(obs[4], self.map_min_max[0], self.map_min_max[1])

        x = jnp.concatenate((action_map[..., None], target_map[..., None], obs[5][..., None]), axis=-1)

        # x = self.conv1(x)
        x = self.resnet(x)
        return x
    

# class SimplifiedMapsNet(nn.Module):
#     """
#     Pre-process one or multiple maps.
#     """
#     map_min_max: Sequence[int] = (-2, 2)  # TODO from config
#     hidden_dim_layers_mlp: Sequence[int] = (128, 32)

#     def setup(self) -> None:
#         self.mlp = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp)

#     def __call__(self, obs: dict[str, Array]):
#         action_map = normalize(obs[3], self.map_min_max[0], self.map_min_max[1])

#         # delta_target_map = jnp.clip(obs[3] - obs[2], a_max=0)
#         # delta_target_map = normalize(delta_target_map, self.map_min_max[0], 0)

#         target_map = normalize(obs[4], self.map_min_max[0], self.map_min_max[1])

#         x = jnp.concatenate((action_map[..., None], target_map[..., None], obs[5][..., None]), axis=-1)
#         # x = self.conv1(x)
#         x = self.mlp(x.reshape(x.shape[0], -1))
#         return x


# class CategoricalNet(nn.Module):
#     """
#     The full net.

#     The obs List follows the following order:
#     0 - obs["agent_state"],
#     1 - obs["local_map_action"],
#     2 - obs["local_map_target"],
#     3 - obs["action_map"],
#     4 - obs["target_map"],
#     5 - obs["traversability_mask"]
#     """
#     action_type: Union[TrackedAction, WheeledAction] = TrackedAction
#     hidden_dim_layers_common: Sequence[int] = (64, 32)
#     hidden_dim_pi: Sequence[int] = (16, 16)
#     hidden_dim_v: Sequence[int] = (16, 4, 1)

#     def setup(self) -> None:
#         num_actions = self.action_type.get_num_actions()

#         self.common_mlp = MLP(hidden_dim_layers=self.hidden_dim_layers_common)
#         self.mlp_v = MLP(hidden_dim_layers=self.hidden_dim_v)
#         self.mlp_pi = MLP(hidden_dim_layers=self.hidden_dim_pi + (num_actions,))

#         self.agent_state_net = AgentStateNet()
#         self.maps_net = MapsNet()
#         self.local_map_net = LocalMapNet()

#         self.activation = nn.relu

#     def __call__(self, obs: Array, action_mask: Array) -> Array:
#         x_agent_state = self.agent_state_net(obs)
#         x_maps = self.maps_net(obs)
#         x_local_map = self.local_map_net(obs)

#         x = jnp.concatenate((x_agent_state, x_local_map, x_maps), axis=-1)
#         x = self.activation(x)

#         x = self.common_mlp(x)
#         x = self.activation(x)

#         v = self.mlp_v(x)

#         x_pi = self.mlp_pi(x)
#         # x_pi = jnp.where(
#         #     action_mask.astype(jnp.bool_),
#         #     x_pi,
#         #     -1e8
#         # )

#         # # Mask out arm extension
#         x_pi = x_pi.at[..., -2].set(-1e8)
#         x_pi = x_pi.at[..., -3].set(-1e8)

#         # jax.debug.print("action_mask={x}", x=action_mask)
#         # jax.debug.print("x_pi={x}", x=x_pi)
        
#         pi = tfp.distributions.Categorical(logits=x_pi)
#         return v, pi


# class SimplifiedCategoricalNet(nn.Module):
#     """
#     The full net.

#     The obs List follows the following order:
#     0 - obs["agent_state"],
#     1 - obs["local_map_action"],
#     2 - obs["local_map_target"],
#     3 - obs["action_map"],
#     4 - obs["target_map"],
#     5 - obs["traversability_mask"]
#     """
#     action_type: Union[TrackedAction, WheeledAction] = TrackedAction
#     hidden_dim_layers_common: Sequence[int] = (64, 32)
#     hidden_dim_pi: Sequence[int] = (16, 16)
#     hidden_dim_v: Sequence[int] = (16, 4, 1)

#     def setup(self) -> None:
#         num_actions = self.action_type.get_num_actions()

#         self.common_mlp = MLP(hidden_dim_layers=self.hidden_dim_layers_common)
#         self.mlp_v = MLP(hidden_dim_layers=self.hidden_dim_v)
#         self.mlp_pi = MLP(hidden_dim_layers=self.hidden_dim_pi + (num_actions,))

#         self.agent_state_net = AgentStateNet()
#         self.maps_net = SimplifiedMapsNet()

#         self.activation = nn.relu

#     def __call__(self, obs: Array, action_mask: Array) -> Array:
#         x_agent_state = self.agent_state_net(obs)
#         x_maps = self.maps_net(obs)

#         x = jnp.concatenate((x_agent_state, x_maps), axis=-1)
#         x = self.activation(x)

#         x = self.common_mlp(x)
#         x = self.activation(x)

#         v = self.mlp_v(x)

#         x_pi = self.mlp_pi(x)
#         # x_pi = jnp.where(
#         #     action_mask.astype(jnp.bool_),
#         #     x_pi,
#         #     -1e8
#         # )

#         # # Mask out arm extension
#         x_pi = x_pi.at[..., -2].set(-1e8)
#         x_pi = x_pi.at[..., -3].set(-1e8)

#         # jax.debug.print("action_mask={x}", x=action_mask)
#         # jax.debug.print("x_pi={x}", x=x_pi)
        
#         pi = tfp.distributions.Categorical(logits=x_pi)
#         return v, pi
    
class SimplifiedDecoupledCategoricalNet(nn.Module):
    """
    The full net.

    The obs List follows the following order:
    0 - obs["agent_state"],
    1 - obs["local_map_action"],
    2 - obs["local_map_target"],
    3 - obs["action_map"],
    4 - obs["target_map"],
    5 - obs["traversability_mask"]
    """
    use_action_masking: bool
    mask_out_arm_extension: bool
    num_embeddings_agent: int
    action_type: Union[TrackedAction, WheeledAction] = TrackedAction
    # hidden_dim_layers_common: Sequence[int] = (256, 64)
    hidden_dim_pi: Sequence[int] = (128, 32)
    hidden_dim_v: Sequence[int] = (128, 32, 1)

    def setup(self) -> None:
        num_actions = self.action_type.get_num_actions()

        # self.common_mlp_v = MLP(hidden_dim_layers=self.hidden_dim_layers_common)
        # self.common_mlp_pi = MLP(hidden_dim_layers=self.hidden_dim_layers_common)

        self.mlp_v = MLP(hidden_dim_layers=self.hidden_dim_v)
        self.mlp_pi = MLP(hidden_dim_layers=self.hidden_dim_pi + (num_actions,))

        self.local_map_net_v = LocalMapNet()
        self.local_map_net_pi = LocalMapNet()

        self.agent_state_net_v = AgentStateNet(num_embeddings=self.num_embeddings_agent)
        self.agent_state_net_pi = AgentStateNet(num_embeddings=self.num_embeddings_agent)

        # Resnet
        self.maps_net_v = MapsNet()
        self.maps_net_pi = MapsNet()
        # MLP for maps
        # self.maps_net_v = SimplifiedMapsNet()
        # self.maps_net_pi = SimplifiedMapsNet()

        self.activation = nn.relu

    def __call__(self, obs: Array, action_mask: Array) -> Array:
        x_agent_state_v = self.agent_state_net_v(obs)
        x_agent_state_pi = self.agent_state_net_pi(obs)
        
        x_maps_v = self.maps_net_v(obs)
        x_maps_pi = self.maps_net_pi(obs)

        x_local_map_v = self.local_map_net_v(obs)
        x_local_map_pi = self.local_map_net_pi(obs)

        xv = jnp.concatenate((x_agent_state_v, x_maps_v, x_local_map_v), axis=-1)
        xv = self.activation(xv)

        xpi = jnp.concatenate((x_agent_state_pi, x_maps_pi, x_local_map_pi), axis=-1)
        xpi = self.activation(xpi)

        # xv = self.common_mlp_v(xv)
        # xv = self.activation(xv)

        # xpi = self.common_mlp_pi(xpi)
        # xpi = self.activation(xpi)

        v = self.mlp_v(xv)

        xpi = self.mlp_pi(xpi)

        # INVALID ACTION MASKING
        if self.use_action_masking:
            action_mask = action_mask.astype(jnp.bool_)
            # OPTION 1
            xpi = xpi * action_mask - 1e8 * (~action_mask)
            # OPTION 2
            # xpi = jnp.where(
            #     action_mask,
            #     xpi,
            #     -1e8
            # )

        if self.mask_out_arm_extension:
            xpi = xpi.at[..., -2].set(-1e8)
            xpi = xpi.at[..., -3].set(-1e8)

        # jax.debug.print("action_mask={x}", x=action_mask)
        # jax.debug.print("xpi={x}", x=xpi)
        
        # Note: returning logits xpi, not distribution pi!
        return v, xpi


# class CategoricalSeparateMLP(nn.Module):
#     """Split Actor-Critic Architecture for PPO."""

#     num_output_units: int
#     num_hidden_units: int
#     num_hidden_layers: int
#     prefix_actor: str = "actor"
#     prefix_critic: str = "critic"
#     model_name: str = "separate-mlp"
#     flatten_2d: bool = False  # Catch case
#     flatten_3d: bool = False  # Rooms/minatar case

#     @nn.compact
#     def __call__(self, x, rng):
#         # Flatten a single 2D image
#         if self.flatten_2d and len(x.shape) == 2:
#             x = x.reshape(-1)
#         # Flatten a batch of 2d images into a batch of flat vectors
#         if self.flatten_2d and len(x.shape) > 2:
#             x = x.reshape(x.shape[0], -1)

#         # Flatten a single 3D image
#         if self.flatten_3d and len(x.shape) == 3:
#             x = x.reshape(-1)
#         # Flatten a batch of 3d images into a batch of flat vectors
#         if self.flatten_3d and len(x.shape) > 3:
#             x = x.reshape(x.shape[0], -1)
#         x_v = nn.relu(
#             nn.Dense(
#                 self.num_hidden_units,
#                 name=self.prefix_critic + "_fc_1",
#                 bias_init=default_mlp_init(),
#             )(x)
#         )
#         # Loop over rest of intermediate hidden layers
#         for i in range(1, self.num_hidden_layers):
#             x_v = nn.relu(
#                 nn.Dense(
#                     self.num_hidden_units,
#                     name=self.prefix_critic + f"_fc_{i+1}",
#                     bias_init=default_mlp_init(),
#                 )(x_v)
#             )
#         v = nn.Dense(
#             1,
#             name=self.prefix_critic + "_fc_v",
#             bias_init=default_mlp_init(),
#         )(x_v)

#         x_a = nn.relu(
#             nn.Dense(
#                 self.num_hidden_units,
#                 bias_init=default_mlp_init(),
#             )(x)
#         )
#         # Loop over rest of intermediate hidden layers
#         for i in range(1, self.num_hidden_layers):
#             x_a = nn.relu(
#                 nn.Dense(
#                     self.num_hidden_units,
#                     bias_init=default_mlp_init(),
#                 )(x_a)
#             )
#         logits = nn.Dense(
#             self.num_output_units,
#             bias_init=default_mlp_init(),
#         )(x_a)
#         # pi = distrax.Categorical(logits=logits)
#         pi = tfp.distributions.Categorical(logits=logits)
#         return v, pi


# class GaussianSeparateMLP(nn.Module):
#     """Split Actor-Critic Architecture for PPO."""

#     num_output_units: int
#     num_hidden_units: int
#     num_hidden_layers: int
#     prefix_actor: str = "actor"
#     prefix_critic: str = "critic"
#     min_std: float = 0.001
#     model_name: str = "separate-mlp"

#     @nn.compact
#     def __call__(self, x, rng):
#         x_v = nn.relu(
#             nn.Dense(
#                 self.num_hidden_units,
#                 name=self.prefix_critic + "_fc_1",
#                 bias_init=default_mlp_init(),
#             )(x)
#         )
#         # Loop over rest of intermediate hidden layers
#         for i in range(1, self.num_hidden_layers):
#             x_v = nn.relu(
#                 nn.Dense(
#                     self.num_hidden_units,
#                     name=self.prefix_critic + f"_fc_{i+1}",
#                     bias_init=default_mlp_init(),
#                 )(x_v)
#             )
#         v = nn.Dense(
#             1,
#             name=self.prefix_critic + "_fc_v",
#             bias_init=default_mlp_init(),
#         )(x_v)

#         x_a = nn.relu(
#             nn.Dense(
#                 self.num_hidden_units,
#                 name=self.prefix_actor + "_fc_1",
#                 bias_init=default_mlp_init(),
#             )(x)
#         )
#         # Loop over rest of intermediate hidden layers
#         for i in range(1, self.num_hidden_layers):
#             x_a = nn.relu(
#                 nn.Dense(
#                     self.num_hidden_units,
#                     name=self.prefix_actor + f"_fc_{i+1}",
#                     bias_init=default_mlp_init(),
#                 )(x_a)
#             )
#         mu = nn.Dense(
#             self.num_output_units,
#             name=self.prefix_actor + "_fc_mu",
#             bias_init=default_mlp_init(),
#         )(x_a)
#         log_scale = nn.Dense(
#             self.num_output_units,
#             name=self.prefix_actor + "_fc_scale",
#             bias_init=default_mlp_init(),
#         )(x_a)
#         scale = jax.nn.softplus(log_scale) + self.min_std
#         pi = tfp.distributions.MultivariateNormalDiag(mu, scale)
#         return v, pi
