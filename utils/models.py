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
# from jax_resnet import ResNet as ResNetBase
from jax_resnet import ModuleDef
from jax_resnet import ConvBlock
from typing import Optional, Callable
from functools import partial
from jax_resnet.common import Sequential



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
            map_min_max = tuple(config["maps_net_normalization_bounds"]) if not config["clip_action_maps"] else (-1, 1)
            jax.debug.print("map normalization min max = {x}", x=map_min_max)
            model = SimplifiedDecoupledCategoricalNet(
                use_action_masking=config["use_action_masking"],
                mask_out_arm_extension=config["mask_out_arm_extension"],
                num_embeddings_agent=num_embeddings_agent,
                map_min_max=map_min_max,
                local_map_min_max=tuple(config["local_map_normalization_bounds"]),
                loaded_max=config["loaded_max"],
            )
        elif config["network_name"] == "SimplifiedCoupledCategoricalNet":
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
            map_min_max = tuple(config["maps_net_normalization_bounds"]) if not config["clip_action_maps"] else (-1, 1)
            jax.debug.print("map normalization min max = {x}", x=map_min_max)
            model = SimplifiedCoupledCategoricalNet(
                use_action_masking=config["use_action_masking"],
                mask_out_arm_extension=config["mask_out_arm_extension"],
                num_embeddings_agent=num_embeddings_agent,
                map_min_max=map_min_max,
                local_map_min_max=tuple(config["local_map_normalization_bounds"]),
                loaded_max=config["loaded_max"],
            )

    if config["network_name"] == "Categorical-MLP":
        obs_shape_cumsum = sum([reduce(lambda x, y: x*y, value) for value in env.observation_shapes.values()])
        params = model.init(rng, jnp.zeros((obs_shape_cumsum,)), rng=rng)
    elif config["network_name"] in ("CategoricalNet", "SimplifiedCategoricalNet", "SimplifiedDecoupledCategoricalNet", "SimplifiedCoupledCategoricalNet"):
        map_width = env.batch_cfg.maps.max_width
        map_height = env.batch_cfg.maps.max_height
        obs = [
            jnp.zeros((config["num_train_envs"], 6,)),
            jnp.zeros((config["num_train_envs"], env.batch_cfg.agent.angles_cabin, env.batch_cfg.agent.max_arm_extension + 1)),
            jnp.zeros((config["num_train_envs"], env.batch_cfg.agent.angles_cabin, env.batch_cfg.agent.max_arm_extension + 1)),
            jnp.zeros((config["num_train_envs"], map_width, map_height)),
            jnp.zeros((config["num_train_envs"], map_width, map_height)),
            jnp.zeros((config["num_train_envs"], map_width, map_height)),
            jnp.zeros((config["num_train_envs"], map_width, map_height)),
            jnp.zeros((config["num_train_envs"], map_width, map_height)),
        ]
        action_mask = jnp.ones((config["num_train_envs"], env.batch_cfg.action_type.get_num_actions(),), dtype=jnp.bool_)
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
    # hidden_dim: int = 8
    # one_hot_num_classes: int
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
        # NOTE: this implementation uses the pos of the agent explicitly

        x_one_hot = obs[0][..., :-1].astype(dtype=jnp.int32)
        x_loaded = obs[0][..., [-1]].astype(dtype=jnp.int32)

        # jax.debug.print("jnp.sum(jnp.isnan(x_one_hot)) BEFORE = {x}", x=jnp.sum(jnp.isnan(x_one_hot)))
        x_one_hot = self.embedding(x_one_hot)
        # jax.debug.print("jnp.sum(jnp.isnan(x_one_hot)) = {x}", x=jnp.sum(jnp.isnan(x_one_hot)))
        x_one_hot = self.mlp_one_hot(x_one_hot.reshape(*x_one_hot.shape[:-2], -1))

        x_loaded = normalize(x_loaded, 0, self.loaded_max)
        # x_pos = normalize(x_pos, 0, self.pos_max)
        x_continuous = self.mlp_continuous(x_loaded)

        return jnp.concatenate([x_one_hot, x_continuous], axis=-1)
        # return x_one_hot


class LocalMapNet(nn.Module):
    """
    Pre-process one or multiple maps.
    """
    map_min_max: Sequence[int]
    mlp_use_layernorm: bool
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
        self.mlp = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp, use_layer_norm=self.mlp_use_layernorm)

    def __call__(self, obs: dict[str, Array]):
        x_action = normalize(obs[1], self.map_min_max[0], self.map_min_max[1])
        x_target = normalize(obs[2], self.map_min_max[0], self.map_min_max[1])
        x = jnp.concatenate((x_action[..., None], x_target[..., None]), -1)

        # x = self.conv1(x)
        # x = x[..., None]
        # x = self.resnet(x)
        x = self.mlp(x.reshape(*x.shape[:-3], -1))
        return x
    
class ResNetStemIdentity(nn.Module):
    # need to name it conv_block_cls for compatibility with ResNet library
    conv_block_cls: ModuleDef

    def setup(self) -> None:
        pass
    
    def __call__(self, x):
        return (x)
    
# class CNN(nn.Module):
#   """A simple CNN model."""

#   @nn.compact
#   def __call__(self, x):
#     x = nn.Conv(features=32, kernel_size=(3, 3))(x)
#     x = nn.relu(x)
#     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#     x = nn.Conv(features=64, kernel_size=(3, 3))(x)
#     x = nn.relu(x)
#     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#     x = x.reshape((x.shape[0], -1))  # flatten
#     x = nn.Dense(features=256)(x)
#     x = nn.relu(x)
#     x = nn.Dense(features=32)(x)
#     return x
  
class AtariCNN(nn.Module):
    """From https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/networks.py"""

    @nn.compact
    def __call__(self, x):
        # TODO careful with padding!
        # x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.Conv(features=8, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        # x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.Conv(features=16, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        # x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        
        # x = nn.Dense(features=512)(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        return x


def MyResNet(
    block_cls: ModuleDef,
    *,
    stage_sizes: Sequence[int],
    n_classes: int,
    hidden_sizes: Sequence[int],
    global_avg_pool: bool,
    normalize_fn: Optional[Callable],
    pool_fn: Optional[Callable],
    conv_cls: ModuleDef = nn.Conv,
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9),
    conv_block_cls: ModuleDef = ConvBlock,
    # stem_cls: ModuleDef = ResNetStem,
) -> Sequential:
    conv_block_cls = partial(conv_block_cls, conv_cls=conv_cls, norm_cls=norm_cls)
    # stem_cls = partial(stem_cls, conv_block_cls=conv_block_cls)
    block_cls = partial(block_cls, conv_block_cls=conv_block_cls)

    if pool_fn is not None:
        # NOTE: we apply pooling before normalization
        layers = [pool_fn,]
    else:
        layers = []
    
    if normalize_fn is not None:
        layers.append(normalize_fn)

    for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
        for b in range(n_blocks):
            strides = (1, 1) if i == 0 or b != 0 else (2, 2)  # TODO is this (2, 2) an issue?
            layers.append(block_cls(n_hidden=hsize, strides=strides))

    if global_avg_pool:
        layers.append(partial(jnp.mean, axis=(1, 2)))  # global average pool
    else:
        layers.append(lambda x: x.reshape(x.shape[0], -1))  # no pooling
    
    # Dense layers
    # mlp_head = MLP(
    #     hidden_dim_layers=(512, 32),
    #     use_layer_norm=True,
    # )
    # layers.append(mlp_head)
    layers.append(nn.Dense(n_classes))
    
    return Sequential(layers)

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

# NOTE: my_pool does not make sense for a radical delta map feature set.
# @jax.jit
# def my_pool(x):
#     """
#     Handles special pooling on x, assumed to be ([N]xWxHxC).

#     List of channels and what we highlight with my_pool:
#     1. action map -> 0 tiles
#     2. target map -> -1 tiles
#     3. traversability map -> 1 tiles
#     4. do_prediction map -> 0 tiles
#     5. dig map -> 0 tiles
#     5. delta map -> 1 tiles
#     """
#     # jax.debug.print("x.shape = {y}", y=x.shape)

#     x = jnp.swapaxes(x, 0, 1)
#     x = jnp.swapaxes(x, 1, 2)

#     action_map = x[..., 0]
#     target_map = x[..., 1]
#     traversability_map = x[..., 2]
#     do_prediction_map = x[..., 3]
#     dig_map = x[..., 4]
#     delta_map = x[..., 5]

#     action_map = zero_pool(action_map)
#     target_map = min_pool(target_map)
#     traversability_map = max_pool(traversability_map)
#     do_prediction_map = zero_pool(do_prediction_map)
#     dig_map = zero_pool(dig_map)
#     delta_map = max_pool(delta_map)

#     x = jnp.concatenate((action_map[..., None], target_map[..., None], traversability_map[..., None],
#                             do_prediction_map[..., None], dig_map[..., None], delta_map[..., None]), axis=-1)

#     x = jnp.swapaxes(x, 1, 2)
#     x = jnp.swapaxes(x, 0, 1)

#     # jax.debug.print("x.shape = {y}", y=x.shape)
    
#     return x

class MapsNet(nn.Module):
    """
    Pre-process one or multiple maps.
    """
    map_min_max: Sequence[int]

    def setup(self) -> None:

        # SET POOL FUNCTION HERE
        # pool_fn = my_pool
        pool_fn = None

        # SET GLOBAL AVG POOL BOOL HERE
        global_avg_pool = True
        
        # SET NORMALIZATION HERE
        # TODO: this way I'm normalizing also traversability -- problems with that?
        # self.normalize_maps_fn = partial(
        #     normalize,
        #     x_min=self.map_min_max[0],
        #     x_max=self.map_min_max[1],
        # )
        self.normalize_maps_fn = None


        # self.cnn = MyResNet(
        #                 block_cls=jax_resnet.ResNetBlock,
        #                 stage_sizes=[2, 2, 2],
        #                 # stage_sizes=[2, 2],
        #                 hidden_sizes=(8, 16, 32),
        #                 # hidden_sizes=(16, 32),
        #                 # stem_cls=ResNetStemIdentity,
        #                 n_classes=32,
        #                 pool_fn=pool_fn,
        #                 global_avg_pool=global_avg_pool,
        #                 normalize_fn=self.normalize_maps_fn,
        #             )
        
        self.cnn = AtariCNN()

        # self.mlp = MLP(hidden_dim_layers=self.hidden_dim_layers_mlp)

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
        # action_map = obs[3]
        target_map = obs[4]
        traversability_map = obs[5]
        do_prediction = obs[6]
        dig_map = obs[7]

        # dig_delta_map_negative = self._generate_delta_map_negative(target_map, dig_map)
        # do_prediction_delta_map_negative = self._generate_delta_map_negative(target_map, do_prediction)

        # dig_delta_map_positive = self._generate_delta_map_positive(target_map, dig_map)
        # do_prediction_delta_map_positive = self._generate_delta_map_positive(target_map, do_prediction)

        # NOTE: if change the following, need to also change my_pool
        x = jnp.concatenate(
            (
                traversability_map[..., None],
                target_map[..., None],
                do_prediction[..., None],
                dig_map[..., None],
            ),
            axis=-1,
            )

        # x = self.conv1(x)
        x = self.cnn(x)
        return x

    
# class SimplifiedDecoupledCategoricalNet(nn.Module):
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
#     use_action_masking: bool
#     mask_out_arm_extension: bool
#     num_embeddings_agent: int
#     map_min_max: Sequence[int]
#     local_map_min_max: Sequence[int]
#     loaded_max: int
#     action_type: Union[TrackedAction, WheeledAction] = TrackedAction
#     # hidden_dim_layers_common: Sequence[int] = (256, 64)
#     hidden_dim_pi: Sequence[int] = (128, 32)
#     hidden_dim_v: Sequence[int] = (128, 32, 1)

#     def setup(self) -> None:
#         num_actions = self.action_type.get_num_actions()

#         # self.common_mlp_v = MLP(hidden_dim_layers=self.hidden_dim_layers_common)
#         # self.common_mlp_pi = MLP(hidden_dim_layers=self.hidden_dim_layers_common)

#         self.mlp_v = MLP(hidden_dim_layers=self.hidden_dim_v)
#         self.mlp_pi = MLP(hidden_dim_layers=self.hidden_dim_pi + (num_actions,))

#         self.local_map_net_v = LocalMapNet(map_min_max=self.local_map_min_max)
#         self.local_map_net_pi = LocalMapNet(map_min_max=self.local_map_min_max)

#         self.agent_state_net_v = AgentStateNet(num_embeddings=self.num_embeddings_agent, loaded_max=self.loaded_max)
#         self.agent_state_net_pi = AgentStateNet(num_embeddings=self.num_embeddings_agent, loaded_max=self.loaded_max)

#         # Resnet
#         self.maps_net_v = MapsNet(self.map_min_max)
#         self.maps_net_pi = MapsNet(self.map_min_max)
#         # MLP for maps
#         # self.maps_net_v = SimplifiedMapsNet()
#         # self.maps_net_pi = SimplifiedMapsNet()

#         self.activation = nn.relu

#     def __call__(self, obs: Array, action_mask: Array) -> Array:
#         x_agent_state_v = self.agent_state_net_v(obs)
#         x_agent_state_pi = self.agent_state_net_pi(obs)
        
#         x_maps_v = self.maps_net_v(obs)
#         x_maps_pi = self.maps_net_pi(obs)

#         x_local_map_v = self.local_map_net_v(obs)
#         x_local_map_pi = self.local_map_net_pi(obs)

#         xv = jnp.concatenate((x_agent_state_v, x_maps_v, x_local_map_v), axis=-1)
#         xv = self.activation(xv)

#         xpi = jnp.concatenate((x_agent_state_pi, x_maps_pi, x_local_map_pi), axis=-1)
#         xpi = self.activation(xpi)

#         # xv = self.common_mlp_v(xv)
#         # xv = self.activation(xv)

#         # xpi = self.common_mlp_pi(xpi)
#         # xpi = self.activation(xpi)

#         v = self.mlp_v(xv)

#         xpi = self.mlp_pi(xpi)

#         # INVALID ACTION MASKING
#         if self.use_action_masking:
#             action_mask = action_mask.astype(jnp.bool_)
#             # OPTION 1
#             xpi = xpi * action_mask - 1e8 * (~action_mask)
#             # OPTION 2
#             # xpi = jnp.where(
#             #     action_mask,
#             #     xpi,
#             #     -1e8
#             # )

#         if self.mask_out_arm_extension:
#             xpi = xpi.at[..., -2].set(-1e8)
#             xpi = xpi.at[..., -3].set(-1e8)

#         # jax.debug.print("action_mask={x}", x=action_mask)
#         # jax.debug.print("xpi={x}", x=xpi)
        
#         # Note: returning logits xpi, not distribution pi!
#         return v, xpi
    

class SimplifiedCoupledCategoricalNet(nn.Module):
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
    map_min_max: Sequence[int]
    local_map_min_max: Sequence[int]
    loaded_max: int
    action_type: Union[TrackedAction, WheeledAction] = TrackedAction
    # hidden_dim_layers_common: Sequence[int] = (256, 64)
    hidden_dim_pi: Sequence[int] = (128, 32)
    hidden_dim_v: Sequence[int] = (128, 32, 1)
    mlp_use_layernorm: bool = True

    def setup(self) -> None:
        num_actions = self.action_type.get_num_actions()

        # self.common_mlp_v = MLP(hidden_dim_layers=self.hidden_dim_layers_common)
        # self.common_mlp_pi = MLP(hidden_dim_layers=self.hidden_dim_layers_common)

        self.mlp_v = MLP(hidden_dim_layers=self.hidden_dim_v, use_layer_norm=self.mlp_use_layernorm, last_layer_init_scaling=0.01)
        self.mlp_pi = MLP(hidden_dim_layers=self.hidden_dim_pi + (num_actions,), use_layer_norm=self.mlp_use_layernorm, last_layer_init_scaling=0.01)

        self.local_map_net = LocalMapNet(map_min_max=self.local_map_min_max, mlp_use_layernorm=self.mlp_use_layernorm)

        self.agent_state_net = AgentStateNet(num_embeddings=self.num_embeddings_agent, loaded_max=self.loaded_max, mlp_use_layernorm=self.mlp_use_layernorm)

        # Resnet
        self.maps_net = MapsNet(self.map_min_max)
        # MLP for maps
        # self.maps_net_v = SimplifiedMapsNet()
        # self.maps_net_pi = SimplifiedMapsNet()

        self.activation = nn.relu

    def __call__(self, obs: Array, action_mask: Array) -> Array:
        x_agent_state = self.agent_state_net(obs)
        
        x_maps = self.maps_net(obs)

        x_local_map = self.local_map_net(obs)

        x = jnp.concatenate((x_agent_state, x_maps, x_local_map), axis=-1)
        x = self.activation(x)

        v = self.mlp_v(x)

        xpi = self.mlp_pi(x)

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
