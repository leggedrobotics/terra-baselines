import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
from typing import Sequence, Union
from terra.actions import TrackedAction, WheeledAction
from terra.env import TerraEnvBatch
from functools import partial


def get_model_ready(rng, config, env: TerraEnvBatch, speed=False):
    """Instantiate a model according to obs shape of environment."""
    num_embeddings_agent = jnp.max(
        jnp.array(
            [
                env.batch_cfg.maps_dims.maps_edge_length,
                env.batch_cfg.agent.angles_cabin,
                env.batch_cfg.agent.angles_base,
            ],
            dtype=jnp.int16,
        )
    ).item()
    jax.debug.print("num_embeddings_agent = {x}", x=num_embeddings_agent)
    map_min_max = (
        tuple(config["maps_net_normalization_bounds"])
        if not config["clip_action_maps"]
        else (-1, 1)
    )
    jax.debug.print("map normalization min max = {x}", x=map_min_max)
    model = CentralizedTwoAgentNet(
        num_prev_actions=config["num_prev_actions"],
        num_embeddings_agent=num_embeddings_agent,
        map_min_max=map_min_max,
        local_map_min_max=tuple(config["local_map_normalization_bounds"]),
        loaded_max=config["loaded_max"],
        action_type=env.batch_cfg.action_type,
    )

    map_width = env.batch_cfg.maps_dims.maps_edge_length
    map_height = env.batch_cfg.maps_dims.maps_edge_length

    # Updated obs structure for 2 agents
    obs = [
        # Agent 1 state
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.num_state_obs)),
        # Agent 2 state  
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.num_state_obs)),
        # Agent 1 local maps
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_action_neg
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_action_pos
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_target_neg
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_target_pos
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_dumpability
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_obstacles
        # Agent 2 local maps
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_action_neg_2
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_action_pos_2
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_target_neg_2
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_target_pos_2
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_dumpability_2
        jnp.zeros((config["num_envs"], env.batch_cfg.agent.angles_cabin)),  # local_map_obstacles_2
        # Global maps (shared)
        jnp.zeros((config["num_envs"], map_width, map_height)),  # action_map
        jnp.zeros((config["num_envs"], map_width, map_height)),  # target_map
        jnp.zeros((config["num_envs"], map_width, map_height)),  # traversability_mask
        jnp.zeros((config["num_envs"], map_width, map_height)),  # dumpability_mask
        # Previous actions for both agents
        jnp.zeros((config["num_envs"], config["num_prev_actions"])),  # prev_actions_1
        jnp.zeros((config["num_envs"], config["num_prev_actions"])),  # prev_actions_2
    ]
    params = model.init(rng, obs)

    print(f"Model: {sum(x.size for x in jax.tree_leaves(params)):,} parameters")
    return model, params


def load_neural_network(config, env):
    """Load neural network model based on config and environment."""
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env)
    return model


def normalize(x: Array, x_min: Array, x_max: Array) -> Array:
    """Normalizes to [-1, 1]"""
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


class MLP(nn.Module):
    """MLP without activation function at the last layer."""
    hidden_dim_layers: Sequence[int]
    use_layer_norm: bool
    last_layer_init_scaling: float = 1.0

    def setup(self) -> None:
        layer_init = nn.initializers.lecun_normal
        last_layer_init = lambda a, b, c: self.last_layer_init_scaling * layer_init()(a, b, c)
        self.activation = nn.relu

        if self.use_layer_norm:
            self.layers = [
                nn.Sequential([
                    nn.Dense(self.hidden_dim_layers[i], kernel_init=layer_init()),
                    nn.LayerNorm(),
                ])
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
    """Pre-process agent state features (for single agent)."""
    num_embeddings: int
    loaded_max: int
    mlp_use_layernorm: bool
    num_embedding_features: int = 8
    hidden_dim_layers_mlp_one_hot: Sequence[int] = (16, 32)
    hidden_dim_layers_mlp_continuous: Sequence[int] = (16, 32)

    def setup(self) -> None:
        self.embedding = nn.Embed(
            num_embeddings=self.num_embeddings, features=self.num_embedding_features
        )
        self.mlp_one_hot = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp_one_hot,
            use_layer_norm=self.mlp_use_layernorm,
        )
        self.mlp_continuous = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp_continuous,
            use_layer_norm=self.mlp_use_layernorm,
        )

    def __call__(self, obs):
        x_one_hot = obs[..., :-1].astype(dtype=jnp.int32)
        x_loaded = obs[..., [-1]].astype(dtype=jnp.int32)

        x_one_hot = self.embedding(x_one_hot)
        x_one_hot = self.mlp_one_hot(x_one_hot.reshape(*x_one_hot.shape[:-2], -1))

        x_loaded = normalize(x_loaded, 0, self.loaded_max)
        x_continuous = self.mlp_continuous(x_loaded)

        return jnp.concatenate([x_one_hot, x_continuous], axis=-1)


class TwoAgentStateNet(nn.Module):
    """Pre-process both agents' state features."""
    num_embeddings: int
    loaded_max: int
    mlp_use_layernorm: bool
    num_embedding_features: int = 8
    hidden_dim_layers_mlp_one_hot: Sequence[int] = (16, 32)
    hidden_dim_layers_mlp_continuous: Sequence[int] = (16, 32)
    hidden_dim_fusion: Sequence[int] = (64, 64)

    def setup(self) -> None:
        self.embedding = nn.Embed(
            num_embeddings=self.num_embeddings, features=self.num_embedding_features
        )
        self.mlp_one_hot = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp_one_hot,
            use_layer_norm=self.mlp_use_layernorm,
        )
        self.mlp_continuous = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp_continuous,
            use_layer_norm=self.mlp_use_layernorm,
        )
        self.fusion_mlp = MLP(
            hidden_dim_layers=self.hidden_dim_fusion,
            use_layer_norm=self.mlp_use_layernorm,
        )

    def _process_single_agent(self, agent_state):
        """Process a single agent's state."""
        x_one_hot = agent_state[..., :-1].astype(dtype=jnp.int32)
        x_loaded = agent_state[..., [-1]].astype(dtype=jnp.int32)

        x_one_hot = self.embedding(x_one_hot)
        x_one_hot = self.mlp_one_hot(x_one_hot.reshape(*x_one_hot.shape[:-2], -1))

        x_loaded = normalize(x_loaded, 0, self.loaded_max)
        x_continuous = self.mlp_continuous(x_loaded)

        return jnp.concatenate([x_one_hot, x_continuous], axis=-1)

    def __call__(self, obs):
        # Process both agents
        agent1_features = self._process_single_agent(obs[0])  # agent_state_1
        agent2_features = self._process_single_agent(obs[1])  # agent_state_2
        
        # Fuse both agent states
        combined_features = jnp.concatenate([agent1_features, agent2_features], axis=-1)
        fused_features = self.fusion_mlp(combined_features)
        
        return fused_features


class LocalMapNet(nn.Module):
    """Pre-process local maps (for single agent)."""
    map_min_max: Sequence[int]
    mlp_use_layernorm: bool
    hidden_dim_layers_mlp: Sequence[int] = (256, 128)

    def setup(self) -> None:
        self.mlp = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp,
            use_layer_norm=self.mlp_use_layernorm,
        )

    def __call__(self, obs):
        x_action_neg = normalize(obs[0], self.map_min_max[0], self.map_min_max[1])
        x_action_pos = normalize(obs[1], self.map_min_max[0], self.map_min_max[1])
        x_target_neg = normalize(obs[2], self.map_min_max[0], self.map_min_max[1])
        x_target_pos = normalize(obs[3], self.map_min_max[0], self.map_min_max[1])
        x_dumpability = obs[4]
        x_obstacles = obs[5]
        
        x = jnp.concatenate(
            (
                x_action_neg[..., None],
                x_action_pos[..., None],
                x_target_neg[..., None],
                x_target_pos[..., None],
                x_dumpability[..., None],
                x_obstacles[..., None],
            ),
            -1,
        )
        
        return self.mlp(x.reshape(*x.shape[:-2], -1))


class TwoAgentLocalMapNet(nn.Module):
    """Pre-process local maps for both agents."""
    map_min_max: Sequence[int]
    mlp_use_layernorm: bool
    hidden_dim_layers_mlp: Sequence[int] = (256, 128)
    hidden_dim_fusion: Sequence[int] = (128, 128)

    def setup(self) -> None:
        self.mlp = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp,
            use_layer_norm=self.mlp_use_layernorm,
        )
        self.fusion_mlp = MLP(
            hidden_dim_layers=self.hidden_dim_fusion,
            use_layer_norm=self.mlp_use_layernorm,
        )

    def _process_maps(self, obs, start_idx):
        """Process 6 consecutive local maps starting from start_idx."""
        x_action_neg = normalize(obs[start_idx], self.map_min_max[0], self.map_min_max[1])
        x_action_pos = normalize(obs[start_idx + 1], self.map_min_max[0], self.map_min_max[1])
        x_target_neg = normalize(obs[start_idx + 2], self.map_min_max[0], self.map_min_max[1])
        x_target_pos = normalize(obs[start_idx + 3], self.map_min_max[0], self.map_min_max[1])
        x_dumpability = obs[start_idx + 4]
        x_obstacles = obs[start_idx + 5]
        
        x = jnp.concatenate(
            (
                x_action_neg[..., None],
                x_action_pos[..., None],
                x_target_neg[..., None],
                x_target_pos[..., None],
                x_dumpability[..., None],
                x_obstacles[..., None],
            ),
            -1,
        )
        
        return self.mlp(x.reshape(*x.shape[:-2], -1))

    def __call__(self, obs):
        # Process agent 1 local maps (indices 2-7)
        agent1_local = self._process_maps(obs, 2)
        # Process agent 2 local maps (indices 8-13)  
        #agent2_local = self._process_maps(obs, 8)
        
        # Fuse both agents' local map features
        combined_local = agent1_local#jnp.concatenate([agent1_local, agent2_local], axis=-1)
        fused_local = self.fusion_mlp(combined_local)
        
        return fused_local


class PreviousActionsNet(nn.Module):
    """Pre-processes the sequence of previous actions (for single agent)."""
    num_actions: int
    mlp_use_layernorm: bool
    num_embedding_features: int = 8
    hidden_dim_layers_mlp: Sequence[int] = (16, 32)

    def setup(self) -> None:
        self.embedding = nn.Embed(
            num_embeddings=self.num_actions,
            features=self.num_embedding_features
        )
        self.mlp = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp,
            use_layer_norm=self.mlp_use_layernorm,
        )
        self.activation = nn.relu

    def __call__(self, obs):
        x_actions = obs.astype(jnp.int32)
        x_actions = self.embedding(x_actions)
        x_flattened = x_actions.reshape(*x_actions.shape[:-2], -1)
        x_flattened = self.mlp(x_flattened)
        return self.activation(x_flattened)


class TwoAgentPreviousActionsNet(nn.Module):
    """Pre-processes the sequence of previous actions for both agents."""
    num_actions: int
    mlp_use_layernorm: bool
    num_embedding_features: int = 8
    hidden_dim_layers_mlp: Sequence[int] = (16, 32)
    hidden_dim_fusion: Sequence[int] = (64, 32)

    def setup(self) -> None:
        self.embedding = nn.Embed(
            num_embeddings=self.num_actions,
            features=self.num_embedding_features
        )
        self.mlp = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp,
            use_layer_norm=self.mlp_use_layernorm,
        )
        self.fusion_mlp = MLP(
            hidden_dim_layers=self.hidden_dim_fusion,
            use_layer_norm=self.mlp_use_layernorm,
        )
        self.activation = nn.relu

    def _process_agent_actions(self, actions):
        """Process one agent's action history."""
        x_actions = actions.astype(jnp.int32)
        x_actions = self.embedding(x_actions)
        x_flattened = x_actions.reshape(*x_actions.shape[:-2], -1)
        x_flattened = self.mlp(x_flattened)
        return self.activation(x_flattened)

    def __call__(self, obs):
        # Process both agents' action histories
        agent1_actions = self._process_agent_actions(obs[-2])  # prev_actions_1
        agent2_actions = self._process_agent_actions(obs[-1])  # prev_actions_2
        
        # Fuse both agents' action features
        combined_actions = jnp.concatenate([agent1_actions, agent2_actions], axis=-1)
        fused_actions = self.fusion_mlp(combined_actions)
        
        return fused_actions


class CentralizedTwoAgentNet(nn.Module):
    """
    Centralized policy network for two agents.
    
    Input structure:
    [0] agent_state_1
    [1] agent_state_2  
    [2-7] agent 1 local maps (action_neg, action_pos, target_neg, target_pos, dumpability, obstacles)
    [8-13] agent 2 local maps  
    [14] action_map (global)
    [15] target_map (global)
    [16] traversability_mask (global)
    [17] dumpability_mask (global)
    [18] prev_actions_1
    [19] prev_actions_2
    """

    num_prev_actions: int
    num_embeddings_agent: int
    map_min_max: Sequence[int]
    local_map_min_max: Sequence[int]
    loaded_max: int
    action_type: Union[TrackedAction, WheeledAction]
    hidden_dim_pi: Sequence[int] = (256, 128)
    hidden_dim_v: Sequence[int] = (256, 128, 1)
    mlp_use_layernorm: bool = False

    def setup(self) -> None:
        num_actions = self.action_type.get_num_actions()

        # Value and policy heads - output single action for active agent
        self.mlp_v = MLP(
            hidden_dim_layers=self.hidden_dim_v,
            use_layer_norm=self.mlp_use_layernorm,
            last_layer_init_scaling=0.01,
        )
        self.mlp_pi = MLP(
            hidden_dim_layers=self.hidden_dim_pi + (num_actions,),
            use_layer_norm=self.mlp_use_layernorm,
            last_layer_init_scaling=0.01,
        )

        # Feature extractors
        self.agent_state_net = TwoAgentStateNet(
            num_embeddings=self.num_embeddings_agent,
            loaded_max=self.loaded_max,
            mlp_use_layernorm=self.mlp_use_layernorm,
        )

        self.local_map_net = TwoAgentLocalMapNet(
            map_min_max=self.local_map_min_max, 
            mlp_use_layernorm=self.mlp_use_layernorm
        )

        self.maps_net = MapsNet(self.map_min_max)

        self.actions_net = TwoAgentPreviousActionsNet(
            num_actions=num_actions,
            mlp_use_layernorm=self.mlp_use_layernorm,
        )

        self.activation = nn.relu

    def __call__(self, obs: Array) -> Array:
        # Extract features from both agents
        x_agent_states = self.agent_state_net(obs)
        x_local_maps = self.local_map_net(obs)
        x_actions = self.actions_net(obs)
        
        # Extract global map features (indices 14-17)
        global_obs = obs[14:18]  # action_map, target_map, traversability_mask, dumpability_mask
        x_global_maps = self.maps_net(global_obs)

        # Combine all features
        x = jnp.concatenate((x_agent_states, x_local_maps, x_global_maps, x_actions), axis=-1)
        x = self.activation(x)

        # Output value and policy for the active agent
        v = self.mlp_v(x)
        xpi = self.mlp_pi(x)

        return v, xpi


class AtariCNN(nn.Module):
    """From https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/networks.py"""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        return x


class MapsNet(nn.Module):
    """Pre-process global maps."""
    map_min_max: Sequence[int]

    def setup(self) -> None:
        self.cnn = AtariCNN()

    def __call__(self, obs):
        """Process global maps: action_map, target_map, traversability_mask, dumpability_mask"""
        action_map = obs[0]
        target_map = obs[1] 
        traversability_map = obs[2]
        dumpability_mask = obs[3]

        x = jnp.concatenate(
            (
                action_map[..., None],
                traversability_map[..., None],
                target_map[..., None],
                dumpability_mask[..., None],
            ),
            axis=-1,
        )

        x = self.cnn(x)
        return x


# Backwards compatibility - keep these for any existing imports
SimplifiedCoupledCategoricalNet = CentralizedTwoAgentNet


@jax.jit
def _single_agent_obs_to_model_input_old(obs, prev_actions):
    """Legacy function for single agent observation processing."""
    obs_list = [
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
        obs["dumpability_mask"],
        prev_actions,
    ]
    return obs_list


@jax.jit
def _multi_agent_obs_to_model_input_old(obs, prev_actions_1, prev_actions_2):
    """Legacy function for multi-agent observation processing."""
    obs_list = [
        obs["agent_state_1"],
        obs["agent_state_2"],
        obs["local_map_action_neg"],
        obs["local_map_action_pos"],
        obs["local_map_target_neg"],
        obs["local_map_target_pos"],
        obs["local_map_dumpability"],
        obs["local_map_obstacles"],
        obs["local_map_action_neg_2"],
        obs["local_map_action_pos_2"],
        obs["local_map_target_neg_2"],
        obs["local_map_target_pos_2"],
        obs["local_map_dumpability_2"],
        obs["local_map_obstacles_2"],
        obs["action_map"],
        obs["target_map"],
        obs["traversability_mask"],
        obs["dumpability_mask"],
        prev_actions_1,
        prev_actions_2,
    ]
    return obs_list


@jax.jit
def apply_jit(params, apply_fn, obs):
    """JIT compiled model application."""
    return apply_fn(params, obs)
