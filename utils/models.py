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
    model = SimplifiedCoupledCategoricalNet(
        num_prev_actions=config["num_prev_actions"],
        num_embeddings_agent=num_embeddings_agent,
        map_min_max=map_min_max,
        local_map_min_max=tuple(config["local_map_normalization_bounds"]),
        loaded_max=config["loaded_max"],
        agent_types_max=3,  # Maximum agent type value (0=tracked, 1=wheeled, 2=skidsteer, 3=truck)
        action_type=env.batch_cfg.action_type,
    )

    map_width = env.batch_cfg.maps_dims.maps_edge_length
    map_height = env.batch_cfg.maps_dims.maps_edge_length

    MAX_AGENTS = 4
    angles_cabin = env.batch_cfg.agent.angles_cabin
    # Build dummy obs matching utils.utils_ppo.obs_to_model_input indexing
    obs = [
        # [0] agent_states
        jnp.zeros((config["num_envs"], MAX_AGENTS, env.batch_cfg.agent.num_state_obs)),
        # [1] agent_active (mask)
        jnp.zeros((config["num_envs"], MAX_AGENTS), dtype=jnp.int8),
        # [2] num_agents
        jnp.zeros((config["num_envs"],), dtype=jnp.int32),
        # [3]-[8] local maps (1D per-angle summaries)
        jnp.zeros((config["num_envs"], angles_cabin)),
        jnp.zeros((config["num_envs"], angles_cabin)),
        jnp.zeros((config["num_envs"], angles_cabin)),
        jnp.zeros((config["num_envs"], angles_cabin)),
        jnp.zeros((config["num_envs"], angles_cabin)),
        jnp.zeros((config["num_envs"], angles_cabin)),
        # [9]-[11] global maps
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        # [12]-[13] agent dims (scalars per env) - not used by model
        jnp.zeros((config["num_envs"],), dtype=jnp.int32),
        jnp.zeros((config["num_envs"],), dtype=jnp.int32),
        # [14]-[16] masks
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        jnp.zeros((config["num_envs"], map_width, map_height)),
        # [17] prev_actions
        jnp.zeros((config["num_envs"], config["num_prev_actions"]), dtype=jnp.int32),
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
    """
    Normalizes to [-1, 1]
    """
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


class MLP(nn.Module):
    """
    MLP without activation function at the last layer.
    """

    hidden_dim_layers: Sequence[int]
    use_layer_norm: bool
    last_layer_init_scaling: float = 1.0

    def setup(self) -> None:
        layer_init = nn.initializers.lecun_normal
        last_layer_init = lambda a, b, c: self.last_layer_init_scaling * layer_init()(
            a, b, c
        )
        self.activation = nn.relu

        if self.use_layer_norm:
            self.layers = [
                nn.Sequential(
                    [
                        nn.Dense(self.hidden_dim_layers[i], kernel_init=layer_init()),
                        nn.LayerNorm(),
                    ]
                )
                for i in range(len(self.hidden_dim_layers) - 1)
            ]
            self.layers += (
                nn.Dense(self.hidden_dim_layers[-1], kernel_init=last_layer_init),
            )
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
    agent_types_max: int  # Maximum agent type value (0..agent_types_max), e.g., 3 includes truck
    mlp_use_layernorm: bool
    num_embedding_features: int = 8
    hidden_dim_layers_mlp_one_hot: Sequence[int] = (16, 32)
    hidden_dim_layers_mlp_continuous: Sequence[int] = (16, 32)
    hidden_dim_layers_mlp_agent_type: Sequence[int] = (8, 16)  # New MLP for agent type

    def setup(self) -> None:
        self.embedding_1 = nn.Embed(
            num_embeddings=self.num_embeddings, features=self.num_embedding_features
        )
        self.embedding_2 = nn.Embed(
            num_embeddings=self.num_embeddings, features=self.num_embedding_features
        )
        # New embedding for agent type
        self.embedding_agent_type = nn.Embed(
            num_embeddings=self.agent_types_max + 1, features=self.num_embedding_features
        )
        
        self.mlp_one_hot = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp_one_hot,
            use_layer_norm=self.mlp_use_layernorm,
        )

        self.mlp_two_hot = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp_one_hot,
            use_layer_norm=self.mlp_use_layernorm,
        )

        self.mlp_continuous = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp_continuous,
            use_layer_norm=self.mlp_use_layernorm,
        )
        
        # New MLP for agent type features
        self.mlp_agent_type = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp_agent_type,
            use_layer_norm=self.mlp_use_layernorm,
        )

    def __call__(self, agent_state_obs: Array):
        # Per-agent feature contains: [pos_x, pos_y, angle_base, angle_cabin, wheel_angle, loaded, agent_type, shovel_lifted]
        x_one_hot = agent_state_obs[..., 0:2].astype(dtype=jnp.int32)  # pos_base (x, y)
        x_two_hot = agent_state_obs[..., 2:5].astype(dtype=jnp.int32)  # angle_base, angle_cabin, wheel_angle
        x_loaded = agent_state_obs[..., [5]].astype(dtype=jnp.int32)   # loaded
        x_agent_type = agent_state_obs[..., [6]].astype(dtype=jnp.int32)  # agent_type
        x_shovel_lifted = agent_state_obs[..., [7]].astype(dtype=jnp.int32)  # shovel_lifted

        # Process embeddings
        x_one_hot = self.embedding_1(x_one_hot)
        x_two_hot = self.embedding_2(x_two_hot)
        x_agent_type_emb = self.embedding_agent_type(x_agent_type)

        # Process through MLPs
        x_one_hot = self.mlp_one_hot(x_one_hot.reshape(*x_one_hot.shape[:-2], -1))
        x_two_hot = self.mlp_two_hot(x_two_hot.reshape(*x_two_hot.shape[:-2], -1))
        
        # Normalize continuous features
        x_loaded = normalize(x_loaded, 0, self.loaded_max)
        x_shovel_lifted = normalize(x_shovel_lifted, 0, 1)  # Binary feature (0 or 1)
        
        # Combine continuous features
        x_continuous = jnp.concatenate([x_loaded, x_shovel_lifted], axis=-1)
        x_continuous = self.mlp_continuous(x_continuous)
        
        # Process agent type embedding
        x_agent_type_processed = self.mlp_agent_type(x_agent_type_emb.reshape(*x_agent_type_emb.shape[:-2], -1))
        
        # Concatenate all processed features
        x_combined = jnp.concatenate(
            (x_one_hot, x_two_hot, x_continuous, x_agent_type_processed), axis=-1
        )
        return x_combined


class LocalMapNet(nn.Module):
    """
    Pre-process six 1D local maps (length-12 each), concatenate, then MLP.
    """

    map_min_max: Sequence[int]
    mlp_use_layernorm: bool
    hidden_dim_layers_mlp: Sequence[int] = (256, 32)

    def setup(self) -> None:
        self.mlp = MLP(
            hidden_dim_layers=self.hidden_dim_layers_mlp,
            use_layer_norm=self.mlp_use_layernorm,
        )

    def __call__(self, local_maps: Sequence[Array]):
        """
        Processes a sequence of six 1D local maps.
        Expects order: action_neg, action_pos, target_neg, target_pos, dumpability, obstacles.
        """
        # Normalize first four maps (heights)
        m0 = normalize(local_maps[0], self.map_min_max[0], self.map_min_max[1])
        m1 = normalize(local_maps[1], self.map_min_max[0], self.map_min_max[1])
        m2 = normalize(local_maps[2], self.map_min_max[0], self.map_min_max[1])
        m3 = normalize(local_maps[3], self.map_min_max[0], self.map_min_max[1])
        m4 = local_maps[4]
        m5 = local_maps[5]

        # Ensure batch dimension
        def ensure_batch(x: Array) -> Array:
            return x[None, :] if x.ndim == 1 else x

        m0 = ensure_batch(m0)
        m1 = ensure_batch(m1)
        m2 = ensure_batch(m2)
        m3 = ensure_batch(m3)
        m4 = ensure_batch(m4)
        m5 = ensure_batch(m5)

        # Concatenate into (B, 72)
        x = jnp.concatenate((m0, m1, m2, m3, m4, m5), axis=-1)
        # Single MLP over concatenated vector
        x = self.mlp(x)
        return x


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


@jax.jit
def min_pool(x):
    pool_fn = partial(
        nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
    )
    return -pool_fn(-x)


@jax.jit
def max_pool(x):
    pool_fn = partial(
        nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
    )
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

    def __call__(self, obs: dict[str, Array]):
        """
        obs["agent_states"],
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
        """
        traversability_map = obs[0]
        action_map = obs[1]
        target_map = obs[2]
        padding_mask = obs[3]
        dumpability_mask = obs[4]
        interaction_mask = obs[5]

        x = jnp.concatenate(
            (
                traversability_map[..., None],
                action_map[..., None],
                target_map[..., None],
                padding_mask[..., None],
                dumpability_mask[..., None],
                interaction_mask[..., None],
            ),
            axis=-1,
        )
        x = self.cnn(x)
        return x


class PreviousActionsNet(nn.Module):
    """
    Pre-processes the sequence of previous actions.
    """
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

    def __call__(self, obs: dict[str, Array]):
        # Use the full single-stream history of previous actions
        x_actions = obs[17].astype(jnp.int32)
        x_actions = self.embedding(x_actions)

        x_flattened = x_actions.reshape(*x_actions.shape[:-2], -1)
        x_flattened = self.mlp(x_flattened)

        x = self.activation(x_flattened)
        return x


class SimplifiedCoupledCategoricalNet(nn.Module):
    """
    The full net for centralized dual-agent policy.
    """

    num_prev_actions: int
    num_embeddings_agent: int
    map_min_max: Sequence[int]
    local_map_min_max: Sequence[int]
    loaded_max: int
    action_type: Union[TrackedAction, WheeledAction]
    agent_types_max: int = 3  # Maximum agent type value (0=tracked, 1=wheeled, 2=skidsteer, 3=truck)
    hidden_dim_pi: Sequence[int] = (128, 32)
    hidden_dim_v: Sequence[int] = (128, 32, 1)
    mlp_use_layernorm: bool = False
    intermediate_mlp_dim: int = 128
    intermediate_mlp_layers: Sequence[int] = (256, 128)

    def setup(self) -> None:
        num_actions = self.action_type.get_num_actions()

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

        self.local_map_net = LocalMapNet(
            map_min_max=self.local_map_min_max, mlp_use_layernorm=self.mlp_use_layernorm
        )

        self.agent_state_net = AgentStateNet(
            num_embeddings=self.num_embeddings_agent,
            loaded_max=self.loaded_max,
            agent_types_max=self.agent_types_max,
            mlp_use_layernorm=self.mlp_use_layernorm,
        )

        self.maps_net = MapsNet(self.map_min_max)

        self.actions_net = PreviousActionsNet(
            num_actions=num_actions,
            mlp_use_layernorm=self.mlp_use_layernorm,
        )

        # New intermediate MLP to process concatenated features
        self.intermediate_mlp = MLP(
            hidden_dim_layers=self.intermediate_mlp_layers + (self.intermediate_mlp_dim,),
            use_layer_norm=self.mlp_use_layernorm,
            last_layer_init_scaling=1.0,
        )

        self.activation = nn.relu

    def __call__(self, obs: Array) -> Array:
        # OPTIMIZED: Batched processing for both agents
        # Variable agents: obs[0] is [B, MAX_AGENTS, num_state_obs], obs[2] is num_agents
        agent_states_all = obs[0]
        # Handle possible extra dims (e.g., (B,1) or (B,1,1)) for num_agents
        num_agents = jnp.squeeze(obs[2]).astype(jnp.int32)
        # Encode all agents individually, mask inactive ones, and concatenate
        B = agent_states_all.shape[0]
        MAX_AGENTS = agent_states_all.shape[1]
        # Flatten agents into batch
        agent_states_flat = agent_states_all.reshape(B * MAX_AGENTS, -1)
        x_agents_flat = self.agent_state_net(agent_states_flat)
        # Restore agent axis
        x_agents = x_agents_flat.reshape(B, MAX_AGENTS, -1)
        # Build active mask from obs[1] directly (robust to shape quirks)
        active_mask_raw = obs[1]
        active_mask = jnp.squeeze(active_mask_raw).astype(jnp.bool_)
        # Ensure shape [B, MAX_AGENTS]
        if active_mask.ndim == 1:
            active_mask = active_mask[None, :].repeat(B, axis=0)
        # Zero out inactive agent embeddings
        x_agents = jnp.where(active_mask[..., None], x_agents, 0)
        # Concatenate all agent embeddings into a single vector
        x_agents_concat = x_agents.reshape(B, -1)
        
        # Process only active agent local maps (ignore other agents' locals)
        # Debug prints removed for cleaner training logs
        
        # Local maps are at indices 3-8: action_neg, action_pos, target_neg, target_pos, dumpability, obstacles
        local_maps_1 = [obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]
        x_local_active = self.local_map_net(local_maps_1)
        
        # Process global maps and actions (unchanged)
        # Global maps are at indices 9, 10, 11, 14, 15, 16: traversability_mask, action_map, target_map, padding_mask, dumpability_mask, interaction_mask
        map_obs = [
            obs[9],   # traversability_mask
            obs[10],  # action_map
            obs[11],  # target_map
            obs[14],  # padding_mask
            obs[15],  # dumpability_mask
            obs[16],  # interaction_mask
        ]
        x_maps = self.maps_net(map_obs)
        x_actions = self.actions_net(obs)
        
        # Flatten local map outputs if they have spatial dimensions
        x_local_active_flat = x_local_active.reshape(x_local_active.shape[0], -1)
        
        # Concatenate features based on user request: AgentState(1), prv action, AgentState(2), CNN(Maps)
        # Local maps are also included.
        # Build a single combined feature vector (all agent states + actions + active local maps)
        combined_features = jnp.concatenate(
            (x_agents_concat, x_actions, x_local_active_flat),
            axis=-1,
        )
        combined_features = self.activation(combined_features)

        # Concatenate combined features with MapNet output
        x = jnp.concatenate((combined_features, x_maps), axis=-1)
        
        # Apply final activation
        x = self.activation(x)

        v = self.mlp_v(x)
        xpi = self.mlp_pi(x)

        return v, xpi


class PreviousActionsNet2(nn.Module):
    """
    Pre-processes the sequence of previous actions for agent 2.
    """
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

    def __call__(self, obs: dict[str, Array]):
        x_actions = obs[19].astype(jnp.int32)  # Agent 2 previous actions
        x_actions = self.embedding(x_actions)

        x_flattened = x_actions.reshape(*x_actions.shape[:-2], -1)
        x_flattened = self.mlp(x_flattened)

        x = self.activation(x_flattened)
        return x
