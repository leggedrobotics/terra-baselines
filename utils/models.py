import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
from typing import Any, Sequence, Union
from terra.actions import TrackedAction, WheeledAction
from terra.env import TerraEnvBatch
from functools import partial


MAP_ENCODER_ALIASES = {
    "atari": "atari",
    "resnet_global_pool": "resnet_global_pool",
    "resnet_delayed": "resnet_global_pool",
    "resnet_spatial_8x8": "resnet_spatial_8x8",
    "resnet_spatial_v2": "resnet_spatial_8x8",
    # Behavior-based canonical name (AGENTS.md): the spatial 8x8 readout plus
    # squeeze-excitation gating and derived/coordinate channels.
    "resnet_spatial_8x8_se": "resnet_spatial_8x8_se",
    # Version-style alias kept for compatibility; resolves to the canonical name.
    "resnet_spatial_v3": "resnet_spatial_8x8_se",
    # Behavior-based canonical name: the _se trunk plus an agent-conditioned
    # cross-attention readout branch (F13).
    "resnet_spatial_8x8_se_xattn": "resnet_spatial_8x8_se_xattn",
    # Version-style alias kept for compatibility; resolves to the canonical name.
    "resnet_spatial_v4": "resnet_spatial_8x8_se_xattn",
    # Behavior-based canonical name: the _se_xattn encoder plus a token
    # self-attention mixing stage over the 8x8 grid (F14). Identity-initialized
    # so it is function-preserving when grown from a v3/v4 checkpoint.
    "resnet_spatial_8x8_se_sa_xattn": "resnet_spatial_8x8_se_sa_xattn",
    # Version-style alias kept for compatibility; resolves to the canonical name.
    "resnet_spatial_v5": "resnet_spatial_8x8_se_sa_xattn",
}


def canonical_map_encoder(name: str) -> str:
    """Return the stable architecture name used by new checkpoints."""
    try:
        return MAP_ENCODER_ALIASES[name]
    except KeyError as error:
        choices = ", ".join(sorted(MAP_ENCODER_ALIASES))
        raise ValueError(
            f"Unsupported map_encoder={name!r}. Expected one of: {choices}."
        ) from error


def _config_option(config, name: str, default):
    """Read an optional config field from dict-like or dataclass configs.

    Attribute-style access is tried first (dataclasses, and the dict subclasses
    with ``__getattr__ = __getitem__`` used in tests). A plain ``dict`` has no
    such attribute and raises ``AttributeError``, so fall back to subscript
    access. Missing fields and stored ``None`` both resolve to ``default``.
    """
    try:
        value = getattr(config, name)
    except (AttributeError, KeyError):
        try:
            value = config[name]
        except (TypeError, KeyError, IndexError):
            return default
    return value if value is not None else default


def get_model_ready(rng, config, env: TerraEnvBatch, speed=False):
    """Instantiate a model according to obs shape of environment."""
    init_batch_size = 1
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
    print(f"num_embeddings_agent = {num_embeddings_agent}")
    map_min_max = (
        tuple(config["maps_net_normalization_bounds"])
        if not config["clip_action_maps"]
        else (-1, 1)
    )
    print(f"map normalization min max = {map_min_max}")
    model_size = getattr(config, "model_size", "base")
    if model_size not in ("base", "medium", "large"):
        raise ValueError(
            f"Unsupported model_size='{model_size}'. Expected 'base', 'medium', or 'large'."
        )
    model_core = getattr(config, "model_core", "mlp")
    if model_core not in ("mlp", "transformer"):
        raise ValueError(
            f"Unsupported model_core='{model_core}'. Expected 'mlp' or 'transformer'."
        )
    map_encoder = canonical_map_encoder(getattr(config, "map_encoder", "atari"))

    model_kwargs = {}
    if model_size == "medium":
        model_kwargs = {
            "cnn_channels": (24, 48, 48),
            "cnn_dense_layers": (192, 48),
            "resnet_stage_channels": (24, 48, 64, 96),
            "resnet_blocks_per_stage": (1, 2, 2, 2),
            "resnet_dense_layers": (192, 160),
            "resnet_attn_qkv": 96,
            "resnet_attn_out": 160,
            "hidden_dim_pi": (160, 48),
            "hidden_dim_v": (160, 48, 1),
            "intermediate_mlp_dim": 160,
            "local_map_hidden_dim_layers_mlp": (320, 64),
            "transformer_model_dim": 192,
            "transformer_num_layers": 2,
            "transformer_num_heads": 4,
            "transformer_ffn_dim": 384,
        }
    if model_size == "large":
        model_kwargs = {
            "cnn_channels": (32, 64, 64),
            "cnn_dense_layers": (256, 64),
            "resnet_stage_channels": (32, 64, 96, 128),
            "resnet_blocks_per_stage": (2, 2, 3, 3),
            "resnet_dense_layers": (256, 192),
            "resnet_attn_qkv": 128,
            "resnet_attn_out": 192,
            "hidden_dim_pi": (192, 64),
            "hidden_dim_v": (192, 64, 1),
            "intermediate_mlp_dim": 192,
            "local_map_hidden_dim_layers_mlp": (384, 96),
            "transformer_model_dim": 256,
            "transformer_num_layers": 3,
            "transformer_num_heads": 8,
            "transformer_ffn_dim": 512,
        }

    # Optional critic-head width override (F4). When unset the model_size preset
    # (or the module default for "base") is left untouched.
    critic_hidden_dims = _config_option(config, "critic_hidden_dims", None)
    if critic_hidden_dims is not None:
        model_kwargs["hidden_dim_v"] = tuple(int(f) for f in critic_hidden_dims) + (1,)
        print(f"critic_hidden_dims override -> hidden_dim_v = {model_kwargs['hidden_dim_v']}")

    # Optional spatial-ResNet stage overrides (F15). When set they replace the
    # model_size preset's stage layout (or the module default for "base"), e.g.
    # a 5-stage (16,32,48,64,64)/(1,1,2,2,2) config keeps 128x128 inputs at an
    # 8x8 readout. None keeps the preset -> default runs stay bit-identical.
    resnet_stage_channels = _config_option(config, "resnet_stage_channels", None)
    if resnet_stage_channels is not None:
        model_kwargs["resnet_stage_channels"] = tuple(int(f) for f in resnet_stage_channels)
        print(
            f"resnet_stage_channels override -> {model_kwargs['resnet_stage_channels']}"
        )
    resnet_blocks_per_stage = _config_option(config, "resnet_blocks_per_stage", None)
    if resnet_blocks_per_stage is not None:
        model_kwargs["resnet_blocks_per_stage"] = tuple(
            int(f) for f in resnet_blocks_per_stage
        )
        print(
            f"resnet_blocks_per_stage override -> {model_kwargs['resnet_blocks_per_stage']}"
        )

    # Encoder mixed-precision compute dtype (F3). Default float32 keeps the
    # existing param tree and numerics identical.
    encoder_compute_dtype_name = _config_option(config, "encoder_compute_dtype", "float32")
    if encoder_compute_dtype_name not in ("float32", "bfloat16"):
        raise ValueError(
            f"Unsupported encoder_compute_dtype={encoder_compute_dtype_name!r}. "
            "Expected 'float32' or 'bfloat16'."
        )
    if encoder_compute_dtype_name == "bfloat16" and map_encoder in (
        "atari",
        "resnet_global_pool",
    ):
        raise ValueError(
            "encoder_compute_dtype='bfloat16' is only supported for the spatial "
            f"ResNet encoders, not map_encoder={map_encoder!r}."
        )
    encoder_compute_dtype = (
        jnp.bfloat16 if encoder_compute_dtype_name == "bfloat16" else jnp.float32
    )

    model = SimplifiedCoupledCategoricalNet(
        num_prev_actions=config["num_prev_actions"],
        num_embeddings_agent=num_embeddings_agent,
        map_min_max=map_min_max,
        local_map_min_max=tuple(config["local_map_normalization_bounds"]),
        loaded_max=config["loaded_max"],
        agent_types_max=2,  # Maximum agent type value (0=excavator, 1=truck, 2=skidsteer)
        action_type=env.batch_cfg.action_type,
        model_core=model_core,
        map_encoder=map_encoder,
        encoder_compute_dtype=encoder_compute_dtype,
        **model_kwargs,
    )

    map_width = env.batch_cfg.maps_dims.maps_edge_length
    map_height = env.batch_cfg.maps_dims.maps_edge_length

    MAX_AGENTS = 4
    angles_cabin = env.batch_cfg.agent.angles_cabin
    # Build dummy obs matching utils.utils_ppo.obs_to_model_input indexing
    # (new layout includes reachability_mask at index [13]).
    obs = [
        # [0] agent_states
        jnp.zeros((init_batch_size, MAX_AGENTS, env.batch_cfg.agent.num_state_obs)),
        # [1] agent_active (mask)
        jnp.zeros((init_batch_size, MAX_AGENTS), dtype=jnp.int8),
        # [2] num_agents
        jnp.zeros((init_batch_size,), dtype=jnp.int32),
        # [3]-[11] local maps (1D per-angle summaries)
        jnp.zeros((init_batch_size, angles_cabin)),
        jnp.zeros((init_batch_size, angles_cabin)),
        jnp.zeros((init_batch_size, angles_cabin)),
        jnp.zeros((init_batch_size, angles_cabin)),
        jnp.zeros((init_batch_size, angles_cabin)),
        jnp.zeros((init_batch_size, angles_cabin)),
        jnp.zeros((init_batch_size, angles_cabin)),
        jnp.zeros((init_batch_size, angles_cabin)),
        jnp.zeros((init_batch_size, angles_cabin)),
        # [12]-[15] global maps
        jnp.zeros((init_batch_size, map_width, map_height)),
        jnp.zeros((init_batch_size, map_width, map_height)),
        jnp.zeros((init_batch_size, map_width, map_height)),
        jnp.zeros((init_batch_size, map_width, map_height)),
        # [16]-[17] agent dims (scalars per env) - not used by model
        jnp.zeros((init_batch_size,), dtype=jnp.int32),
        jnp.zeros((init_batch_size,), dtype=jnp.int32),
        # [18]-[20] masks
        jnp.zeros((init_batch_size, map_width, map_height)),
        jnp.zeros((init_batch_size, map_width, map_height)),
        jnp.zeros((init_batch_size, map_width, map_height)),
        # [21] prev_actions
        jnp.zeros((init_batch_size, config["num_prev_actions"]), dtype=jnp.int32),
    ]
    print(f"model.init obs_len = {len(obs)}")
    print(f"model.init obs_shapes = {[tuple(x.shape) for x in obs]}")
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
    agent_types_max: int  # Maximum agent type value (0..agent_types_max), e.g., 2 includes skidsteer
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
    Pre-process local 1D maps, concatenate, then MLP.
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
        Processes local 1D maps.
        Expects order:
        action_neg, action_pos, target_neg, target_pos, dumpability, obstacles,
        border_workspace, edge_alignment_error, border_diggable.
        """
        # Normalize first four maps (heights)
        m0 = normalize(local_maps[0], self.map_min_max[0], self.map_min_max[1])
        m1 = normalize(local_maps[1], self.map_min_max[0], self.map_min_max[1])
        m2 = normalize(local_maps[2], self.map_min_max[0], self.map_min_max[1])
        m3 = normalize(local_maps[3], self.map_min_max[0], self.map_min_max[1])
        m4 = local_maps[4]
        m5 = local_maps[5]
        m6 = local_maps[6]
        m7 = local_maps[7]
        m8 = local_maps[8]

        # Ensure batch dimension
        def ensure_batch(x: Array) -> Array:
            if x.ndim == 1:
                return x[None, :]
            if x.ndim > 2:
                return x.reshape((x.shape[0], -1))
            return x

        m0 = ensure_batch(m0)
        m1 = ensure_batch(m1)
        m2 = ensure_batch(m2)
        m3 = ensure_batch(m3)
        m4 = ensure_batch(m4)
        m5 = ensure_batch(m5)
        m6 = ensure_batch(m6)
        m7 = ensure_batch(m7)
        m8 = ensure_batch(m8)

        # Concatenate into (B, large vector)
        x = jnp.concatenate((m0, m1, m2, m3, m4, m5, m6, m7, m8), axis=-1)
        # Single MLP over concatenated vector
        x = self.mlp(x)
        return x


class AtariCNN(nn.Module):
    """From https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/networks.py"""
    conv_channels: Sequence[int] = (16, 32, 32)
    dense_layers: Sequence[int] = (128, 32)

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.conv_channels[0], kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.conv_channels[1], kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.conv_channels[2], kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=self.dense_layers[0])(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.dense_layers[1])(x)
        return x


class ResidualMapBlock(nn.Module):
    """Two map convolutions with a projected skip connection when needed."""

    features: int
    strides: tuple[int, int] = (1, 1)
    use_se: bool = False
    se_reduction: int = 4
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=self.strides,
            padding="SAME",
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )(x)
        x = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )(x)
        x = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(x)

        if self.use_se:
            # Squeeze-and-excitation gate: recalibrate channels using a global
            # descriptor before the residual add (residual is post-relu >= 0).
            se = jnp.mean(x, axis=(1, 2))
            se = nn.Dense(
                features=self.features // self.se_reduction,
                dtype=self.compute_dtype,
                param_dtype=jnp.float32,
            )(se)
            se = nn.relu(se)
            se = nn.Dense(
                features=self.features,
                dtype=self.compute_dtype,
                param_dtype=jnp.float32,
            )(se)
            se = nn.sigmoid(se)
            x = x * se[:, None, None, :]

        if residual.shape[-1] != self.features or self.strides != (1, 1):
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=self.strides,
                padding="SAME",
                use_bias=False,
                dtype=self.compute_dtype,
                param_dtype=jnp.float32,
            )(residual)
            residual = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(
                residual
            )

        return nn.relu(x + residual)


class GlobalPoolMapResNet(nn.Module):
    """PR #15 residual encoder with global mean+max pooling."""

    @nn.compact
    def __call__(self, x):
        channels = (16, 32, 32)
        x = nn.Conv(
            features=channels[0],
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        for stage, features in enumerate(channels):
            for block in range(2):
                strides = (2, 2) if stage > 0 and block == 0 else (1, 1)
                x = ResidualMapBlock(features=features, strides=strides)(x)

        x = jnp.concatenate(
            (jnp.mean(x, axis=(1, 2)), jnp.max(x, axis=(1, 2))), axis=-1
        )
        x = nn.relu(nn.Dense(features=128)(x))
        return nn.Dense(features=128)(x)


class _TokenSelfAttentionBlock(nn.Module):
    """Pre-norm residual self-attention block over the 8x8 map tokens (F14).

    ``x = x + Attn(LN(x))`` then ``x = x + MLP(LN(x))``. Both residual
    contributions are exactly zero at init, so the block is the identity
    function at init and the mixing stage is function-preserving when grown from
    a v3/v4 checkpoint.

    Zero-init detail: flax 0.8.2's ``MultiHeadDotProductAttention`` shares one
    ``kernel_init`` across its q/k/v/out ``DenseGeneral`` layers (there is no
    ``out_kernel_init``), and ``DenseGeneral`` flattens every kernel to 2D
    before calling the initializer, so the output projection cannot be zeroed in
    isolation inside the MHA. We therefore keep the attention fully live and use
    an explicit zero-initialized ``Dense`` as the OUTPUT PROJECTION: at init it
    forces the whole attention residual to exactly zero (identity), while the
    live q/k/v/value path means that once this output kernel is trained or
    perturbed, token mixing propagates. The MLP's second ``Dense`` kernel is
    likewise zero-initialized (biases are zero-init by default). All submodules
    run in the encoder compute dtype with float32 params (bf16 path per F3).
    """

    num_heads: int
    qkv_features: int
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        channels = x.shape[-1]

        h = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            out_features=channels,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )(h)
        # Zero-initialized output projection -> attention residual is 0 at init.
        attn = nn.Dense(
            features=channels,
            kernel_init=nn.initializers.zeros,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )(attn)
        x = x + attn

        h = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(x)
        h = nn.Dense(
            features=2 * channels,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )(h)
        h = nn.gelu(h)
        h = nn.Dense(
            features=channels,
            kernel_init=nn.initializers.zeros,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )(h)
        x = x + h
        return x


class Spatial8x8MapResNet(nn.Module):
    """Residual map encoder with a flattened 8x8 spatial readout.

    When ``use_xattn`` is set (F13) the flatten readout is augmented with an
    agent-conditioned cross-attention branch that reads the final 8x8 feature
    grid as 64 tokens. The flatten path is left byte-identical to the non-xattn
    encoder; the attention branch is concatenated in only before the final dense
    projection so the encoder output dim (``dense_layers[-1]``) is unchanged.

    When ``use_token_mixer`` is set (F14, requires ``use_xattn``) the 8x8xC grid
    is viewed as 64 tokens, the learned positional table is added ONCE here (the
    cross-attention readout then reuses the mixed tokens WITHOUT re-adding
    position), and ``mixer_blocks`` identity-initialized pre-norm self-attention
    blocks let the tokens interact before BOTH readouts consume them. The mixer
    is the identity at init, so v4 param trees/numerics are unchanged when it is
    off and warm-starting from v3/v4 checkpoints is function-preserving.
    """

    stage_channels: Sequence[int] = (16, 32, 48, 64)
    blocks_per_stage: Sequence[int] = (1, 1, 2, 2)
    dense_layers: Sequence[int] = (128, 128)
    use_se: bool = False
    compute_dtype: Any = jnp.float32
    use_xattn: bool = False
    attn_qkv: int = 64
    attn_out: int = 128
    attn_num_heads: int = 4
    use_token_mixer: bool = False
    mixer_blocks: int = 2

    @nn.compact
    def __call__(self, x, agent_embedding=None):
        x = nn.Conv(
            features=self.stage_channels[0],
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )(x)
        x = nn.LayerNorm(dtype=self.compute_dtype, param_dtype=jnp.float32)(x)
        x = nn.relu(x)

        # Keep the first residual stage at full resolution so narrow map
        # details survive; each later stage halves the grid (64 -> 8).
        for stage, (features, num_blocks) in enumerate(
            zip(self.stage_channels, self.blocks_per_stage)
        ):
            for block in range(num_blocks):
                strides = (2, 2) if stage > 0 and block == 0 else (1, 1)
                x = ResidualMapBlock(
                    features=features,
                    strides=strides,
                    use_se=self.use_se,
                    compute_dtype=self.compute_dtype,
                )(x)

        # Keep the final 8x8xC grid around for the optional readouts/mixer.
        feature_grid = x

        # F14 token self-attention mixer. View the grid as 64 tokens, add the
        # learned positional table ONCE here, run identity-initialized pre-norm
        # self-attention blocks so the tokens interact, then reshape back so BOTH
        # readouts consume the mixed grid. The cross-attention readout must not
        # re-add position in this mode (it is already applied here). At init the
        # blocks contribute exactly zero, so the mixed grid equals ``feature_grid``
        # (positional table is zero-init) and the encoder matches the non-mixer
        # (v4) path bit-for-bit.
        if self.use_token_mixer:
            batch = feature_grid.shape[0]
            channels = feature_grid.shape[-1]
            tokens = feature_grid.reshape((batch, -1, channels))
            pos_embed = self.param(
                "attn_pos_embed",
                nn.initializers.zeros,
                (tokens.shape[1], channels),
                jnp.float32,
            )
            tokens = tokens + pos_embed.astype(self.compute_dtype)
            for i in range(self.mixer_blocks):
                tokens = _TokenSelfAttentionBlock(
                    num_heads=self.attn_num_heads,
                    qkv_features=self.attn_qkv,
                    compute_dtype=self.compute_dtype,
                    name=f"token_mixer_{i}",
                )(tokens)
            feature_grid = tokens.reshape(feature_grid.shape)

        # Flatten readout (unchanged from the non-xattn encoder).
        h = feature_grid.reshape((feature_grid.shape[0], -1))
        for features in self.dense_layers[:-1]:
            h = nn.relu(
                nn.Dense(
                    features=features,
                    dtype=self.compute_dtype,
                    param_dtype=jnp.float32,
                )(h)
            )

        if self.use_xattn:
            attn_branch = self._cross_attention_readout(
                feature_grid,
                agent_embedding,
                add_pos_embed=not self.use_token_mixer,
            )
            h = jnp.concatenate((h, attn_branch), axis=-1)

        return nn.Dense(
            features=self.dense_layers[-1],
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )(h)

    def _cross_attention_readout(self, feature_grid, agent_embedding, add_pos_embed=True):
        """Agent-conditioned cross-attention over the 8x8 feature grid (F13).

        Tokens are the 64 grid cells (dim C) plus a learned positional table.
        Five queries attend over them: one projected from the active agent's
        embedding and four learned latent queries. All submodules run in the
        encoder compute dtype with float32 params (bf16 path per F3).

        ``add_pos_embed`` adds the learned positional table here (v4). With the
        F14 token mixer the table is applied once before the mixer instead, so
        the readout is called with ``add_pos_embed=False`` and simply reuses the
        already-positioned, mixed tokens (the ``attn_pos_embed`` param keeps the
        same name/shape either way, so v4 checkpoints stay valid).
        """
        compute_dtype = self.compute_dtype
        batch = feature_grid.shape[0]
        channels = feature_grid.shape[-1]

        # Tokens: [B, 64, C] from the 8x8 grid (+ learned positional embedding
        # unless it was already applied before the token mixer).
        tokens = feature_grid.reshape((batch, -1, channels))
        num_tokens = tokens.shape[1]
        if add_pos_embed:
            pos_embed = self.param(
                "attn_pos_embed",
                nn.initializers.zeros,
                (num_tokens, channels),
                jnp.float32,
            )
            tokens = tokens + pos_embed.astype(compute_dtype)
        tokens = nn.LayerNorm(dtype=compute_dtype, param_dtype=jnp.float32)(tokens)

        # Queries: active-agent projection + 4 learned latent vectors -> [B, 5, Q].
        agent_query = nn.Dense(
            features=self.attn_qkv,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
        )(agent_embedding.astype(compute_dtype))[:, None, :]
        latent_queries = self.param(
            "attn_latent_queries",
            nn.initializers.lecun_normal(),
            (4, self.attn_qkv),
            jnp.float32,
        )
        latent_queries = jnp.broadcast_to(
            latent_queries.astype(compute_dtype)[None],
            (batch, 4, self.attn_qkv),
        )
        queries = jnp.concatenate((agent_query, latent_queries), axis=1)
        queries = nn.LayerNorm(dtype=compute_dtype, param_dtype=jnp.float32)(queries)

        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.attn_num_heads,
            qkv_features=self.attn_qkv,
            out_features=self.attn_qkv,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
        )(queries, tokens)
        attended = attended.reshape((batch, -1))
        return nn.relu(
            nn.Dense(
                features=self.attn_out,
                dtype=compute_dtype,
                param_dtype=jnp.float32,
            )(attended)
        )


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
    cnn_channels: Sequence[int] = (16, 32, 32)
    cnn_dense_layers: Sequence[int] = (128, 32)
    encoder_type: str = "atari"
    resnet_stage_channels: Sequence[int] = (16, 32, 48, 64)
    resnet_blocks_per_stage: Sequence[int] = (1, 1, 2, 2)
    resnet_dense_layers: Sequence[int] = (128, 128)
    resnet_attn_qkv: int = 64
    resnet_attn_out: int = 128
    encoder_compute_dtype: Any = jnp.float32

    def setup(self) -> None:
        encoder_type = canonical_map_encoder(self.encoder_type)
        if encoder_type == "atari":
            self.cnn = AtariCNN(
                conv_channels=self.cnn_channels,
                dense_layers=self.cnn_dense_layers,
            )
            return
        if encoder_type == "resnet_global_pool":
            self.cnn = GlobalPoolMapResNet()
            return
        if encoder_type in (
            "resnet_spatial_8x8",
            "resnet_spatial_8x8_se",
            "resnet_spatial_8x8_se_xattn",
            "resnet_spatial_8x8_se_sa_xattn",
        ):
            use_se = encoder_type in (
                "resnet_spatial_8x8_se",
                "resnet_spatial_8x8_se_xattn",
                "resnet_spatial_8x8_se_sa_xattn",
            )
            # Both the cross-attention (v4) and self-attention+cross-attention
            # (v5) encoders use the cross-attention readout; only v5 adds the
            # token self-attention mixer before it.
            use_xattn = encoder_type in (
                "resnet_spatial_8x8_se_xattn",
                "resnet_spatial_8x8_se_sa_xattn",
            )
            use_token_mixer = encoder_type == "resnet_spatial_8x8_se_sa_xattn"
            self.cnn = Spatial8x8MapResNet(
                stage_channels=self.resnet_stage_channels,
                blocks_per_stage=self.resnet_blocks_per_stage,
                dense_layers=self.resnet_dense_layers,
                use_se=use_se,
                compute_dtype=self.encoder_compute_dtype,
                use_xattn=use_xattn,
                attn_qkv=self.resnet_attn_qkv,
                attn_out=self.resnet_attn_out,
                use_token_mixer=use_token_mixer,
            )
            return
        raise AssertionError(f"Unhandled map encoder: {encoder_type}")

    def __call__(self, obs: dict[str, Array], agent_embedding: Array = None):
        """
        Expects 7 global maps in order:
        traversability_mask, reachability_mask, action_map, target_map,
        padding_mask, dumpability_mask, interaction_mask.

        ``agent_embedding`` is the active agent's AgentStateNet output and is
        consumed only by the cross-attention (xattn) encoder (F13); every other
        encoder path ignores it, keeping their param trees byte-identical.
        """
        def as_map_batch(x: Array) -> Array:
            if x.ndim == 2:
                return x[None, :, :]
            if x.ndim == 3:
                return x
            return x.reshape((-1,) + x.shape[-2:])

        encoder_type = canonical_map_encoder(self.encoder_type)
        # The cross-attention readout (v4 and v5) conditions on the active-agent
        # embedding, so both require it to be passed through.
        is_spatial_xattn = encoder_type in (
            "resnet_spatial_8x8_se_xattn",
            "resnet_spatial_8x8_se_sa_xattn",
        )
        # The _se, _se_xattn and _se_sa_xattn encoders share the 11-channel
        # derived-input assembly; only the readout/mixer differs.
        is_spatial_se = encoder_type in (
            "resnet_spatial_8x8_se",
            "resnet_spatial_8x8_se_xattn",
            "resnet_spatial_8x8_se_sa_xattn",
        )

        traversability_map = as_map_batch(obs[0])
        reachability_map = as_map_batch(obs[1])
        action_map = as_map_batch(obs[2])
        target_map = as_map_batch(obs[3])

        # Derived channels are computed from the action/target maps BEFORE the
        # normalize() that follows (F1). Note these are not raw heights: with
        # clip_action_maps=True (obs_to_model_input) the action map has already
        # been clipped to [-1, 1] upstream, so remaining_dig/dump_deficit are
        # exact only for depth-1 dig targets (target values in {-1, 0, 1}); for
        # deeper targets the clipped action height still yields a usable, if
        # slightly conservative, membership mask.
        if is_spatial_se:
            # Kept boolean here; cast straight to the compute dtype at assembly
            # so no float32 intermediate is materialized (F12).
            remaining_dig = (target_map < 0) & (action_map > target_map)
            dump_deficit = (target_map > 0) & (action_map <= 0)
            grid_h = action_map.shape[-2]
            grid_w = action_map.shape[-1]
            coord_rows = jnp.linspace(-1.0, 1.0, grid_h, dtype=jnp.float32)
            coord_cols = jnp.linspace(-1.0, 1.0, grid_w, dtype=jnp.float32)
            coord_y, coord_x = jnp.meshgrid(coord_rows, coord_cols, indexing="ij")
            coord_x = jnp.broadcast_to(coord_x, action_map.shape)
            coord_y = jnp.broadcast_to(coord_y, action_map.shape)

        # Version preprocessing together with the topology. Atari and the
        # global-pool ResNet consumed raw unclipped height maps historically;
        # both spatial ResNet encoders normalize action/target heights.
        if encoder_type in (
            "resnet_spatial_8x8",
            "resnet_spatial_8x8_se",
            "resnet_spatial_8x8_se_xattn",
            "resnet_spatial_8x8_se_sa_xattn",
        ):
            action_map = normalize(
                action_map, self.map_min_max[0], self.map_min_max[1]
            )
            target_map = normalize(
                target_map, self.map_min_max[0], self.map_min_max[1]
            )
        padding_mask = as_map_batch(obs[4])
        dumpability_mask = as_map_batch(obs[5])
        interaction_mask = as_map_batch(obs[6])

        # Encoder mixed precision (F3/F12): cast each channel to the compute
        # dtype BEFORE concatenating (bool masks / derived channels convert
        # straight to bf16), so the full multi-channel f32 tensor is never
        # materialized. The encoder returns f32 again below.
        compute_dtype = self.encoder_compute_dtype
        channels = [
            traversability_map[..., None].astype(compute_dtype),
            reachability_map[..., None].astype(compute_dtype),
            action_map[..., None].astype(compute_dtype),
            target_map[..., None].astype(compute_dtype),
            padding_mask[..., None].astype(compute_dtype),
            dumpability_mask[..., None].astype(compute_dtype),
            interaction_mask[..., None].astype(compute_dtype),
        ]
        if is_spatial_se:
            channels.extend(
                [
                    remaining_dig[..., None].astype(compute_dtype),
                    dump_deficit[..., None].astype(compute_dtype),
                    coord_x[..., None].astype(compute_dtype),
                    coord_y[..., None].astype(compute_dtype),
                ]
            )

        x = jnp.concatenate(channels, axis=-1)
        if is_spatial_xattn:
            if agent_embedding is None:
                raise ValueError(
                    f"map_encoder={encoder_type!r} requires the active-agent "
                    "embedding, but MapsNet.__call__ was given "
                    "agent_embedding=None."
                )
            x = self.cnn(x, agent_embedding)
        else:
            x = self.cnn(x)
        x = x.astype(jnp.float32)
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
        # Support both layouts:
        # - new layout (len=22): prev_actions at [21]
        # - legacy layout (len=21): prev_actions at [20]
        action_idx = 21 if len(obs) >= 22 else 20
        x_actions = obs[action_idx].astype(jnp.int32)
        x_actions = self.embedding(x_actions)

        x_flattened = x_actions.reshape(*x_actions.shape[:-2], -1)
        x_flattened = self.mlp(x_flattened)

        x = self.activation(x_flattened)
        return x


class TransformerEncoderBlock(nn.Module):
    model_dim: int
    num_heads: int
    ffn_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
        )(h)
        x = x + h

        h2 = nn.LayerNorm()(x)
        h2 = nn.Dense(self.ffn_dim)(h2)
        h2 = nn.gelu(h2)
        h2 = nn.Dense(self.model_dim)(h2)
        x = x + h2
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
    agent_types_max: int = 2  # Maximum agent type value (0=excavator, 1=truck, 2=skidsteer)
    hidden_dim_pi: Sequence[int] = (128, 32)
    hidden_dim_v: Sequence[int] = (128, 32, 1)
    mlp_use_layernorm: bool = False
    intermediate_mlp_dim: int = 128
    cnn_channels: Sequence[int] = (16, 32, 32)
    cnn_dense_layers: Sequence[int] = (128, 32)
    resnet_stage_channels: Sequence[int] = (16, 32, 48, 64)
    resnet_blocks_per_stage: Sequence[int] = (1, 1, 2, 2)
    resnet_dense_layers: Sequence[int] = (128, 128)
    resnet_attn_qkv: int = 64
    resnet_attn_out: int = 128
    local_map_hidden_dim_layers_mlp: Sequence[int] = (256, 32)
    model_core: str = "mlp"  # "mlp" or "transformer"
    map_encoder: str = "atari"
    encoder_compute_dtype: Any = jnp.float32
    transformer_model_dim: int = 128
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_ffn_dim: int = 256

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
            map_min_max=self.local_map_min_max,
            mlp_use_layernorm=self.mlp_use_layernorm,
            hidden_dim_layers_mlp=self.local_map_hidden_dim_layers_mlp,
        )

        self.agent_state_net = AgentStateNet(
            num_embeddings=self.num_embeddings_agent,
            loaded_max=self.loaded_max,
            agent_types_max=self.agent_types_max,
            mlp_use_layernorm=self.mlp_use_layernorm,
        )

        self.maps_net = MapsNet(
            self.map_min_max,
            cnn_channels=self.cnn_channels,
            cnn_dense_layers=self.cnn_dense_layers,
            encoder_type=self.map_encoder,
            resnet_stage_channels=self.resnet_stage_channels,
            resnet_blocks_per_stage=self.resnet_blocks_per_stage,
            resnet_dense_layers=self.resnet_dense_layers,
            resnet_attn_qkv=self.resnet_attn_qkv,
            resnet_attn_out=self.resnet_attn_out,
            encoder_compute_dtype=self.encoder_compute_dtype,
        )

        self.actions_net = PreviousActionsNet(
            num_actions=num_actions,
            mlp_use_layernorm=self.mlp_use_layernorm,
        )

        self.activation = nn.relu
        if self.model_core == "transformer":
            self.token_agent_proj = nn.Dense(self.transformer_model_dim)
            self.token_actions_proj = nn.Dense(self.transformer_model_dim)
            self.token_local_proj = nn.Dense(self.transformer_model_dim)
            self.token_maps_proj = nn.Dense(self.transformer_model_dim)
            self.transformer_blocks = [
                TransformerEncoderBlock(
                    model_dim=self.transformer_model_dim,
                    num_heads=self.transformer_num_heads,
                    ffn_dim=self.transformer_ffn_dim,
                )
                for _ in range(self.transformer_num_layers)
            ]
            self.transformer_out_proj = nn.Dense(self.intermediate_mlp_dim)

    def __call__(self, obs: Array) -> Array:
        # OPTIMIZED: Batched processing for both agents
        # Variable agents: obs[0] is [B, MAX_AGENTS, num_state_obs], obs[2] is num_agents
        agent_states_all = obs[0]
        if agent_states_all.ndim == 2:
            agent_states_all = agent_states_all[None, :, :]
        elif agent_states_all.ndim > 3:
            agent_states_all = agent_states_all.reshape(
                (-1,) + agent_states_all.shape[-2:]
            )
        # Encode all agents individually, mask inactive ones, and concatenate
        B = agent_states_all.shape[0]
        MAX_AGENTS = agent_states_all.shape[1]
        # Handle possible extra dims (e.g., (B,1) or (B,1,1)) for num_agents
        num_agents = jnp.asarray(obs[2], dtype=jnp.int32).reshape((B, -1))[:, 0]
        # Flatten agents into batch
        agent_states_flat = agent_states_all.reshape(B * MAX_AGENTS, -1)
        x_agents_flat = self.agent_state_net(agent_states_flat)
        # Restore agent axis
        x_agents = x_agents_flat.reshape(B, MAX_AGENTS, -1)
        # Build active mask from obs[1] directly (robust to shape quirks)
        active_mask_raw = obs[1]
        active_mask = jnp.asarray(active_mask_raw).astype(jnp.bool_)
        active_mask = active_mask.reshape((B, MAX_AGENTS))
        # Zero out inactive agent embeddings
        x_agents = jnp.where(active_mask[..., None], x_agents, 0)
        # Concatenate all agent embeddings into a single vector
        x_agents_concat = x_agents.reshape(B, -1)
        
        # Process only active agent local maps (ignore other agents' locals)
        # Debug prints removed for cleaner training logs
        
        # Local maps are at indices 3-11
        local_maps_1 = [obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10], obs[11]]
        x_local_active = self.local_map_net(local_maps_1)
        
        # Process global maps. Support both observation layouts:
        # - New layout (len=22): includes reachability at [13]
        # - Legacy layout (len=21): no reachability channel
        has_reachability = len(obs) >= 22
        if has_reachability:
            map_obs = [
                obs[12],  # traversability_mask
                obs[13],  # reachability_mask
                obs[14],  # action_map
                obs[15],  # target_map
                obs[18],  # padding_mask
                obs[19],  # dumpability_mask
                obs[20],  # interaction_mask
            ]
        else:
            map_obs = [
                obs[12],  # traversability_mask
                jnp.zeros_like(obs[12]),  # reachability_mask (absent in legacy layout)
                obs[13],  # action_map
                obs[14],  # target_map
                obs[17],  # padding_mask
                obs[18],  # dumpability_mask
                obs[19],  # interaction_mask
            ]
        # The cross-attention readout (F13/F14) conditions on the active agent's
        # embedding (index 0, obs are acting-agent-first). Pass it only for the
        # encoders that consume it so every other encoder path stays
        # byte-identical (None arg).
        maps_agent_embedding = None
        if canonical_map_encoder(self.map_encoder) in (
            "resnet_spatial_8x8_se_xattn",
            "resnet_spatial_8x8_se_sa_xattn",
        ):
            maps_agent_embedding = x_agents[:, 0, :]
        x_maps = self.maps_net(map_obs, agent_embedding=maps_agent_embedding)
        x_actions = self.actions_net(obs)
        
        # Force feature branches to [B, F] before concatenation.
        def to_2d(x: Array, name: str) -> Array:
            if x.ndim == 1:
                return x[None, :]
            if x.ndim > 2:
                return x.reshape((x.shape[0], -1))
            return x

        x_local_active_flat = to_2d(x_local_active, "x_local_active")
        x_actions = to_2d(x_actions, "x_actions")
        x_maps = to_2d(x_maps, "x_maps")

        if not (x_agents_concat.ndim == x_actions.ndim == x_local_active_flat.ndim == x_maps.ndim == 2):
            raise ValueError(
                f"Feature rank mismatch before concat: "
                f"x_agents_concat={x_agents_concat.shape}, "
                f"x_actions={x_actions.shape}, "
                f"x_local_active={x_local_active_flat.shape}, "
                f"x_maps={x_maps.shape}"
            )
        
        # Concatenate features based on user request: AgentState(1), prv action, AgentState(2), CNN(Maps)
        # Local maps are also included.
        if self.model_core == "transformer":
            # Build lightweight token sequence:
            # - per-agent tokens
            # - one global token each for prev-actions, local map summary, global maps
            agent_tokens = self.token_agent_proj(x_agents)  # [B, MAX_AGENTS, D]
            actions_token = self.token_actions_proj(x_actions)[:, None, :]  # [B,1,D]
            local_token = self.token_local_proj(x_local_active_flat)[:, None, :]  # [B,1,D]
            maps_token = self.token_maps_proj(x_maps)[:, None, :]  # [B,1,D]
            tokens = jnp.concatenate(
                (agent_tokens, actions_token, local_token, maps_token),
                axis=1,
            )
            for block in self.transformer_blocks:
                tokens = block(tokens, train=True)
            pooled = jnp.mean(tokens, axis=1)
            x = self.activation(self.transformer_out_proj(pooled))
        else:
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
        action_idx = 21 if len(obs) >= 22 else 20
        x_actions = obs[action_idx].astype(jnp.int32)  # Agent 2 previous actions
        x_actions = self.embedding(x_actions)

        x_flattened = x_actions.reshape(*x_actions.shape[:-2], -1)
        x_flattened = self.mlp(x_flattened)

        x = self.activation(x_flattened)
        return x
