import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from terra.config import BatchConfig, MapsDimsConfig

from utils.models import (
    MapsNet,
    ResidualMapBlock,
    Spatial8x8MapResNet,
    canonical_map_encoder,
    get_model_ready,
)


class _IdentityEncoder(nn.Module):
    """Returns the assembled map stack unchanged (probe for channel wiring)."""

    @nn.compact
    def __call__(self, x):
        return x


class _ProbeMapsNet(MapsNet):
    """MapsNet variant whose encoder is the identity, exposing assembled channels."""

    def setup(self) -> None:
        self.cnn = _IdentityEncoder()


def _full_model_config(map_encoder, model_size="base", **extra):
    class Config(dict):
        __getattr__ = dict.__getitem__

    return Config(
        clip_action_maps=False,
        loaded_max=6,
        local_map_normalization_bounds=(-1, 1),
        map_encoder=map_encoder,
        maps_net_normalization_bounds=(-1, 1),
        model_core="mlp",
        model_size=model_size,
        num_prev_actions=5,
        **extra,
    )


def _dummy_env():
    return SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
    )


def _param_paths(params):
    return [jax.tree_util.keystr(path) for path, _ in
            jax.tree_util.tree_flatten_with_path(params)[0]]


class ResidualMapEncoderTest(unittest.TestCase):
    def test_output_shape_and_backward_pass(self):
        model = Spatial8x8MapResNet()
        inputs = jax.random.normal(
            jax.random.PRNGKey(1), (2, 64, 64, 7), dtype=jnp.float32
        )
        params = model.init(jax.random.PRNGKey(0), inputs)

        output = model.apply(params, inputs)
        self.assertEqual(output.shape, (2, 128))
        self.assertTrue(bool(jnp.all(jnp.isfinite(output))))

        loss, gradients = jax.jit(
            jax.value_and_grad(
                lambda variables: jnp.mean(model.apply(variables, inputs) ** 2)
            )
        )(params)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(
            all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(gradients))
        )

    def test_spatial_readout_flattens_the_8x8_feature_grid(self):
        model = Spatial8x8MapResNet()
        inputs = jnp.zeros((1, 64, 64, 7), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), inputs)
        dense_kernels = [
            leaf
            for leaf in jax.tree.leaves(params)
            if getattr(leaf, "ndim", 0) == 2
        ]
        self.assertTrue(any(kernel.shape == (8 * 8 * 64, 128) for kernel in dense_kernels))

    def test_scaled_configuration_changes_output_and_params(self):
        model = Spatial8x8MapResNet(
            stage_channels=(32, 64, 96, 128),
            blocks_per_stage=(2, 2, 3, 3),
            dense_layers=(256, 192),
        )
        inputs = jnp.zeros((2, 64, 64, 7), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), inputs)

        output = model.apply(params, inputs)
        self.assertEqual(output.shape, (2, 192))
        self.assertGreater(sum(x.size for x in jax.tree.leaves(params)), 1_000_000)

    def test_maps_net_selects_residual_encoder(self):
        model = MapsNet(map_min_max=(-1, 1), encoder_type="resnet_global_pool")
        maps = [jnp.zeros((2, 64, 64), dtype=jnp.float32) for _ in range(7)]
        params = model.init(jax.random.PRNGKey(0), maps)

        output = model.apply(params, maps)
        self.assertEqual(output.shape, (2, 128))

    def test_spatial_v2_has_a_distinct_checkpoint_identity(self):
        maps = [jnp.zeros((1, 64, 64), dtype=jnp.float32) for _ in range(7)]
        legacy = MapsNet(map_min_max=(-1, 1), encoder_type="resnet_global_pool")
        spatial = MapsNet(map_min_max=(-1, 1), encoder_type="resnet_spatial_8x8")
        legacy_params = legacy.init(jax.random.PRNGKey(0), maps)
        spatial_params = spatial.init(jax.random.PRNGKey(0), maps)
        self.assertNotEqual(
            sum(x.size for x in jax.tree.leaves(legacy_params)),
            sum(x.size for x in jax.tree.leaves(spatial_params)),
        )

    def test_maps_net_keeps_atari_encoder_as_default(self):
        model = MapsNet(map_min_max=(-1, 1))
        maps = [jnp.zeros((2, 64, 64), dtype=jnp.float32) for _ in range(7)]
        params = model.init(jax.random.PRNGKey(0), maps)

        output = model.apply(params, maps)
        self.assertEqual(output.shape, (2, 32))

    def test_height_normalization_is_versioned_with_spatial_v2(self):
        maps = [jnp.zeros((1, 64, 64), dtype=jnp.float32) for _ in range(7)]
        maps[2] = jnp.full((1, 64, 64), 5.0)
        maps[3] = jnp.full((1, 64, 64), -3.0)

        atari_raw = MapsNet(map_min_max=(-10, 10), encoder_type="atari")
        atari_identity = MapsNet(map_min_max=(-1, 1), encoder_type="atari")
        atari_params = atari_raw.init(jax.random.PRNGKey(0), maps)
        np.testing.assert_array_equal(
            atari_raw.apply(atari_params, maps),
            atari_identity.apply(atari_params, maps),
        )

        spatial_scaled = MapsNet(
            map_min_max=(-10, 10), encoder_type="resnet_spatial_8x8"
        )
        spatial_identity = MapsNet(
            map_min_max=(-1, 1), encoder_type="resnet_spatial_8x8"
        )
        spatial_params = spatial_scaled.init(jax.random.PRNGKey(0), maps)
        self.assertGreater(
            float(
                jnp.max(
                    jnp.abs(
                        spatial_scaled.apply(spatial_params, maps)
                        - spatial_identity.apply(spatial_params, maps)
                    )
                )
            ),
            1e-6,
        )

    def test_full_policy_initializes_with_residual_encoder(self):
        class Config(dict):
            __getattr__ = dict.__getitem__

        config = Config(
            clip_action_maps=False,
            loaded_max=6,
            local_map_normalization_bounds=(-1, 1),
            map_encoder="resnet_spatial_8x8",
            maps_net_normalization_bounds=(-1, 1),
            model_core="mlp",
            model_size="base",
            num_prev_actions=5,
        )
        env = SimpleNamespace(
            batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64))
        )

        model, params = get_model_ready(jax.random.PRNGKey(0), config, env)
        self.assertEqual(model.map_encoder, "resnet_spatial_8x8")
        base_param_count = sum(x.size for x in jax.tree.leaves(params))
        self.assertGreater(base_param_count, 0)

        # model_size presets must reach the residual encoder, not only heads.
        config_large = Config(dict(config, model_size="large"))
        model_large, params_large = get_model_ready(
            jax.random.PRNGKey(0), config_large, env
        )
        self.assertEqual(model_large.resnet_stage_channels, (32, 64, 96, 128))
        large_param_count = sum(x.size for x in jax.tree.leaves(params_large))
        self.assertGreater(large_param_count, base_param_count)

    def test_old_encoder_names_are_exact_aliases(self):
        self.assertEqual(
            canonical_map_encoder("resnet_delayed"), "resnet_global_pool"
        )
        self.assertEqual(
            canonical_map_encoder("resnet_spatial_v2"), "resnet_spatial_8x8"
        )

        maps = [jnp.zeros((1, 64, 64), dtype=jnp.float32) for _ in range(7)]
        canonical = MapsNet(
            map_min_max=(-1, 1), encoder_type="resnet_global_pool"
        )
        alias = MapsNet(map_min_max=(-1, 1), encoder_type="resnet_delayed")
        map_params = canonical.init(jax.random.PRNGKey(0), maps)
        np.testing.assert_array_equal(
            canonical.apply(map_params, maps), alias.apply(map_params, maps)
        )


class SpatialV3EncoderTest(unittest.TestCase):
    def _crafted_maps(self):
        # 8x8 action/target maps embedded as full 64x64 obs maps is unnecessary:
        # MapsNet derives H/W from the map itself, so an 8x8 map is a valid,
        # smaller instance where the derived channels are hand-checkable.
        action = jnp.zeros((8, 8), dtype=jnp.float32)
        target = jnp.zeros((8, 8), dtype=jnp.float32)
        # (0,0): must-dig cell -> target<0 and action above target.
        target = target.at[0, 0].set(-1.0)
        action = action.at[0, 0].set(0.0)
        # (1,1): already dug to target -> not remaining.
        target = target.at[1, 1].set(-1.0)
        action = action.at[1, 1].set(-1.0)
        # (2,2): dump-zone cell not yet filled -> dump_deficit.
        target = target.at[2, 2].set(1.0)
        action = action.at[2, 2].set(0.0)
        # (3,3): dump-zone cell already filled -> not a deficit.
        target = target.at[3, 3].set(1.0)
        action = action.at[3, 3].set(1.0)

        maps = [jnp.zeros((1, 8, 8), dtype=jnp.float32) for _ in range(7)]
        maps[2] = action[None]
        maps[3] = target[None]
        return maps

    def test_v3_assembles_eleven_channels_with_correct_derived_values(self):
        maps = self._crafted_maps()
        probe = _ProbeMapsNet(map_min_max=(-1, 1), encoder_type="resnet_spatial_v3")
        params = probe.init(jax.random.PRNGKey(0), maps)
        stack = probe.apply(params, maps)

        # 7 base channels + remaining_dig, dump_deficit, coord_x, coord_y.
        self.assertEqual(stack.shape, (1, 8, 8, 11))

        remaining_dig = stack[0, :, :, 7]
        dump_deficit = stack[0, :, :, 8]
        coord_x = stack[0, :, :, 9]
        coord_y = stack[0, :, :, 10]

        expected_remaining = np.zeros((8, 8), dtype=np.float32)
        expected_remaining[0, 0] = 1.0
        expected_deficit = np.zeros((8, 8), dtype=np.float32)
        expected_deficit[2, 2] = 1.0
        np.testing.assert_array_equal(np.asarray(remaining_dig), expected_remaining)
        np.testing.assert_array_equal(np.asarray(dump_deficit), expected_deficit)

        # Coordinate channels are a [-1, 1] meshgrid: x varies along columns,
        # y along rows.
        np.testing.assert_allclose(
            np.asarray(coord_x[0, :]), np.linspace(-1, 1, 8), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(coord_y[:, 0]), np.linspace(-1, 1, 8), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(coord_x[3, :]), np.linspace(-1, 1, 8), atol=1e-6
        )  # x constant across rows

    def test_v2_assembles_only_seven_channels(self):
        maps = self._crafted_maps()
        probe = _ProbeMapsNet(map_min_max=(-1, 1), encoder_type="resnet_spatial_8x8")
        params = probe.init(jax.random.PRNGKey(0), maps)
        stack = probe.apply(params, maps)
        self.assertEqual(stack.shape, (1, 8, 8, 7))

    def test_v3_full_model_first_conv_reads_eleven_channels(self):
        model, params = get_model_ready(
            jax.random.PRNGKey(0),
            _full_model_config("resnet_spatial_v3"),
            _dummy_env(),
        )
        self.assertEqual(model.map_encoder, "resnet_spatial_v3")
        conv_kernels = [
            leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
            if "cnn" in jax.tree_util.keystr(path)
            and getattr(leaf, "ndim", 0) == 4
        ]
        # The stem conv has an input-channel dim equal to the assembled stack.
        self.assertTrue(any(kernel.shape[2] == 11 for kernel in conv_kernels))


class SqueezeExcitationTest(unittest.TestCase):
    def _se_param_paths(self, params):
        return [
            path
            for path in _param_paths(params)
            if "ResidualMapBlock" in path and "Dense" in path
        ]

    def test_se_params_present_only_in_v3(self):
        maps = [jnp.zeros((1, 64, 64), dtype=jnp.float32) for _ in range(7)]

        v2 = MapsNet(map_min_max=(-1, 1), encoder_type="resnet_spatial_8x8")
        v2_params = v2.init(jax.random.PRNGKey(0), maps)
        self.assertEqual(self._se_param_paths(v2_params), [])

        v3 = MapsNet(map_min_max=(-1, 1), encoder_type="resnet_spatial_v3")
        v3_params = v3.init(jax.random.PRNGKey(0), maps)
        self.assertGreater(len(self._se_param_paths(v3_params)), 0)

    def test_residual_block_se_toggle_controls_dense_params(self):
        inputs = jnp.zeros((2, 8, 8, 16), dtype=jnp.float32)

        plain = ResidualMapBlock(features=16)
        plain_params = plain.init(jax.random.PRNGKey(0), inputs)
        self.assertFalse(any("Dense" in p for p in _param_paths(plain_params)))

        gated = ResidualMapBlock(features=16, use_se=True)
        gated_params = gated.init(jax.random.PRNGKey(0), inputs)
        se_dense = [p for p in _param_paths(gated_params) if "Dense" in p]
        # SE gate: Dense(features//4) -> Dense(features), each has kernel+bias.
        self.assertEqual(len(se_dense), 4)
        output = gated.apply(gated_params, inputs)
        self.assertEqual(output.shape, (2, 8, 8, 16))


class EncoderComputeDtypeTest(unittest.TestCase):
    def test_bf16_output_is_f32_with_finite_forward_and_grads(self):
        maps = [jnp.zeros((2, 64, 64), dtype=jnp.float32) for _ in range(7)]
        maps[2] = jax.random.normal(jax.random.PRNGKey(1), (2, 64, 64))
        maps[3] = jax.random.normal(jax.random.PRNGKey(2), (2, 64, 64))

        model = MapsNet(
            map_min_max=(-1, 1),
            encoder_type="resnet_spatial_v3",
            encoder_compute_dtype=jnp.bfloat16,
        )
        params = model.init(jax.random.PRNGKey(0), maps)

        # Params stay float32 even under bf16 compute.
        self.assertTrue(
            all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
        )

        output = model.apply(params, maps)
        self.assertEqual(output.dtype, jnp.float32)
        self.assertTrue(bool(jnp.all(jnp.isfinite(output))))

        loss, grads = jax.value_and_grad(
            lambda variables: jnp.mean(model.apply(variables, maps).astype(jnp.float32) ** 2)
        )(params)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(
            all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
        )

    def test_bf16_and_f32_share_the_same_param_tree(self):
        maps = [jnp.zeros((1, 64, 64), dtype=jnp.float32) for _ in range(7)]
        f32 = MapsNet(map_min_max=(-1, 1), encoder_type="resnet_spatial_v3")
        bf16 = MapsNet(
            map_min_max=(-1, 1),
            encoder_type="resnet_spatial_v3",
            encoder_compute_dtype=jnp.bfloat16,
        )
        f32_params = f32.init(jax.random.PRNGKey(0), maps)
        bf16_params = bf16.init(jax.random.PRNGKey(0), maps)
        self.assertEqual(
            jax.tree_util.tree_structure(f32_params),
            jax.tree_util.tree_structure(bf16_params),
        )
        self.assertEqual(
            [leaf.shape for leaf in jax.tree.leaves(f32_params)],
            [leaf.shape for leaf in jax.tree.leaves(bf16_params)],
        )

    def test_full_model_bf16_keeps_f32_params(self):
        _, params = get_model_ready(
            jax.random.PRNGKey(0),
            _full_model_config("resnet_spatial_v3", encoder_compute_dtype="bfloat16"),
            _dummy_env(),
        )
        self.assertTrue(
            all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
        )
        # Identical to the float32 v3 param tree (compute dtype is not a param).
        _, f32_params = get_model_ready(
            jax.random.PRNGKey(0),
            _full_model_config("resnet_spatial_v3"),
            _dummy_env(),
        )
        self.assertEqual(
            sum(x.size for x in jax.tree.leaves(params)),
            sum(x.size for x in jax.tree.leaves(f32_params)),
        )

    def test_bf16_rejected_for_atari_and_global_pool(self):
        for encoder in ("atari", "resnet_global_pool"):
            with self.assertRaisesRegex(ValueError, "bfloat16"):
                get_model_ready(
                    jax.random.PRNGKey(0),
                    _full_model_config(encoder, encoder_compute_dtype="bfloat16"),
                    _dummy_env(),
                )

    def test_invalid_dtype_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "encoder_compute_dtype"):
            get_model_ready(
                jax.random.PRNGKey(0),
                _full_model_config("resnet_spatial_v3", encoder_compute_dtype="float16"),
                _dummy_env(),
            )


class CheckpointIdentityTest(unittest.TestCase):
    """Default flags must leave existing encoders byte-identical to main."""

    # Recorded from the pre-change code path (full SimplifiedCoupledCategoricalNet).
    EXPECTED = {
        ("atari", "base"): 480_137,
        ("resnet_global_pool", "base"): 320_169,
        ("resnet_spatial_8x8", "base"): 994_825,
        ("resnet_spatial_8x8", "medium"): 2_051_937,
    }
    EXPECTED_LEAVES = {
        ("resnet_spatial_8x8", "base"): 92,
        ("resnet_spatial_8x8", "medium"): 98,
    }

    def test_existing_encoder_param_counts_are_unchanged(self):
        for (encoder, size), expected in self.EXPECTED.items():
            _, params = get_model_ready(
                jax.random.PRNGKey(0),
                _full_model_config(encoder, size),
                _dummy_env(),
            )
            count = sum(x.size for x in jax.tree.leaves(params))
            self.assertEqual(count, expected, f"{encoder}/{size}")

    def test_spatial_v2_tree_structure_and_leaf_shapes_unchanged(self):
        for (encoder, size), leaves in self.EXPECTED_LEAVES.items():
            _, params = get_model_ready(
                jax.random.PRNGKey(0),
                _full_model_config(encoder, size),
                _dummy_env(),
            )
            all_leaves = jax.tree.leaves(params)
            self.assertEqual(len(all_leaves), leaves, f"{encoder}/{size}")
            # The v2 encoder must not have grown SE Dense params.
            se_paths = [
                p
                for p in _param_paths(params)
                if "ResidualMapBlock" in p and "Dense" in p
            ]
            self.assertEqual(se_paths, [], f"{encoder}/{size}")

    def test_spatial_v2_standalone_output_shape_and_params(self):
        model = Spatial8x8MapResNet()
        inputs = jnp.zeros((2, 64, 64, 7), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), inputs)
        self.assertEqual(model.apply(params, inputs).shape, (2, 128))
        self.assertEqual(sum(x.size for x in jax.tree.leaves(params)), 781_168)


class CriticHiddenDimsOverrideTest(unittest.TestCase):
    def test_override_changes_value_head_and_param_count(self):
        _, base_params = get_model_ready(
            jax.random.PRNGKey(0),
            _full_model_config("resnet_spatial_8x8"),
            _dummy_env(),
        )
        base_count = sum(x.size for x in jax.tree.leaves(base_params))

        model, params = get_model_ready(
            jax.random.PRNGKey(0),
            _full_model_config("resnet_spatial_8x8", critic_hidden_dims=(512, 256)),
            _dummy_env(),
        )
        self.assertEqual(model.hidden_dim_v, (512, 256, 1))
        override_count = sum(x.size for x in jax.tree.leaves(params))
        self.assertGreater(override_count, base_count)

    def test_none_keeps_model_size_preset(self):
        base_model, _ = get_model_ready(
            jax.random.PRNGKey(0),
            _full_model_config("resnet_spatial_8x8"),
            _dummy_env(),
        )
        self.assertEqual(base_model.hidden_dim_v, (128, 32, 1))

        medium_model, _ = get_model_ready(
            jax.random.PRNGKey(0),
            _full_model_config("resnet_spatial_8x8", model_size="medium"),
            _dummy_env(),
        )
        self.assertEqual(medium_model.hidden_dim_v, (160, 48, 1))


if __name__ == "__main__":
    unittest.main()
