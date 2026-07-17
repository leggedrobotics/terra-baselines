import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from terra.config import BatchConfig, MapsDimsConfig

from utils.models import (
    MapsNet,
    Spatial8x8MapResNet,
    canonical_map_encoder,
    get_model_ready,
)


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


if __name__ == "__main__":
    unittest.main()
