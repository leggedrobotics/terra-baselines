import os
import tempfile
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig, MapsDimsConfig
from utils.models import Spatial8x8MapResNet, get_model_ready
import utils.helpers as helpers
from scripts.grow_checkpoint import (
    build_target_config,
    grow_params,
    _leaf_map,
    _param_count,
    _update_out_train_config,
    _interp_pos_embed,
    _sequential_block_remap,
)


class _Config(dict):
    __getattr__ = dict.__getitem__


def _full_model_config(map_encoder, model_size="base", **extra):
    return _Config(
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


def _dummy_env(maps_edge_length=64):
    return SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=maps_edge_length))
    )


class GrowFunctionPreservationTest(unittest.TestCase):
    def test_v2_deepened_reproduces_source_output(self):
        # Only depth is added (one extra block appended to the last stage), so
        # zero-init of the added block's second conv makes it an exact identity.
        source = Spatial8x8MapResNet(blocks_per_stage=(1, 1, 2, 2))
        target = Spatial8x8MapResNet(blocks_per_stage=(1, 1, 2, 3))
        inputs = jax.random.normal(jax.random.PRNGKey(5), (3, 64, 64, 7))
        source_params = source.init(jax.random.PRNGKey(0), inputs)
        target_params = target.init(jax.random.PRNGKey(1), inputs)

        grown_params, report = grow_params(source_params, target_params)

        source_out = source.apply(source_params, inputs)
        grown_out = target.apply(grown_params, inputs)
        self.assertLess(
            float(jnp.max(jnp.abs(source_out - grown_out))), 1e-5
        )

        # The added block's second conv kernel is exactly zero.
        grown_leaves = _leaf_map(grown_params)
        added_key = "['params']['ResidualMapBlock_6']['Conv_1']['kernel']"
        self.assertIn(added_key, grown_leaves)
        self.assertTrue(bool(jnp.all(grown_leaves[added_key] == 0)))
        # Exactly one zero-init leaf (the added block's second conv).
        zero_leaves = [e for e in report if e["category"] == "zero-init"]
        self.assertEqual(len(zero_leaves), 1)

        # Existing shape-matched leaves are verbatim copies.
        source_leaves = _leaf_map(source_params)
        copied = [e for e in report if e["category"] == "copied"]
        self.assertGreater(len(copied), 0)
        for entry in copied:
            np.testing.assert_array_equal(
                np.asarray(grown_leaves[entry["key"]]),
                np.asarray(source_leaves[entry["key"]]),
            )

    def test_deepened_added_block_is_identity_on_positive_input(self):
        # Sanity on the exactness of the identity claim.
        source = Spatial8x8MapResNet(blocks_per_stage=(1, 1, 2, 2))
        target = Spatial8x8MapResNet(blocks_per_stage=(1, 1, 2, 3))
        inputs = jnp.abs(jax.random.normal(jax.random.PRNGKey(7), (2, 64, 64, 7)))
        sp = source.init(jax.random.PRNGKey(0), inputs)
        tp = target.init(jax.random.PRNGKey(1), inputs)
        grown, _ = grow_params(sp, tp)
        np.testing.assert_allclose(
            np.asarray(source.apply(sp, inputs)),
            np.asarray(target.apply(grown, inputs)),
            atol=1e-6,
        )

    def test_nonfinal_stage_depth_growth_is_exactly_function_preserving(self):
        # F1: growing a NON-final stage (1,1,2,2)->(1,2,2,2) shifts every later
        # block's flax auto-name. Stage-aware remap + zero-init of the added
        # block's second conv must yield an EXACT identity (max abs diff 0).
        source = Spatial8x8MapResNet(blocks_per_stage=(1, 1, 2, 2))
        target = Spatial8x8MapResNet(blocks_per_stage=(1, 2, 2, 2))
        inputs = jax.random.normal(jax.random.PRNGKey(11), (3, 64, 64, 7))
        source_params = source.init(jax.random.PRNGKey(0), inputs)
        target_params = target.init(jax.random.PRNGKey(1), inputs)

        grown_params, report = grow_params(source_params, target_params)

        source_out = source.apply(source_params, inputs)
        grown_out = target.apply(grown_params, inputs)
        self.assertEqual(float(jnp.max(jnp.abs(source_out - grown_out))), 0.0)

        # Exactly one added block: stage 1, block 1 -> sequential index 2.
        zero_leaves = [e for e in report if e["category"] == "zero-init"]
        self.assertEqual(len(zero_leaves), 1)
        grown_leaves = _leaf_map(grown_params)
        added_key = "['params']['ResidualMapBlock_2']['Conv_1']['kernel']"
        self.assertIn(added_key, grown_leaves)
        self.assertTrue(bool(jnp.all(grown_leaves[added_key] == 0)))

        # The later pre-existing blocks are copied verbatim onto their SHIFTED
        # target indices (source block 2 -> target block 3).
        source_leaves = _leaf_map(source_params)
        np.testing.assert_array_equal(
            np.asarray(
                grown_leaves["['params']['ResidualMapBlock_3']['Conv_0']['kernel']"]
            ),
            np.asarray(
                source_leaves["['params']['ResidualMapBlock_2']['Conv_0']['kernel']"]
            ),
        )

    def test_channel_growth_dense_embed_places_source_rows(self):
        # F1: growing the final-stage channel count reorders the flatten-Dense
        # input rows (C interleaved fastest). The embed must place the source
        # kernel at [:, :, :c_src, :out_src] of the (H, W, C, out) view.
        source = Spatial8x8MapResNet(
            stage_channels=(16, 32, 48, 64),
            blocks_per_stage=(1, 1, 2, 2),
            dense_layers=(128, 128),
        )
        target = Spatial8x8MapResNet(
            stage_channels=(16, 32, 48, 96),
            blocks_per_stage=(1, 1, 2, 2),
            dense_layers=(192, 128),
        )
        inputs = jax.random.normal(jax.random.PRNGKey(13), (2, 64, 64, 7))
        source_params = source.init(jax.random.PRNGKey(0), inputs)
        target_params = target.init(jax.random.PRNGKey(1), inputs)

        grown_params, report = grow_params(source_params, target_params)

        dense_key = "['params']['Dense_0']['kernel']"
        grown_dense = _leaf_map(grown_params)[dense_key]
        src_dense = _leaf_map(source_params)[dense_key]
        tgt_dense = _leaf_map(target_params)[dense_key]

        # Manual reference embed: (H, W, C, out) with C fastest.
        c_src, c_tgt = 64, 96
        out_src, out_tgt = 128, 192
        h = 8  # 64x64 -> 8x8 after three downsampling stages
        self.assertEqual(src_dense.shape, (h * h * c_src, out_src))
        self.assertEqual(tgt_dense.shape, (h * h * c_tgt, out_tgt))
        expected = (0.1 * jnp.asarray(tgt_dense)).reshape(h, h, c_tgt, out_tgt)
        expected = expected.at[:, :, :c_src, :out_src].set(
            jnp.asarray(src_dense).reshape(h, h, c_src, out_src)
        )
        expected = expected.reshape(h * h * c_tgt, out_tgt)
        np.testing.assert_array_equal(np.asarray(grown_dense), np.asarray(expected))

        embed_entries = [
            e
            for e in report
            if e["category"] == "dense-embed" and e["key"] == dense_key
        ]
        self.assertEqual(len(embed_entries), 1)


class GrowFullModelTest(unittest.TestCase):
    def test_base_v2_to_medium_v3_grows_without_errors(self):
        _, source_params = get_model_ready(
            jax.random.PRNGKey(0),
            _full_model_config("resnet_spatial_8x8", "base"),
            _dummy_env(),
        )
        _, target_params = get_model_ready(
            jax.random.PRNGKey(1),
            _full_model_config("resnet_spatial_v3", "medium"),
            _dummy_env(),
        )

        grown_params, report = grow_params(source_params, target_params)

        # Structure and total param count follow the target exactly.
        self.assertEqual(
            jax.tree_util.tree_structure(grown_params),
            jax.tree_util.tree_structure(target_params),
        )
        self.assertEqual(_param_count(grown_params), _param_count(target_params))

        # Exact copies where shapes matched.
        source_leaves = _leaf_map(source_params)
        grown_leaves = _leaf_map(grown_params)
        for entry in report:
            if entry["category"] == "copied":
                np.testing.assert_array_equal(
                    np.asarray(grown_leaves[entry["key"]]),
                    np.asarray(source_leaves[entry["key"]]),
                )

        # Added-depth block: second conv kernel zeroed.
        zero_leaves = [e for e in report if e["category"] == "zero-init"]
        self.assertEqual(len(zero_leaves), 1)
        for entry in zero_leaves:
            self.assertIn("ResidualMapBlock", entry["key"])
            self.assertIn("Conv_1", entry["key"])
            self.assertTrue(bool(jnp.all(grown_leaves[entry["key"]] == 0)))

        # v3 stem conv reads the 11 assembled channels; it is a sliced copy that
        # keeps the source's 7 input channels in the leading slice.
        sliced = [e for e in report if e["category"] == "sliced"]
        self.assertGreater(len(sliced), 0)

    def test_output_checkpoint_is_consumable(self):
        # Build a source checkpoint, grow it, and confirm the output pkl rebuilds
        # into the target architecture (as --resume_from / --teacher_checkpoint do).
        src_config = _full_model_config("resnet_spatial_8x8", "base")
        _, source_params = get_model_ready(
            jax.random.PRNGKey(0), src_config, _dummy_env()
        )
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.pkl")
            out_path = os.path.join(tmp, "grown.pkl")
            helpers.save_pkl_object(
                {"model": source_params, "train_config": dict(src_config), "env_config": None},
                src_path,
            )
            checkpoint = helpers.load_pkl_object(src_path)

            overrides = {
                "map_encoder": "resnet_spatial_v3",
                "model_size": "medium",
                "model_core": None,
                "critic_hidden_dims": (512, 256),
                "encoder_compute_dtype": None,
            }
            target_config = build_target_config(checkpoint["train_config"], overrides)
            _, target_params = get_model_ready(
                jax.random.PRNGKey(1), target_config, _dummy_env()
            )
            grown_params, _ = grow_params(checkpoint["model"], target_params)

            out_checkpoint = dict(checkpoint)
            out_checkpoint["model"] = grown_params
            out_checkpoint["train_config"] = _update_out_train_config(
                checkpoint["train_config"], target_config
            )
            helpers.save_pkl_object(out_checkpoint, out_path)

            # Reload and rebuild the model from the stored train_config.
            reloaded = helpers.load_pkl_object(out_path)
            self.assertEqual(
                reloaded["train_config"]["map_encoder"], "resnet_spatial_8x8_se"
            )
            self.assertEqual(reloaded["train_config"]["model_size"], "medium")
            self.assertEqual(
                tuple(reloaded["train_config"]["critic_hidden_dims"]), (512, 256)
            )
            rebuild_config = build_target_config(reloaded["train_config"], {})
            _, rebuilt_params = get_model_ready(
                jax.random.PRNGKey(2), rebuild_config, _dummy_env()
            )
            self.assertEqual(
                jax.tree_util.tree_structure(reloaded["model"]),
                jax.tree_util.tree_structure(rebuilt_params),
            )


class AddedStageGrowthTest(unittest.TestCase):
    """F15: 4-stage -> 5-stage growth for 128x128 resolution scaling."""

    def test_sequential_block_remap_appends_stage_without_shifting_existing(self):
        # 4 stages (1,1,2,2) -> 5 stages (1,1,2,2,2). The shared leading stages
        # must map one-to-one; only the appended stage's blocks are new.
        remap, added_depth, added_stage = _sequential_block_remap(
            (1, 1, 2, 2), (1, 1, 2, 2, 2)
        )
        self.assertEqual(remap, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
        self.assertEqual(added_depth, set())
        self.assertEqual(added_stage, {6, 7})

    def test_grow_4stage_to_5stage_copies_flatten_readout_heads_exactly(self):
        # Source: 4-stage base encoder at 64. Target: 5-stage encoder at 128 with
        # the SAME final channel count (64) -> the 8x8x64 readout, flatten Dense,
        # and all downstream heads keep their shapes and must copy EXACTLY.
        source_config = _full_model_config("resnet_spatial_8x8", "base")
        target_config = _full_model_config(
            "resnet_spatial_8x8",
            "base",
            resnet_stage_channels=(16, 32, 48, 64, 64),
            resnet_blocks_per_stage=(1, 1, 2, 2, 2),
        )
        _, source_params = get_model_ready(
            jax.random.PRNGKey(0), source_config, _dummy_env(64)
        )
        _, target_params = get_model_ready(
            jax.random.PRNGKey(1), target_config, _dummy_env(128)
        )

        grown_params, report = grow_params(source_params, target_params)

        # Structure follows the (larger) target exactly.
        self.assertEqual(
            jax.tree_util.tree_structure(grown_params),
            jax.tree_util.tree_structure(target_params),
        )

        source_leaves = _leaf_map(source_params)
        grown_leaves = _leaf_map(grown_params)

        # The flatten Dense, the readout Dense, and the policy/value heads must
        # be EXACT copies (max abs diff 0 on those subtrees).
        def _is_readout_or_head(key):
            in_maps = "maps_net" in key
            is_flatten = in_maps and "Dense_0" in key
            is_readout = in_maps and "Dense_1" in key
            is_head = "mlp_pi" in key or "mlp_v" in key
            return is_flatten or is_readout or is_head

        exact_keys = [k for k in grown_leaves if _is_readout_or_head(k)]
        self.assertGreater(len(exact_keys), 0)
        for key in exact_keys:
            self.assertIn(key, source_leaves, key)
            self.assertEqual(
                float(jnp.max(jnp.abs(grown_leaves[key] - source_leaves[key]))),
                0.0,
                key,
            )

        # Every shape-matched leaf is a verbatim copy (no silent perturbation).
        for entry in report:
            if entry["category"] == "copied":
                np.testing.assert_array_equal(
                    np.asarray(grown_leaves[entry["key"]]),
                    np.asarray(source_leaves[entry["key"]]),
                )

        # The appended 5th stage (blocks 6 and 7) is fresh-init "added-stage".
        added_stage = [e for e in report if e["category"] == "added-stage"]
        self.assertGreater(len(added_stage), 0)
        for entry in added_stage:
            self.assertIn("ResidualMapBlock", entry["key"])
            seq = int(entry["key"].split("ResidualMapBlock_")[1].split("'")[0])
            self.assertIn(seq, (6, 7))
        # No spurious zero-init (that is for added DEPTH inside an existing stage).
        self.assertEqual([e for e in report if e["category"] == "zero-init"], [])


class PosEmbedInterpolationTest(unittest.TestCase):
    """F15: positional-table bilinear interpolation across token counts."""

    def test_interp_shapes_and_corner_values(self):
        # 8x8 (64 tokens, C=8) -> 16x16 (256 tokens). Corner tokens are preserved
        # by bilinear resize (align_corners-style at the grid extremes).
        c = 8
        src = jnp.arange(64 * c, dtype=jnp.float32).reshape(64, c)
        tgt = jnp.zeros((256, c), dtype=jnp.float32)
        out = _interp_pos_embed(src, tgt)
        self.assertEqual(out.shape, (256, c))
        src_grid = np.asarray(src).reshape(8, 8, c)
        out_grid = np.asarray(out).reshape(16, 16, c)
        np.testing.assert_allclose(out_grid[0, 0], src_grid[0, 0], atol=1e-4)
        np.testing.assert_allclose(out_grid[-1, -1], src_grid[-1, -1], atol=1e-4)
        # Interpolated interior stays within the source value range.
        self.assertGreaterEqual(float(out_grid.min()), float(src_grid.min()) - 1e-3)
        self.assertLessEqual(float(out_grid.max()), float(src_grid.max()) + 1e-3)

    def test_interp_with_channel_growth_slices_channels(self):
        # Token count 64 -> 256 AND channels 8 -> 12: interpolate spatially, then
        # slice-copy the interpolated table into the leading channels of the
        # zero-init target (extra channels stay at the fresh init).
        src = jnp.arange(64 * 8, dtype=jnp.float32).reshape(64, 8)
        tgt = jnp.zeros((256, 12), dtype=jnp.float32)
        out = _interp_pos_embed(src, tgt)
        self.assertEqual(out.shape, (256, 12))
        self.assertTrue(bool(jnp.all(out[:, 8:] == 0)))

    def test_pos_interp_category_via_grow_params(self):
        # A synthetic param tree with only a mismatched attn_pos_embed leaf goes
        # through the "pos-interp" branch of grow_params.
        source = {"params": {"attn_pos_embed": jnp.ones((64, 8), dtype=jnp.float32)}}
        target = {"params": {"attn_pos_embed": jnp.zeros((256, 8), dtype=jnp.float32)}}
        grown, report = grow_params(source, target)
        self.assertEqual(grown["params"]["attn_pos_embed"].shape, (256, 8))
        self.assertEqual([e["category"] for e in report], ["pos-interp"])


if __name__ == "__main__":
    unittest.main()
