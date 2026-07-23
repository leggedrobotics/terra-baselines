#!/usr/bin/env python3
"""Function-preserving growth of a Terra policy checkpoint (spec F8).

Grow a trained checkpoint into a larger architecture (wider channels, deeper
ResNet stages, extra derived channels, SE gates, wider critic head) while
preserving the source function as closely as possible so the grown network can
be warm-started via ``--resume_from`` or used as a ``--teacher_checkpoint``.

Growth rules (per-leaf, matched by parameter path):
  * identical shape                 -> copy the source leaf verbatim;
  * larger target (channel growth)  -> copy the source into the leading slices
                                       and 0.1-scale the fresh init elsewhere;
  * new ResidualMapBlock (added
    depth) second conv kernel        -> zero-init, so the block computes
                                       ``relu(0 + residual) = residual`` exactly
                                       (residual is post-relu >= 0), i.e. the
                                       added depth is function-preserving;
  * everything else that is new
    (SE params, added-block first
    conv / LayerNorms, ...)          -> fresh init.

Usage:
    python scripts/grow_checkpoint.py --src ckpt.pkl \
        --map_encoder resnet_spatial_v3 --model_size medium \
        [--critic_hidden_dims 512,256] [--encoder_compute_dtype bfloat16] \
        [--attention_compute_dtype float32] \
        [--token_mixer_residual_init_scale 0.001] \
        [--maps_edge_length 64] --out grown.pkl
"""

import argparse
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import utils.helpers as helpers
from terra.actions import TrackedAction, WheeledAction
from terra.config import BatchConfig, MapsDimsConfig
from utils.models import _config_option, canonical_map_encoder, get_model_ready


# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------

class _DictConfig(dict):
    """Dict that also supports attribute access (get_model_ready reads both)."""

    __getattr__ = dict.__getitem__


# Spatial ResNet stage presets, mirroring get_model_ready's model_size kwargs.
# The base preset matches Spatial8x8MapResNet's class defaults.
_RESNET_STAGE_PRESETS = {
    "base": ((1, 1, 2, 2), (16, 32, 48, 64)),
    "medium": ((1, 2, 2, 2), (24, 48, 64, 96)),
    "large": ((2, 2, 3, 3), (32, 64, 96, 128)),
}

_SPATIAL_ENCODERS = ("resnet_spatial_8x8", "resnet_spatial_8x8_se")


def _resnet_stage_spec(model_size):
    """Return (blocks_per_stage, stage_channels) for a model_size preset."""
    return _RESNET_STAGE_PRESETS.get(model_size, _RESNET_STAGE_PRESETS["base"])


def build_target_config(source_train_config, overrides):
    """Build a get_model_ready-compatible config for the grown architecture.

    ``overrides`` may set map_encoder / model_size / model_core /
    critic_hidden_dims / encoder_compute_dtype / attention_compute_dtype /
    token_mixer_residual_init_scale / resnet_stage_channels /
    resnet_blocks_per_stage; anything left ``None`` inherits the source
    checkpoint value.
    """
    target = _DictConfig(
        clip_action_maps=_config_option(source_train_config, "clip_action_maps", True),
        maps_net_normalization_bounds=_config_option(
            source_train_config, "maps_net_normalization_bounds", (-10, 10)
        ),
        local_map_normalization_bounds=_config_option(
            source_train_config, "local_map_normalization_bounds", (-16, 16)
        ),
        loaded_max=_config_option(source_train_config, "loaded_max", 100),
        num_prev_actions=_config_option(source_train_config, "num_prev_actions", 10),
        model_core=_config_option(source_train_config, "model_core", "mlp"),
        model_size=_config_option(source_train_config, "model_size", "base"),
        map_encoder=_config_option(source_train_config, "map_encoder", "atari"),
        encoder_compute_dtype=_config_option(
            source_train_config, "encoder_compute_dtype", "float32"
        ),
        attention_compute_dtype=_config_option(
            source_train_config, "attention_compute_dtype", "encoder"
        ),
        token_mixer_residual_init_scale=_config_option(
            source_train_config, "token_mixer_residual_init_scale", 0.0
        ),
        critic_hidden_dims=_config_option(source_train_config, "critic_hidden_dims", None),
        # F15: spatial-ResNet stage overrides. None inherits the source (which
        # itself falls back to the model_size preset inside get_model_ready).
        resnet_stage_channels=_config_option(
            source_train_config, "resnet_stage_channels", None
        ),
        resnet_blocks_per_stage=_config_option(
            source_train_config, "resnet_blocks_per_stage", None
        ),
    )
    for key, value in overrides.items():
        if value is not None:
            target[key] = value
    target["map_encoder"] = canonical_map_encoder(target["map_encoder"])
    return target


def _normalize_action_types(value):
    """Coerce a recorded action_types field into a flat list of ints."""
    if value is None:
        return []
    try:
        if isinstance(value, (tuple, list)):
            return [int(v) for v in value]
        if hasattr(value, "ndim"):
            arr = np.asarray(value)
            if arr.ndim == 0:
                return [int(arr.item())]
            if arr.ndim >= 2:
                arr = arr[0]
            return [int(v) for v in arr.tolist()]
        if isinstance(value, int):
            return [int(value)]
    except Exception:
        pass
    return []


def _derive_action_type(checkpoint):
    """Pick WheeledAction/TrackedAction from the source checkpoint (F5).

    Mirrors eval_mcts.py: read the recorded action_types (from env_config, or the
    train_config's action_types_override) and use WheeledAction only when every
    agent is wheeled. Defaults to TrackedAction when nothing is recorded. Using
    the default BatchConfig would pin TrackedAction and mis-size the policy head
    for wheeled checkpoints.
    """
    action_types = None
    env_config = checkpoint.get("env_config")
    if env_config is not None:
        action_types = getattr(env_config, "action_types", None)
    if action_types is None:
        action_types = _config_option(
            checkpoint.get("train_config"), "action_types_override", None
        )
    types_list = _normalize_action_types(action_types)
    all_wheeled = len(types_list) > 0 and all(t == 1 for t in types_list)
    return WheeledAction if all_wheeled else TrackedAction


def _dummy_env(maps_edge_length, action_type):
    return SimpleNamespace(
        batch_cfg=BatchConfig(
            action_type=action_type,
            maps_dims=MapsDimsConfig(maps_edge_length=maps_edge_length),
        )
    )


# ----------------------------------------------------------------------------
# Growth core
# ----------------------------------------------------------------------------

def _leaf_map(params):
    """Return {keystr: leaf} for every parameter leaf."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {jax.tree_util.keystr(path): leaf for path, leaf in leaves}


_BLOCK_TOKEN = re.compile(r"ResidualMapBlock_(\d+)")


def _is_added_block_second_conv(key: str) -> bool:
    """True for the SECOND conv kernel (``Conv_1``) of a ResidualMapBlock.

    Zeroing this kernel makes an added stride-1 same-channel block an identity
    (``relu(0 + residual) = residual``), preserving the source function.
    """
    return (
        "ResidualMapBlock" in key
        and "Conv_1" in key
        and key.rstrip("]'").endswith("kernel")
    )


def _is_flatten_readout_dense(key: str) -> bool:
    """True for the spatial encoder's post-flatten ``Dense_0`` kernel.

    Spatial8x8MapResNet flattens (H, W, C) then applies ``Dense_0``. That is the
    only ``Dense_0`` kernel outside a ResidualMapBlock in the (mlp-core) spatial
    model, and its input rows interleave channels fastest, so it needs the
    reshape-aware embed rather than a plain leading-row slice.
    """
    return (
        "Dense_0" in key
        and "ResidualMapBlock" not in key
        and key.rstrip("]'").endswith("kernel")
    )


def _block_seq(key: str):
    """Sequential ResidualMapBlock index in ``key``, or None for non-block leaves."""
    match = _BLOCK_TOKEN.search(key)
    return int(match.group(1)) if match is not None else None


def _is_attn_pos_embed(key: str) -> bool:
    """True for the cross-attention / token-mixer learned positional table.

    Named ``attn_pos_embed`` (shape ``(num_tokens, C)``) in the v4/v5 encoders.
    When the token count differs between source and target it is bilinearly
    interpolated on its 2D grid rather than sliced.
    """
    return key.rstrip("]'").endswith("attn_pos_embed")


def _interp_pos_embed(source_leaf, target_leaf):
    """Bilinearly interpolate a positional table across differing token counts.

    Reshape the source ``(T_s, C_s)`` table to its square grid
    ``(g_s, g_s, C_s)``, ``jax.image.resize`` (bilinear) to the target grid
    ``(g_t, g_t, C_s)``, and flatten back to ``(T_t, C_s)``. When the channel
    count also grows, place the interpolated table in the leading channel slice
    of the 0.1-scaled fresh target (combining the slice rule), matching how
    channel growth is handled elsewhere. The tables are square by construction
    (8x8 -> 64 tokens); non-square token counts are rejected.
    """
    tokens_src, c_src = source_leaf.shape
    tokens_tgt, c_tgt = target_leaf.shape
    g_src = int(round(tokens_src ** 0.5))
    g_tgt = int(round(tokens_tgt ** 0.5))
    if g_src * g_src != tokens_src:
        raise ValueError(f"attn_pos_embed source token count {tokens_src} is not square")
    if g_tgt * g_tgt != tokens_tgt:
        raise ValueError(f"attn_pos_embed target token count {tokens_tgt} is not square")
    src_grid = jnp.asarray(source_leaf, dtype=target_leaf.dtype).reshape(
        g_src, g_src, c_src
    )
    resized = jax.image.resize(src_grid, (g_tgt, g_tgt, c_src), method="bilinear")
    resized = resized.reshape(tokens_tgt, c_src)
    if c_src == c_tgt:
        return resized
    grown = 0.1 * target_leaf
    return grown.at[:, :c_src].set(resized)


def _sequential_block_remap(source_blocks_per_stage, target_blocks_per_stage):
    """Map source ResidualMapBlock sequential indices onto the target's.

    Flax auto-names blocks sequentially across ALL stages in
    ``Spatial8x8MapResNet.__call__``, so the sequential index of (stage, block)
    is ``sum(blocks_per_stage[:stage]) + block``. Growing depth in a non-final
    stage shifts every later index; adding a whole stage (target has MORE stages
    than the source, e.g. the 4->5 stage 128x128 grow) appends entirely new
    stages after the existing ones. Handling unequal stage counts is what keeps
    the shared leading stages mapped one-to-one (``src_count`` is 0 for the
    appended stages, so their target blocks are all new and never collide with a
    remapped source block).

    Returns ``(remap, added_depth, added_stage)`` where ``remap[src_seq] =
    tgt_seq`` for blocks present in both, ``added_depth`` is the set of new target
    sequential indices in a stage that ALSO exists in the source (added depth ->
    stride-1 same-channel blocks that can be zero-init identities), and
    ``added_stage`` is the set of new target sequential indices in a stage BEYOND
    the source's stage count (its first block is stride-2, so it cannot be an
    identity -> the whole stage is fresh-init).
    """
    remap = {}
    added_depth = set()
    added_stage = set()
    num_source_stages = len(source_blocks_per_stage)
    for stage in range(len(target_blocks_per_stage)):
        src_count = (
            source_blocks_per_stage[stage] if stage < num_source_stages else 0
        )
        src_base = sum(source_blocks_per_stage[:stage])
        tgt_base = sum(target_blocks_per_stage[:stage])
        for block in range(target_blocks_per_stage[stage]):
            tgt_seq = tgt_base + block
            if block < src_count:
                remap[src_base + block] = tgt_seq
            elif stage >= num_source_stages:
                added_stage.add(tgt_seq)
            else:
                added_depth.add(tgt_seq)
    return remap, added_depth, added_stage


def _remap_source_leaves(source_leaves, block_remap):
    """Relabel ResidualMapBlock indices in source leaf keys per ``block_remap``.

    Non-block keys pass through unchanged. Block leaves whose source index is not
    in the remap (target is shallower at that stage) are dropped so they cannot
    collide with a remapped target key.
    """
    remapped = {}
    for key, leaf in source_leaves.items():
        match = _BLOCK_TOKEN.search(key)
        if match is None:
            remapped[key] = leaf
            continue
        src_seq = int(match.group(1))
        if src_seq not in block_remap:
            continue
        new_key = _BLOCK_TOKEN.sub(
            f"ResidualMapBlock_{block_remap[src_seq]}", key, count=1
        )
        remapped[new_key] = leaf
    return remapped


def _derive_stage_spec_from_params(params):
    """Infer (blocks_per_stage, stage_channels) from a Spatial8x8MapResNet tree.

    Blocks are grouped into stages by the presence of a residual-projection conv
    (``Conv_2``), which is only added at the first block of each downsampling
    stage. Returns ``(None, None)`` when no residual blocks are present (e.g.
    the atari encoder), which disables both remapping and the flatten embed.
    """
    leaves = _leaf_map(params)
    block_features = {}
    block_has_proj = {}
    for key, leaf in leaves.items():
        match = _BLOCK_TOKEN.search(key)
        if match is None:
            continue
        idx = int(match.group(1))
        if "Conv_0" in key and key.rstrip("]'").endswith("kernel"):
            block_features[idx] = int(leaf.shape[-1])
        if "Conv_2" in key:
            block_has_proj[idx] = True
    if not block_features:
        return None, None
    blocks_per_stage = []
    stage_channels = []
    for pos, idx in enumerate(sorted(block_features)):
        if pos == 0 or block_has_proj.get(idx, False):
            blocks_per_stage.append(1)
            stage_channels.append(block_features[idx])
        else:
            blocks_per_stage[-1] += 1
    return tuple(blocks_per_stage), tuple(stage_channels)


def _slice_copy(source_leaf, target_leaf):
    """Copy the source into the leading slices; 0.1-scale fresh init elsewhere."""
    grown = 0.1 * target_leaf
    slices = tuple(slice(0, s) for s in source_leaf.shape)
    return grown.at[slices].set(source_leaf)


def _embed_flatten_dense(source_kernel, target_kernel, c_src, c_tgt):
    """Embed the source flatten-Dense kernel into a channel-grown target kernel.

    The flatten of Spatial8x8MapResNet interleaves (H, W, C) with C fastest, so
    growing C reorders the input rows. Reshape both kernels to (H, W, C, out),
    place the source block at ``[:, :, :c_src, :out_src]`` over the 0.1-scaled
    fresh init, then flatten back. Refuses to grow when the implied H (== W)
    differs, i.e. the source and target maps_edge_length differ.
    """
    rows_src, out_src = source_kernel.shape
    rows_tgt, out_tgt = target_kernel.shape
    h_src = int(round((rows_src / c_src) ** 0.5))
    h_tgt = int(round((rows_tgt / c_tgt) ** 0.5))
    if h_src * h_src * c_src != rows_src:
        raise ValueError(
            f"flatten Dense source rows {rows_src} are not H*W*C for c_src={c_src}"
        )
    if h_tgt * h_tgt * c_tgt != rows_tgt:
        raise ValueError(
            f"flatten Dense target rows {rows_tgt} are not H*W*C for c_tgt={c_tgt}"
        )
    if h_src != h_tgt:
        raise ValueError(
            "Refusing to grow: source/target maps_edge_length differ "
            f"(flatten grid {h_src}x{h_src} vs {h_tgt}x{h_tgt}). Rebuild the "
            "target at the source checkpoint's map edge length."
        )
    grown_4d = (0.1 * target_kernel).reshape(h_tgt, h_tgt, c_tgt, out_tgt)
    src_4d = jnp.asarray(source_kernel, dtype=target_kernel.dtype).reshape(
        h_src, h_src, c_src, out_src
    )
    grown_4d = grown_4d.at[:, :, :c_src, :out_src].set(src_4d)
    return grown_4d.reshape(rows_tgt, out_tgt)


def grow_params(
    source_params,
    target_params,
    source_stage_spec=None,
    target_stage_spec=None,
):
    """Grow ``source_params`` into the shape of ``target_params``.

    ``target_params`` supplies the fresh init and the output structure.
    ``*_stage_spec`` are ``(blocks_per_stage, stage_channels)`` tuples for the
    spatial ResNet encoder; when omitted they are inferred from the param trees.
    They drive (F1) the stage-aware ResidualMapBlock relabeling and the
    channel-interleave-aware flatten-Dense embed. Returns ``(grown_params,
    report)`` where report is a list of per-leaf dicts.
    """
    if source_stage_spec is None:
        source_stage_spec = _derive_stage_spec_from_params(source_params)
    if target_stage_spec is None:
        target_stage_spec = _derive_stage_spec_from_params(target_params)
    source_blocks_per_stage, source_stage_channels = source_stage_spec
    target_blocks_per_stage, target_stage_channels = target_stage_spec

    source_leaves = _leaf_map(source_params)

    # F1a: stage-aware block relabeling, applied BEFORE any shape matching so a
    # non-final-stage depth increase does not slice mismatched blocks together.
    # F15: added_stage holds the sequential indices of blocks in target stages
    # BEYOND the source's stage count (the appended 5th stage of the 128x128
    # grow). Those blocks are fresh-init (their stride-2 entry is not identity).
    added_stage_blocks = set()
    if source_blocks_per_stage is not None and target_blocks_per_stage is not None:
        block_remap, _added_depth, added_stage_blocks = _sequential_block_remap(
            source_blocks_per_stage, target_blocks_per_stage
        )
        source_leaves = _remap_source_leaves(source_leaves, block_remap)

    # F1b: the last-stage channel counts drive the flatten-Dense embed. Only
    # meaningful for the spatial encoder (None otherwise).
    c_src = source_stage_channels[-1] if source_stage_channels else None
    c_tgt = target_stage_channels[-1] if target_stage_channels else None

    report = []

    def _grow_leaf(path, target_leaf):
        key = jax.tree_util.keystr(path)
        source_leaf = source_leaves.get(key)
        if source_leaf is not None:
            if source_leaf.shape == target_leaf.shape:
                category = "copied"
                grown = jnp.asarray(source_leaf, dtype=target_leaf.dtype)
            elif _is_attn_pos_embed(key) and source_leaf.shape[0] != target_leaf.shape[0]:
                # Token count changed (e.g. 64 -> 256): bilinearly interpolate
                # the positional table on its 2D grid (F15), slice-copying any
                # channel growth.
                category = "pos-interp"
                grown = _interp_pos_embed(source_leaf, target_leaf)
            elif (
                c_src is not None
                and c_tgt is not None
                and c_src != c_tgt
                and _is_flatten_readout_dense(key)
            ):
                # Channels grew: reorder the interleaved input rows (F1b).
                category = "dense-embed"
                grown = _embed_flatten_dense(source_leaf, target_leaf, c_src, c_tgt)
            elif source_leaf.ndim == target_leaf.ndim and all(
                t >= s for s, t in zip(source_leaf.shape, target_leaf.shape)
            ):
                category = "sliced"
                grown = _slice_copy(
                    jnp.asarray(source_leaf, dtype=target_leaf.dtype), target_leaf
                )
            else:
                # Incompatible ranks/shrinking dims: fall back to fresh init.
                category = "fresh"
                grown = target_leaf
        elif _block_seq(key) in added_stage_blocks:
            # F15: an entire appended stage. Its first block downsamples (stride
            # 2), so no zero-init makes it an identity -> fresh-init the stage.
            category = "added-stage"
            grown = target_leaf
        elif _is_added_block_second_conv(key):
            category = "zero-init"
            grown = jnp.zeros_like(target_leaf)
        else:
            category = "fresh"
            grown = target_leaf
        report.append(
            {
                "key": key,
                "category": category,
                "source_shape": None if source_leaf is None else tuple(source_leaf.shape),
                "target_shape": tuple(target_leaf.shape),
            }
        )
        return grown

    grown_params = jax.tree_util.tree_map_with_path(_grow_leaf, target_params)
    return grown_params, report


def _param_count(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def print_report(
    report,
    source_params,
    target_params,
    grown_params,
    source_stage_spec=None,
    target_stage_spec=None,
):
    """Print the per-leaf growth report and parameter counts."""
    print("\nPer-leaf growth report:")
    counts = {
        "copied": 0,
        "sliced": 0,
        "dense-embed": 0,
        "pos-interp": 0,
        "zero-init": 0,
        "added-stage": 0,
        "fresh": 0,
    }
    for entry in report:
        counts[entry["category"]] += 1
        print(
            f"  {entry['category']:9s} {entry['key']}  "
            f"src={entry['source_shape']} tgt={entry['target_shape']}"
        )
    print("\nCategory totals:")
    for name, count in counts.items():
        print(f"  {name:9s}: {count}")
    # Report genuinely unused source params against the post-remap key space so
    # relabeled ResidualMapBlocks are not spuriously flagged.
    effective_source = _leaf_map(source_params)
    if source_stage_spec is not None and target_stage_spec is not None:
        source_blocks_per_stage = source_stage_spec[0]
        target_blocks_per_stage = target_stage_spec[0]
        if source_blocks_per_stage is not None and target_blocks_per_stage is not None:
            block_remap, _added_depth, _added_stage = _sequential_block_remap(
                source_blocks_per_stage, target_blocks_per_stage
            )
            effective_source = _remap_source_leaves(effective_source, block_remap)
    unused = set(effective_source) - {e["key"] for e in report}
    if unused:
        print(f"\nSource leaves not used ({len(unused)}):")
        for key in sorted(unused):
            print(f"  unused    {key}")
    print("\nParameter counts:")
    print(f"  source: {_param_count(source_params):,}")
    print(f"  target: {_param_count(target_params):,}")
    print(f"  grown : {_param_count(grown_params):,}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_critic_hidden_dims(value):
    if value is None:
        return None
    dims = tuple(int(tok.strip()) for tok in value.split(",") if tok.strip())
    if not dims:
        raise ValueError(f"Failed to parse --critic_hidden_dims '{value}'.")
    return dims


def _update_out_train_config(source_train_config, target_config):
    """Return the source train_config with the grown-architecture overrides."""
    overrides = {
        "map_encoder": target_config["map_encoder"],
        "model_size": target_config["model_size"],
        "model_core": target_config["model_core"],
        "encoder_compute_dtype": target_config["encoder_compute_dtype"],
        "attention_compute_dtype": target_config["attention_compute_dtype"],
        "token_mixer_residual_init_scale": target_config[
            "token_mixer_residual_init_scale"
        ],
        "critic_hidden_dims": target_config["critic_hidden_dims"],
        # F15: persist the stage overrides so --resume_from / --teacher_checkpoint
        # rebuild the grown (5-stage) trunk and _validate_checkpoint_architecture
        # matches.
        "resnet_stage_channels": target_config["resnet_stage_channels"],
        "resnet_blocks_per_stage": target_config["resnet_blocks_per_stage"],
    }
    if source_train_config is None:
        return _DictConfig(**target_config)
    if isinstance(source_train_config, dict):
        updated = dict(source_train_config)
        updated.update(overrides)
        return updated
    # Mutate the dataclass in place (avoids re-running __post_init__ device checks).
    for name, value in overrides.items():
        setattr(source_train_config, name, value)
    return source_train_config


def main():
    parser = argparse.ArgumentParser(description="Grow a Terra checkpoint (spec F8).")
    parser.add_argument("--src", required=True, help="Source checkpoint .pkl.")
    parser.add_argument("--out", required=True, help="Output grown checkpoint .pkl.")
    parser.add_argument(
        "--map_encoder", default=None, help="Target map encoder (default: keep source)."
    )
    parser.add_argument(
        "--model_size",
        default=None,
        choices=["base", "medium", "large"],
        help="Target model_size preset (default: keep source).",
    )
    parser.add_argument(
        "--model_core",
        default=None,
        choices=["mlp", "transformer"],
        help="Target model_core (default: keep source).",
    )
    parser.add_argument(
        "--critic_hidden_dims",
        default=None,
        help="Comma-separated critic-head widths, e.g. '512,256' (default: keep source).",
    )
    parser.add_argument(
        "--encoder_compute_dtype",
        default=None,
        choices=["float32", "bfloat16"],
        help="Target encoder compute dtype (default: keep source).",
    )
    parser.add_argument(
        "--attention_compute_dtype",
        default=None,
        choices=["encoder", "float32", "bfloat16"],
        help=(
            "Target v4/v5 attention compute dtype (default: keep source, "
            "usually 'encoder')."
        ),
    )
    parser.add_argument(
        "--token_mixer_residual_init_scale",
        type=float,
        default=None,
        help=(
            "Target v5 token-mixer residual init scale. Set a small value "
            "such as 0.001 to make newly grown mixer residual projections "
            "nonzero."
        ),
    )
    parser.add_argument(
        "--resnet_stage_channels",
        default=None,
        help=(
            "Comma-separated target spatial-ResNet stage channels, e.g. "
            "'16,32,48,64,64' for the 4->5 stage 128x128 grow (default: keep source)."
        ),
    )
    parser.add_argument(
        "--resnet_blocks_per_stage",
        default=None,
        help=(
            "Comma-separated target spatial-ResNet blocks per stage, e.g. "
            "'1,1,2,2,2' (default: keep source)."
        ),
    )
    parser.add_argument(
        "--maps_edge_length",
        type=int,
        default=64,
        help=(
            "Map edge length used to build the TARGET model. With a 5-stage "
            "override this is the finer grid (e.g. 128); the readout stays 8x8."
        ),
    )
    args = parser.parse_args()

    # F6: shared __main__-unpickling shim so train.py- and train_mixed.py-saved
    # checkpoints both load here.
    helpers.register_checkpoint_config_classes()
    checkpoint = helpers.load_pkl_object(args.src)
    if "model" not in checkpoint:
        raise KeyError("source checkpoint has no 'model' parameters")
    source_params = checkpoint["model"]
    source_train_config = checkpoint.get("train_config")

    overrides = {
        "map_encoder": args.map_encoder,
        "model_size": args.model_size,
        "model_core": args.model_core,
        "critic_hidden_dims": _parse_critic_hidden_dims(args.critic_hidden_dims),
        "encoder_compute_dtype": args.encoder_compute_dtype,
        "attention_compute_dtype": args.attention_compute_dtype,
        "token_mixer_residual_init_scale": args.token_mixer_residual_init_scale,
        "resnet_stage_channels": _parse_critic_hidden_dims(args.resnet_stage_channels),
        "resnet_blocks_per_stage": _parse_critic_hidden_dims(args.resnet_blocks_per_stage),
    }
    target_config = build_target_config(source_train_config, overrides)

    # F5: size the policy head for the source checkpoint's action type instead of
    # the default (Tracked) BatchConfig.
    action_type = _derive_action_type(checkpoint)
    print(f"derived action_type: {action_type.__name__}")
    env = _dummy_env(args.maps_edge_length, action_type)
    _, target_params = get_model_ready(jax.random.PRNGKey(0), target_config, env)

    # F1/F15: derive the spatial stage layout from the actual param trees so
    # ResidualMapBlock relabeling and the flatten-Dense embed are correct across
    # non-final-stage depth growth, channel growth, AND added stages (the 4->5
    # stage 128x128 grow via --resnet_stage_channels). Param-derivation returns
    # (None, None) for the atari/global-pool encoders, disabling both remaps.
    source_stage_spec = _derive_stage_spec_from_params(source_params)
    target_stage_spec = _derive_stage_spec_from_params(target_params)

    grown_params, report = grow_params(
        source_params, target_params, source_stage_spec, target_stage_spec
    )
    print_report(
        report,
        source_params,
        target_params,
        grown_params,
        source_stage_spec,
        target_stage_spec,
    )

    out_checkpoint = dict(checkpoint)
    out_checkpoint["model"] = grown_params
    out_checkpoint["train_config"] = _update_out_train_config(
        source_train_config, target_config
    )

    # F2: a grown network restarts optimization from scratch, so drop any
    # optimizer/step bookkeeping carried by the source checkpoint.
    stale_keys = ["optimizer_state", "train_state_step", "update", "next_update"]
    dropped = [key for key in stale_keys if key in out_checkpoint]
    for key in dropped:
        del out_checkpoint[key]
    if dropped:
        print(f"Stripped stale optimization state (fresh restart): {', '.join(dropped)}")

    helpers.save_pkl_object(out_checkpoint, args.out)
    print(f"\n✅ Grown checkpoint written to {args.out}")


if __name__ == "__main__":
    main()
