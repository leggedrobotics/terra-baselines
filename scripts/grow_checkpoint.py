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
        [--maps_edge_length 64] --out grown.pkl
"""

import argparse
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp

import utils.helpers as helpers
from terra.config import BatchConfig, MapsDimsConfig
from utils.models import canonical_map_encoder, get_model_ready


def _register_checkpoint_config_classes():
    """Checkpoints pickle their config dataclass under ``__main__``.

    train.py / train_mixed.py run as scripts, so their TrainConfig /
    MixedAgentTrainConfig classes are referenced as ``__main__.<name>`` inside
    the pkl. Alias them into this script's ``__main__`` so unpickling works.
    """
    main_module = sys.modules["__main__"]
    try:
        from train import TrainConfig

        main_module.TrainConfig = TrainConfig
    except ImportError:
        pass
    try:
        from train_mixed import MixedAgentTrainConfig

        main_module.MixedAgentTrainConfig = MixedAgentTrainConfig
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------

class _DictConfig(dict):
    """Dict that also supports attribute access (get_model_ready reads both)."""

    __getattr__ = dict.__getitem__


def _cfg_get(config, name, default):
    """Read a field from a dataclass-like or dict-like config."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def build_target_config(source_train_config, overrides):
    """Build a get_model_ready-compatible config for the grown architecture.

    ``overrides`` may set map_encoder / model_size / model_core /
    critic_hidden_dims / encoder_compute_dtype; anything left ``None`` inherits
    the source checkpoint value.
    """
    target = _DictConfig(
        clip_action_maps=_cfg_get(source_train_config, "clip_action_maps", True),
        maps_net_normalization_bounds=_cfg_get(
            source_train_config, "maps_net_normalization_bounds", (-10, 10)
        ),
        local_map_normalization_bounds=_cfg_get(
            source_train_config, "local_map_normalization_bounds", (-16, 16)
        ),
        loaded_max=_cfg_get(source_train_config, "loaded_max", 100),
        num_prev_actions=_cfg_get(source_train_config, "num_prev_actions", 10),
        model_core=_cfg_get(source_train_config, "model_core", "mlp"),
        model_size=_cfg_get(source_train_config, "model_size", "base"),
        map_encoder=_cfg_get(source_train_config, "map_encoder", "atari"),
        encoder_compute_dtype=_cfg_get(
            source_train_config, "encoder_compute_dtype", "float32"
        ),
        critic_hidden_dims=_cfg_get(source_train_config, "critic_hidden_dims", None),
    )
    for key, value in overrides.items():
        if value is not None:
            target[key] = value
    target["map_encoder"] = canonical_map_encoder(target["map_encoder"])
    return target


def _dummy_env(maps_edge_length):
    return SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=maps_edge_length))
    )


# ----------------------------------------------------------------------------
# Growth core
# ----------------------------------------------------------------------------

def _leaf_map(params):
    """Return {keystr: leaf} for every parameter leaf."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {jax.tree_util.keystr(path): leaf for path, leaf in leaves}


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


def _slice_copy(source_leaf, target_leaf):
    """Copy the source into the leading slices; 0.1-scale fresh init elsewhere."""
    grown = 0.1 * target_leaf
    slices = tuple(slice(0, s) for s in source_leaf.shape)
    return grown.at[slices].set(source_leaf)


def grow_params(source_params, target_params):
    """Grow ``source_params`` into the shape of ``target_params``.

    ``target_params`` supplies the fresh init and the output structure. Returns
    ``(grown_params, report)`` where report is a list of per-leaf dicts.
    """
    source_leaves = _leaf_map(source_params)
    report = []

    def _grow_leaf(path, target_leaf):
        key = jax.tree_util.keystr(path)
        source_leaf = source_leaves.get(key)
        if source_leaf is not None:
            if source_leaf.shape == target_leaf.shape:
                category = "copied"
                grown = jnp.asarray(source_leaf, dtype=target_leaf.dtype)
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


def print_report(report, source_params, target_params, grown_params):
    """Print the per-leaf growth report and parameter counts."""
    print("\nPer-leaf growth report:")
    counts = {"copied": 0, "sliced": 0, "zero-init": 0, "fresh": 0}
    for entry in report:
        counts[entry["category"]] += 1
        print(
            f"  {entry['category']:9s} {entry['key']}  "
            f"src={entry['source_shape']} tgt={entry['target_shape']}"
        )
    print("\nCategory totals:")
    for name, count in counts.items():
        print(f"  {name:9s}: {count}")
    unused = set(_leaf_map(source_params)) - {e["key"] for e in report}
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
        "critic_hidden_dims": target_config["critic_hidden_dims"],
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
        "--maps_edge_length",
        type=int,
        default=64,
        help="Map edge length used to build the model (must match training).",
    )
    args = parser.parse_args()

    _register_checkpoint_config_classes()
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
    }
    target_config = build_target_config(source_train_config, overrides)
    env = _dummy_env(args.maps_edge_length)
    _, target_params = get_model_ready(jax.random.PRNGKey(0), target_config, env)

    grown_params, report = grow_params(source_params, target_params)
    print_report(report, source_params, target_params, grown_params)

    out_checkpoint = dict(checkpoint)
    out_checkpoint["model"] = grown_params
    out_checkpoint["train_config"] = _update_out_train_config(
        source_train_config, target_config
    )
    helpers.save_pkl_object(out_checkpoint, args.out)
    print(f"\n✅ Grown checkpoint written to {args.out}")


if __name__ == "__main__":
    main()
