#!/usr/bin/env python3
"""Zero selected attention parameters in a Terra checkpoint for rollout probes.

This is an analysis helper, not a training path. It lets us answer questions
like "does the trained cross-attention branch matter?" by saving a copy of a
checkpoint with only the selected attention leaves zeroed, then running the
usual benchmark/eval tooling on the original and ablated checkpoints.

Examples:
    python scripts/analysis/ablate_attention_checkpoint.py ckpt.pkl xattn.pkl \
        --mode xattn
    python scripts/analysis/ablate_attention_checkpoint.py ckpt.pkl mixer.pkl \
        --mode token_mixer
    python scripts/analysis/ablate_attention_checkpoint.py ckpt.pkl /tmp/noop.pkl \
        --mode xattn --dry_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

import utils.helpers as helpers


MODE_PATTERNS = {
    # Direct cross-attention branch inside Spatial8x8MapResNet. Dense_1 is the
    # agent-query projection, Dense_2 is the branch projection, and Dense_3 is
    # the final fusion projection that must be left intact.
    "xattn": (
        "['cnn']['attn_latent_queries']",
        "['cnn']['attn_pos_embed']",
        "['cnn']['Dense_1']",
        "['cnn']['Dense_2']",
        "['cnn']['MultiHeadDotProductAttention_0']",
    ),
    # V5 self-attention mixer blocks. Zeroing all block params makes each block
    # an identity map while leaving the shared xattn readout and pos table live.
    "token_mixer": ("['cnn']['token_mixer_",),
}
MODE_PATTERNS["all_attention"] = MODE_PATTERNS["xattn"] + MODE_PATTERNS["token_mixer"]


def _patterns_from_args(args) -> tuple[str, ...]:
    patterns = []
    for mode in args.mode:
        patterns.extend(MODE_PATTERNS[mode])
    patterns.extend(args.pattern)
    # Keep order stable while dropping duplicates.
    return tuple(dict.fromkeys(patterns))


def _matches(key: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in key for pattern in patterns)


def _zero_matching_params(params, patterns: tuple[str, ...]):
    matched = []

    def maybe_zero(path, leaf):
        key = jax.tree_util.keystr(path)
        if _matches(key, patterns):
            matched.append((key, tuple(leaf.shape), str(leaf.dtype), int(leaf.size)))
            return jnp.zeros_like(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(maybe_zero, params), matched


def _print_matches(matched):
    if not matched:
        print("Matched 0 parameter leaves.")
        return
    total = sum(size for _, _, _, size in matched)
    print(f"Matched {len(matched)} parameter leaves ({total:,} scalars):")
    for key, shape, dtype, size in matched:
        print(f"  {key} shape={shape} dtype={dtype} size={size:,}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="Input checkpoint .pkl")
    parser.add_argument(
        "out",
        nargs="?",
        help="Output checkpoint .pkl. Required unless --dry_run is set.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        choices=sorted(MODE_PATTERNS),
        help=(
            "Named ablation pattern. May be repeated. xattn zeros the direct "
            "cross-attention branch; token_mixer zeros v5 mixer blocks; "
            "all_attention does both."
        ),
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help=(
            "Extra jax.tree_util.keystr substring to zero. May be repeated; "
            "use with --dry_run first when targeting exact Flax names."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print matched leaves but do not write an output checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output checkpoint.",
    )
    parser.add_argument(
        "--allow_empty",
        action="store_true",
        help="Do not fail if no parameter leaves match.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    patterns = _patterns_from_args(args)
    if not patterns:
        raise SystemExit("Choose at least one --mode or --pattern.")
    if not args.dry_run and args.out is None:
        raise SystemExit("out is required unless --dry_run is set.")
    if (
        not args.dry_run
        and args.out is not None
        and Path(args.out).exists()
        and not args.force
    ):
        raise SystemExit(f"Output exists: {args.out}. Pass --force to overwrite.")

    helpers.register_checkpoint_config_classes()
    checkpoint = helpers.load_pkl_object(args.src)
    if "model" not in checkpoint:
        raise KeyError("checkpoint has no 'model' parameter tree")

    ablated_params, matched = _zero_matching_params(checkpoint["model"], patterns)
    _print_matches(matched)
    if not matched and not args.allow_empty:
        raise SystemExit("No parameter leaves matched. Re-run with --dry_run/--pattern.")
    if args.dry_run:
        print("Dry run: no checkpoint written.")
        return

    out_checkpoint = dict(checkpoint)
    out_checkpoint["model"] = ablated_params
    out_checkpoint["ablation"] = {
        "source": args.src,
        "modes": tuple(args.mode),
        "patterns": patterns,
        "matched_leaves": len(matched),
        "matched_scalars": sum(size for _, _, _, size in matched),
    }
    helpers.save_pkl_object(out_checkpoint, args.out)
    print(f"Ablated checkpoint written to {args.out}")


if __name__ == "__main__":
    main()
