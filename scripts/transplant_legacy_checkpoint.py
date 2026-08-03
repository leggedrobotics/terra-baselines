#!/usr/bin/env python3
"""Transplant a historical Terra model into a current evaluation skeleton.

This is deliberately an evaluation-only compatibility path.  It reads the
historical checkpoint with one narrowly scoped pickle exception for the old
``terra.config.EnvConfig`` tuple arity, verifies that its model tree exactly
matches a current checkpoint skeleton, and writes only:

* the historical model and ``next_update``;
* the current skeleton's train/environment configuration; and
* a provenance receipt.

Historical optimizer, environment, sampler, loss, and runtime state are never
copied.

Example::

    python scripts/transplant_legacy_checkpoint.py \
        --source old_E8.pkl \
        --source-sha256 f364a5db... \
        --skeleton current_eval_skeleton.pkl \
        --skeleton-sha256 1a417c88... \
        --output E8_current_compat.pkl \
        --source-revision spatial-v3-3a21cd6 \
        --skeleton-revision 70d8099 \
        --role legacy-easy-e8-current-compat
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import pickle
import re
import tempfile
from pathlib import Path

import jax
import numpy as np

from utils import helpers

RECEIPT_SCHEMA = "terra_legacy_checkpoint_compatibility_transplant_v1"


class _HistoricalEnvConfig(tuple):
    """Tuple sink for historical EnvConfig values that are intentionally dropped."""

    def __new__(cls, *values):
        return tuple.__new__(cls, values)


class _HistoricalCheckpointUnpickler(pickle.Unpickler):
    """Use ordinary pickle resolution except for the incompatible EnvConfig."""

    def find_class(self, module, name):
        if (module, name) == ("terra.config", "EnvConfig"):
            return _HistoricalEnvConfig
        return super().find_class(module, name)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validated_sha256(label: str, digest: str) -> str:
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError(f"{label} must be exactly 64 hexadecimal characters")
    return digest.lower()


def _read_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_bytes()


def _load_historical(data: bytes):
    helpers.register_checkpoint_config_classes()
    return _HistoricalCheckpointUnpickler(io.BytesIO(data)).load()


def _load_current(data: bytes):
    helpers.register_checkpoint_config_classes()
    return pickle.loads(data)


def _model_inventory(model):
    leaves_with_paths, tree = jax.tree_util.tree_flatten_with_path(model)
    inventory = []
    for path, leaf in leaves_with_paths:
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.hasobject:
            raise TypeError(f"model leaf {jax.tree_util.keystr(path)} has object dtype")
        inventory.append(
            (
                jax.tree_util.keystr(path),
                tuple(int(size) for size in array.shape),
                str(array.dtype),
                array.tobytes(order="C"),
            )
        )
    return tree, inventory


def model_content_sha256(model) -> tuple[str, int, int]:
    """Hash model paths, shapes, dtypes, and exact C-order leaf bytes."""

    _, inventory = _model_inventory(model)
    digest = hashlib.sha256()
    parameter_count = 0
    for path, shape, dtype, content in inventory:
        header = json.dumps(
            {"dtype": dtype, "path": path, "shape": shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        parameter_count += int(np.prod(shape, dtype=np.int64))
    return digest.hexdigest(), len(inventory), parameter_count


def _require_exact_model_contract(source_model, skeleton_model) -> None:
    source_tree, source_inventory = _model_inventory(source_model)
    skeleton_tree, skeleton_inventory = _model_inventory(skeleton_model)
    if source_tree != skeleton_tree:
        raise ValueError("source and skeleton model pytree structures differ")

    source_contract = [
        (path, shape, dtype) for path, shape, dtype, _ in source_inventory
    ]
    skeleton_contract = [
        (path, shape, dtype) for path, shape, dtype, _ in skeleton_inventory
    ]
    if source_contract == skeleton_contract:
        return

    for source_leaf, skeleton_leaf in zip(source_contract, skeleton_contract):
        if source_leaf != skeleton_leaf:
            raise ValueError(
                "source and skeleton model leaf contracts differ: "
                f"source={source_leaf}, skeleton={skeleton_leaf}"
            )
    raise ValueError(
        "source and skeleton model leaf counts differ: "
        f"{len(source_contract)} != {len(skeleton_contract)}"
    )


def _presentation_train_config(train_config, role: str):
    presented = copy.deepcopy(train_config)
    if isinstance(presented, dict):
        previous = {field: presented.get(field) for field in ("name", "config_name")}
        presented["name"] = role
        presented["config_name"] = role
        return presented, previous

    for field in ("name", "config_name"):
        if not hasattr(presented, field):
            raise TypeError(f"skeleton train_config has no {field}")
    previous = {field: getattr(presented, field) for field in ("name", "config_name")}
    presented.name = role
    presented.config_name = role
    return presented, previous


def _config_value(config, field: str):
    return config[field] if isinstance(config, dict) else getattr(config, field)


def transplant_checkpoint(
    *,
    source: Path,
    expected_source_sha256: str,
    skeleton: Path,
    expected_skeleton_sha256: str,
    output: Path,
    source_revision: str,
    skeleton_revision: str,
    role: str,
) -> dict:
    """Write one current-loadable evaluation checkpoint and return its receipt."""

    source = source.resolve()
    skeleton = skeleton.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not source_revision or not skeleton_revision or not role:
        raise ValueError("source_revision, skeleton_revision, and role are required")

    expected_source_sha256 = _validated_sha256("source_sha256", expected_source_sha256)
    expected_skeleton_sha256 = _validated_sha256(
        "skeleton_sha256", expected_skeleton_sha256
    )
    source_bytes = _read_bytes(source)
    source_sha256 = _sha256(source_bytes)
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            "source checkpoint SHA-256 mismatch: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )
    skeleton_bytes = _read_bytes(skeleton)
    skeleton_sha256 = _sha256(skeleton_bytes)
    if skeleton_sha256 != expected_skeleton_sha256:
        raise ValueError(
            "skeleton checkpoint SHA-256 mismatch: "
            f"expected {expected_skeleton_sha256}, got {skeleton_sha256}"
        )

    source_checkpoint = _load_historical(source_bytes)
    skeleton_checkpoint = _load_current(skeleton_bytes)
    for name, checkpoint in (
        ("source", source_checkpoint),
        ("skeleton", skeleton_checkpoint),
    ):
        if not isinstance(checkpoint, dict):
            raise TypeError(f"{name} checkpoint must be a dict")
        if "model" not in checkpoint:
            raise KeyError(f"{name} checkpoint has no model")
    if "next_update" not in source_checkpoint:
        raise KeyError("source checkpoint has no next_update")
    for field in ("train_config", "env_config"):
        if field not in skeleton_checkpoint:
            raise KeyError(f"skeleton checkpoint has no {field}")

    _require_exact_model_contract(
        source_checkpoint["model"], skeleton_checkpoint["model"]
    )
    model_sha256, leaf_count, parameter_count = model_content_sha256(
        source_checkpoint["model"]
    )
    next_update = int(source_checkpoint["next_update"])
    train_config, previous_presentation = _presentation_train_config(
        skeleton_checkpoint["train_config"], role
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "role": role,
        "source_checkpoint": {
            "path": str(source),
            "raw_sha256": source_sha256,
            "revision": source_revision,
            "next_update": next_update,
        },
        "evaluation_skeleton": {
            "path": str(skeleton),
            "raw_sha256": skeleton_sha256,
            "revision": skeleton_revision,
        },
        "model_content_sha256": model_sha256,
        "model_leaf_count": leaf_count,
        "model_parameter_count": parameter_count,
        "copied_from_source": ["model", "next_update"],
        "copied_from_skeleton": ["train_config", "env_config"],
        "train_config_presentation_overrides": {
            field: {"from": previous_presentation[field], "to": role}
            for field in ("name", "config_name")
        },
        "historical_state_discarded": True,
    }
    output_checkpoint = {
        "checkpoint_version": skeleton_checkpoint.get("checkpoint_version", 2),
        "train_config": train_config,
        "env_config": skeleton_checkpoint["env_config"],
        "model": source_checkpoint["model"],
        "next_update": next_update,
        "compatibility_transplant": receipt,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            pickle.dump(output_checkpoint, stream, pickle.HIGHEST_PROTOCOL)

        reloaded = helpers.load_pkl_object(str(temporary_path))
        reloaded_sha256, _, _ = model_content_sha256(reloaded["model"])
        if reloaded_sha256 != model_sha256:
            raise RuntimeError(
                "written model content differs from the historical source"
            )
        if int(reloaded["next_update"]) != next_update:
            raise RuntimeError("written next_update differs from the historical source")
        for field in ("name", "config_name"):
            if _config_value(reloaded["train_config"], field) != role:
                raise RuntimeError(f"written train_config.{field} differs from role")
        if output.exists():
            raise FileExistsError(output)
        temporary_path.rename(output)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--skeleton-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--skeleton-revision", required=True)
    parser.add_argument("--role", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = transplant_checkpoint(
        source=args.source,
        expected_source_sha256=args.source_sha256,
        skeleton=args.skeleton,
        expected_skeleton_sha256=args.skeleton_sha256,
        output=args.output,
        source_revision=args.source_revision,
        skeleton_revision=args.skeleton_revision,
        role=args.role,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
