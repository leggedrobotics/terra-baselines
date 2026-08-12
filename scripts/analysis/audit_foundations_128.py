#!/usr/bin/env python3
"""Audit the E9 foundations_128 dataset against its 64x64 source family."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from terra.env_generation.distance import compute_distance_map_taxicab


ARRAY_SUBDIRS = ("images", "occupancy", "dumpability", "distance")


def _load(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    if arr.dtype.kind in {"f", "c"} and not np.isfinite(arr).all():
        raise ValueError(f"{path} contains non-finite values")
    return arr


def _audit_counts(dataset_128: Path, expected_count: int) -> None:
    for subdir in ARRAY_SUBDIRS:
        files = sorted((dataset_128 / subdir).glob("img_*.npy"))
        if len(files) != expected_count:
            raise ValueError(
                f"{dataset_128 / subdir}: expected {expected_count} files, "
                f"found {len(files)}"
            )
        for path in files:
            arr = _load(path)
            if arr.shape != (128, 128):
                raise ValueError(f"{path}: expected shape (128, 128), got {arr.shape}")


def _audit_source_match(dataset_128: Path, source_64: Path, expected_count: int) -> None:
    for subdir in ("images", "occupancy", "dumpability"):
        for idx in range(1, expected_count + 1):
            arr128 = _load(dataset_128 / subdir / f"img_{idx}.npy")
            arr64 = _load(source_64 / subdir / f"img_{idx}.npy")
            if arr64.shape != (64, 64):
                raise ValueError(
                    f"{source_64 / subdir / f'img_{idx}.npy'}: "
                    f"expected shape (64, 64), got {arr64.shape}"
                )
            if not np.array_equal(arr128[::2, ::2], arr64):
                raise ValueError(
                    f"{subdir}/img_{idx}.npy is not the doubled 64x64 source map"
                )


def _audit_distance(dataset_128: Path, expected_count: int) -> None:
    for idx in range(1, expected_count + 1):
        target = _load(dataset_128 / "images" / f"img_{idx}.npy")
        actual = _load(dataset_128 / "distance" / f"img_{idx}.npy")
        expected = compute_distance_map_taxicab(target, realistic_max_distance=48)
        if not np.allclose(actual, expected, atol=1e-6):
            max_abs = float(np.max(np.abs(actual - expected)))
            raise ValueError(
                f"distance/img_{idx}.npy does not match 128x128 taxicab "
                f"recompute with realistic_max_distance=48; max_abs={max_abs}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-128", type=Path, required=True)
    parser.add_argument("--source-64", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=600)
    args = parser.parse_args()

    _audit_counts(args.dataset_128, args.expected_count)
    _audit_source_match(args.dataset_128, args.source_64, args.expected_count)
    _audit_distance(args.dataset_128, args.expected_count)
    metadata_count = len(list((args.dataset_128 / "metadata").glob("*.json")))
    print(
        "FOUNDATIONS_128_AUDIT_PASSED "
        f"dataset={args.dataset_128} source={args.source_64} "
        f"count={args.expected_count} metadata_json={metadata_count} "
        "distance_norm=48"
    )


if __name__ == "__main__":
    main()
