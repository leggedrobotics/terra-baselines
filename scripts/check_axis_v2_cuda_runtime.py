#!/usr/bin/env python3
"""Numerically validate the CUDA convolution shapes used by axis-v2 PPO."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def expected_ones_kernel_gradient(spatial_size: int, channels: int) -> np.ndarray:
    """Return the closed-form 3x3 SAME-convolution gradient for all-one inputs."""
    valid = np.array(
        [
            sum(0 <= position + offset - 1 < spatial_size for offset in range(3))
            for position in range(spatial_size)
        ],
        dtype=np.float64,
    )
    output = channels * np.outer(valid, valid)
    expected = np.empty((3, 3), dtype=np.float64)
    scale = 2.0 / (spatial_size * spatial_size * channels)
    for kernel_row in range(3):
        for kernel_col in range(3):
            rows = [
                row
                for row in range(spatial_size)
                if 0 <= row + kernel_row - 1 < spatial_size
            ]
            cols = [
                col
                for col in range(spatial_size)
                if 0 <= col + kernel_col - 1 < spatial_size
            ]
            expected[kernel_row, kernel_col] = (
                scale * output[np.ix_(rows, cols)].sum()
            )
    return expected


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _write_receipt(path: Path | None, receipt: dict[str, object]) -> None:
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    print(text, end="", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-devices", type=int, required=True)
    parser.add_argument("--envs-per-device", type=int, required=True)
    parser.add_argument(
        "--expected-device-kind", default="NVIDIA GeForce RTX 4090"
    )
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_devices <= 0 or args.envs_per_device <= 0:
        raise ValueError("device and environment counts must be positive")

    started = time.monotonic()
    devices = jax.devices()
    backend = jax.lib.xla_bridge.get_backend()
    receipt: dict[str, object] = {
        "schema": "terra_axis_v2_cuda_runtime_validation_v1",
        "status": "running",
        "num_devices": args.num_devices,
        "envs_per_device": args.envs_per_device,
        "encoder_convolution_batch_per_device": min(
            args.envs_per_device,
            64,
        ),
        "device_kinds": [device.device_kind for device in devices],
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "cudnn_package_version": _package_version("nvidia-cudnn-cu12"),
        "backend_platform_version": backend.platform_version,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "checks": [],
    }

    try:
        if len(devices) != args.num_devices:
            raise RuntimeError(
                f"expected {args.num_devices} JAX devices, got {devices}"
            )
        if any(device.device_kind != args.expected_device_kind for device in devices):
            raise RuntimeError(
                f"expected only {args.expected_device_kind!r}, got {devices}"
            )

        def loss(x: jax.Array, kernel: jax.Array) -> jax.Array:
            output = lax.conv_general_dilated(
                x,
                kernel,
                (1, 1),
                "SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            ).astype(jnp.float32)
            return lax.psum(jnp.mean(output * output), "devices")

        kernel_grad = jax.pmap(
            jax.grad(loss, argnums=1), axis_name="devices", in_axes=(0, None)
        )

        checks = receipt["checks"]
        assert isinstance(checks, list)
        convolution_batch = min(args.envs_per_device, 64)
        for spatial_size, channels in ((16, 64), (8, 96)):
            case_started = time.monotonic()
            x = jnp.ones(
                (
                    args.num_devices,
                    convolution_batch,
                    spatial_size,
                    spatial_size,
                    channels,
                ),
                dtype=jnp.bfloat16,
            )
            kernel = jnp.ones((3, 3, channels, channels), dtype=jnp.bfloat16)
            gradient = kernel_grad(x, kernel)
            expected = expected_ones_kernel_gradient(spatial_size, channels)
            observed = np.asarray(gradient[0], dtype=np.float32) / args.num_devices
            expected_full = np.broadcast_to(
                expected[:, :, None, None], observed.shape
            )
            np.testing.assert_allclose(
                observed,
                expected_full,
                rtol=2e-2,
                atol=2e-2,
            )
            checks.append(
                {
                    "name": f"bf16_backward_filter_3x3_{spatial_size}x{spatial_size}_{channels}",
                    "input_shape": list(x.shape),
                    "kernel_shape": list(kernel.shape),
                    "gradient_shape": list(gradient.shape),
                    "expected_representative": expected.tolist(),
                    "observed_representative": observed[:, :, 0, 0].tolist(),
                    "max_abs_error": float(np.max(np.abs(observed - expected_full))),
                    "elapsed_seconds": time.monotonic() - case_started,
                    "passed": True,
                }
            )

        case_started = time.monotonic()
        x = jnp.ones(
            (args.num_devices, convolution_batch, 8, 8, 96),
            dtype=jnp.bfloat16,
        )
        kernel = jnp.ones((1, 1, 96, 32), dtype=jnp.bfloat16)
        gradient = kernel_grad(x, kernel)
        observed = np.asarray(gradient[0], dtype=np.float32) / args.num_devices
        np.testing.assert_allclose(observed, 6.0, rtol=2e-2, atol=2e-2)
        checks.append(
            {
                "name": "bf16_backward_filter_1x1_8x8_96_to_32",
                "input_shape": list(x.shape),
                "kernel_shape": list(kernel.shape),
                "gradient_shape": list(gradient.shape),
                "expected": 6.0,
                "observed_min": float(observed.min()),
                "observed_max": float(observed.max()),
                "observed_mean": float(observed.mean()),
                "max_abs_error": float(np.max(np.abs(observed - 6.0))),
                "elapsed_seconds": time.monotonic() - case_started,
                "passed": True,
            }
        )
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["elapsed_seconds"] = time.monotonic() - started
        _write_receipt(args.receipt, receipt)
        raise

    receipt["status"] = "passed"
    receipt["elapsed_seconds"] = time.monotonic() - started
    _write_receipt(args.receipt, receipt)


if __name__ == "__main__":
    main()
