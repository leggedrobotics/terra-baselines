#!/usr/bin/env python3
"""Exercise the CUDA paths Terra needs before starting a training update."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import os
import site
import sys
from pathlib import Path


REQUIRED_LIBS = {
    "cuDNN": ("cudnn", "libcudnn.so*", "nvidia/cudnn/lib/libcudnn.so*"),
    "CUPTI": ("cupti", "libcupti.so*", "nvidia/cuda_cupti/lib/libcupti.so*"),
    "cuBLAS": ("cublas", "libcublas.so*", "nvidia/cublas/lib/libcublas.so*"),
    "NVRTC": ("nvrtc", "libnvrtc.so*", "nvidia/cuda_nvrtc/lib/libnvrtc.so*"),
    "NCCL": ("nccl", "libnccl.so*", "nvidia/nccl/lib/libnccl.so*"),
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def candidate_roots() -> list[Path]:
    roots = [Path(path) for path in site.getsitepackages()]
    roots.extend(
        Path(path)
        for path in (
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/targets/aarch64-linux/lib",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/local/lib",
        )
    )
    roots.extend(Path(path) for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path)
    return list(dict.fromkeys(root.resolve() for root in roots if root.exists()))


def check_library_paths() -> None:
    roots = candidate_roots()
    missing = []
    for label, (lookup_name, filename, pip_pattern) in REQUIRED_LIBS.items():
        candidates: list[Path] = []
        for root in roots:
            candidates.extend(Path(path) for path in glob.glob(str(root / filename)))
            candidates.extend(Path(path) for path in glob.glob(str(root / pip_pattern)))
        system_name = ctypes.util.find_library(lookup_name)
        load_target = str(candidates[0]) if candidates else system_name
        if not load_target:
            missing.append(label)
            continue
        try:
            ctypes.CDLL(load_target)
        except OSError as exc:
            fail(f"{label} found as {load_target}, but the loader rejected it: {exc}")
        print(f"PASS {label} loader: {load_target}")
    if missing:
        fail("missing CUDA libraries: " + ", ".join(missing))


def check_jax_devices(min_devices: int) -> list[object]:
    import jax

    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    print("JAX version:", jax.__version__)
    print("JAX devices:", devices)
    if len(gpu_devices) < min_devices:
        fail(f"expected at least {min_devices} GPU devices, got {len(gpu_devices)}")
    print(f"PASS JAX devices: {len(gpu_devices)} GPU device(s)")
    return gpu_devices


def check_cudnn_conv(devices: list[object]) -> None:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def conv_grad(weight, values):
        def loss_fn(current_weight):
            result = jax.lax.conv_general_dilated(
                values,
                current_weight,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            return jnp.mean(result * result)

        return jax.grad(loss_fn)(weight)

    device = devices[0]
    values = jax.device_put(jnp.ones((4, 8, 16, 16), dtype=jnp.float32), device)
    weight = jax.device_put(jnp.ones((8, 8, 3, 3), dtype=jnp.float32) * 0.01, device)
    conv_grad(weight, values).block_until_ready()
    print("PASS cuDNN path: jitted convolution backward completed")


def check_nccl_all_reduce(num_devices: int) -> None:
    if num_devices < 2:
        print("SKIP NCCL all-reduce: only one GPU visible")
        return

    import jax
    import jax.numpy as jnp
    import numpy as np

    all_reduce = jax.pmap(lambda value: jax.lax.psum(value, "i"), axis_name="i")
    result = np.asarray(all_reduce(jnp.arange(num_devices, dtype=jnp.float32)))
    expected = float(num_devices * (num_devices - 1) / 2)
    if not np.allclose(result, expected):
        fail(f"NCCL all-reduce returned {result}, expected {expected}")
    print("PASS NCCL path: pmap all-reduce completed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-devices", type=int, default=1)
    args = parser.parse_args()

    check_library_paths()
    devices = check_jax_devices(args.min_devices)
    check_cudnn_conv(devices)
    check_nccl_all_reduce(len(devices))
    print("PASS JAX CUDA runtime preflight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
