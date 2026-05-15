#!/usr/bin/env python3
"""Check JAX CUDA runtime paths needed by Terra RL training."""

from __future__ import annotations

import argparse
import glob
import os
import site
import sys
from pathlib import Path


REQUIRED_LIBS = {
    "cuDNN": "nvidia/cudnn/lib/libcudnn.so*",
    "CUPTI": "nvidia/cuda_cupti/lib/libcupti.so*",
    "cuBLAS": "nvidia/cublas/lib/libcublas.so*",
    "NVRTC": "nvidia/cuda_nvrtc/lib/libnvrtc.so*",
    "NCCL": "nvidia/nccl/lib/libnccl.so*",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def site_packages() -> str:
    paths = site.getsitepackages()
    if not paths:
        fail("site.getsitepackages() returned no paths")
    return paths[0]


def check_library_paths(sp: str) -> None:
    ld_paths = {Path(p).resolve() for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p}
    missing = []
    missing_ld = []

    for label, pattern in REQUIRED_LIBS.items():
        matches = [Path(p).resolve() for p in glob.glob(str(Path(sp) / pattern))]
        if not matches:
            missing.append(f"{label} ({pattern})")
            continue
        lib_dir = matches[0].parent
        if lib_dir not in ld_paths:
            missing_ld.append(f"{label} dir not in LD_LIBRARY_PATH: {lib_dir}")

    if missing:
        fail("missing venv CUDA libraries: " + ", ".join(missing))
    if missing_ld:
        fail("; ".join(missing_ld))

    print("PASS library paths: cuDNN/CUPTI/cuBLAS/NVRTC/NCCL present and exported")


def check_jax_devices(min_devices: int) -> list[object]:
    import jax

    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    print("JAX devices:", devices)
    if len(gpu_devices) < min_devices:
        fail(f"expected at least {min_devices} GPU devices, got {len(gpu_devices)}")
    print(f"PASS JAX devices: {len(gpu_devices)} GPU device(s)")
    return gpu_devices


def check_cudnn_conv(devices: list[object]) -> None:
    import jax
    import jax.numpy as jnp

    device = devices[0]

    @jax.jit
    def conv_grad(weight, x):
        def loss_fn(w):
            y = jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            return jnp.mean(y * y)

        return jax.grad(loss_fn)(weight)

    x = jax.device_put(jnp.ones((4, 8, 16, 16), dtype=jnp.float32), device)
    weight = jax.device_put(jnp.ones((8, 8, 3, 3), dtype=jnp.float32) * 0.01, device)
    grad = conv_grad(weight, x)
    grad.block_until_ready()
    print("PASS cuDNN path: jitted conv backward completed")


def check_nccl_all_reduce(num_devices: int) -> None:
    if num_devices < 2:
        print("SKIP NCCL all-reduce: only one GPU visible")
        return

    import jax
    import jax.numpy as jnp
    import numpy as np

    all_reduce = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")
    values = jnp.arange(num_devices, dtype=jnp.float32)
    result = np.asarray(all_reduce(values))
    expected = float(num_devices * (num_devices - 1) / 2)
    if not np.allclose(result, expected):
        fail(f"NCCL all-reduce returned {result}, expected {expected}")
    print("PASS NCCL path: pmap all-reduce completed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-devices", type=int, default=1)
    parser.add_argument("--skip-library-path-check", action="store_true")
    args = parser.parse_args()

    sp = site_packages()
    print("site-packages:", sp)
    if not args.skip_library_path_check:
        check_library_paths(sp)
    devices = check_jax_devices(args.min_devices)
    check_cudnn_conv(devices)
    check_nccl_all_reduce(len(devices))
    print("PASS JAX CUDA runtime preflight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
