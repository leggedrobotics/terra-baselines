#!/usr/bin/env python3
"""Fail unless the CUDA paths used by Terra PPO work on this allocation."""

import argparse
import glob
import os
import site
from pathlib import Path


REQUIRED_LIBS = {
    "cuDNN": "nvidia/cudnn/lib/libcudnn.so*",
    "CUPTI": "nvidia/cuda_cupti/lib/libcupti.so*",
    "cuBLAS": "nvidia/cublas/lib/libcublas.so*",
    "NVRTC": "nvidia/cuda_nvrtc/lib/libnvrtc.so*",
    "NCCL": "nvidia/nccl/lib/libnccl.so*",
}


def fail(message):
    raise SystemExit(f"FAIL: {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-devices", type=int, default=1)
    args = parser.parse_args()

    site_packages = site.getsitepackages()[0]
    exported = {
        Path(path).resolve()
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if path
    }
    for label, pattern in REQUIRED_LIBS.items():
        matches = glob.glob(str(Path(site_packages) / pattern))
        if not matches:
            fail(f"missing {label}: {pattern}")
        library_dir = Path(matches[0]).resolve().parent
        if library_dir not in exported:
            fail(f"{label} directory is absent from LD_LIBRARY_PATH")

    import jax
    import jax.numpy as jnp
    import numpy as np

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(devices) < args.min_devices:
        fail(f"expected {args.min_devices} GPUs, got {devices}")

    @jax.jit
    def conv_grad(weight, inputs):
        def loss(candidate):
            output = jax.lax.conv_general_dilated(
                inputs,
                candidate,
                (1, 1),
                "SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            return jnp.mean(output * output)

        return jax.grad(loss)(weight)

    inputs = jax.device_put(
        jnp.ones((4, 8, 16, 16), dtype=jnp.float32), devices[0]
    )
    weights = jax.device_put(
        jnp.ones((8, 8, 3, 3), dtype=jnp.float32) * 0.01, devices[0]
    )
    conv_grad(weights, inputs).block_until_ready()

    all_reduce = jax.pmap(lambda value: jax.lax.psum(value, "i"), axis_name="i")
    result = np.asarray(all_reduce(jnp.arange(len(devices), dtype=jnp.float32)))
    expected = len(devices) * (len(devices) - 1) / 2
    if not np.allclose(result, expected):
        fail(f"NCCL all-reduce returned {result}, expected {expected}")

    print(
        f"PASS CUDA runtime: libraries, {len(devices)} GPUs, conv backward, NCCL"
    )


if __name__ == "__main__":
    main()
