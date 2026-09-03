# Pinned Euler JAX runtime

The trench-v2 launcher uses a pip-managed CUDA runtime rather than mixing the
Euler `cuda` module with CUDA libraries from Python wheels. The validated stack
is Python 3.12.8, JAX 0.4.33, CUDA 12.6.77, cuDNN 9.5.0.50, and NCCL 2.23.4.

Build it on an Euler login node with only the base stack and proxy loaded:

```bash
module purge
module load stack/2024-06 eth_proxy

runtime=/cluster/project/rsl/$USER/terra_runtime/terra_jax0433_cuda126_cudnn950_20260903
uv venv --python 3.12 "$runtime"
uv pip install --python "$runtime/bin/python" \
  -r cluster/euler_runtime/requirements-jax0433-cuda126-cudnn950.txt
uv pip check --python "$runtime/bin/python"
cp cluster/euler_runtime/requirements-jax0433-cuda126-cudnn950.txt \
  "$runtime/requirements.lock.txt"
sha256sum "$runtime/requirements.lock.txt"
```

The expected lock digest is
`36413dbcd02339dd6c899c9015ea2c5119bdeb90116a93104b676065036c6189`.
Launchers must verify that digest and the core package versions inside every
allocation. Do not load an Euler CUDA module or prepend CUDA, cuDNN, cuBLAS,
NCCL, or NVRTC directories to `LD_LIBRARY_PATH`; either action can override the
wheel-managed libraries and recreate an unsupported mixed runtime.
