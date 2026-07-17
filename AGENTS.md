# Agent Notes for `terra-baselines`

This repository contains PPO training, evaluation, checkpointing, and model
code for the sibling Terra environment at `/home/lorenzo/moleworks/terra`.
Use the `terra-rl` workflow for changes spanning both repositories.

## Map Encoder Contract

Use behavior-based canonical names in new commands, configs, logs, and docs:

- `atari`: fast default convolutional encoder.
- `resnet_global_pool`: PR #15 residual topology with global mean+max pooling.
  Preserve its parameter tree and raw-map preprocessing so existing checkpoints
  remain loadable.
- `resnet_spatial_8x8`: residual encoder with a flattened 8x8 spatial readout.
  This is the residual candidate for new runs when retaining map location is
  worth the additional compute.

Compatibility aliases are part of the checkpoint API:

- `resnet_delayed` -> `resnet_global_pool`
- `resnet_spatial_v2` -> `resnet_spatial_8x8`

Do not assign a different topology or preprocessing rule to an existing
canonical name or alias. Add a new canonical name when parameter shapes or
input semantics change, then update checkpoint validation and alias tests.

The base `resnet_spatial_8x8` stages use channels `(16, 32, 48, 64)` and block
counts `(1, 1, 2, 2)`. Only stages after the first perform downsampling, producing
64x64 -> 32x32 -> 16x16 -> 8x8 grids. The encoder flattens the final grid; it
does not perform global pooling.

## Validation

Run focused correctness checks on CPU:

```bash
export PYTHONPATH=/home/lorenzo/moleworks/terra:/home/lorenzo/moleworks/terra-baselines
export JAX_PLATFORMS=cpu
/home/lorenzo/moleworks/.venv-terra-uv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

For architecture or runtime changes, also use the CUDA environment
`/home/lorenzo/moleworks/.venv-terra-gpu-uv`, complete the Terra RL CUDA
preflight, and run through at least the first training update on the RTX 4090.
Do not compare throughput while another process is saturating the GPU.

Before committing, verify `git status -sb` and leave unrelated files such as
local lockfiles or run artifacts untouched.
