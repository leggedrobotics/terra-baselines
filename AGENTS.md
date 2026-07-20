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

## Euler Storage Contract

`/cluster/home/lterenzi` is a hard 50 GB cap and is for code and small config
only. On 2026-07-20 a Terra job died with `Disk quota exceeded` because
`WANDB_DIR` and checkpoints were written under home (home was 44.2/50 GB). Run
artifacts belong on scratch.

- sbatch scripts MUST set `WANDB_DIR` and checkpoint output dirs to
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/` (symlinked from the home
  workspace), never `/cluster/home`. Run logs go there too.
- Scratch is purged when a file is not accessed for ~15 days. Anything needed
  long-term must be `rsync`'d to `/cluster/work/rsl/lterenzi` (final large
  checkpoints, tars) or `/cluster/project/rsl/lterenzi` (venvs, many small
  files) — both verified writable — or be rebuildable. A venv left on scratch
  was already corrupted by the purge (empty `jax` namespace package).
- The dataset stays read-only at
  `/cluster/project/rsl/alesweber/TerraProject/...`; do not copy it into home.
- Smoke gates MUST fail fast if home is near full: parse `lquota` and abort
  before training when `/cluster/home/lterenzi` space usage exceeds ~90% of the
  hard quota (i.e. above ~45 GB of 50 GB).

## Training Metric Contract

- `train/episode_success_rate` is the bounded online ratio of successful task
  completions to all completed episodes (successes plus timeouts) in the latest
  PPO rollout, aggregated across devices. A window with no completed episode is
  reported as NaN so it is not confused with zero success. Use W&B smoothing
  when individual rollout windows are sparse.
- `eval/success_within_horizon_rate` is the primary bounded evaluation metric:
  the fraction of initial reset episodes that succeed within the fixed eval step
  budget. Auto-reset episodes are excluded. Inspect
  `eval/initial_episode_completion_rate` to see how much of the initial cohort
  terminated before the horizon.
- `eval/completed_episode_success_rate` is success among all completed episodes
  in the auto-reset eval stream. It is secondary because unfinished episodes are
  censored and a horizon shorter than the timeout can make it trivially one.
- `eval/positive_terminations` and `eval/total_terminations` are legacy
  episodes-per-initial-environment metrics. Evaluation environments auto-reset,
  so these values can exceed one. The explicit aliases are
  `eval/successful_episodes_per_env` and `eval/completed_episodes_per_env`.
- `progress/episode_completion_rate` is a legacy name for the fraction of
  environments terminal on the final training-rollout step. It includes
  timeouts and is not a success rate; use
  `progress/last_step_termination_fraction` when that quantity is needed.
