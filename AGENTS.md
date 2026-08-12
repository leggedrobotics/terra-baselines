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

Euler launchers must select the execution account explicitly and derive
account-owned storage through `cluster/euler_account.sh`. The current default
is `alesweber`; `lterenzi` remains a supported fallback. Never infer the
storage owner from a historical path embedded in an experiment receipt.

- Reproducible code snapshots go under
  `$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation/`.
- Pinned inputs, `WANDB_DIR`, checkpoints, run logs, and other live run artifacts go under
  `$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs/`, never `/cluster/home`.
- Long-lived environments and archives go under the selected account's
  project storage. A group-readable pinned runtime may be shared across
  accounts, but its exact path and package/runtime check remain part of the
  run contract.
- Scratch is purged when a file is not accessed for ~15 days. Anything needed
  long-term must be copied to writable project/work storage or be rebuildable.
  A venv left on scratch was already corrupted by the purge (empty `jax`
  namespace package).
- The dataset stays read-only at
  `/cluster/project/rsl/alesweber/TerraProject/...`; do not copy it into home.
- Launchers MUST verify `id -un`, `$HOME`, scratch writability, and the selected
  runtime before staging or submission. Smoke gates parse the selected
  account's `lquota` row and abort above 45 GB of the 50 GB hard home quota.
- `SUBMIT=stage` may stage pinned code and inputs and run read-only Slurm
  association/partition/GPU-inventory checks, but must not create a per-run
  directory, contact W&B, or submit a job. A new account needs its own
  update-1 smoke before any production phase; do not reuse a smoke receipt from
  another Unix account or private scratch tree.

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
