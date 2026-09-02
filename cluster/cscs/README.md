# Terra training on CSCS Daint

This workflow runs `terra-baselines/train_mixed.py` as one JAX process with all
four GH200 GPUs on a Daint node. `terra-baselines` owns the launcher; every job
stages both sibling repositories and records their Git revisions and dirty-entry
counts in `SOURCE_REVISIONS.txt`.

The image uses NVIDIA JAX 24.10 (`jaxlib` 0.4.33) because it is close to Terra's
tested JAX 0.4 runtime. It does not install either source package; the staged
repositories are mounted and added to `PYTHONPATH`, so Terra's stale x86 conda
lock and `python_requires` metadata do not control the aarch64 runtime.
The launcher disables the image's experimental MLIR fusion emitter because
Terra's gather/select PPO graph fails in that emitter; the established GPU
emitter remains enabled for training.

## One-time setup

```bash
cd ~/moleworks/terra-baselines
cp cluster/cscs/config.env.example cluster/cscs/config.env
$EDITOR cluster/cscs/config.env

# Build on the Daint login node; this does not allocate a compute node.
cluster/cscs/build_image.sh

# Upload a dataset root. This example produces .../datasets/src64/foundations.
cluster/cscs/sync_dataset.sh ~/moleworks/terra_data/_src64 src64
```

Authenticate W&B without putting a key in the repository. Either run `wandb
login` in the CSCS account or export `WANDB_API_KEY` when submitting production
work. The smoke profile defaults to offline W&B.

## Four-GPU smoke

`--test-only` is the default and asks Slurm to validate the request without
allocating a node:

```bash
cluster/cscs/submit.sh \
  --account d130 \
  --partition debug \
  --time 00:20:00 \
  --dataset-path /capstor/scratch/cscs/lterenzi/terra-training/datasets/src64 \
  --dataset-size 8 \
  --profile smoke
```

Add `--submit` to run it. The smoke performs these gates before one PPO update:

1. Four JAX GPU devices are visible.
2. CUDA, cuDNN, cuBLAS, NVRTC, CUPTI, and NCCL load.
3. A jitted convolution backward pass succeeds.
4. A four-GPU `pmap` all-reduce succeeds.
5. One PPO update and checkpoint save complete.

Inspect the paths printed by `submit.sh`, or use:

```bash
ssh daint 'squeue -u "$USER"'
ssh daint 'tail -f "$SCRATCH"/terra-training/runs/RUN_ID/logs/slurm-JOB_ID.out'
```

## Production profile

The production profile defaults to the established single-node parameters:
four devices, 1024 environments per device, 32 rollout steps, two PPO epochs,
16 minibatches, and 50 billion total timesteps. Arguments after `--` override
or extend the profile.

```bash
cluster/cscs/submit.sh \
  --dataset-path /capstor/scratch/cscs/lterenzi/terra-training/datasets/full-train \
  --dataset-size 600 \
  --profile production \
  --submit \
  -- --config solo_excavator_rectangles_2stage --name rectangles-2stage
```

Use a new `CSCS_IMAGE_TAG` for image changes. Source snapshots are immutable by
run ID. Checkpoints, W&B files, and Slurm logs are written under
`$SCRATCH/terra-training/runs/RUN_ID`.

## Continuation past the 24 h cap

The `normal` partition caps a job at 24 h, so long runs continue in chained
jobs. `--resume-latest` sets `TERRA_RESUME_LATEST=1` in the job environment;
`run_training.sh` then picks the newest `*_update_*.pkl` under
`RUN_DIR/checkpoints` and appends `--resume_from` to the trainer arguments, so
the model, the optimizer state, and the absolute update counter carry over and
training continues towards the same `--total_timesteps`. It prints
`resume_from=<path>`, or `resume_from=none` when the run has no checkpoint yet
and training starts from scratch. `--dependency` adds an `#SBATCH --dependency`
directive, so the follow-up is queued before its predecessor finishes.

Reuse the run ID and the existing snapshot, and repeat the trainer arguments of
the first job verbatim:

```bash
cluster/cscs/submit.sh \
  --profile production --partition normal --account d130 --time 24:00:00 \
  --dataset-path /capstor/scratch/cscs/lterenzi/terra-training/datasets/DATASET \
  --dataset-size 1440 \
  --run-id RUN_ID \
  --no-sync \
  --wandb-mode offline \
  --resume-latest \
  --dependency afterany:PREVIOUS_JOB_ID \
  --submit \
  -- TRAIN_ARGS...
```

Chain the next job on the one just submitted. Each submission rewrites
`job.sbatch` and `terra.edf.toml` in the run root; copy them aside first if a
predecessor's exact files must be kept.
