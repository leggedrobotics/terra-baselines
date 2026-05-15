# Running Experiments

## 2026-05-15 ResMap64 Four-GPU Architecture Run

Active Slurm jobs:

- `66574910`: online W&B four-GPU residual-map architecture run from paired `codex/mask-speedup-wip` worktrees.
  - Script: `scripts/euler/terra_train_resmap64_phase_4gpu.sbatch`
  - Shape: `4` GPUs requested, `1024` envs/GPU, `4096` envs total, `num_steps=32`, `total_timesteps=50000000000`.
  - Architecture: `--map_encoder resnet_delayed --map_feature_dim 128 --use_map_derived_channels --separate_actor_critic_trunks`.
  - Masking: explicit `--disable_action_mask`; this is an architecture run, not a mask run.
  - PPO minibatching: `num_minibatches=32` to keep the residual-map gradient batch at `1024` map frames instead of the failed `2048`.
  - Eval/checkpointing: every `100` updates, `eval_episodes=100`, `num_rollouts_eval=550`.
  - W&B: online, entity `aless-weber-eth`, project `mixed-agents`.
  - CUDA gate: script runs `scripts/euler/check_jax_runtime.py --min-devices 4` before any training.
  - Smoke gate: script runs a one-update W&B-disabled full-shape resmap smoke before the full W&B run.
  - GPU guard: script hard-fails unless `nvidia-smi` reports exactly four RTX 3090/4090 GPUs.
  - Node restriction: `eu-g4-[001-032],eu-g6-[001-080]`.
  - Logs: `/cluster/home/lterenzi/codex_terra_edge_validation/logs/66574910_resmap64_phase_4gpu.out`.
  - Local 4090 smoke: passed on `2026-05-15 09:36 CEST` with the same per-GPU shape, `num_devices=1`, `1024` envs, `num_steps=32`, `num_minibatches=32`; one update completed in `176.15s` at `194.41` steps/s and wrote `checkpoints/terra-solo-resmap64-phase-mb32-local4090-smoke-local4090-2026-05-15-09-33-19_FINAL.pkl`.
  - Status on `2026-05-15 09:38 CEST`: running on `eu-g6-061`, elapsed about `51m`, Slurm allocation `4 x NVIDIA GeForce RTX 4090`.
  - W&B run: `d1dkjojl`, `https://wandb.ai/aless-weber-eth/mixed-agents/runs/d1dkjojl`.
  - Euler health: GPU guard passed, CUDA/cuDNN/NCCL preflight passed, full-shape W&B-disabled smoke passed in `422.93s` at `310.04` steps/s, and the full online run is past update `300`.
  - Log note: W&B emits `ERROR:root:Driver not initialized (amdgpu not found in modules)` from system monitoring; this is not a training failure.

Recent failed predecessor:

- `66537274`: same architecture with `num_minibatches=16`.
  - Allocation: `eu-g4-030`, `4 x NVIDIA GeForce RTX 3090`.
  - Result: CUDA/JAX/cuDNN/NCCL preflight passed and the small one-update smoke passed, but the full online run failed on the first PPO update with GPU OOM.
  - Fix applied in `66574910`: use `num_minibatches=32` and make the smoke use the full `1024` envs/GPU shape.

## 2026-05-14 Masked Four-GPU Full Run

Active Slurm jobs:

- `66536725`: online W&B four-GPU RTX 4090 masked full run from `codex/mask-speedup-wip`.
  - Script: `scripts/euler/terra_train_mask_4gpu_full.sbatch`
  - Shape: `4` RTX 4090s requested, `1024` envs/GPU, `4096` envs total, `num_steps=32`, `total_timesteps=50000000000`.
  - Masking: explicit `--enable_action_mask`.
  - Checkpointing/eval: every `100` updates, `eval_episodes=100`.
  - W&B: online, entity `aless-weber-eth`, project `mixed-agents`.
  - CUDA gate: script runs `scripts/euler/check_jax_runtime.py --min-devices 4` before training.
  - GPU guard: script hard-fails unless `nvidia-smi` reports exactly four `NVIDIA GeForce RTX 4090` GPUs.
  - Node restriction: `eu-g6-[001-080]`.
  - Logs: `/cluster/home/lterenzi/codex_terra_edge_validation/logs/66536725_mask_4gpu_full.out` and `.err`.
  - Status on 2026-05-15 08:51 CEST: running on `eu-g6-059`, elapsed about `8h`.
  - W&B run: `ti3k3tdp`, `https://wandb.ai/aless-weber-eth/mixed-agents/runs/ti3k3tdp`.
  - Latest checkpoint benchmarked: `terra-mask-multiagent-4gpu-online-euler-pr-2026-05-15-00-50-06.pkl`.
  - Current W&B summary at step `15998`: `eval/success_rate=0.302`, `eval/max_reward=7.08`, `performance/steps_per_second=97395`.

## 2026-05-12 Ringmaps Mask Parity and Training A/B

No Slurm job from this older ledger is currently expected to remain active.

Historical workspace:

- Terra workspace: `/cluster/home/lterenzi/codex_terra_edge_validation`
- Euler venv: `/cluster/scratch/lterenzi/codex_terra_edge_venv`
- Dataset: `/cluster/project/rsl/alesweber/TerraProject/terra/data/terra/train`
- Historical dataset size: `DATASET_SIZE=600`
- Historical trainer: `train_mixed.py --config solo_excavator`
- Historical hardware: `4` GPUs
- Historical envs: `1024` envs/GPU, `4096` envs total

Completed historical jobs:

- `66301608_[0-1]`: offline two-GPU RTX 4090 pretrained ringmaps A/B.
  Status: both tasks completed successfully; mask elapsed `00:19:52`, no-mask elapsed `00:19:29`.
- `66308783_[0-1]`: online two-GPU RTX 4090 100-update rerun.
  Status: cancelled while pending because the next run should be full-length.
- `66311700_[0-1]`: online two-GPU RTX 4090 full A/B.
  Status: cancelled after about `16h36m`; both jobs were advancing but still showed the no-success learning signature after about `2B` env steps.
- `66277765`: one-GPU checkpoint parity gate, `DATASET_SIZE=64`, `num_envs=8`, `num_steps=64`, seed `123`.
  Status: failed parity because both legacy checkpoints eventually selected actions blocked by the new coarse mask.
- `66279187_0`: `--enable_action_mask`, `4` GPUs, `1024` envs/GPU, `100` updates.
  Status: failed before update 1 with `CUDNN_STATUS_INTERNAL_ERROR`.
- `66279187_1`: `--disable_action_mask`, `4` GPUs, `1024` envs/GPU, `100` updates.
  Status: completed `100/100` updates in `00:17:02`; warm throughput after update 10 was about `74.9k` env steps/s.
- `66281228_0`: same-default mask retry on a different node.
  Status: reproduced `CUDNN_STATUS_INTERNAL_ERROR`.
- `66282813_0`: same-default mask retry after finite masked-logit floor.
  Status: reproduced `CUDNN_STATUS_INTERNAL_ERROR`.
- `66284065`: one-GPU masked isolation smoke, `128` envs/GPU, `64` ringmaps, `2` updates.
  Status: completed.
- `66286333`: four-GPU masked isolation smoke, `128` envs/GPU, `64` ringmaps, `2` updates.
  Status: completed.
