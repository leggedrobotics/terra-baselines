# Running Experiments

## 2026-05-12 Ringmaps Two-GPU 4090 Mask A/B

Active Slurm jobs:

- `66397924`: online W&B four-GPU RTX 4090 no-mask baseline from current `multi-agent` worktree.
  - Script: `scripts/terra_train_ringmaps_4gpu4090_baseline.sbatch`
  - Shape: `4` RTX 4090s, `1024` envs/GPU, `4096` envs total, `num_steps=32`, `total_timesteps=50000000000`.
  - Masking: explicit `--disable_action_mask`.
  - Checkpointing/eval: every `100` updates, `eval_episodes=100`.
  - W&B: online, entity `aless-weber-eth`, project `mixed-agents`.
  - Logs: `/cluster/home/lterenzi/codex_terra_edge_validation/logs/66397924_ring_4gpu4090_baseline.out`
  - Status at submission: pending on priority in `gpuhe.120h`.

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
