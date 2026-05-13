# Experiment Log

## 2026-05-12 Ringmaps Two-GPU 4090 Mask A/B

Goal: test whether the fixed no-mask default path and the experimental coarse-mask path can run the pretrained ringmaps recipe on exactly two RTX 4090 GPUs.

Setup:

- Synced local `terra` and `terra-baselines` `multi-agent` worktrees to `/cluster/home/lterenzi/codex_terra_edge_validation`.
- Remote gates passed in the Euler workspace:
  - `python -m py_compile` for changed Python files.
  - `scripts/validate_edge_mask_changes.py --case ppo-mask --jax-platforms cpu`
  - `scripts/validate_edge_mask_changes.py --case model-policy --jax-platforms cpu`
  - `scripts/validate_edge_mask_changes.py --case training-accounting --jax-platforms cpu`
- Submitted initial offline Slurm array `66301608_[0-1]` with `scripts/terra_train_ringmaps_2gpu4090_mask_ab.sbatch`.
- Each task requests `--gpus=gpu:nvidia_geforce_rtx_4090:2`, runs CUDA/cuDNN/NCCL preflight with `--min-devices 2`, then trains `train_mixed.py --config solo_excavator`.
- Training shape: `2` GPUs, `1024` envs/GPU, `num_steps=32`, `update_epochs=2`, `num_minibatches=16`, `total_timesteps=6553600` (`100` updates).
- Task `0`: `--enable_action_mask`.
- Task `1`: `--disable_action_mask`.
- Offline results:
  - `66301608_0` mask completed, elapsed `00:19:52`, final warm throughput about `61.7k` env steps/s.
  - `66301608_1` no-mask completed, elapsed `00:19:29`, final warm throughput about `61.7k` env steps/s.
  - These runs used `WANDB_MODE=offline`, so they are useful runtime evidence but not live W&B evidence.
- Updated the script to `WANDB_MODE=online`, `WANDB_ENTITY=aless-weber-eth`, `WANDB_PROJECT=mixed-agents`, and run names `terra-ringmaps-2gpu4090-{mask,nomask}-pretrained-online-ab`.
- Submitted online Slurm array `66308783_[0-1]`, then cancelled it while still pending because the requested run should be full-length rather than a 100-update A/B.
- Added full-length online script `scripts/terra_train_ringmaps_2gpu4090_full_mask_ab.sbatch`.
- Submitted full-length online Slurm array `66311700_[0-1]`:
  - Partition/time: `gpuhe.120h`, `10-00:00:00`.
  - Shape: `2` RTX 4090s, `1024` envs/GPU, `num_steps=32`, `update_epochs=2`, `num_minibatches=16`.
  - Horizon: `total_timesteps=50000000000`.
  - Checkpoint/eval cadence: every `100` updates.
  - Task `0`: `--enable_action_mask`; task `1`: `--disable_action_mask`.

Pending:

- Wait for full online tasks `66311700_[0-1]` to start, pass runtime preflight, and complete at least update 1 before treating either online run as healthy.
- Compare W&B run ids, first-update health, warm steps/s, learning metrics, checkpoint cadence, and whether the masked path remains stable at this two-GPU 4090 full-training shape.

Update:

- Cancelled `66311700_[0-1]` after about `16h36m` because both runs were advancing but still showed the no-success learning signature after about `2B` env steps.
  - Mask: about `31k` updates, `eval/positive_terminations=0`, `eval/rewards=0.0034`, `eval/max_reward=0.86`, `eval/DO_NOTHING %=0.52`.
  - No-mask: about `30k` updates, `eval/positive_terminations=0`, `eval/rewards=0.00044`, `eval/max_reward=0.91`, `eval/DO_NOTHING %=0.58`.
- Added `scripts/terra_train_ringmaps_4gpu4090_baseline.sbatch` to launch a current-branch no-mask baseline with the historical 4-GPU shape.
- Submitted baseline Slurm job `66397924`.
  - Shape: `4` RTX 4090s, `1024` envs/GPU, `4096` envs total, `total_timesteps=50000000000`.
  - Masking: explicit `--disable_action_mask`.
  - W&B: online, `aless-weber-eth/mixed-agents`.
  - Status at submission: pending on priority in `gpuhe.120h`.

## 2026-05-12 Ringmaps Mask Parity and Training A/B

Goal: collect PR evidence for the action-mask and speedup changes by checking old ringmaps policy parity with and without masks, then running a short pretrained training A/B with the historical `solo_excavator` recipe.

Setup:

- Synced current local `multi-agent` worktrees for `terra` and `terra-baselines` to `/cluster/home/lterenzi/codex_terra_edge_validation`.
- Synced the two downloaded checkpoints to `/cluster/home/lterenzi/codex_terra_edge_validation/checkpoints`.
- Historical recipe source: `/home/lorenzo/Downloads/train_cluster.sh`.
- Historical checkpoint inputs:
  - `/home/lorenzo/Downloads/solo-excavator-2026-04-21-13-36-45-ringmaps.pkl`
  - `/home/lorenzo/Downloads/solo-excavator-2026-04-22-10-20-06-ringmaps-hongg-archi3-finetuned.pkl`
- Default training shape: `4` GPUs, `1024` envs/GPU, `num_steps=32`, `update_epochs=2`, `num_minibatches=16`, `DATASET_SIZE=600`.

Results:

- `66275406`: first parity submission with `DATASET_SIZE=600`, `num_envs=16`, `num_steps=128`.
  Passed one-GPU CUDA runtime preflight, then was cancelled because map loading was too slow for a quick gate.
- `66277765`: smaller parity submission with `DATASET_SIZE=64`, `num_envs=8`, `num_steps=64`.
  Passed one-GPU CUDA preflight but failed behavioral parity.
  The 2026-04-21 checkpoint diverged at step `16`; the 2026-04-22 finetuned checkpoint diverged at step `11`.
  Conclusion: do not claim masked replay parity for these legacy ringmaps policies.
- `66279187_1`: default-shape no-mask training smoke.
  Passed 4-GPU CUDA/NCCL preflight, loaded the 2026-04-21 checkpoint and all `600` ringmaps, replaced model parameters, and completed `100/100` updates in `00:17:02`.
  Warm throughput after the first 10 updates was about `74.9k` env steps/s.
- `66279187_0`: default-shape mask training smoke.
  Passed 4-GPU CUDA/NCCL preflight, loaded the same checkpoint and maps, but failed before update 1 with `CUDNN_STATUS_INTERNAL_ERROR`.
- `66281228_0`: default-shape mask retry on the node that had just completed the no-mask job.
  Reproduced `CUDNN_STATUS_INTERNAL_ERROR`.
- `66282813_0`: default-shape mask retry after changing masked logits from `-inf` to a finite floor.
  Local and Euler CPU `ppo-mask` gates passed, but the job still reproduced `CUDNN_STATUS_INTERNAL_ERROR`.
- `66284065`: one-GPU masked isolation smoke with `128` envs/GPU, `64` ringmaps, `2` updates.
  Completed.
- `66286333`: four-GPU masked isolation smoke with `128` envs/GPU, `64` ringmaps, `2` updates.
  Completed.

Conclusion:

- The no-mask default-shape path is the merge candidate for the speedup PR.
- The coarse action-mask path remains experimental until the `4 x 1024` envs/GPU masked PPO update-0 cuDNN failure is understood.
