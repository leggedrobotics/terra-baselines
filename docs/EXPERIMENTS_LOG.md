# Experiment Log

## 2026-05-14 ResMap64 Four-GPU Architecture Run

Goal: launch a stronger but still PPO-compatible architecture for the solo-excavator ringmaps task, without changing the primitive action interface or enabling the experimental action mask.

Architecture selected after repo/research review:

- Delayed-downsample residual global-map encoder: stride-1 stem, residual blocks before downsampling, two late stride-2 stages, global average/max pooling, `128` map features.
- Derived global-map channels for current residual/error/dirt state, built from existing observation maps.
- Separate actor and critic trunks after the shared feature encoder.
- Existing scalar categorical primitive action head and PPO objective retained.

Setup:

- Edited `utils/models.py` and `train_mixed.py` in `terra-baselines_mask_wip`.
- Added `scripts/euler/terra_train_resmap64_phase_4gpu.sbatch`.
- Synced paired local WIP worktrees to `/cluster/home/lterenzi/codex_terra_edge_validation`:
  - `/home/lorenzo/moleworks/terra_mask_wip` -> `terra`
  - `/home/lorenzo/moleworks/terra-baselines_mask_wip` -> `terra-baselines`
- Local CPU gates before submission:
  - `python -m py_compile train_mixed.py utils/models.py utils/utils_ppo.py eval_ppo.py scripts/validation/validate_edge_mask_changes.py`
  - `scripts/validation/validate_edge_mask_changes.py --case training-accounting --jax-platforms cpu`
  - `--case ppo-mask --jax-platforms cpu`
  - `--case model-policy --jax-platforms cpu`
  - `--case state-action-mask --jax-platforms cpu --disable-jit`
  - `--case state-step-dispatch --jax-platforms cpu --disable-jit`
  - `--case synthetic-env-action-mask --jax-platforms cpu --disable-jit`
  - `--case env-action-mask --jax-platforms cpu --dataset-path /home/lorenzo/moleworks/terra_data/train --dataset-size 1`
  - `--case synthetic-step-fast-reset --jax-platforms cpu --disable-jit`
  - `--case synthetic-batch-step-fast-reset --jax-platforms cpu`
  - Direct resmap model-init smoke: `534,873` parameters on the local one-map dataset.
- Remote Euler `py_compile` passed after sync.
- First submission with the script's generic `#SBATCH --gpus=4` was rejected by Slurm as unavailable.
- Resubmitted with typed GPU request as job `66537274`; `scontrol` shows `PENDING` on `gpuhe.120h`, one node, four GPUs, node list constrained to `eu-g4-[001-032],eu-g6-[001-080]`.

Update 2026-05-15:

- Job `66537274` started on `eu-g4-030` with `4 x NVIDIA GeForce RTX 3090`.
- Runtime preflight passed:
  - CUDA library paths exported.
  - JAX saw 4 GPU devices.
  - jitted cuDNN conv backward completed.
  - NCCL pmap all-reduce completed.
- The first W&B-disabled smoke passed, but it used a small `8` envs/GPU shape and did not catch full-update memory.
- The full online run failed on the first PPO update with GPU OOM:
  - `num_minibatches=16` produced a residual-map update batch with `2048` map frames through the full-resolution stem.
  - XLA reported OOM while allocating about `15.3 GB`; peak buffers were dominated by `f32[2048,16,64,64]` residual-map activations.
- Applied fix:
  - Kept `1024` envs/GPU.
  - Changed `num_minibatches` from `16` to `32`, reducing the update map batch from `2048` to `1024`.
  - Changed the W&B-disabled smoke to use the full `1024` envs/GPU, `num_minibatches=32`, one-update shape.
- Submitted replacement job `66574910`.
  - Status at submission: pending on priority in `gpuhe.120h`.
  - `sbatch --test-only` resolved to a four-GPU node.

Update 2026-05-15 09:38 CEST:

- Local RTX 4090 full-shape smoke passed before treating the run as healthy:
  - Command shape: `num_devices=1`, `1024` envs/GPU, `num_steps=32`, `num_minibatches=32`, `update_epochs=2`, W&B disabled.
  - Architecture flags: `--map_encoder resnet_delayed --map_feature_dim 128 --use_map_derived_channels --separate_actor_critic_trunks --disable_action_mask`.
  - Result: one update completed in `176.15s` at `194.41` steps/s.
  - Checkpoint: `checkpoints/terra-solo-resmap64-phase-mb32-local4090-smoke-local4090-2026-05-15-09-33-19_FINAL.pkl`.
- Replacement job `66574910` started on `eu-g6-061`; Slurm `AllocTRES` reports `gres/gpu:nvidia_geforce_rtx_4090=4`.
- Euler runtime health:
  - GPU guard saw `4 x NVIDIA GeForce RTX 4090`.
  - CUDA/cuDNN/NCCL preflight passed.
  - Full-shape W&B-disabled smoke passed in `422.93s` at `310.04` steps/s.
  - Full online run is past update `300` with no OOM or traceback in the checked log.
- W&B run id: `d1dkjojl`, `https://wandb.ai/aless-weber-eth/mixed-agents/runs/d1dkjojl`.
- The W&B/system-monitoring message `ERROR:root:Driver not initialized (amdgpu not found in modules)` appears in the log, but training is continuing normally.

## 2026-05-14 Masked Four-GPU Full Run

Goal: launch the experimental coarse action-mask path as a full default solo-excavator training run on Euler, matching the four-GPU baseline shape while logging online to W&B.

Setup:

- Synced local `terra_mask_wip` and `terra-baselines_mask_wip` to `/cluster/home/lterenzi/codex_terra_edge_validation`.
- Included the observation-mutation fix in `utils/utils_ppo.py` so policy input clipping no longer mutates rollout observations.
- Local CPU gates before submission:
  - `python -m py_compile utils/utils_ppo.py train_mixed.py scripts/validation/validate_edge_mask_changes.py`
  - `scripts/validation/validate_edge_mask_changes.py --case ppo-mask --jax-platforms cpu`
  - `--case training-accounting --jax-platforms cpu`
  - `--case model-policy --jax-platforms cpu`
  - `--case state-action-mask --jax-platforms cpu --disable-jit`
  - `--case state-step-dispatch --jax-platforms cpu --disable-jit`
  - `--case synthetic-env-action-mask --jax-platforms cpu --disable-jit`
  - `--case env-action-mask --jax-platforms cpu --dataset-path /home/lorenzo/moleworks/terra_data/train --dataset-size 1`
  - `--case synthetic-step-fast-reset --jax-platforms cpu --disable-jit`
  - `--case synthetic-batch-step-fast-reset --jax-platforms cpu`
- Added `scripts/euler/terra_train_mask_4gpu_full.sbatch`.
  - Partition/time: `gpuhe.120h`, `10-00:00:00`.
  - GPU request: `--gpus=gpu:nvidia_geforce_rtx_4090:4`.
  - Node restriction: `--nodelist=eu-g6-[001-080]`.
  - Runtime guard: exits before JAX if the allocation is not exactly four `NVIDIA GeForce RTX 4090` GPUs.
  - Runtime preflight: `scripts/euler/check_jax_runtime.py --min-devices 4` checks CUDA library paths, JAX GPU visibility, jitted conv backward, and NCCL all-reduce.
  - Training: `train_mixed.py --config solo_excavator --num_devices 4 --num_envs_per_device 1024 --num_steps 32 --update_epochs 2 --num_minibatches 16 --total_timesteps 50000000000 --enable_action_mask`.
  - W&B: `WANDB_MODE=online`, `WANDB_ENTITY=aless-weber-eth`, `WANDB_PROJECT=mixed-agents`.
- Slurm dry-run resolved to a four-GPU RTX 4090 node.
- Submitted job `66536523`, then cancelled it before start because `squeue --start` predicted an `eu-g3` node when the first script lacked a node restriction.
- Added the explicit `eu-g6-[001-080]` node restriction, reran `sbatch --test-only`, and resubmitted as job `66536725`.

Pending:

- Wait for `66536725` to start.
- Verify `nvidia-smi` GPU guard, CUDA/NCCL preflight, W&B run URL, and completion of update 1 before treating the run as healthy.

Update 2026-05-15:

- Job `66536725` started on `eu-g6-059`; Slurm `AllocTRES` reports `gres/gpu:nvidia_geforce_rtx_4090=4`.
- W&B run id: `ti3k3tdp`, `https://wandb.ai/aless-weber-eth/mixed-agents/runs/ti3k3tdp`.
- CUDA/NCCL preflight passed and training is well past update 1.
- Latest checkpoint pulled locally:
  `/home/lorenzo/moleworks/terra-baselines_mask_wip/checkpoints/terra-mask-multiagent-4gpu-online-euler-pr-2026-05-15-00-50-06.pkl`.
- Local deterministic benchmark:
  `scripts/analysis/benchmark_checkpoint_masked.py ... --config solo_excavator --num-envs 16 --num-steps 550 --seed 0 --modes masked unmasked`.
  Results:
  - Masked rollout: `2/16` success, `avg_return=2.390`, `max_return=9.959`, `invalid_selected_rate=0.0`, `dig_coverage=0.774`, `dump_coverage=0.935`.
  - Raw unmasked logits: `0/16` success, `avg_return=-2.819`, `invalid_selected_rate=0.599`, `dig_coverage=0.462`, `dump_coverage=0.874`.
- Current W&B summary around update `15998`:
  `eval/success_rate=0.302`, `eval/positive_terminations=0.302`, `eval/max_reward=7.08`, `eval/rewards=0.0079`, `eval/DO_NOTHING %=0.668`, `performance/steps_per_second=97395`.
- Control comparison on 2026-05-15:
  - Live control run `66438404` / W&B `tkzxeaas` is still running on `eu-g4-031` with `4x RTX 3090`.
  - Live masked run `66536725` / W&B `ti3k3tdp` is still running on `eu-g6-059` with `4x RTX 4090`.
  - W&B summary: control at update `35973` has `eval/positive_terminations=0.436`, `eval/rewards=0.0336`, `eval/max_reward=7.07`, `eval/DO_NOTHING %=0.235`, `eval/DO=0.170`, `steps/s=217375`.
  - W&B summary: masked at update `17412` has `eval/success_rate=0.352`, `eval/rewards=0.0088`, `eval/max_reward=7.08`, `eval/DO_NOTHING %=0.675`, `eval/DO=0.062`, `steps/s=97959`.
  - Same local deterministic benchmark, `16` envs, `550` steps, seed `0`:
    - Control checkpoint intended unmasked mode: `1/16` success, `avg_return=1.400`, `invalid_selected_rate=0.0395`.
    - Control checkpoint with forced mask: `1/16` success, `avg_return=1.306`, `invalid_selected_rate=0.0`.
    - Masked checkpoint intended masked mode: `2/16` success, `avg_return=2.390`, `invalid_selected_rate=0.0`.
    - Masked checkpoint with raw unmasked logits: `0/16` success, `avg_return=-2.819`, `invalid_selected_rate=0.599`.

## 2026-05-12 Ringmaps Two-GPU 4090 Mask A/B

Goal: test whether the fixed no-mask default path and the experimental coarse-mask path can run the pretrained ringmaps recipe on exactly two RTX 4090 GPUs.

Setup:

- Synced local `terra` and `terra-baselines` `multi-agent` worktrees to `/cluster/home/lterenzi/codex_terra_edge_validation`.
- Remote gates passed in the Euler workspace:
  - `python -m py_compile` for changed Python files.
  - `scripts/validation/validate_edge_mask_changes.py --case ppo-mask --jax-platforms cpu`
  - `scripts/validation/validate_edge_mask_changes.py --case model-policy --jax-platforms cpu`
  - `scripts/validation/validate_edge_mask_changes.py --case training-accounting --jax-platforms cpu`
- Submitted initial offline Slurm array `66301608_[0-1]` with an earlier one-off 2-GPU mask A/B Slurm script.
- Each task requests `--gpus=gpu:nvidia_geforce_rtx_4090:2`, runs CUDA/cuDNN/NCCL preflight with `--min-devices 2`, then trains `train_mixed.py --config solo_excavator`.
- Training shape: `2` GPUs, `1024` envs/GPU, `num_steps=32`, `update_epochs=2`, `num_minibatches=16`, `total_timesteps=6553600` (`100` updates).
- Task `0`: `--enable_action_mask`.
- Task `1`: `--disable_action_mask`.
- Offline results:
  - `66301608_0` mask completed, elapsed `00:19:52`, final warm throughput about `61.7k` env steps/s.
  - `66301608_1` no-mask completed, elapsed `00:19:29`, final warm throughput about `61.7k` env steps/s.
  - These runs used `WANDB_MODE=offline`, so they are useful runtime evidence but not live W&B evidence.
- Updated that one-off script to `WANDB_MODE=online`, `WANDB_ENTITY=aless-weber-eth`, `WANDB_PROJECT=mixed-agents`, and run names `terra-ringmaps-2gpu4090-{mask,nomask}-pretrained-online-ab`.
- Submitted online Slurm array `66308783_[0-1]`, then cancelled it while still pending because the requested run should be full-length rather than a 100-update A/B.
- Added a full-length online one-off Slurm script for the 2-GPU A/B.
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
- Added a one-off 4-GPU Slurm script to launch a current-branch no-mask baseline with the historical 4-GPU shape.
- Submitted baseline Slurm job `66397924`.
  - Shape: `4` RTX 4090s, `1024` envs/GPU, `4096` envs total, `total_timesteps=50000000000`.
  - Masking: explicit `--disable_action_mask`.
  - W&B: online, `aless-weber-eth/mixed-agents`.
  - Status at submission: pending on priority in `gpuhe.120h`.
- Cleanup note: the one-off ringmaps Slurm scripts above were retired before PR handoff. The
  reusable Euler scripts kept in this branch are the current masked full run and resmap run.

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
