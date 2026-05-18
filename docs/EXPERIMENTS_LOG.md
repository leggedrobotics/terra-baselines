# Experiment Log

## 2026-05-18 Larger ResNet Distillation Setup

Goal: launch larger delayed-ResNet policies and test whether teacher imitation
from the trained ResMap64 policy stabilizes larger PPO.

Changes:

- Added explicit delayed-ResNet capacity scaling to `model_size=medium` and
  `model_size=large`; the previous presets widened heads/trunks but left the
  delayed ResNet itself fixed.
- Added optional teacher-policy imitation warm start to `train_mixed.py`.
  The teacher rolls out actions, the student matches teacher logits with KL and
  teacher values with an auxiliary MSE, then the Adam optimizer is reset before
  PPO starts.
- Oracle review required tightening the experiment before launch:
  - PPO now restores the pre-imitation RNG/env/previous-action state so
    distill-vs-scratch tests parameter warm start only.
  - The final imitation metrics are logged with the first PPO row, and
    `_POST_DISTILL.pkl` is saved.
  - Teacher/student `num_prev_actions` and parameter-shape compatibility are
    checked before distillation.
  - Unknown CLI args now fail, and PPO minibatch divisibility is validated.
  - Slurm smoke now covers tiny eval and checkpoint paths before online runs.
- Added `scripts/euler/terra_train_larger_resnet_4gpu.sbatch`, parameterized by
  `RUN_KIND=medium_distill`, `medium_scratch`, `large_distill`, or
  `large_scratch`.
- Added `docs/LARGER_RESNET_DISTILLATION_PLAN.md` with the experiment matrix,
  local memory findings, and decision criteria.

Local validation:

- `python -m py_compile train_mixed.py utils/models.py utils/utils_ppo.py`
  passed.
- `bash -n scripts/euler/terra_train_larger_resnet_4gpu.sbatch` passed.
- `git diff --check` passed.
- Tiny CPU imitation+PPO smoke passed before and after the Oracle fixes; the
  post-fix run saved both `_POST_DISTILL.pkl` and `_FINAL.pkl`.
- RTX 4090 medium smoke:
  - `model_size=medium`, `map_feature_dim=192`, `1024` envs/GPU,
    `num_steps=32`.
  - Failed with `num_minibatches=32` due to ResNet activation OOM.
  - Passed with `num_minibatches=64`; one imitation update and one PPO update
    completed.
- RTX 4090 large smoke:
  - `model_size=large`, `map_feature_dim=256`, `1024` envs/GPU,
    `num_steps=32`, `num_minibatches=128`.
  - Failed without autotune mitigation with `CUDNN_STATUS_INTERNAL_ERROR`.
  - Passed with `XLA_FLAGS=--xla_gpu_autotune_level=0`; one imitation update
    and one PPO update completed.

Planned Euler jobs:

- `medium_distill`: medium ResNet, `64` minibatches, `200` imitation updates.
- `medium_scratch`: same medium ResNet and minibatches, no teacher; this is the
  warm-start control.
- `large_distill`: large ResNet, `128` minibatches, `100` imitation updates,
  cuDNN autotune disabled.
- `large_scratch`: same large ResNet/autotune setup, no teacher; this is the
  large-model warm-start control.

Launch:

- Synced paired WIP worktrees and the teacher checkpoint to
  `/cluster/home/lterenzi/codex_terra_edge_validation`.
- Remote Euler preflight passed:
  - `py_compile train_mixed.py utils/models.py utils/utils_ppo.py`
  - `bash -n scripts/euler/terra_train_larger_resnet_4gpu.sbatch`
  - teacher checkpoint exists under `$WORK/checkpoints`
- `sbatch --test-only` resolved all four run kinds to a four-GPU node.
- Submitted:
  - `66965292` `terra-medium-distill`
  - `66965294` `terra-medium-scratch`
  - `66965296` `terra-large-distill`
  - `66965300` `terra-large-scratch`
- Initial Slurm state: all four `PENDING (Priority)` in `gpuhe.120h` with
  `4` GPUs requested.

Update 2026-05-18 15:32 CEST:

- Cancelled the four-GPU matrix because the next useful step is a cheaper
  single-GPU calibration before spending four GPUs on each larger model.
  - `66965292` had started on `eu-g4-027` with `4 x RTX 3090` and was
    cancelled after `00:06:23`, during the first imitation smoke compile; no
    online W&B run was produced.
  - `66965294`, `66965296`, and `66965300` were cancelled while pending.
- Added `scripts/euler/terra_train_larger_resnet_1gpu_4h.sbatch`.
  - `RUN_KIND` is required so a plain `sbatch` cannot silently pick a variant.
  - The script requests one RTX 3090 in `gpuhe.4h`, hard-fails unless exactly
    one allowed GPU is allocated, runs `check_jax_runtime.py --min-devices 1`,
    and runs a W&B-disabled one-update smoke before online W&B.
  - Online calibration defaults to `1B` env steps with eval/checkpoint every
    `50` PPO updates; the four-hour queue will bound wall time if the run does
    not finish.
- Remote `bash -n` passed, and `sbatch --test-only` resolved all four run
  kinds to a one-GPU RTX 3090 node in `gpuhe.4h`.
- Submitted one-GPU `gpuhe.4h` calibration jobs:
  - `66969658` `terra1g-medium-distill`, `imitation_updates=200`.
  - `66969660` `terra1g-medium-scratch`.
  - `66969663` `terra1g-large-distill`, `imitation_updates=200`.
  - `66969665` `terra1g-large-scratch`.
- Initial state: all four `PENDING (Priority)` with no start estimate from
  `squeue --start`.

Update 2026-05-18 15:47 CEST:

- All four one-GPU `gpuhe.4h` calibration jobs started:
  - `66969658` on `eu-g4-015`, `1 x RTX 3090`.
  - `66969660` on `eu-g4-030`, `1 x RTX 3090`.
  - `66969663` on `eu-g4-027`, `1 x RTX 3090`.
  - `66969665` on `eu-g4-025`, `1 x RTX 3090`.
- The hard GPU guard and `check_jax_runtime.py --min-devices 1` passed for
  all four jobs.
- Current state: the W&B-disabled full-shape smoke is compiling/running. The
  distill variants have started their one-update teacher imitation warm-start
  smoke; no online W&B run has started yet.

## 2026-05-18 Default-Unmasked PPO Cleanup

Goal: make the ResMap path the default unmasked-actor path and remove
backwards-compatibility plumbing that could silently re-enable actor action
masking from an old checkpoint.

Changes:

- Removed `use_action_mask` from PPO train configs and policy calls.
- Removed `--enable_action_mask`, `--disable_action_mask`, and the tri-state
  `action_mask_cli_override`.
- Stopped restoring action-mask state from checkpoint train configs.
- Removed legacy observation-layout fallback and legacy edge-width inference;
  checkpoint model-shape fields are now required.
- Kept `action_mask` in the observation layout and kept `apply_action_mask` for
  diagnostics/analysis scripts, not PPO training/eval.
- Removed the masked four-GPU Euler launcher and dropped the redundant
  `--disable_action_mask` from the ResMap launcher.
- Ensured edge/progress affordances are critic-only when
  `--use_critic_affordances` is active; they no longer enter the actor through
  the old shared-affordance branch.

Validation:

- `python -m py_compile` passed for train/eval/visualization/inference/model
  helpers and validation/analysis scripts.
- Focused CPU gates passed: `training-accounting`, `model-policy`,
  `model-edge-no-mask`, `model-critic-affordance-shapes`,
  `checkpoint-config-restore`, `reward-logging-accounting`,
  `timeout-bootstrap-value`, `gae-timeout-bootstrap`, `ppo-mask`, and
  `synthetic-batch-step-fast-reset`.
- Full CPU validation passed:
  `validate_edge_mask_changes.py --case all --jax-platforms cpu --dataset-path /home/lorenzo/moleworks/terra_data/train --dataset-size 1`.

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

Update 2026-05-15 11:48 CEST:

- Cancelled replacement job `66574910` intentionally after `03:01:32` on `eu-g6-061`; Slurm reports `CANCELLED by 542598`.
- Reason: stop the partial ResMap-only ablation and relaunch after adding the critic affordance and timeout-spike fixes together.
- Last diagnostic read before cancellation:
  - W&B run `d1dkjojl` was around update `998`.
  - Eval positive terminations and total terminations were still `0`.
  - Value-loss spikes appeared periodically around updates such as `964`, `981`, and `998`, matching the `550 / 32 ~= 17` update episode-horizon cadence.
  - These spikes looked like timeout/truncation target shocks rather than successful-task terminal shocks; the run was too early to judge learning quality.
- Next run plan:
  - Detailed plan: `docs/RESMAP_AFFORDANCE_TIMEOUT_PLAN.md`.
  - Keep ResMap64: delayed-downsample residual global-map encoder, derived global-map channels, separate actor/critic trunks, `num_minibatches=32`.
  - Add critic-only affordance conditioning using the existing `10` env-computed `edge_features`.
  - Add `episode_progress = env_steps / max_steps_in_episode`.
  - Add training diagnostics for `task_done_rate`, `timeout_rate`, value loss by timeout bucket, and explained variance by timeout bucket.
  - Fix GAE time-limit semantics: bootstrap through max-step timeouts, but not through true `task_done` success.
  - Defer actor-visible affordances to a later ablation and only use robot-computable signals there.
  - Gate the relaunch with CPU checks and a local RTX 4090 full-shape smoke before submitting the four-GPU Euler job.

Update 2026-05-15 13:31 CEST:

- Implemented the combined R1/R2 local patch before relaunch:
  - Critic-only affordances: existing `edge_features` plus `episode_progress`, width `11`, appended only after the actor/critic split.
  - Timeout handling: preserve `final_observation` through env reset and bootstrap value targets from that pre-reset observation on max-step truncations.
  - GAE semantics: bootstrap max-step timeouts, zero-bootstrap true task terminals, and stop recursive GAE at reset boundaries to avoid reset-episode leakage.
  - Diagnostics: added done/task_done/timeout rates, episode progress, timeout-bucket value loss/explained variance, and affordance summary metrics.
  - Checkpoint tools: eval, visualization, inference, parity, and analysis scripts now preserve saved R1/R2 model-shape fields instead of re-inferring edge width from params.
- Validation:
  - `python3 -m py_compile` passed for env/state, train/train_mixed, model/utils, eval/inference/analysis, and validation files.
  - Full CPU validation suite passed: `validate_edge_mask_changes.py --case all --jax-platforms cpu --dataset-path /home/lorenzo/moleworks/terra_data/train --dataset-size 1`.
  - Local JAX CUDA preflight passed on one RTX 4090.
  - Local RTX 4090 full-shape smoke passed after the final diagnostic patch: one update, `1024` envs, `num_steps=32`, `num_minibatches=32`, ResMap64, critic affordances/progress enabled, `158.97s`, `215.44` steps/s.
- Updated `scripts/euler/terra_train_resmap64_phase_4gpu.sbatch` for the combined R1/R2 flags and run names. No Euler job was launched in this step.

Update 2026-05-15 14:10 CEST:

- Synced the reviewed paired WIP worktrees to `/cluster/home/lterenzi/codex_terra_edge_validation`:
  - `/home/lorenzo/moleworks/terra_mask_wip` -> `terra`
  - `/home/lorenzo/moleworks/terra-baselines_mask_wip` -> `terra-baselines`
- Remote sanity before submission:
  - Remote `py_compile` passed for env/state, train/train_mixed, model/utils, and validation files.
  - Euler sbatch contains the R1/R2 flags: `--use_critic_affordances --include_episode_progress --edge_features_dim 10`.
  - `sbatch --test-only` resolved to a four-GPU node (`eu-g6-042`) with preemption listed.
- Submitted job `66591675`.
  - Initial state: `PENDING (Priority)`.
  - Log path once started: `/cluster/home/lterenzi/codex_terra_edge_validation/logs/66591675_resmap64_phase_4gpu.out`.
  - Next checks: allocation GPU type, CUDA/cuDNN/NCCL preflight, W&B-disabled smoke completion, online W&B run URL, and first online update.

Update 2026-05-15 15:16 CEST:

- Cancelled saturated clean multiagent run `66438404` (`terra-clean-ma-4x4090`), checkpoint/W&B name `terra-clean-multiagent-4x4090-autotune0-euler-pr-2026-05-13-19-49-55`.
  - Slurm state: `CANCELLED by 542598`.
  - Runtime before cancellation: `1-19:26:22` on `eu-g4-031`, `4 x NVIDIA GeForce RTX 3090`.
- Reduced pending R1/R2 job `66591675` from `5-00:00:00` to `1-00:00:00` to improve backfill probability.
  - Updated both the live Slurm job and `scripts/euler/terra_train_resmap64_phase_4gpu.sbatch`.
  - Slurm after update: `PENDING`, `TimeLimit=1-00:00:00`, `SchedNodeList=eu-g6-051`, no start time yet.

Update 2026-05-15 21:32 CEST:

- Job `66591675` started on `eu-g4-027` with `4 x NVIDIA GeForce RTX 3090`.
- Runtime gates passed:
  - GPU guard accepted the allocation.
  - CUDA/cuDNN/NCCL preflight passed.
  - W&B-disabled full-shape smoke passed in `754.43s` at `177.86` steps/s.
- Online W&B run: `faen4uin`, `https://wandb.ai/aless-weber-eth/mixed-agents/runs/faen4uin`.
- Diagnosis after about `1700` updates:
  - Spikes are still timeout-synchronized, not task-success terminations.
  - Top value-loss spikes have `task_done_rate=0` and `timeout_rate ~= 0.031 = 1/32`.
  - Timeout updates have mean `value_loss ~= 0.28`; quiet updates have mean `value_loss ~= 0.0038`.
  - The error is concentrated in `train/value_loss_timeout_bucket_75_100`; that bucket often has negative explained variance.
- Implemented staggered initial episode age:
  - Randomize `state.env_steps` once after initial `reset_prepared`, per env.
  - Update `observation["episode_progress"]`, `info["episode_progress"]`, and `info["final_observation"]["episode_progress"]`.
  - Normal reset behavior after that remains age zero.
- Validation:
  - `python3 -m py_compile train.py train_mixed.py scripts/validation/validate_edge_mask_changes.py`.
  - CPU `--case initial-episode-progress-randomization`.
  - CPU `--case training-accounting`.
  - Two review agents reported no blocking findings.
  - Local RTX 4090 full-shape smoke passed: one update, `1024` envs, `num_steps=32`, `num_minibatches=32`, ResMap64 + critic affordances, `432.00s`, `79.36` steps/s.
- Next action: sync the patch to Euler, cancel `66591675`, and submit a fresh 24h run to test whether timeout spikes are desynchronized.

Update 2026-05-15 21:45 CEST:

- Synced the staggered-timeout patch to `/cluster/home/lterenzi/codex_terra_edge_validation`.
- Remote sanity after sync:
  - Remote `py_compile` passed for `train.py`, `train_mixed.py`, and `scripts/validation/validate_edge_mask_changes.py`.
  - Remote code contains `randomize_initial_episode_progress` in both `train.py` and `train_mixed.py`.
- Cancelled superseded job `66591675` after `06:02:00`; Slurm reports `CANCELLED by 542598`.
- Submitted replacement job `66660010` with the same R1/R2 architecture and 24h wall time.
  - Initial state: `PENDING (Priority)`.
  - Latest scheduler estimate at `2026-05-15 21:50 CEST`: start at `2026-05-16 01:32:54` on `eu-g4-031`.
  - Requested TRES: `cpu=16, gres/gpu=4, mem=128G, node=1`.
  - Log path once started: `/cluster/home/lterenzi/codex_terra_edge_validation/logs/66660010_resmap64_phase_4gpu.out`.
  - Next checks: allocation GPU type, CUDA/cuDNN/NCCL preflight, W&B-disabled smoke completion, online W&B run URL, and timeout/value-loss desynchronization.

Update 2026-05-15 21:57 CEST:

- Moved live pending job `66660010` to the 120h queue at user request:
  - `Partition=gpuhe.120h`
  - `TimeLimit=5-00:00:00`
  - `SchedNodeList=eu-g4-027`
  - `StartTime=2026-05-16T01:35:21`
- Updated `scripts/euler/terra_train_resmap64_phase_4gpu.sbatch` to `#SBATCH --time=5-00:00:00` so future submissions match the live job.

Update 2026-05-15 22:23 CEST:

- Created early-learning diagnostic: `docs/policy_diagnostics/early_learning_diagnostic_2026-05-15.md`.
- Copied latest saved checkpoints for masked, clean, base ResMap, and R1/R2 sync runs from Euler for local probing.
- W&B run files contained no checkpoint artifacts; existing training checkpoints were rolling files, not historical early snapshots.
- Added early checkpoint preservation to `train_mixed.py`: when checkpointing, keep update-indexed snapshots at `0`, `100`, `500`, `1000`, `2000`, `5000`, and `10000` in addition to `checkpoints/{name}.pkl`.
- Validation/sync:
  - Local `python3 -m py_compile train_mixed.py` passed.
  - Synced `train_mixed.py` to Euler before job start.
  - Remote `/cluster/scratch/lterenzi/codex_terra_edge_venv/bin/python -m py_compile train_mixed.py` passed.
- Latest live job estimate after scheduler movement: `66660010` pending in `gpuhe.120h`, `5-00:00:00`, estimated start `2026-05-16T03:18:24` on `eu-g6-021`.

Update 2026-05-15 22:43 CEST:

- Strengthened the early-learning diagnostic with real downloaded checkpoint rollouts run locally on the RTX 4090:
  - Added `scripts/analysis/probe_checkpoint_rollouts.py`, a jitted local rollout probe.
  - Wrote local rollout outputs under `docs/policy_diagnostics/local_rollout_*_16x550_seed0.json`.
  - Updated `docs/policy_diagnostics/early_learning_diagnostic_2026-05-15.md` with the local rollout table and checkpoint hashes.
- Local rollout summary, `16` envs, `550` steps, seed `0`:
  - refreshed masked checkpoint in masked mode: `4/16` success, `15/16` terrain changed, avg return `4.90`, invalid selected `0.0`.
  - clean checkpoint: `3/16` success, `15/16` terrain changed, avg return `4.69`.
  - R1/R2 sync checkpoint: `0/16` success, `15/16` terrain changed, avg return `0.91`, entropy `1.852`, invalid selected `0.0005`.
  - base ResMap checkpoint: `0/16` success, `15/16` terrain changed, avg return `-0.44`, entropy `1.896`, invalid selected `0.028`.
- Interpretation: R1/R2 is not a dead/no-op policy; it edits terrain locally but does not convert edits into completed episodes at that early checkpoint.
- `66660010` allocated on `eu-g4-027` and failed after `00:00:48`:
  - GPU guard and JAX CUDA/cuDNN/NCCL preflight passed.
  - Failure was `ImportError: cannot import name 'eval_component_metrics' from 'train'`.
  - Cause: partial remote sync had updated `train_mixed.py` without the matching current `train.py`.
- Fix and replacement:
  - Resynced both paired WIP trees to Euler.
  - Verified remote `train.py` contains `eval_component_metrics` and `randomize_initial_episode_progress`.
  - Submitted replacement job `66670662`.
  - Initial state: `PENDING (Priority)` in `gpuhe.120h`, `5-00:00:00`, estimated start `2026-05-16T02:15:12` on `eu-g4-029`.
  - Log path once started: `/cluster/home/lterenzi/codex_terra_edge_validation/logs/66670662_resmap64_phase_4gpu.out`.

Update 2026-05-15 23:35 CEST:

- Job `66670662` started on `eu-g4-019` with `4 x NVIDIA GeForce RTX 3090`.
- GPU guard and CUDA/cuDNN/NCCL preflight passed.
- W&B-disabled full-shape smoke passed in `765.12s`, `175.32` steps/s.
- Online W&B run started:
  - Run id: `xxf7eoap`
  - URL: `https://wandb.ai/aless-weber-eth/mixed-agents/runs/xxf7eoap`
- Online run is in the first compilation/training phase; next checks are first online update, timeout desynchronization metrics, and early checkpoint files.

Update 2026-05-16 10:31 CEST:

- Job `66670662` is still running on `eu-g4-019`; Slurm elapsed `11:21:11`, time limit `5-00:00:00`, allocation `4 x NVIDIA GeForce RTX 3090`.
- Online run is past the first update and around step/update `3070`.
- W&B summary for `xxf7eoap`:
  - `performance/steps_per_second=12348.9` in normal training rows; eval/checkpoint rows temporarily report lower throughput.
  - `train/timeout_rate=0.00187`, `train/done_rate=0.00187`, `train/task_done_rate=0`.
  - `explained_variance=0.9994`, `value_loss=1.38`, `entropy=1.91`.
  - `eval/success_rate=0`, `eval/rewards=-0.00033`.
- Timeout-stagger check: the timeout rate is near the expected per-transition `1/550 ~= 0.00182`, not the old synchronized `1/32 ~= 0.03125` wave.
- Early snapshots exist for updates `0`, `100`, `500`, `1000`, and `2000`; the rolling checkpoint was updated at `2026-05-16 10:05 CEST`.

Update 2026-05-16 12:07 CEST:

- Pulled and locally probed the latest `xxf7eoap` checkpoint plus update `0`, `500`, and `2000` snapshots on the RTX 4090.
- Local latest checkpoint probe, `32` envs, `550` steps, seeds `0` and `1`:
  - unmasked mode: `0/64` success, entropy about `1.90`, invalid selected about `3%`, terrain changed in `39/64` envs, `DO` only about `4%` of actions.
  - forced-mask diagnostic: `0/64` success, invalid selected `0`, similar terrain changes and returns.
  - Conclusion: entropy is not collapsed and invalid actions are not the main blocker; the policy edits terrain but does not learn a finishing sequence.
- W&B/code diagnosis:
  - `xxf7eoap` fixed timeout synchronization: `train/timeout_rate ~= 0.0018`, not the old `1/32` wave.
  - `eval/success_rate=0`, but `eval/failure/rewards/terminal` was nonzero, which means timeout failures were receiving terminal-shaped reward.
  - Root cause: `terra/state.py` paid `terminal_r` on `done`, and `done` includes max-step timeout.
- Implemented terminal-reward fix:
  - `terminal_r` is now gated on `task_done`.
  - Added a high-completion timeout validation in `env-episode-progress`: `done=True`, `timeout_done=True`, `task_done=False`, `completion > 0.5`, and terminal reward must be `0`.
- Validation:
  - Local `py_compile` passed for `terra/state.py` and `validate_edge_mask_changes.py`.
  - CPU `env-episode-progress --disable-jit` passed.
  - CPU `training-accounting` passed.
  - CPU `synthetic-env-action-mask --disable-jit` passed.
  - Two review agents reported no blocking findings after the high-completion validation fix.
- Synced both paired WIP trees to Euler and verified remote `py_compile` plus the terminal gate/name greps.
- Cancelled `66670662` after `12:58:11`; Slurm state `CANCELLED by 542598`.
- Submitted replacement job `66756388`.
  - Initial state: `PENDING (Priority)` in `gpuhe.120h`, `5-00:00:00`; `squeue --start` reports `N/A (Resources)`.
  - Online run name will include `terminalfix`: `terra-solo-resmap64-r1r2-terminalfix-mb32-unmasked-4gpu-50B-20260516`.
  - Log path once started: `/cluster/home/lterenzi/codex_terra_edge_validation/logs/66756388_resmap64_phase_4gpu.out`.

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

## 2026-05-16 ResMap64 R1/R2 Terminal-Fix Relaunch

Goal: replace `66670662` after diagnosing that terminal reward was paid on max-step timeout, then verify the same ResMap64 R1/R2 architecture starts cleanly with terminal reward gated on true `task_done`.

Setup:

- Slurm job `66756388`, script `scripts/euler/terra_train_resmap64_phase_4gpu.sbatch`.
- Node `eu-g6-072`, `4 x NVIDIA GeForce RTX 4090`, partition `gpuhe.120h`, time limit `5-00:00:00`.
- W&B run `04e8dada`: `https://wandb.ai/aless-weber-eth/mixed-agents/runs/04e8dada`.
- Architecture: `--map_encoder resnet_delayed --map_feature_dim 128 --use_map_derived_channels --separate_actor_critic_trunks`.
- PPO/run shape: `--num_devices 4 --num_envs_per_device 1024 --num_steps 32 --num_minibatches 32 --total_timesteps 50000000000`.
- Semantics: `--disable_action_mask`, critic-only affordances, staggered initial timeout phase, timeout bootstrap from final observation, and terminal reward only on `task_done`.

Validation:

- Local gates before submit: `py_compile`, CPU `env-episode-progress` high-completion timeout terminal-zero fixture, CPU `training-accounting`, CPU `synthetic-env-action-mask`, and two-agent review.
- Euler gates after allocation: GPU guard passed, CUDA/cuDNN/NCCL preflight passed, and W&B-disabled full-shape smoke completed one update in `416.07s` at `315.15` steps/s.
- Online training reached W&B step `0` and saved both rolling and `update_000000` checkpoints.

First online row:

- `entropy=2.0771589` versus `ln(8)=2.07944`; entropy is not collapsed.
- `value_loss=0.0018877`, `explained_variance=0.0001058`, `sched/entropy_coef=0.15`.
- `train/timeout_rate=train/done_rate=0.002037`, `train/task_done_rate=0`.
- `eval/success_rate=0`, `eval/positive_terminations=0`.
- `eval/failure/rewards/terminal=0` and `eval/rewards/terminal=0`, confirming the timeout terminal-reward leak is not present in the live run.

Update `100` evidence:

- W&B step `100`: `entropy=1.9799`, `value_loss=0.00231`, `explained_variance=0.954`, `performance/steps_per_second=19261.7`, `train/timeout_rate=0.0017166`, `train/task_done_rate=0`.
- Eval at step `100`: `eval/success_rate=0`, `eval/positive_terminations=0`, `eval/max_reward=0.86875`, `eval/rewards=-0.00295`, `eval/failure/rewards/terminal=0`, `eval/rewards/terminal=0`.
- Local checkpoint probe saved to `docs/policy_diagnostics/04e8dada_update100_rollout_32x550_seed0_1.json`.
- Probe setup: local RTX 4090, `32` envs, `550` steps, seeds `0` and `1`, unmasked and forced-mask modes.
- Probe implementation note: `probe_checkpoint_rollouts.py` now records `info["final_observation"]["action_map"]` at first done per env before computing final terrain scores, so dig/dump coverage is measured on the terminal episode state rather than the post-reset observation.
- Probe result: no successes in any rollout; forced masking removes the one small invalid-action rate but does not improve success; terrain changes occur in `8-10/32` envs per seed/mode; final dig coverage is only about `0.07-0.08`; final dump coverage is about `0.22-0.28`; `DO` remains sparse at about `1.2-2.2%` and `DO_NOTHING` remains high at about `31.7-41.4%`.
- Interpretation: update `100` is not learning evidence yet. It confirms no entropy collapse and no timeout terminal-reward leak, but behavior is still early diffuse exploration rather than a finishing strategy.

Update `500` evidence:

- W&B around step `536`: `entropy=1.8695`, `value_loss=0.00244`, `explained_variance=0.9907`, `performance/steps_per_second=19187.7`, `train/timeout_rate=0.001724`, `train/task_done_rate=0`.
- Eval remains no-success: `eval/success_rate=0`, `eval/positive_terminations=0`, `eval/max_reward=0.823`, `eval/rewards=-0.00046`, `eval/failure/rewards/terminal=0`, `eval/rewards/terminal=0`.
- Eval behavior moved in the right direction but is still weak: `eval/DO=0.0725`, `eval/DO_NOTHING %=0.156`, `behavior/terrain_changed_rate=0.0309`, `behavior/dig_success_rate=0.0155`, `behavior/dump_success_rate=0.0154`.
- Local checkpoint probe saved to `docs/policy_diagnostics/04e8dada_update500_rollout_32x550_seed0_1.json`.
- Probe setup: local RTX 4090, `32` envs, `550` steps, seeds `0` and `1`, unmasked and forced-mask modes.
- Probe result: still no successes, but it is no longer mostly inert. Terrain changed in `30/32` envs for every seed/mode, terrain-change/action rose to `0.0146-0.0159` from about `0.0014-0.0017` at update `100`, final dig coverage rose to about `0.50`, final dump coverage rose to about `0.93-0.94`, average returns improved to about `-1.41` to `-0.89`, and max returns are positive up to `2.46`.
- Mask diagnostic: unmasked logits select more `DO` (`4.6-5.6%`) but include `4-5%` invalid selections; forced mask removes invalid selections but lowers `DO` to `1.6-2.1%` and increases `DO_NOTHING`.
- Interpretation: continue training. The run is not learning completions yet, but update `500` shows real terrain interaction and healthier behavior than update `100`. Next decision checkpoints are `1000` and `2000`.

Update `1000` evidence:

- W&B has tiny nonzero eval success rows at step `600` (`eval/success_rate=0.000244`), step `900` (`0.000244`, `eval/max_reward=4.59`), and step `1000` (`0.000977`, `eval/max_reward=5.22`). Treat this as promising early signal, not persistent success yet.
- Step `1100` eval returned to `0` success with `eval/max_reward=0.847`, `eval/rewards=0.000135`, `eval/DO=0.0886`, and `eval/DO_NOTHING %=0.155`. Completion still trends upward, so this weakens any "persistent success" claim but does not by itself justify intervention before update `2000`.
- Step `1200` eval returned to tiny success: `eval/success_rate=0.000244`, `eval/max_reward=5.45`, `eval/rewards=0.000303`, `eval/DO=0.0829`, and `eval/DO_NOTHING %=0.153`.
- Step `1300` eval/latest summary at step `1320`: tiny success rose to `0.001709`, `eval/max_reward=5.54`, `eval/rewards=0.000518`, `eval/DO=0.0820`, `eval/DO_NOTHING %=0.159`, `entropy=1.8642`, `value_loss=0.00148`, `explained_variance=0.9956`, `train/timeout_rate=0.00182`, `train/task_done_rate=0`, `progress/completion=0.6711`, `core=0.8164`, and `edge=0.6129`.
- Value/termination history through W&B step `1053`: no return of the old synchronized timeout spike pattern. `value_loss` median is `0.00185`, p95 `0.00264`, max `0.00627`; `explained_variance` median is `0.992`; `train/timeout_rate` stays near `1/550` (`0.00153-0.00208`); `train/task_done_rate` is mostly `0` with rare `1.5e-5` rows.
- Local checkpoint probe saved to `docs/policy_diagnostics/04e8dada_update1000_rollout_32x550_seed0_1.json`.
- Probe setup: local RTX 4090, `32` envs, `550` steps, seeds `0` and `1`, unmasked and forced-mask modes.
- Probe result:
  - seed `0`, unmasked: `0/32` success, avg return `-1.290`, max return `0.981`, mean entropy `1.724`, invalid selected `2.55%`, `DO=7.40%`, `DO_NOTHING=10.8%`, terrain changed `27/32`, final dig coverage `0.587`, final dump coverage `0.790`.
  - seed `0`, forced mask: `0/32` success, avg return `-1.180`, max return `1.528`, invalid selected `0`, `DO=4.85%`, `DO_NOTHING=21.0%`, terrain changed `27/32`, final dig coverage `0.587`, final dump coverage `0.790`.
  - seed `1`, unmasked: `0/32` success, avg return `-0.986`, max return `1.562`, mean entropy `1.859`, invalid selected `1.90%`, `DO=7.45%`, `DO_NOTHING=5.02%`, terrain changed `29/32`, final dig coverage `0.627`, final dump coverage `0.897`.
  - seed `1`, forced mask: `0/32` success, avg return `-0.841`, max return `1.562`, invalid selected `0`, `DO=4.94%`, `DO_NOTHING=20.5%`, terrain changed `29/32`, final dig coverage `0.621`, final dump coverage `0.897`.
- Failure summary: final undug fraction is still high at about `0.37-0.41`. Positive moved dirt exists in `26/32` seed-0 failures and `29/32` seed-1 failures; within those positive-dirt failures, only about `1.0-2.7%` of moved dirt is off the dump zones, and all positive-dirt failures are below the simple `<=20%` off-dump threshold. Only `9-11/32` failures meet the "mostly dug" threshold (`<=20%` undug), and `8-10/32` envs are still loaded at timeout. The current local evidence points more to unfinished digging/sequence completion than to wrong-place dumping, but it does not prove every failed env completed dumping.
- Interpretation: update `1000` still has no deterministic local completions in the `128` rollout samples, so the policy is not solved. It is also not entropy-collapsed and not inert. Compared with update `500`, unmasked `DO` increased from about `4.6-5.6%` to about `7.4%`, invalid raw selections fell from about `4-5%` to about `2%`, dig coverage improved from about `0.50` to about `0.59-0.63`, and terrain-change/action improved from about `0.015` to about `0.021-0.025`. The W&B eval blips are now appearing in more than one eval row, so continue to update `2000` before changing architecture or reward unless behavior metrics regress sharply.

Latest rolling-checkpoint evidence at W&B step about `1258`:

- Pulled the rolling checkpoint from Euler to `logs/euler_checkpoints/terra-solo-resmap64-r1r2-terminalfix-mb32-unmasked-4gpu-50B-20260516-euler-4gpu-2026-05-16-12-18-34_latest_step1258.pkl`.
- Local rollout JSON: `docs/policy_diagnostics/04e8dada_latest_step1258_rollout_32x550_seed0_1.json`.
- Result: still `0/128` successes across seeds/modes, but the final maps are closer than update `1000`.
  - seed `0`, unmasked: avg return `-0.173`, max return `1.652`, final dig coverage `0.704`, final dump coverage `0.965`, final undug fraction `0.296`, off-dump moved dirt positive mean `0.0036`, loaded failures `1/32`, `DO=4.09%`, `DO_NOTHING=27.5%`.
  - seed `0`, forced mask: avg return `-0.059`, max return `1.588`, final dig coverage `0.698`, final dump coverage `0.965`, final undug fraction `0.302`, loaded failures `1/32`, `DO=2.34%`, `DO_NOTHING=38.9%`.
  - seed `1`, unmasked: avg return `-0.337`, max return `1.475`, final dig coverage `0.650`, final dump coverage `0.893`, final undug fraction `0.350`, off-dump moved dirt positive mean `0.0144`, loaded failures `0/32`, `DO=4.78%`, `DO_NOTHING=32.3%`.
  - seed `1`, forced mask: avg return `-0.206`, max return `1.475`, final dig coverage `0.647`, final dump coverage `0.893`, final undug fraction `0.353`, loaded failures `0/32`, `DO=2.20%`, `DO_NOTHING=43.3%`.
- Interpretation: the latest rolling checkpoint strengthens the "unfinished digging/sequence completion" diagnosis and shows progress in map final state. It also adds a new concern: `DO` selection has fallen while `DO_NOTHING` rose sharply, especially under forced masking. Continue to update `2000`; at that checkpoint, judge whether map progress continues despite high no-op mass or whether exploration/control is stalling.

Stochastic rolling-checkpoint diagnostic:

- W&B eval samples from the policy distribution, while the local probes above use deterministic argmax. Added `--stochastic` to `scripts/analysis/probe_checkpoint_rollouts.py` to match the W&B action-selection path.
- Local rollout JSON: `docs/policy_diagnostics/04e8dada_latest_step1258_stochastic_rollout_32x550_seed0_1.json`.
- Result, unmasked stochastic, `32` envs x `550` steps, seeds `0` and `1`: still `0/64` successes, but action mix now matches W&B much better: `DO=8.0-8.1%`, `DO_NOTHING=15.2-15.7%`, mean entropy about `1.88`, invalid selections about `7.1-7.3%`.
- Final terrain state is closer than argmax: final dig coverage `0.74-0.77`, final dump coverage `0.995`, final undug fraction `0.23-0.26`, no loaded-at-timeout failures, all envs have positive moved dirt, and off-dump moved dirt is only about `0.4-0.5%`.
- Interpretation: the high deterministic no-op rate is partly an argmax artifact, not the W&B eval behavior. The W&B-comparable failure mode is now clearer: the policy can move dirt to the right place but still leaves about a quarter of required dig tiles unfinished by timeout.

Clean-baseline stochastic target:

- Local rollout JSON: `docs/policy_diagnostics/clean_baseline_stochastic_rollout_32x550_seed0_1.json`.
- Same local dataset size and stochastic unmasked probe on `terra-clean-multiagent-4x4090-autotune0-euler-pr-2026-05-13-19-49-55.pkl`.
- Result: seed `0` succeeds `14/32`, seed `1` succeeds `12/32`; max return is about `9.0`, average success step is about `111-133`, and average episode length is about `368-386`.
- Clean target final state: final dig coverage `0.89-0.90`, final undug fraction `0.095-0.112`, final dump coverage `0.966-0.968`, and off-dump dirt only `0.1-0.3%`.
- Clean target policy behavior: entropy is nearly deterministic (`0.032-0.041`), invalid selection is low (`0.3-0.8%`), `DO` is similar to current stochastic (`7.2-8.0%`), but productive terrain-change/action is about `0.058-0.064` versus current `0.029-0.031`.
- Interpretation: current `04e8dada` does not need more raw `DO`; it needs more *productive* DO and a sharper finishing strategy. The clean policy uses a similar amount of DO but turns roughly twice as many actions into terrain changes, leaves roughly half as much undug target, and becomes much more deterministic.

Next checks:

- Do not judge learning from update `0`; update `100` is the first useful eval after warm compile.
- Watch whether update `2000` makes the tiny success persistent, entropy decay, value-loss/explained-variance spikes, timeout bucket losses, `DO` rising toward historical healthy levels, `DO_NOTHING` falling, and early checkpoints at updates `2000`, `5000`, and `10000`.

## 2026-05-18 Eval Success-Rate Accounting Fix

- Diagnosis: current W&B rows from `04e8dada` report `eval/success_rate > 1` because the active eval loop runs a fixed `550`-step window, resets completed envs, and logs `positive_terminations / initial_eval_envs`. That is a useful throughput count, but not a bounded success probability.
- Local WIP fix: `eval_ppo.RolloutStats` now tracks first completion per initial eval env. `eval/success_rate` is the bounded first-episode success fraction, `eval/timeout_rate` is the bounded first-episode timeout/failure fraction, `eval/episode_success_rate` is successes over all completed eval episodes, and the old count-style metric is exposed as `eval/successes_per_env` plus `eval/terminations_per_env`.
- Euler workspace fix: applied the same metric semantics to `/cluster/home/lterenzi/codex_terra_edge_validation/terra-baselines/{eval_ppo.py,train.py,train_mixed.py}` for future launches. The already-running jobs will keep logging with the old in-memory code until relaunched.
- Validation: local `reward-logging-accounting` passed, local `py_compile` passed for touched files, and Euler `py_compile` passed for the patched runtime files.
