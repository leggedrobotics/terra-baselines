# Running Experiments

## 2026-05-19 Deep ResNet Four-GPU Warm-Start PPO

Goal: launch two full 120h-queue Terra PPO runs to test deeper delayed-ResNet
students warm-started from the trained ResMap64 teacher.

- Script: `scripts/euler/terra_train_deep_resnet_4gpu_120h.sbatch`
- Queue shape: `gpuhe.120h`, `5-00:00:00`, one node, `4` GPUs, restricted to
  `eu-g4-[001-032]` and `eu-g6-[001-080]`.
- Runtime guard: hard-fails unless exactly four `NVIDIA GeForce RTX 3090` or
  `NVIDIA GeForce RTX 4090` GPUs are allocated before JAX/W&B training.
- Teacher checkpoint:
  `terra-solo-resmap64-r1r2-terminalfix-mb32-unmasked-4gpu-50B-20260516-euler-4gpu-2026-05-16-12-18-34.pkl`.
- Shared PPO recipe: unmasked actor, delayed ResNet global-map encoder,
  derived terrain channels, separate actor/critic trunks, critic-only
  edge/progress affordances, timeout bootstrap and initial timeout-phase
  staggering from the current branch.
- Medium-deep run:
  - `RUN_KIND=medium_deep`
  - `model_size=medium_deep`, `map_feature_dim=224`
  - measured local size: `2,135,385` parameters
  - `num_minibatches=64`, `imitation_updates=100`
- Large-deep run:
  - `RUN_KIND=large_deep`
  - `model_size=large_deep`, `map_feature_dim=512`
  - measured local size: `9,958,521` parameters
  - `num_minibatches=128`, `imitation_updates=100`
  - exports `XLA_FLAGS=--xla_gpu_autotune_level=0`
- Local validation on `2026-05-19 00:17 CEST`:
  - `python3 -m py_compile utils/models.py train_mixed.py`
  - `bash -n scripts/euler/terra_train_deep_resnet_4gpu_120h.sbatch`
  - full CPU validation:
    `validate_edge_mask_changes.py --case all --jax-platforms cpu --dataset-path /home/lorenzo/moleworks/terra_data/train --dataset-size 1`
  - local RTX 4090 runtime preflight passed with one GPU.
  - Medium-deep local smoke passed: one imitation update plus one PPO update,
    `64` envs, `8` steps, `2,135,385` parameters.
  - Large-deep local smoke passed with `--xla_gpu_autotune_level=0`: one
    imitation update plus one PPO update, `64` envs, `8` steps,
    `9,958,521` parameters. A prior non-autotune large smoke initialized and
    loaded the teacher but failed in cuDNN during imitation, so the Euler
    launcher keeps autotune level `0` for this variant.
- Superseded jobs cancelled before this launch:
  - `66536725` `terra-mask-4gpu-full`, cancelled after `3-22:58:56`.
  - `66756388` `terra-resmap64-mb32`, cancelled after `2-11:38:11`.
- Submission status on `2026-05-19 00:21 CEST`:
  - `67032208` `terra-meddeep-4gpu`, `RUN_KIND=medium_deep`, submitted at
    `2026-05-19T00:19:53`, pending in `gpuhe.120h` with reason `Priority`,
    `StartTime=Unknown`.
  - `67032210` `terra-lgdeep-4gpu`, `RUN_KIND=large_deep`, submitted at
    `2026-05-19T00:20:00`, pending in `gpuhe.120h` with reason `Priority`,
    `StartTime=Unknown`.
- Queue diagnosis on `2026-05-19 00:48 CEST`:
  - Both jobs are still pending with empty `AllocTRES` and no Slurm output
    files, so runtime verification has not started.
  - Detailed Slurm request remains the intended one-node/four-GPU shape:
    `NumNodes=1-1`, `ReqTRES=cpu=16,mem=128G,node=1,gres/gpu=4`,
    `TresPerJob=gres/gpu:4`.
  - Candidate nodes are resource-limited:
    `eu-g4-014` has `7/8` GPUs allocated and `eu-g4-015` has `6/8` GPUs
    allocated.
  - Fresh `sbatch --test-only` with the current launcher estimates
    `2026-05-20T19:07` on `eu-g6-054`, so cancelling/relaunching is not a
    clear improvement.

## 2026-05-18 Larger ResNet Distillation Jobs

Current single-GPU supervised warmup calibration from paired
`codex/mask-speedup-wip` worktrees:

- Script: `scripts/euler/terra_train_larger_resnet_supervised_1gpu_4h.sbatch`
- Queue: `gpuhe.4h`, one RTX 3090 per job, `1024` envs/GPU,
  `num_steps=32`.
- Goal: save larger-student checkpoints during teacher imitation so the next
  step can benchmark warmup lengths such as `10`, `20`, `50`, `100`, and `200`
  updates before launching PPO.
- Teacher checkpoint:
  `terra-solo-resmap64-r1r2-terminalfix-mb32-unmasked-4gpu-50B-20260516-euler-4gpu-2026-05-16-12-18-34.pkl`
- Shared semantics: unmasked PPO actor, ResMap derived channels, separate
  actor/critic trunks, critic-only edge/progress affordances, timeout bootstrap
  and initial timeout-phase staggering from the current branch.
- Local gates on `2026-05-18`:
  - CPU imitation+PPO smoke passed before and after the Oracle fixes; the
    post-fix run saved `_POST_DISTILL.pkl` and `_FINAL.pkl`.
  - Medium RTX 4090 smoke passed with `1024` envs/GPU,
    `num_minibatches=64`.
  - Large RTX 4090 smoke passed with `1024` envs/GPU,
    `num_minibatches=128`, and `XLA_FLAGS=--xla_gpu_autotune_level=0`.
  - Full CPU validation sweep passed after the Oracle fixes.
- The first one-GPU calibration matrix used
  `scripts/euler/terra_train_larger_resnet_1gpu_4h.sbatch`; it was superseded
  because it moved into PPO after the supervised warmup instead of producing a
  checkpoint ladder for warmup-length selection.
  - `66969660` and `66969665` scratch PPO controls were cancelled after their
    W&B-disabled smoke because they do not answer the supervised-warmup-length
    question.
  - `66969658` and `66969663` distill jobs entered the `200`-update imitation
    block, but only save the final `_POST_DISTILL.pkl`; replace them with the
    supervised-only checkpoint-ladder launcher.
- Corrected Slurm submissions on `2026-05-18 16:27 CEST`:
  - `66981998`, `terra1g-sup-medium`, `RUN_KIND=medium`,
    `imitation_updates=200`, checkpoint interval `10`.
  - `66982006`, `terra1g-sup-large`, `RUN_KIND=large`,
    `imitation_updates=200`, checkpoint interval `10`.
- Status on `2026-05-18 16:28 CEST`: both corrected jobs are `RUNNING`.
  - `66981998`: `eu-g4-022`, `1 x RTX 3090`.
  - `66982006`: `eu-g4-026`, `1 x RTX 3090`.
- Runtime gates: both passed the hard GPU guard and
  `check_jax_runtime.py --min-devices 1`; the W&B-disabled supervised-only
  smoke completed and saved `_POST_DISTILL.pkl` checkpoints without entering
  PPO. Both jobs then started the online `200`-update supervised-only warmup.
- Checkpoint ladder status on `2026-05-18 16:58 CEST`:
  - Medium has written `_POST_DISTILL_update_0010.pkl`,
    `_POST_DISTILL_update_0020.pkl`, `_POST_DISTILL_update_0030.pkl`, and
    `_POST_DISTILL_update_0040.pkl`.
  - Large has written `_POST_DISTILL_update_0010.pkl`.
- The initial four-GPU `gpuhe.120h` matrix was cancelled when the experiment
  was narrowed to one-GPU calibration:
  - `66965292` started on `eu-g4-027` with `4 x RTX 3090` and was cancelled
    after `00:06:23`, while compiling the first imitation smoke; no online
    W&B run was produced.
  - `66965294`, `66965296`, and `66965300` were cancelled while pending.

## 2026-05-15 ResMap64 Four-GPU Architecture Run

Active Slurm jobs:

- `66756388`: running 120h-queue four-GPU ResMap64 R1/R2 terminal-reward-fix replacement from paired `codex/mask-speedup-wip` worktrees.
  - Script: `scripts/euler/terra_train_resmap64_phase_4gpu.sbatch`
  - Shape: `4` GPUs requested, `1024` envs/GPU, `4096` envs total, `num_steps=32`, `num_minibatches=32`, `total_timesteps=50000000000`.
  - Architecture: `--map_encoder resnet_delayed --map_feature_dim 128 --use_map_derived_channels --separate_actor_critic_trunks`.
  - Masking: PPO actor is unmasked by construction; current code has no action-mask train/eval switch.
  - R1/R2 flags: `--use_critic_affordances --include_episode_progress --edge_features_dim 10`.
  - Timeout semantics: pre-reset `final_observation` bootstrap for max-step timeouts; true task terminals do not bootstrap; recursive GAE stops at reset boundaries; initial `env_steps`/`episode_progress` are randomized once after startup reset to desynchronize max-step timeouts; terminal success reward is paid only on `task_done`, not on max-step timeout.
  - W&B: online full run `04e8dada`, `https://wandb.ai/aless-weber-eth/mixed-agents/runs/04e8dada`, entity `aless-weber-eth`, project `mixed-agents`.
  - Local gates on `2026-05-16 12:07 CEST`: `py_compile`, CPU `env-episode-progress` high-completion timeout terminal-zero validation, CPU `training-accounting`, CPU `synthetic-env-action-mask`, and two-agent review passed.
  - Local 4090 smoke result: one update, `1024` envs, `num_steps=32`, `num_minibatches=32`, ResMap64 + critic affordances, `432.00s`, `79.36` steps/s.
  - Current status on `2026-05-16 12:34 CEST`: running on `eu-g6-072` in `gpuhe.120h`, `5-00:00:00`, with `4 x NVIDIA GeForce RTX 4090`.
  - Euler gates: GPU guard passed, CUDA/cuDNN/NCCL preflight passed, and the W&B-disabled full-shape smoke completed one update in `416.07s` at `315.15` steps/s.
  - First online row, W&B step `0`: `entropy=2.0771589` (`ln(8)=2.07944`, not collapsed), `value_loss=0.0018877`, `explained_variance=0.0001058`, `sched/entropy_coef=0.15`, `train/timeout_rate=train/done_rate=0.002037`, `train/task_done_rate=0`, `eval/success_rate=0`, `eval/positive_terminations=0`, `eval/failure/rewards/terminal=0`, and `eval/rewards/terminal=0`.
  - Update `100` health on `2026-05-16 12:46 CEST`: `entropy=1.9799`, `value_loss=0.00231`, `explained_variance=0.954`, `train/timeout_rate=0.0017166`, `train/task_done_rate=0`, `eval/success_rate=0`, `eval/max_reward=0.86875`, `eval/rewards=-0.00295`, and terminal reward on eval failures remains `0`.
  - Update `100` local rollout probe: `docs/policy_diagnostics/04e8dada_update100_rollout_32x550_seed0_1.json`, `32` envs, `550` steps, seeds `0` and `1`, unmasked plus forced-mask. Result: `0/128` total successes across all mode/seed rollouts, invalid selections are low or zero, forced masking does not change success, final dig coverage is about `0.07-0.08`, final dump coverage is about `0.22-0.28`, and the policy still uses sparse `DO` with high `DO_NOTHING`.
  - Update `500` health on `2026-05-16 13:36 CEST`: W&B around step `536` has `entropy=1.8695`, `value_loss=0.00244`, `explained_variance=0.9907`, `train/timeout_rate=0.001724`, `train/task_done_rate=0`, `eval/success_rate=0`, `eval/max_reward=0.823`, `eval/rewards=-0.00046`, `eval/DO=0.0725`, `eval/DO_NOTHING %=0.156`, and terminal reward on eval failures remains `0`.
  - Update `500` local rollout probe: `docs/policy_diagnostics/04e8dada_update500_rollout_32x550_seed0_1.json`. Result: still `0` successes, but terrain changed in `30/32` envs in every seed/mode, terrain-change/action rose to `0.0146-0.0159`, final dig coverage rose to about `0.50`, final dump coverage rose to about `0.93-0.94`, returns improved versus update `100`, and max returns became positive up to `2.46`. Unmasked logits use more `DO` (`4.6-5.6%`) but include `4-5%` invalid selections; forced mask removes invalid selections but pushes more mass to `DO_NOTHING`.
  - Update `900-1000` health on `2026-05-16 14:52 CEST`: W&B now has tiny nonzero eval success rows at step `600` (`0.000244`), step `900` (`0.000244`, `eval/max_reward=4.59`), and step `1000` (`0.000977`, `eval/max_reward=5.22`).
  - Update `1100` eval on `2026-05-16 15:08 CEST`: success returned to `0`, `eval/max_reward=0.847`, `eval/rewards=0.000135`, `eval/DO=0.0886`, and `eval/DO_NOTHING %=0.155`. Completion still trends up (`completion=0.6467`, `core=0.8069`, `edge=0.5855` on the eval row; latest summary completion `0.666`). Treat the `600/900/1000` successes as rare discovery, not persistent learning yet.
  - Update `1200` eval on `2026-05-16 15:18 CEST`: tiny success returned (`eval/success_rate=0.000244`, `eval/max_reward=5.45`, `eval/rewards=0.000303`, `eval/DO=0.0829`, `eval/DO_NOTHING %=0.153`).
  - Update `1300` eval/latest summary at step `1320`: tiny success rose to `0.001709`, `eval/max_reward=5.54`, `eval/rewards=0.000518`, `eval/DO=0.0820`, `eval/DO_NOTHING %=0.159`, `entropy=1.8642`, `value_loss=0.00148`, `explained_variance=0.9956`, `train/timeout_rate=0.00182`, `train/task_done_rate=0`, `progress/completion=0.6711`, `core=0.8164`, and `edge=0.6129`.
  - Latest W&B summary at step `1474`: tiny success remains nonzero at `0.000732`, `eval/max_reward=5.44`, `eval/rewards=0.000463`, `eval/DO=0.0880`, `eval/DO_NOTHING %=0.155`, `entropy=1.8593`, `value_loss=0.00142`, `explained_variance=0.9953`, `train/timeout_rate=0.00168`, `train/task_done_rate=0`, `progress/completion=0.6883`, `core=0.8257`, and `edge=0.6370`.
  - Value/termination history through W&B step `1053`: no return of the old synchronized timeout spike pattern. `value_loss` median is `0.00185`, p95 `0.00264`, max `0.00627`; `explained_variance` median is `0.992`; `train/timeout_rate` stays near `1/550` (`0.00153-0.00208`); `train/task_done_rate` is mostly `0` with rare `1.5e-5` rows.
  - Update `1000` local rollout probe: `docs/policy_diagnostics/04e8dada_update1000_rollout_32x550_seed0_1.json`. Result: still `0/128` successes across unmasked/forced-mask seeds, so local deterministic completions are not reliable yet. Several behavior metrics moved in the useful direction versus update `500`: unmasked `DO` is about `7.4%`, invalid raw selections are down to about `1.9-2.6%`, terrain-change/action rose to `0.021-0.025`, and final dig coverage rose to about `0.59-0.63`; terrain changed in `27-29/32` envs and max returns remain positive up to `1.56`. Forced masking still removes invalid selections but lowers `DO` to about `4.9%` and increases `DO_NOTHING` to about `20-21%`, so re-enabling a naive action mask is not yet the obvious rescue.
  - Update `1000` failure summary: final undug fraction is still high at about `0.37-0.41`. Positive moved dirt exists in `26/32` seed-0 failures and `29/32` seed-1 failures; within those positive-dirt failures, only about `1.0-2.7%` of moved dirt is off the dump zones, and all positive-dirt failures are below the simple `<=20%` off-dump threshold. `8-10/32` envs are still loaded at timeout. Current local evidence points more to unfinished digging/sequence completion than to wrong-place dumping, but it does not prove every failed env completed dumping.
  - Latest rolling-checkpoint probe at W&B step about `1258`: `docs/policy_diagnostics/04e8dada_latest_step1258_rollout_32x550_seed0_1.json`, `0/128` successes. It improves the final state versus update `1000`: avg returns are about `-0.34` to `-0.06`, final dig coverage is about `0.65-0.70`, final dump coverage about `0.89-0.97`, final undug fraction about `0.30-0.35`, off-dump moved dirt about `0.4-1.4%`, and loaded-at-timeout is `0-1/32`. The concern is action distribution: unmasked `DO` fell to `4.1-4.8%`, forced-mask `DO` fell to `2.2-2.3%`, and `DO_NOTHING` rose to `27-43%`, so the policy is improving final maps but may be drifting toward too much waiting.
  - Stochastic rolling-checkpoint probe: `docs/policy_diagnostics/04e8dada_latest_step1258_stochastic_rollout_32x550_seed0_1.json`, unmasked only, `0/64` successes. This better matches W&B eval action sampling: `DO=8.0-8.1%`, `DO_NOTHING=15.2-15.7%`, entropy `1.88`, invalid selections about `7.1-7.3%`. Final maps are closer than argmax: final dig coverage `0.74-0.77`, dump coverage `0.995`, final undug fraction `0.23-0.26`, no loaded-at-timeout failures, and off-dump moved dirt `0.4-0.5%`. The W&B-comparable bottleneck is still finishing the remaining dig, not dumping placement or entropy collapse.
  - Clean-baseline stochastic comparison: `docs/policy_diagnostics/clean_baseline_stochastic_rollout_32x550_seed0_1.json` succeeds `12-14/32` per seed on the same local probe. It reaches final dig coverage `0.89-0.90`, final undug fraction `0.095-0.112`, terrain-change/action `0.058-0.064`, low invalid selection `0.3-0.8%`, and near-deterministic entropy `0.03-0.04`. Current `04e8dada` has similar raw `DO` in stochastic mode, but only about half the productive terrain-change/action and much more remaining dig.
  - Early checkpoint preservation: `train_mixed.py` now keeps additional snapshots at updates `0`, `100`, `500`, `1000`, `2000`, `5000`, and `10000` in addition to the rolling checkpoint.
  - Logs once started: `/cluster/home/lterenzi/codex_terra_edge_validation/logs/66756388_resmap64_phase_4gpu.out`.
  - Next health checks: monitor update `2000`, whether the tiny `600/900/1000` success rows become persistent, value-loss/explained-variance spikes, entropy trend, `DO` rising toward historical healthy levels, `DO_NOTHING` falling, and early checkpoints at updates `2000`, `5000`, and `10000`.

Recently cancelled:

- `66670662`: online W&B four-GPU ResMap64 R1/R2 staggered-timeout run, W&B `xxf7eoap`.
  - Cancelled on `2026-05-16 12:05 CEST` after `12:58:11` on `eu-g4-019`.
  - Reason: latest-checkpoint diagnosis found zero success despite timeout desync; W&B showed nonzero `eval/failure/rewards/terminal` while `eval/success_rate=0`, and code paid terminal reward on `done` where `done` includes timeouts. Replaced by `66756388` with terminal reward gated on `task_done`.

- `66591675`: online W&B four-GPU ResMap64 R1/R2 combined-fix run, W&B `faen4uin`.
  - Cancelled on `2026-05-15 21:41 CEST` after `06:02:00` on `eu-g4-027`.
  - Reason: superseded by the staggered-start timeout fix. Diagnosis before cancellation: value-loss/explained-variance spikes remained timeout-synchronized; spike updates had `timeout_rate ~= 1/32`, `task_done_rate=0`, and high error in `train/value_loss_timeout_bucket_75_100`.

- `66660010`: first staggered-timeout 120h replacement.
  - Failed on `2026-05-15 22:11 CEST` after `00:00:48` on `eu-g4-027`.
  - Reason: partial code sync left remote `train.py` older than `train_mixed.py`, causing `ImportError: cannot import name 'eval_component_metrics' from 'train'`.
  - Fix: resynced both paired WIP trees to Euler and submitted replacement `66670662`.

- `66438404`: `terra-clean-ma-4x4090`, W&B/checkpoint name `terra-clean-multiagent-4x4090-autotune0-euler-pr-2026-05-13-19-49-55`.
  - Cancelled on `2026-05-15 15:16 CEST` after `1-19:26:22` on `eu-g4-031`.
  - Reason: run appeared saturated and freeing its four RTX 3090 GPUs/fairshare should help the pending R1/R2 launch.

- `66574910`: online W&B four-GPU residual-map architecture run from paired `codex/mask-speedup-wip` worktrees.
  - Script: `scripts/euler/terra_train_resmap64_phase_4gpu.sbatch`
  - Shape: `4` GPUs requested, `1024` envs/GPU, `4096` envs total, `num_steps=32`, `total_timesteps=50000000000`.
  - Architecture: `--map_encoder resnet_delayed --map_feature_dim 128 --use_map_derived_channels --separate_actor_critic_trunks`.
  - Masking: unmasked PPO actor; this is an architecture run, not a mask run.
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
  - Cancelled on `2026-05-15 11:48 CEST` after `03:01:32` on `eu-g6-061`; Slurm state `CANCELLED by 542598`.
  - Last diagnostic read before cancellation: W&B was around update `998`; periodic value-loss spikes aligned with the episode horizon, eval terminations were still zero, and the run was too early to judge learning.

Combined-fix run lineage:

- Detailed plan: `docs/RESMAP_AFFORDANCE_TIMEOUT_PLAN.md`.
- Keep the ResMap64 spatial architecture, derived map channels, separate actor/critic trunks, and `num_minibatches=32`.
- Feed existing `edge_features` plus a new `episode_progress = env_steps / max_steps_in_episode` scalar to the critic trunk only.
- Add training diagnostics for `task_done_rate`, `timeout_rate`, value loss by timeout bucket, and explained variance by timeout bucket.
- Fix time-limit handling as a separate training semantics change: bootstrap through max-step truncations, but still zero bootstrap for true task success.
- Keep actor-visible affordances as a later deployability-gated ablation; anything fed to the actor must be computable on the real robot with the same semantics.
- `66591675` tested this combined R1/R2 setup without initial timeout staggering. It reached online training but was cancelled when W&B diagnostics showed value-loss spikes phase-locked to max-step timeouts.
- `66670662` added startup `env_steps`/`episode_progress` randomization and fixed timeout synchronization, but still paid timeout terminal reward.
- `66756388` is the active replacement and adds the terminal-reward gate fix on top of the same architecture and PPO shape.

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
