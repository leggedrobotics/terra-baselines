# Larger ResNet Distillation Plan

Date: 2026-05-18

## Objective

Train larger delayed-downsample ResNet policies for the solo-excavator
ringmaps task without changing the primitive action interface. Test the
hypothesis that larger PPO policies are harder to optimize from scratch and
benefit from an imitation warm start from the smaller trained ResMap policy.

## Research Takeaways

- Larger RL networks can help, but the optimization is less forgiving than in
  supervised learning; capacity should be paired with stable initialization,
  conservative learning-rate/minibatch choices, and stronger auxiliary signal.
- Policy distillation is the lowest-friction warm-start mechanism here because
  it does not require matching parameter shapes between small and large
  networks. The larger student matches the trained teacher's action
  distribution on live environment observations, then PPO takes over.
- A scratch larger-network control is needed. Otherwise a better or worse
  larger run cannot be attributed to the imitation stage.

Reference threads:

- Nauman and Raffel, 2024, "The Case for Bigger Models in Reinforcement
  Learning": https://arxiv.org/abs/2407.15134
- Rusu et al., 2015, "Policy Distillation": https://arxiv.org/abs/1511.06295
- Oh et al., 2018, "Self-Imitation Learning": https://arxiv.org/abs/1806.05635

## Implemented Setup

Teacher:

- Checkpoint:
  `/home/lorenzo/moleworks/checkpoints/terra-solo-resmap64-r1r2-terminalfix-mb32-unmasked-4gpu-50B-20260516-euler-4gpu-2026-05-16-12-18-34.pkl`
- Architecture: base delayed ResMap64, `537,689` parameters.
- Actor remains unmasked; `action_mask` is not used to mask PPO logits.

Student warm start:

- `train_mixed.py` accepts `--teacher_checkpoint` and `--imitation_updates`.
- The teacher rolls out actions in the live env.
- The student minimizes KL from teacher logits plus a weighted teacher-value
  regression loss.
- The PPO start RNG, env state, previous actions, and previous reward are saved
  before imitation and restored afterward. This makes the distill-vs-scratch
  comparison a policy-parameter warm-start test rather than a teacher-driven
  curriculum/env-state warm start.
- The imitation optimizer uses its own learning rate, then Adam is reset before
  PPO starts. This keeps PPO optimizer state from inheriting distillation
  transients.
- The final imitation metrics are logged on W&B step `0`, and the post-distill
  student checkpoint is saved as `checkpoints/<name>_POST_DISTILL.pkl`.

Architecture presets:

- `model_size=medium`, `map_feature_dim=192`:
  - ResNet channels `(24, 48, 64)`, two blocks/stage.
  - Local measured size: `951,081` parameters.
  - Local 4090 smoke passes at `1024` envs/GPU, `32` steps,
    `64` minibatches.
- `model_size=large`, `map_feature_dim=256`:
  - ResNet channels `(32, 64, 96, 128)`, three blocks/stage.
  - Local measured size: `3,157,145` parameters.
  - Local 4090 smoke passes at `1024` envs/GPU, `32` steps,
    `128` minibatches, with `XLA_FLAGS=--xla_gpu_autotune_level=0`.

## Local Validation

- `python -m py_compile train_mixed.py utils/models.py utils/utils_ppo.py`
  passed.
- `bash -n scripts/euler/terra_train_larger_resnet_4gpu.sbatch` passed.
- `git diff --check` passed.
- Tiny CPU imitation+PPO smoke passed after the PPO-start reset fix:
  - `model_size=medium`, `4` envs, `2` steps, one imitation update, one PPO
    update.
  - Imitation loss: `2.3258`; PPO completed and saved both
    `_POST_DISTILL.pkl` and `_FINAL.pkl`.
- Local RTX 4090 medium smoke:
  - Failed with `1024` envs/GPU and `32` minibatches due to ResNet activation
    OOM.
  - Passed with `64` minibatches.
  - Imitation loss: `2.0055`; one PPO update completed in `60.25s`,
    `545.87` steps/s.
- Local RTX 4090 large smoke:
  - Failed without autotune mitigation with `CUDNN_STATUS_INTERNAL_ERROR`.
  - Passed with `XLA_FLAGS=--xla_gpu_autotune_level=0` and `128` minibatches.
  - Imitation loss: `1.8941`; one PPO update completed in `63.94s`,
    `513.88` steps/s.

## Euler Supervised-Warmup Matrix

Current single-GPU supervised-warmup script:

- `scripts/euler/terra_train_larger_resnet_supervised_1gpu_4h.sbatch`
- Queue: `gpuhe.4h`, one RTX 3090 per job, `1024` envs/GPU.
- Purpose: decide how many teacher-imitation updates to run before PPO.
- Output: `_POST_DISTILL_update_XXXX.pkl` slices every `10` imitation updates
  by default, plus a final `_POST_DISTILL.pkl`.

Calibration runs:

- `RUN_KIND=medium`
  - `model_size=medium`, `map_feature_dim=192`, `num_minibatches=64`.
  - `imitation_updates=200`.
- `RUN_KIND=large`
  - `model_size=large`, `map_feature_dim=256`, `num_minibatches=128`.
  - `imitation_updates=200`.
  - Uses `--xla_gpu_autotune_level=0` because the local 4090 smoke otherwise
    hit a cuDNN internal error.

Every single-GPU calibration job runs:

- hard GPU guard for exactly one RTX 3090 GPU;
- `check_jax_runtime.py --min-devices 1`;
- one W&B-disabled full-shape supervised-only smoke;
- online supervised imitation only, with no PPO after the warmup.

Four-GPU follow-up:

- Keep `scripts/euler/terra_train_larger_resnet_4gpu.sbatch` as the later
  full-scale PPO launcher once the single-GPU checkpoint ladder identifies a
  healthy architecture and warm-start length.

## Decision Criteria

Early health:

- The W&B-disabled full-shape smoke must complete update 1.
- Online row 0 should have entropy close to `ln(8)` unless imitation has made
  the actor deliberately sharper.
- No synchronized timeout value-loss spike pattern should reappear.
- The smoke now exercises tiny eval and checkpoint paths before starting the
  online run.

Learning comparison:

- Compare medium-distill against medium-scratch by W&B update, not wall clock.
- Compare large-distill against large-scratch by W&B update, not wall clock.
- The warm-start path should show earlier nonzero completion/success or better
  productive terrain-change/action without collapsing entropy.
- Large-distill is not a pure warm-start ablation; it tests whether the much
  larger spatial encoder can exploit the same teacher signal without becoming
  unstable.
