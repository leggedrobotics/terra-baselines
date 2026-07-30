#!/usr/bin/env bash
# Local single-GPU smoke for the M2 dose curriculum (wave-long / dose / dose-fast).
# Small env count, 8 PPO updates, offline W&B, real checkpoint write. Proves:
# the bank loads under the exact-dataset contract, horizon 450 is effective,
# reward-v2 is ACTIVE (--require_reward_v2 turns a reward-v1 config into a
# launch failure), the per-condition telemetry reports all 32 conditions on the
# dose arms, and the promotion rule the preset asks for is the one in force.
#
#   scripts/smoke_m2_local.sh wave|dose|dose_fast
#
# Run it from the checkout that carries the condition telemetry; the dose arms'
# 32-condition check needs it.
set -euo pipefail

ARM="${1:?usage: smoke_m2_local.sh wave|dose|dose_fast}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TERRA=/home/lorenzo/moleworks/.worktrees/terra_v5m_screen_20260730
VENV=/home/lorenzo/moleworks/.venv-terra-gpu-uv

case "$ARM" in
    wave) PRESET=m2_wave_long; NAME=terra-v6m-wave-long-smoke; BANK=curriculum_v5m; SEED=20260730 ;;
    dose) PRESET=m2_dose;      NAME=terra-v6m-dose-smoke;      BANK=curriculum_v6m; SEED=20260731 ;;
    fast|dose_fast)
          ARM=dose_fast
          PRESET=m2_dose_fast; NAME=terra-v6m-dose-fast-smoke; BANK=curriculum_v6m; SEED=20260732 ;;
    *) echo "unknown arm $ARM (want wave|dose|dose_fast)" >&2; exit 2 ;;
esac

export PYTHONPATH="$TERRA:$REPO"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export DATASET_PATH="$REPO/terra_data/$BANK"
export DATASET_SIZE=864
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_DIR="$REPO/logs/smoke_m2"
# REVIEW_V6 F5-13: the condition-telemetry A/B runs landed in the shared
# mixed-agents project that M1/M2 comparisons read. Smokes stay offline unless
# someone asks for otherwise.
export WANDB_MODE="${WANDB_MODE:-offline}"

# REVIEW_V6 R-8: the venv's editable `terra` resolves to the un-fixed main repo,
# and a run that silently imports it produces reward-v1 numbers that look fine.
"$VENV/bin/python" - "$TERRA" <<'PY'
import sys
import terra
expected, actual = sys.argv[1], terra.__file__
if not actual.startswith(expected):
    raise SystemExit(f"terra resolves to {actual}, expected a path under {expected}")
print(f"terra = {actual}")
PY

mkdir -p "$WANDB_DIR" "$REPO/logs/smoke_m2/checkpoints_$ARM"

cd "$REPO"
exec "$VENV/bin/python" -u train_mixed.py \
    --config "$PRESET" \
    --name "$NAME" \
    --machine local4090 \
    --seed "$SEED" \
    --num_devices 1 \
    --num_envs_per_device 128 \
    --num_steps 32 \
    --update_epochs 2 \
    --num_minibatches 8 \
    --total_timesteps 32768 \
    --lr 3e-4 \
    --require_reward_v2 \
    --model_size medium \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8_se \
    --encoder_compute_dtype bfloat16 \
    --critic_hidden_dims 512,256 \
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.005 \
    --ent_schedule_steps 7600 \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --fail_on_nonfinite \
    --finite_check_interval 1 \
    --log_train_interval 1 \
    --log_eval_interval 4 \
    --eval_episodes 8 \
    --checkpoint_interval 4 \
    --checkpoint_dir "$REPO/logs/smoke_m2/checkpoints_$ARM" \
    --cache_clear_interval 0
