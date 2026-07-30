#!/usr/bin/env bash
# Local single-GPU smoke for the M1 v5-main screen (A/B/C).
# Small env count, 3 PPO updates, real W&B, real checkpoint write. Proves:
# maps load under the exact-dataset contract, horizon 450 is effective
# (environment/effective_horizon_{min,max} in W&B), the checkpoint is written,
# and the arch matches the E8 template (2,441,223 params).
#
#   scripts/smoke_v5m_local.sh a|b|c
set -euo pipefail

ARM="${1:?usage: smoke_v5m_local.sh a|b|c}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TERRA=/home/lorenzo/moleworks/.worktrees/terra_v5m_screen_20260730
VENV=/home/lorenzo/moleworks/.venv-terra-gpu-uv

case "$ARM" in
    a) PRESET=curriculum_v5m_a_t0;      NAME=terra-v5m-A-t0base-smoke ;;
    b) PRESET=curriculum_v5m_b_uniform; NAME=terra-v5m-B-uniform-smoke ;;
    c) PRESET=curriculum_v5m_c_waves;   NAME=terra-v5m-C-curriculum-smoke ;;
    *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

export PYTHONPATH="$TERRA:$REPO"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export DATASET_PATH="$REPO/terra_data/curriculum_v5m"
export DATASET_SIZE=864
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_DIR="$REPO/logs/smoke_v5m"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$WANDB_DIR" "$REPO/logs/smoke_v5m/checkpoints_$ARM"

cd "$REPO"
exec "$VENV/bin/python" -u train_mixed.py \
    --config "$PRESET" \
    --name "$NAME" \
    --machine local4090 \
    --seed 20260730 \
    --num_devices 1 \
    --num_envs_per_device 128 \
    --num_steps 32 \
    --update_epochs 2 \
    --num_minibatches 8 \
    --total_timesteps 12288 \
    --lr 3e-4 \
    --model_size medium \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8_se \
    --encoder_compute_dtype bfloat16 \
    --critic_hidden_dims 512,256 \
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.005 \
    --ent_schedule_steps 3800 \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --fail_on_nonfinite \
    --finite_check_interval 1 \
    --log_train_interval 1 \
    --log_eval_interval 2 \
    --eval_episodes 8 \
    --checkpoint_interval 2 \
    --checkpoint_dir "$REPO/logs/smoke_v5m/checkpoints_$ARM" \
    --cache_clear_interval 0
