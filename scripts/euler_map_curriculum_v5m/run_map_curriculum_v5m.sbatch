#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-v5m
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v5m_20260730/logs/%x_%j.out

# M1 v5-main curriculum screen. Three arms, all trained FROM SCRATCH (Lorenzo's
# 2026-07-30 directive supersedes the TRAINING_DESIGN §2.1 E8 warm-start
# contract for this screen). Only map exposure differs between arms.
#
#   sbatch --job-name=terra-v5m-A-t0base   --export=ALL,ARM=A run_map_curriculum_v5m.sbatch
#   sbatch --job-name=terra-v5m-B-uniform  --export=ALL,ARM=B run_map_curriculum_v5m.sbatch
#   sbatch --job-name=terra-v5m-C-curriculum --export=ALL,ARM=C run_map_curriculum_v5m.sbatch

set -euo pipefail
umask 002

: "${ARM:?submit with --export=ALL,ARM=A|B|C}"
case "$ARM" in
    A) PRESET=curriculum_v5m_a_t0;      RUN_NAME=terra-v5m-A-t0base ;;
    B) PRESET=curriculum_v5m_b_uniform; RUN_NAME=terra-v5m-B-uniform ;;
    C) PRESET=curriculum_v5m_c_waves;   RUN_NAME=terra-v5m-C-curriculum ;;
    *) echo "Unsupported ARM=$ARM" >&2; exit 2 ;;
esac

RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v5m_20260730
SOURCE="$RUN_ROOT/source"
BANK="$RUN_ROOT/bank"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/$ARM"

module load eth_proxy
module load stack/2024-06 cuda/12.1.1

SITE_PACKAGES="$("$VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
export PYTHONPATH="$SOURCE/terra:$SOURCE/terra-baselines"
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_ENTITY=aless-weber-eth
export WANDB_PROJECT=mixed-agents
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_MODE=online
export DATASET_PATH="$BANK"
export DATASET_SIZE=864

mkdir -p "$CHECKPOINT_DIR" "$RUN_ROOT/wandb" "$RUN_ROOT/logs"
test -f "$RUN_ROOT/LOCAL_SMOKE_PASSED"
test -f "$RUN_ROOT/SOURCE_SYNC_VERIFIED"
test -f "$BANK/curriculum_v5m_provenance.json"
test -f "$BANK/train/L0/dataset.json"
test -f "$BANK/train/L3/dataset.json"
test -f "$BANK/held_out/all/manifest.jsonl"

HOME_USED_GB="$(
    lquota |
        awk -F'|' '/\/cluster\/home\/lterenzi/ && /space/ {
            value=$4
            gsub(/[^0-9.]/, "", value)
            print value
        }'
)"
test -n "$HOME_USED_GB"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "Home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 40
}

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "Unsupported GPU: $GPU_NAME" >&2
        exit 41
    }
done
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 4

echo "=== Terra M1 v5-main screen: arm $ARM ($RUN_NAME) ==="
echo "job_id=$SLURM_JOB_ID"
echo "preset=$PRESET  init=scratch (no warm start, no teacher)"
echo "bank=$BANK  DATASET_SIZE=$DATASET_SIZE"
echo "arch=medium + mlp + resnet_spatial_8x8_se (bf16 encoder, critic 512,256) = 2,441,223 params"
echo "shape=4 devices x 1024 envs x 32 steps = 131,072 transitions/update, 4000 updates"
echo "horizon=450 (set on every curriculum level; curriculum.py rewrites env_cfg from that table)"
cat "$RUN_ROOT/source_sync_manifest.txt" || true
scontrol show job "$SLURM_JOB_ID" | grep -E 'JobId=|Partition=|ReqTRES=|AllocTRES=' || true

cd "$SOURCE/terra-baselines"
unset WANDB_RUN_ID WANDB_RESUME
"$VENV/bin/python" -u train_mixed.py \
    --config "$PRESET" \
    --name "$RUN_NAME" \
    --machine euler \
    --seed 20260730 \
    --num_devices 4 \
    --num_envs_per_device 1024 \
    --num_steps 32 \
    --update_epochs 2 \
    --num_minibatches 32 \
    --total_timesteps 524288000 \
    --lr 3e-4 \
    --log_train_interval 1 \
    --log_eval_interval 0 \
    --checkpoint_interval 250 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --keep_checkpoint_history \
    --cache_clear_interval 1000 \
    --eval_episodes 64 \
    --ent_schedule_start 0.15 \
    --ent_schedule_end 0.005 \
    --ent_schedule_steps 3800 \
    --model_size medium \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8_se \
    --encoder_compute_dtype bfloat16 \
    --critic_hidden_dims 512,256 \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --fail_on_nonfinite \
    --finite_check_interval 10

FINAL_COUNT="$(
    find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*_FINAL.pkl' | wc -l
)"
test "$FINAL_COUNT" -eq 1
touch "$RUN_ROOT/${ARM}_TRAINING_COMPLETE"
echo "ARM_${ARM}_TRAINING_COMPLETE"
