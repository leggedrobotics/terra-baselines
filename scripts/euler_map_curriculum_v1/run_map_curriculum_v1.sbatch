#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-map-v1
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724/logs/%x_%j.out

set -euo pipefail
umask 002

: "${ARM:?submit with --export=ALL,ARM=flat or ARM=staged}"
case "$ARM" in
    flat)
        PRESET=map_curriculum_flat_v1
        ;;
    staged)
        PRESET=map_curriculum_staged_v1
        ;;
    *)
        echo "Unsupported ARM=$ARM" >&2
        exit 2
        ;;
esac

RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724
SOURCE="$RUN_ROOT/source"
BANK="$RUN_ROOT/bank"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py
PARENT="$RUN_ROOT/parent/E8_FINAL.pkl"
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
export DATASET_SIZE=256

mkdir -p "$CHECKPOINT_DIR" "$RUN_ROOT/wandb" "$RUN_ROOT/logs"
test -f "$RUN_ROOT/SMOKE_GATE_PASSED"
test -f "$PARENT"
test -f "$BANK/validation.json"

HOME_USED_GB="$(
    lquota |
        awk -F'|' '/\\/cluster\\/home\\/lterenzi/ && /space/ {
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

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "Unsupported GPU: $GPU_NAME" >&2
        exit 41
    }
done
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 4

cd "$SOURCE/terra-baselines"
unset WANDB_RUN_ID WANDB_RESUME
"$VENV/bin/python" -u train_mixed.py \
    --config "$PRESET" \
    --name "terra-map-v1-${ARM}" \
    --machine euler \
    --num_devices 4 \
    --num_envs_per_device 1024 \
    --num_steps 32 \
    --update_epochs 2 \
    --num_minibatches 32 \
    --total_timesteps 524288000 \
    --log_train_interval 1 \
    --log_eval_interval 0 \
    --checkpoint_interval 500 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --keep_checkpoint_history \
    --cache_clear_interval 1000 \
    --eval_episodes 64 \
    --ent_schedule_start 0.02 \
    --ent_schedule_end 0.005 \
    --ent_schedule_steps 10000 \
    --model_size medium \
    --model_core mlp \
    --map_encoder resnet_spatial_8x8_se \
    --encoder_compute_dtype bfloat16 \
    --critic_hidden_dims 512,256 \
    --warm_start_from "$PARENT" \
    --no_value_clip \
    --flat_minibatch_shuffle \
    --fail_on_nonfinite \
    --finite_check_interval 10

FINAL_COUNT="$(
    find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*_FINAL.pkl' |
        wc -l
)"
test "$FINAL_COUNT" -eq 1
touch "$RUN_ROOT/${ARM}_TRAINING_COMPLETE"
echo "${ARM^^}_TRAINING_COMPLETE"
