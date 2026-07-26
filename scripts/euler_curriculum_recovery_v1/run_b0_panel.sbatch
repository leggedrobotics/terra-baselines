#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-recovery-b0
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_v1/logs/%x_%j.out

set -euo pipefail
umask 002

: "${PANEL:?submit with a declared B0 panel}"
TRAIN_RELATIVE_PATH="panels/train/$PANEL"
case "$PANEL" in
    foundation_geometry)
        PRESET=b0_foundation_geometry_v1
        TRAIN_SEED=2026072701
        PANEL_COUNT=16
        ;;
    foundation_distance)
        PRESET=b0_foundation_distance_v1
        TRAIN_SEED=2026072702
        PANEL_COUNT=32
        ;;
    trench_distance)
        PRESET=b0_trench_distance_v1
        TRAIN_SEED=2026072703
        PANEL_COUNT=32
        ;;
    trench_side)
        PRESET=b0_trench_side_v1
        TRAIN_SEED=2026072704
        PANEL_COUNT=16
        ;;
    trench_one_d02_isolate)
        PRESET=b0_trench_one_d02_isolate_v1
        TRAIN_SEED=2026072704
        PANEL_COUNT=64
        TRAIN_RELATIVE_PATH=cells/train/t_straight_one_d02
        ;;
    trench_topology)
        PRESET=b0_trench_topology_v1
        TRAIN_SEED=2026072705
        PANEL_COUNT=48
        ;;
    *)
        echo "Unsupported B0 panel: $PANEL" >&2
        exit 2
        ;;
esac
B0_TRAIN_VARIANT="${B0_TRAIN_VARIANT:-base_v1}"
case "$B0_TRAIN_VARIANT" in
    base_v1) ;;
    trench_side_diversity_v1)
        if [[ "$PANEL" != trench_side ]]; then
            echo "trench_side_diversity_v1 requires PANEL=trench_side" >&2
            exit 4
        fi
        PANEL_COUNT=128
        ;;
    foundation_distance_diversity_v1)
        if [[ "$PANEL" != foundation_distance ]]; then
            echo "foundation_distance_diversity_v1 requires PANEL=foundation_distance" >&2
            exit 4
        fi
        PANEL_COUNT=256
        ;;
    *)
        echo "Unsupported B0 training variant: $B0_TRAIN_VARIANT" >&2
        exit 4
        ;;
esac
B0_UPDATES="${B0_UPDATES:-500}"
case "$B0_UPDATES" in
    500 | 1000 | 2000 | 5000) ;;
    *)
        echo "Unsupported bounded B0 update target: $B0_UPDATES" >&2
        exit 3
        ;;
esac
if [[ "$PANEL" == trench_one_d02_isolate && "$B0_UPDATES" != 2000 ]]; then
    echo "trench_one_d02_isolate requires B0_UPDATES=2000" >&2
    exit 3
fi
B0_TIMESTEPS=$((B0_UPDATES * 131072))
EXPECTED_CHECKPOINTS=$((B0_UPDATES / 100))

RUN_ROOT="${RUN_ROOT:-/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_v1}"
SOURCE="$RUN_ROOT/source"
BANK="$RUN_ROOT/bank"
PANEL_ROOT="$RUN_ROOT/runs/$PANEL"
SMOKE_DIR="$PANEL_ROOT/smoke"
CHECKPOINT_DIR="$PANEL_ROOT/checkpoints"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py

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
export DATASET_PATH="$BANK"
export DATASET_SIZE="$PANEL_COUNT"

mkdir -p \
    "$RUN_ROOT/logs" \
    "$RUN_ROOT/wandb" \
    "$SMOKE_DIR" \
    "$CHECKPOINT_DIR"
test -x "$VENV/bin/python"
test -f "$RUNTIME_CHECK"
test -s "$SOURCE/terra/REVISION"
test -s "$SOURCE/terra-baselines/REVISION"
test -f "$RUN_ROOT/manifests/source_files.sha256"
test -f "$RUN_ROOT/manifests/bank_files.sha256"
test -f "$BANK/$TRAIN_RELATIVE_PATH/manifest.jsonl"
test ! -e "$PANEL_ROOT/TRAINING_COMPLETE"
(
    cd "$RUN_ROOT"
    sha256sum --check manifests/source_files.sha256
    sha256sum --check manifests/bank_files.sha256
)

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

"$VENV/bin/python" -m unittest discover \
    -s tests -p 'test_b0_panel_eval.py' -q
"$VENV/bin/python" -m unittest discover \
    -s tests -p 'test_verify_b0_training.py' -q

COMMON_ARGS=(
    --config "$PRESET"
    --machine euler
    --seed "$TRAIN_SEED"
    --num_devices 4
    --num_envs_per_device 1024
    --num_steps 32
    --lr 3e-4
    --update_epochs 2
    --num_minibatches 32
    --log_train_interval 1
    --log_eval_interval 0
    --eval_episodes 32
    --ent_schedule_start 0.15
    --ent_schedule_end 0.005
    --ent_schedule_steps 950
    --model_size base
    --model_core mlp
    --map_encoder resnet_spatial_8x8
    --encoder_compute_dtype float32
    --no_value_clip
    --flat_minibatch_shuffle
    --fail_on_nonfinite
    --finite_check_interval 1
)

echo "=== B0 $PANEL exact update-1 smoke ==="
export WANDB_MODE=disabled
"$VENV/bin/python" -u train_mixed.py \
    "${COMMON_ARGS[@]}" \
    --name "terra-recovery-b0-$PANEL-smoke" \
    --total_timesteps 131072 \
    --checkpoint_interval 1 \
    --checkpoint_dir "$SMOKE_DIR" \
    --keep_checkpoint_history \
    --cache_clear_interval 0

mapfile -t SMOKE_FINALS < <(
    find "$SMOKE_DIR" -maxdepth 1 -type f -name '*_FINAL.pkl'
)
mapfile -t SMOKE_AGGREGATES < <(
    find "$SMOKE_DIR/episode_aggregates" \
        -maxdepth 1 -type f -name '*_update_000001.json'
)
test "${#SMOKE_FINALS[@]}" -eq 1
test "${#SMOKE_AGGREGATES[@]}" -eq 1
"$VENV/bin/python" -u scripts/verify_b0_smoke.py \
    --checkpoint "${SMOKE_FINALS[0]}" \
    --aggregate "${SMOKE_AGGREGATES[0]}" \
    --dataset "$BANK/$TRAIN_RELATIVE_PATH" \
    --panel "$PANEL" \
    --expected-dataset-count "$PANEL_COUNT" \
    --output "$SMOKE_DIR/SMOKE_GATE.json"

echo "=== B0 $PANEL $B0_UPDATES-update bounded panel ==="
export WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME
"$VENV/bin/python" -u train_mixed.py \
    "${COMMON_ARGS[@]}" \
    --name "terra-recovery-b0-$PANEL-u$B0_UPDATES" \
    --total_timesteps "$B0_TIMESTEPS" \
    --checkpoint_interval 100 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --keep_checkpoint_history \
    --cache_clear_interval 1000

UPDATE_CHECKPOINT_COUNT="$(
    find "$CHECKPOINT_DIR" -maxdepth 1 -type f \
        -name '*_update_*.pkl' | wc -l
)"
AGGREGATE_COUNT="$(
    find "$CHECKPOINT_DIR/episode_aggregates" -maxdepth 1 -type f \
        -name '*_update_*.json' | wc -l
)"
FINAL_COUNT="$(
    find "$CHECKPOINT_DIR" -maxdepth 1 -type f \
        -name '*_FINAL.pkl' | wc -l
)"
test "$UPDATE_CHECKPOINT_COUNT" -eq "$EXPECTED_CHECKPOINTS"
test "$AGGREGATE_COUNT" -eq "$B0_UPDATES"
test "$FINAL_COUNT" -eq 1
"$VENV/bin/python" -u scripts/verify_b0_training.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --panel "$PANEL" \
    --expected-updates "$B0_UPDATES" \
    --dataset "$BANK/$TRAIN_RELATIVE_PATH" \
    --expected-dataset-count "$PANEL_COUNT" \
    --output "$PANEL_ROOT/TRAINING_GATE.json"
touch "$PANEL_ROOT/TRAINING_COMPLETE"
echo "B0_${PANEL^^}_TRAINING_COMPLETE"
