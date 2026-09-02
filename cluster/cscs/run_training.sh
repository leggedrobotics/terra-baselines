#!/usr/bin/env bash
# Runs inside the CSCS container with all four GPUs visible to one JAX process.

set -euo pipefail

PROFILE="${1:-}"
[[ -n "$PROFILE" ]] || { echo "Usage: $0 smoke|production [train_mixed.py arguments...]" >&2; exit 2; }
shift

NUM_DEVICES="${NUM_DEVICES:-4}"
RUN_DIR="${TERRA_RUN_DIR:?TERRA_RUN_DIR is required}"
mkdir -p "$RUN_DIR" "$RUN_DIR/checkpoints" "$RUN_DIR/wandb"
cd "$RUN_DIR"

export PYTHONPATH="/workspace/terra:/workspace/terra-baselines${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export PYTHONUNBUFFERED=1
# NVIDIA JAX 24.10 enables its experimental MLIR fusion emitter at level 4.
# Terra's gather/select-heavy PPO update hits a known lowering failure there;
# level 0 selects the established emitter used by Terra's JAX 0.4 runtime.
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_mlir_emitter_level=0"

echo "run_dir=$RUN_DIR"
echo "dataset_path=${DATASET_PATH:?DATASET_PATH is required}"
echo "dataset_size=${DATASET_SIZE:?DATASET_SIZE is required}"
echo "profile=$PROFILE"
echo "xla_flags=$XLA_FLAGS"
echo "host=$(hostname)"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

python -u /workspace/terra-baselines/cluster/cscs/check_jax_runtime.py \
    --min-devices "$NUM_DEVICES"

case "$PROFILE" in
    smoke)
        export WANDB_MODE="${WANDB_MODE:-offline}"
        PROFILE_ARGS=(
            --config solo_excavator
            --name cscs-smoke
            --machine daint
            --num_devices "$NUM_DEVICES"
            --num_envs_per_device 8
            --num_steps 4
            --update_epochs 1
            --num_minibatches 1
            --total_timesteps 128
            --log_train_interval 1
            --log_eval_interval 0
            --checkpoint_interval 1
            --eval_episodes 4
        )
        ;;
    production)
        PROFILE_ARGS=(
            --config solo_excavator
            --machine daint
            --num_devices "$NUM_DEVICES"
            --num_envs_per_device 1024
            --num_steps 32
            --update_epochs 2
            --num_minibatches 16
            --total_timesteps 50000000000
            --log_train_interval 1
            --log_eval_interval 100
            --checkpoint_interval 100
            --eval_episodes 100
        )
        ;;
    *)
        echo "Unknown profile: $PROFILE (expected smoke or production)" >&2
        exit 2
        ;;
esac

# TERRA_RESUME_LATEST continues a run past the partition wall-time cap: pick the
# newest checkpoint of this run and hand it to the trainer, which restores the
# optimizer state and the absolute update counter.
RESUME_ARGS=()
if [[ "${TERRA_RESUME_LATEST:-0}" == "1" ]]; then
    LATEST_CHECKPOINT="$(
        find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -name '*_update_*.pkl' \
            | sed -E 's|.*_update_0*([0-9]+)\.pkl$|\1 &|' \
            | sort -n -k1,1 \
            | tail -n 1 \
            | cut -d' ' -f2-
    )"
    if [[ -n "$LATEST_CHECKPOINT" ]]; then
        RESUME_ARGS=(--resume_from "$LATEST_CHECKPOINT")
        echo "resume_from=$LATEST_CHECKPOINT"
    else
        echo "resume_from=none"
    fi
fi

echo "Launching Terra training profile: $PROFILE"
python -u /workspace/terra-baselines/train_mixed.py "${PROFILE_ARGS[@]}" "$@" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
