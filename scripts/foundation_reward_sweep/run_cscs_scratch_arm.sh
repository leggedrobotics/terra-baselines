#!/usr/bin/env bash
# Four independent one-GPU policies: foundation/trench, each control/2x.
set -euo pipefail
: "${CAMPAIGN_ROOT:?}" "${BASELINES_ROOT:?}" "${TERRA_ROOT:?}"
case "${SLURM_PROCID:?}" in
    0) TASK_FAMILY=foundation; COST_MULTIPLIER=0 ;;
    1) TASK_FAMILY=foundation; COST_MULTIPLIER=2 ;;
    2) TASK_FAMILY=trench; COST_MULTIPLIER=0 ;;
    3) TASK_FAMILY=trench; COST_MULTIPLIER=2 ;;
    *) exit 2 ;;
esac
export TASK_FAMILY SEED=20260909
export RUN_NAME="${TASK_FAMILY}-scratch-c${COST_MULTIPLIER}-s${SEED}"
ARM_DIR="$CAMPAIGN_ROOT/segments/$SLURM_JOB_ID/$RUN_NAME"
mkdir -p "$ARM_DIR"
exec > "$ARM_DIR/process.log" 2>&1
export TERRA_PYTHON=python MACHINE=daint
export DATASET_PATH="$CAMPAIGN_ROOT/inputs/$TASK_FAMILY"
if [[ "$TASK_FAMILY" == foundation ]]; then
    export DATASET_SIZE=256 DISTANCE_SIDECAR_SHA=6b2675998403ed2d6125d955fca446404fbdf260e0a0c2cf7b9864cbdd1fb2bf
else
    export DATASET_SIZE=1440 DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
fi
export EXECUTABLE_DIG_OBSERVATION=1 BANK_TRANSFER=0
export LATERAL_DIG_COST=0 BASE_TRAVEL_COST=0 BASE_TURN_COST=0
if [[ "$COST_MULTIPLIER" == 2 ]]; then
    export LATERAL_DIG_COST=0.5 BASE_TRAVEL_COST=0.01 BASE_TURN_COST=0.04
fi
export PYTHONPATH="$TERRA_ROOT:$BASELINES_ROOT"
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_COMPILATION_CACHE=true JAX_THREEFRY_PARTITIONABLE=true
export JAX_COMPILATION_CACHE_DIR="$CAMPAIGN_ROOT/jax-cache/$RUN_NAME"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy
export WANDB_ENTITY=aless-weber-eth WANDB_PROJECT=mixed-agents
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
printf '%s\n' "arm=$RUN_NAME" "host=$(hostname)" "job=$SLURM_JOB_ID" \
    "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:?}" "started=$(date -Is)"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || exit 3
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name --format=csv,noheader
python - <<'PY'
import jax
devices = jax.devices()
assert len(devices) == 1 and devices[0].platform == 'gpu', devices
assert 'GH200' in devices[0].device_kind, devices
print('PASS one GH200 per independent policy:', devices, flush=True)
PY
python -u "$BASELINES_ROOT/cluster/cscs/check_jax_runtime.py" --min-devices 1

export INITIALIZATION=scratch START_UPDATE=0 TARGET_UPDATE=2 CHECKPOINT_INTERVAL=1
export RUN_DIR="$ARM_DIR/smoke" WANDB_MODE=disabled JAX_LOG_COMPILES=1
export WANDB_DIR="$RUN_DIR/wandb" TRAIN_ENTRY="$BASELINES_ROOT/train_mixed.py"
unset RESUME_FROM WANDB_RUN_ID WANDB_RESUME
bash "$BASELINES_ROOT/scripts/foundation_reward_sweep/train.sh"
JAX_PLATFORMS=cpu python -u "$BASELINES_ROOT/scripts/foundation_reward_sweep/verify_scratch_smoke.py" \
    "$RUN_DIR/checkpoints/${RUN_NAME}_update_000001.pkl" \
    "$RUN_DIR/checkpoints/${RUN_NAME}_FINAL.pkl" --seed "$SEED" --cost-multiplier "$COST_MULTIPLIER" \
    --task-family "$TASK_FAMILY" \
    > "$ARM_DIR/smoke_check.json"

# Resume only this campaign's two fresh updates; no old policy is imported.
export RESUME_FROM="$RUN_DIR/checkpoints/${RUN_NAME}_FINAL.pkl"
export INITIALIZATION=resume START_UPDATE=2 TARGET_UPDATE=500000 CHECKPOINT_INTERVAL=500
export RUN_DIR="$ARM_DIR/training" WANDB_MODE=offline JAX_LOG_COMPILES=0
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_RUN_ID="terra-scratch-${TASK_FAMILY}-c${COST_MULTIPLIER}-s${SEED}-${SLURM_JOB_ID}"
printf '%s\n' 'Finite scratch smoke passed; continuing this new run from u2.'
exec bash "$BASELINES_ROOT/scripts/foundation_reward_sweep/train.sh"
