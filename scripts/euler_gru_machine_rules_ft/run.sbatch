#!/usr/bin/env bash
# GRU fine-tune under Terra's machine working rules: native resume of the
# released GRU u100000 (model, Adam state, update counter) with the rules of
# preset gru_generalist_512_machine_rules and --dug_clearance_m, to u110000.
# The release recipe otherwise; the teacher is retained for the native resume
# check only (its KL weight has been 0 since u10000, so no teacher runs).
set -euo pipefail
umask 027

for REQUIRED in TARGET_UPDATE RUN_DIR RUN_NAME BASELINES_ROOT BASELINES_REVISION \
    RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION SEED VENV RUNTIME_LOCK_SHA \
    TERRA_EULER_USER TERRA_EULER_HOME_ROOT WANDB_ENTITY WANDB_PROJECT WANDB_MODE \
    BANK_ARCHIVE BANK_ARCHIVE_SHA BANK_MAPS_PATH BANK_DATASET_SIZE \
    BANK_DISTANCE_SIDECAR_SHA PARENT PARENT_SHA TEACHER TEACHER_SHA NUM_DEVICES \
    ENVS_PER_DEVICE KL_ANNEAL_UPDATES GPU_TYPE PRESET \
    DUG_CLEARANCE_M SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

# One segment trains a run at a time; a second one exits.
LOCK="$RUN_DIR/active_segment"
mkdir -p "$RUN_DIR/checkpoints"
if ! mkdir "$LOCK" 2>/dev/null; then
    holder="$(cat "$LOCK/job" 2>/dev/null || true)"
    if [ -n "$holder" ] && squeue -h -j "$holder" -o %T 2>/dev/null | grep -q RUNNING; then
        echo "segment $holder is training this run; nothing to do"
        exit 0
    fi
    echo "taking over a stale lock from segment ${holder:-unknown}"
fi
echo "$SLURM_JOB_ID" > "$LOCK/job"
release_lock() {
    if [ "$(cat "$LOCK/job" 2>/dev/null)" = "$SLURM_JOB_ID" ]; then
        rm -f "$LOCK/job"
        rmdir "$LOCK" 2>/dev/null || true
    fi
}
trap release_lock EXIT
trap 'release_lock; exit 143' TERM INT

if compgen -G "$RUN_DIR/checkpoints/*_FINAL.pkl" > /dev/null; then
    echo "run already reached its target update"
    exit 0
fi
# Checkpoint names carry zero-padded updates and are written atomically. The
# first segment resumes the parent; later ones this run's newest checkpoint.
RESUME_FROM="$( (ls "$RUN_DIR"/checkpoints/*_update_*.pkl 2>/dev/null || true) | sort | tail -n 1)"
if [ -z "$RESUME_FROM" ]; then
    RESUME_FROM="$PARENT"
    test "$(sha256sum "$PARENT" | awk '{print $1}')" = "$PARENT_SHA"
fi
test -r "$RESUME_FROM"
echo "resume_from=$RESUME_FROM"
case "$GPU_TYPE" in
    rtx_4090) GPU_MODEL="RTX 4090" ;;
    rtx_3090) GPU_MODEL="RTX 3090" ;;
    *) echo "GPU_TYPE must be rtx_4090 or rtx_3090" >&2; exit 2 ;;
esac

test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
test "$(sha256sum "$TEACHER" | awk '{print $1}')" = "$TEACHER_SHA"
test "$(sha256sum "$VENV/requirements.lock.txt" | awk '{print $1}')" = "$RUNTIME_LOCK_SHA"

# Hard GPU guard before JAX or W&B start.
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq "$NUM_DEVICES"
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qv "$GPU_MODEL"; then
    echo "unexpected GPU model; expected $GPU_MODEL" >&2
    exit 3
fi

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$RUN_DIR/jax-cache}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
mkdir -p "$RUN_DIR/wandb" "$JAX_COMPILATION_CACHE_DIR"
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
export DATASET_PATH="$TMPDIR/bank"
export DATASET_SIZE="$BANK_DATASET_SIZE"
test -d "$DATASET_PATH/$BANK_MAPS_PATH/metadata"

module purge
module load stack/2024-06 eth_proxy
PYTHON="$VENV/bin/python"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache" WANDB_CONFIG_DIR="$TMPDIR/wandb-config"
if printf '%s' "${LD_LIBRARY_PATH:-}" | grep -Eqi 'cuda|cudnn|cublas|nccl|nvrtc|cupti|cusolver|cusparse|cufft|nvjitlink'; then
    echo "CUDA libraries leaked into LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >&2
    exit 3
fi
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_mlir_emitter_level=0"

"$PYTHON" - "$VENV/requirements.lock.txt" <<'PY'
import importlib.metadata as metadata
import sys
from packaging.requirements import Requirement
from packaging.version import Version

mismatches = []
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    requirement = Requirement(line.strip())
    (pin,) = list(requirement.specifier)
    observed = metadata.version(requirement.name)
    if pin.operator != "==" or Version(observed) != Version(pin.version):
        mismatches.append(f"{requirement.name}: expected {pin}, observed {observed}")
if mismatches:
    raise SystemExit("runtime lock mismatch:\n" + "\n".join(mismatches))
print("PASS runtime lock")
PY
"$PYTHON" "$BASELINES_ROOT/cluster/cscs/check_jax_runtime.py" --min-devices "$NUM_DEVICES"
# The Terra runtime must carry the machine working rules and the centring.
"$PYTHON" -c "from terra.config import AgentConfig as A; f = A._fields; assert {'dig_min_radius_m', 'dug_clearance_m', 'centre_chassis_on_base'} <= set(f), f; print('PASS Terra machine working rules')"

# One W&B run per Slurm segment; continuations carry the job id.
if [ "$RESUME_FROM" = "$PARENT" ]; then
    export WANDB_RUN_ID="$RUN_NAME" WANDB_RESUME=never
else
    export WANDB_RUN_ID="${RUN_NAME}_resume_j${SLURM_JOB_ID}" WANDB_RESUME=never
fi
export WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_DIR="$RUN_DIR/wandb"

scontrol show job "$SLURM_JOB_ID" -o > "$RUN_DIR/slurm_job.$SLURM_JOB_ID.txt"
TRANSITIONS_PER_UPDATE=$((NUM_DEVICES * ENVS_PER_DEVICE * 32))
TOTAL_TIMESTEPS=$((TRANSITIONS_PER_UPDATE * TARGET_UPDATE))
printf '%s\n' \
    "status=RUNNING" \
    "experiment=gru_machine_rules_finetune" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "resume_from=$RESUME_FROM" \
    "parent_sha256=$PARENT_SHA" \
    "wandb_run_id=$WANDB_RUN_ID" \
    "wandb_mode=$WANDB_MODE" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "runtime_venv=$VENV" \
    "bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "bank_maps_path=$BANK_MAPS_PATH" \
    "bank_dataset_size=$BANK_DATASET_SIZE" \
    "distance_sidecar_sha256=$BANK_DISTANCE_SIDECAR_SHA" \
    "teacher_sha256=$TEACHER_SHA" \
    "kickstart_kl=parent_schedule_1_to_0_over_${KL_ANNEAL_UPDATES}_updates_inactive" \
    "preset=$PRESET" \
    "rules=dig_min_radius_m=4.0,dump_max_radius_m=6.0,centre_chassis_on_base=true,dug_clearance_m=$DUG_CLEARANCE_M" \
    "seed=$SEED" \
    "gpu=${NUM_DEVICES}x$GPU_TYPE" \
    "envs_per_device=$ENVS_PER_DEVICE" \
    "transitions_per_update=$TRANSITIONS_PER_UPDATE" \
    "target_update=$TARGET_UPDATE" > "$RUN_DIR/run_contract.env"
cp "$RUN_DIR/run_contract.env" "$RUN_DIR/run_contract.job${SLURM_JOB_ID}.env"

CHECKPOINT_DIR="$RUN_DIR/checkpoints"
source "$BASELINES_ROOT/scripts/euler_gru_machine_rules_ft/train_args.sh"
TRAIN_RC=0
"$PYTHON" -u "$BASELINES_ROOT/train_mixed.py" "${TRAIN_ARGS[@]}" \
    --machine euler \
    --finite_check_interval 10 \
    --log_train_interval 10 \
    --checkpoint_interval 500 || TRAIN_RC=$?

if [ "$TRAIN_RC" -eq 0 ]; then
    echo "status=COMPLETED" >> "$RUN_DIR/run_contract.env"
    exit 0
fi
echo "status=FAILED train_rc=$TRAIN_RC" >> "$RUN_DIR/run_contract.env"
exit "$TRAIN_RC"
