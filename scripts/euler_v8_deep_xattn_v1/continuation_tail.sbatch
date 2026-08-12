#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.24h
#SBATCH --time=23:45:00
#SBATCH --no-requeue
#SBATCH --job-name=terra-v8-cont-tail

set -euo pipefail
umask 002

for REQUIRED in CONTINUATION_JOB_ID CONTINUATION_RUN_DIR ARM BASELINES_ROOT \
    BASELINES_REVISION SEED BANK_ARCHIVE BANK_SHA BANK_DATASET_SHA \
    BANK_RELEASE_ID QUALIFIED_RECEIPT QUALIFIED_RECEIPT_SHA SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
[[ "$CONTINUATION_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$SEED" =~ ^[0-9]+$ ]]
test "${SLURM_JOB_PARTITION:-}" = gpuhe.24h

case "$ARM" in
    G-DEEP-V8-DENSE-WARM|G-DEEP-XATTN-V8-DENSE-WARM) ;;
    *) echo "unsupported arm '$ARM'" >&2; exit 2 ;;
esac

TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
EXPECTED_BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
EXPECTED_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa
P5_ARCHIVE=/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
RUN_CONTRACT="$CONTINUATION_RUN_DIR/run_contract.env"
TAIL_DIR="$CONTINUATION_RUN_DIR/continuation_tail"
INVENTORY="$TAIL_DIR/checkpoint_inventory.json"

test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$BANK_SHA" = "$EXPECTED_BANK_SHA"
test "$BANK_DATASET_SHA" = "$EXPECTED_DATASET_SHA"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "$(sha256sum "$QUALIFIED_RECEIPT" | awk '{print $1}')" = "$QUALIFIED_RECEIPT_SHA"
test -f "$RUN_CONTRACT"
test -d "$CONTINUATION_RUN_DIR/checkpoints"
test ! -e "$TAIL_DIR"
test "${SLURM_CPUS_PER_TASK:-}" = 4
mkdir "$TAIL_DIR"

mapfile -t PARENT_ROWS < <(
    sacct -n -X -P -j "$CONTINUATION_JOB_ID" \
        --format=JobIDRaw,State,ExitCode,Partition |
        awk -F'|' -v id="$CONTINUATION_JOB_ID" '$1 == id { print }'
)
test "${#PARENT_ROWS[@]}" -eq 1 || {
    echo "expected one continuation accounting row, got ${#PARENT_ROWS[@]}" >&2
    exit 3
}
printf '%s\n' "${PARENT_ROWS[0]}" > "$TAIL_DIR/continuation_sacct.txt"
IFS='|' read -r ACCOUNTED_JOB_ID PARENT_STATE PARENT_EXIT_CODE PARENT_PARTITION \
    <<< "${PARENT_ROWS[0]}"
PARENT_STATE="${PARENT_STATE%%+*}"
test "$ACCOUNTED_JOB_ID" = "$CONTINUATION_JOB_ID"
case "$PARENT_STATE" in
    COMPLETED|TIMEOUT) ;;
    *) echo "continuation ended $PARENT_STATE; refusing evaluation" >&2; exit 3 ;;
esac
test "$PARENT_PARTITION" = gpuhe.120h

PYTHON="$VENV/bin/python"
"$PYTHON" "$BASELINES_ROOT/scripts/euler_v8_deep_xattn_v1/continuation_tail_eval.py" \
    inventory --qualified-receipt "$QUALIFIED_RECEIPT" \
    --run-dir "$CONTINUATION_RUN_DIR" --run-contract "$RUN_CONTRACT" \
    --job-id "$CONTINUATION_JOB_ID" --job-state "$PARENT_STATE" \
    --job-exit-code "$PARENT_EXIT_CODE" --job-partition "$PARENT_PARTITION" \
    --evaluator-job-id "$SLURM_JOB_ID" --output "$INVENTORY" \
    > "$TAIL_DIR/selected_checkpoints.txt"
mapfile -t CHECKPOINTS < "$TAIL_DIR/selected_checkpoints.txt"
test "${#CHECKPOINTS[@]}" -ge 2

tar --zstd -xf "$P5_ARCHIVE" -C "$TMPDIR" \
    campaign/source/terra campaign/runtime/check_jax_runtime.py
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
TERRA="$TMPDIR/campaign/source/terra"
BANK="$TMPDIR/bank"
RUNTIME_CHECK="$TMPDIR/campaign/runtime/check_jax_runtime.py"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

module load stack/2024-06 cuda/12.1.1
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$TERRA:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 4
for GPU_NAME in "${GPU_NAMES[@]}"; do
    test "$GPU_NAME" = "NVIDIA GeForce RTX 4090" || {
        echo "unsupported GPU: $GPU_NAME" >&2
        exit 3
    }
done
export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 4
MAPS_PER_CONDITION="$(PYTHONPATH="$BASELINES_ROOT" "$PYTHON" "$BASELINES_ROOT/scripts/euler_accepted_bank_v1/validate_training_bank.py" "$BANK" --expected-maps-per-condition 96 --expected-release-id "$EXPECTED_RELEASE")"
test "$MAPS_PER_CONDITION" = 96

scontrol show job "$SLURM_JOB_ID" -o > "$TAIL_DIR/evaluator_slurm_job.txt"
nvidia-smi -q > "$TAIL_DIR/nvidia_smi.txt"
CHECKPOINT_ARGS=()
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_ARGS+=(--checkpoint "$CHECKPOINT")
done
mkdir "$TAIL_DIR/eval"
for PANEL in promotion development; do
    JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --accepted-panel "$PANEL" \
        --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --output "$TAIL_DIR/eval/$PANEL.json"
    JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        "${CHECKPOINT_ARGS[@]}" --bank-root "$BANK" --capability-panel "$PANEL" \
        --terra-revision "$TERRA_REVISION" --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --output "$TAIL_DIR/eval/capability_$PANEL.json"
done

JAX_PLATFORMS=cpu "$PYTHON" \
    "$BASELINES_ROOT/scripts/euler_v8_deep_xattn_v1/continuation_tail_eval.py" \
    summarize --inventory "$INVENTORY" \
    --qualified-receipt "$QUALIFIED_RECEIPT" --bank-root "$BANK" \
    --promotion "$TAIL_DIR/eval/promotion.json" \
    --development "$TAIL_DIR/eval/development.json" \
    --capability-promotion "$TAIL_DIR/eval/capability_promotion.json" \
    --capability-development "$TAIL_DIR/eval/capability_development.json" \
    --output-dir "$TAIL_DIR/benchmark"

QUALIFIED="$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["qualified_for_reward_curriculum"]).lower())' "$TAIL_DIR/benchmark/dense_reward_gate_receipt.json")"
printf '%s\n' \
    "status=PASSED" \
    "continuation_job_id=$CONTINUATION_JOB_ID" \
    "continuation_state=$PARENT_STATE" \
    "evaluated_checkpoints=${#CHECKPOINTS[@]}" \
    "qualified_for_reward_curriculum=$QUALIFIED" \
    "reward_launched=false" > "$TAIL_DIR/status.env"
echo "V8_CONTINUATION_TAIL_PASSED arm=$ARM checkpoints=${#CHECKPOINTS[@]} reward_gate=$QUALIFIED"
