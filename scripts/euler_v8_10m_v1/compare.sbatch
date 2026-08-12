#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.24h
#SBATCH --time=23:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 002

for REQUIRED in CONTROL_JOB_ID TREATMENT_JOB_ID CONTROL_RUN TREATMENT_RUN \
    BASELINES_ROOT BASELINES_REVISION SEED BANK_ARCHIVE BANK_SHA \
    BANK_DATASET_SHA BANK_RELEASE_ID SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
[[ "$CONTROL_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$TREATMENT_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$SEED" =~ ^[0-9]+$ ]]
test "${SLURM_JOB_PARTITION:-}" = gpuhe.24h

CONTROL_ARM=G-V8-XATTN-REWARM-CONTROL
TREATMENT_ARM=G-V8-10M-XATTN-WARM
TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
EXPECTED_RELEASE=terra_v8_v6_constraints_v7_adjacent_train96_v5
EXPECTED_BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
EXPECTED_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa
P5_ARCHIVE=/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
RUN_PARENT="$(dirname "$CONTROL_RUN")"
COMPARE_ROOT="$RUN_PARENT/paired_benchmark"
INVENTORY="$COMPARE_ROOT/checkpoint_inventory.json"

test "$BANK_RELEASE_ID" = "$EXPECTED_RELEASE"
test "$BANK_SHA" = "$EXPECTED_BANK_SHA"
test "$BANK_DATASET_SHA" = "$EXPECTED_DATASET_SHA"
test "$(dirname "$TREATMENT_RUN")" = "$RUN_PARENT"
test "$(basename "$CONTROL_RUN")" = "$CONTROL_ARM"
test "$(basename "$TREATMENT_RUN")" = "$TREATMENT_ARM"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test -f "$CONTROL_RUN/run_contract.env"
test -f "$TREATMENT_RUN/run_contract.env"
test ! -e "$COMPARE_ROOT"
mkdir "$COMPARE_ROOT"

accounting_row() {
    local job_id="$1"
    local -a rows
    mapfile -t rows < <(
        sacct -n -X -P -j "$job_id" --format=JobIDRaw,State,ExitCode,Partition,ElapsedRaw |
            awk -F'|' -v id="$job_id" '$1 == id { print }'
    )
    test "${#rows[@]}" -eq 1 || {
        echo "expected one accounting row for $job_id, got ${#rows[@]}" >&2
        exit 3
    }
    printf '%s\n' "${rows[0]}"
}

CONTROL_ROW="$(accounting_row "$CONTROL_JOB_ID")"
TREATMENT_ROW="$(accounting_row "$TREATMENT_JOB_ID")"
printf '%s\n' "$CONTROL_ROW" "$TREATMENT_ROW" > "$COMPARE_ROOT/screen_sacct.txt"
IFS='|' read -r _ CONTROL_STATE _ CONTROL_PARTITION CONTROL_ELAPSED _ <<< "$CONTROL_ROW"
IFS='|' read -r _ TREATMENT_STATE _ TREATMENT_PARTITION TREATMENT_ELAPSED _ <<< "$TREATMENT_ROW"
CONTROL_STATE="${CONTROL_STATE%%+*}"
TREATMENT_STATE="${TREATMENT_STATE%%+*}"
case "$CONTROL_STATE" in COMPLETED|TIMEOUT) ;; *) echo "control ended $CONTROL_STATE" >&2; exit 3 ;; esac
case "$TREATMENT_STATE" in COMPLETED|TIMEOUT) ;; *) echo "treatment ended $TREATMENT_STATE" >&2; exit 3 ;; esac
test "$CONTROL_PARTITION" = gpuhe.24h
test "$TREATMENT_PARTITION" = gpuhe.24h
[[ "$CONTROL_ELAPSED" =~ ^[1-9][0-9]*$ ]]
[[ "$TREATMENT_ELAPSED" =~ ^[1-9][0-9]*$ ]]

PYTHON="$VENV/bin/python"
JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/v8_10m_compare.py" \
    inventory --control-run "$CONTROL_RUN" --treatment-run "$TREATMENT_RUN" \
    --control-job-id "$CONTROL_JOB_ID" --treatment-job-id "$TREATMENT_JOB_ID" \
    --control-job-state "$CONTROL_STATE" \
    --treatment-job-state "$TREATMENT_STATE" \
    --control-elapsed-seconds "$CONTROL_ELAPSED" \
    --treatment-elapsed-seconds "$TREATMENT_ELAPSED" \
    --evaluator-job-id "$SLURM_JOB_ID" --output "$INVENTORY" \
    > "$COMPARE_ROOT/selected_checkpoints.tsv"

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
scontrol show job "$SLURM_JOB_ID" -o > "$COMPARE_ROOT/evaluator_slurm_job.txt"
nvidia-smi -q > "$COMPARE_ROOT/nvidia_smi.txt"
mkdir "$COMPARE_ROOT/eval"

evaluate_arm() {
    local arm="$1"
    local slug="$2"
    local -a checkpoints checkpoint_args
    mapfile -t checkpoints < <(
        "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(x["path"]) for x in d["jobs"][sys.argv[2]]["selected_checkpoints"]]' "$INVENTORY" "$arm"
    )
    test "${#checkpoints[@]}" -ge 1
    checkpoint_args=()
    for checkpoint in "${checkpoints[@]}"; do
        checkpoint_args+=(--checkpoint "$checkpoint")
    done
    for panel in promotion development; do
        JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
            "${checkpoint_args[@]}" --bank-root "$BANK" \
            --accepted-panel "$panel" --terra-revision "$TERRA_REVISION" \
            --horizon 450 --seed "$SEED" \
            --expect-completion-contract exact_visible_dump_v1 \
            --output "$COMPARE_ROOT/eval/${slug}_${panel}.json"
        JAX_PLATFORMS=cuda "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
            "${checkpoint_args[@]}" --bank-root "$BANK" \
            --capability-panel "$panel" --terra-revision "$TERRA_REVISION" \
            --horizon 450 --seed "$SEED" \
            --expect-completion-contract exact_visible_dump_v1 \
            --output "$COMPARE_ROOT/eval/${slug}_capability_${panel}.json"
    done
}

evaluate_arm "$CONTROL_ARM" control
evaluate_arm "$TREATMENT_ARM" treatment

JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/v8_10m_compare.py" \
    summarize --inventory "$INVENTORY" \
    --control-promotion "$COMPARE_ROOT/eval/control_promotion.json" \
    --control-development "$COMPARE_ROOT/eval/control_development.json" \
    --control-capability-promotion "$COMPARE_ROOT/eval/control_capability_promotion.json" \
    --control-capability-development "$COMPARE_ROOT/eval/control_capability_development.json" \
    --treatment-promotion "$COMPARE_ROOT/eval/treatment_promotion.json" \
    --treatment-development "$COMPARE_ROOT/eval/treatment_development.json" \
    --treatment-capability-promotion "$COMPARE_ROOT/eval/treatment_capability_promotion.json" \
    --treatment-capability-development "$COMPARE_ROOT/eval/treatment_capability_development.json" \
    --output-dir "$COMPARE_ROOT/leaderboard"

printf '%s\n' \
    "status=PASSED" \
    "control_job_id=$CONTROL_JOB_ID" \
    "control_job_state=$CONTROL_STATE" \
    "treatment_job_id=$TREATMENT_JOB_ID" \
    "treatment_job_state=$TREATMENT_STATE" \
    "reward_launched=false" > "$COMPARE_ROOT/status.env"
echo "V8_10M_PAIRED_EVALUATION_PASSED"
