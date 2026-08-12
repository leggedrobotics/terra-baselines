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

for REQUIRED in CONTROL_RUN TREATMENT_RUN CONTROL_JOB_ID TREATMENT_JOB_ID \
    BASELINES_ROOT BASELINES_REVISION SEED BANK_ARCHIVE BANK_SHA \
    BANK_DATASET_SHA SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
test "${SLURM_JOB_PARTITION:-}" = gpuhe.24h

CONTROL_ARM=G-V8-XATTN-REWARM-CONTROL
TREATMENT_ARM=G-V8-10M-XATTN-WARM
TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa
P5_ARCHIVE=/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426
OUTPUT_ROOT="$(dirname "$CONTROL_RUN")/whole_v8_fixed_$BASELINES_REVISION"

test "$(dirname "$CONTROL_RUN")" = "$(dirname "$TREATMENT_RUN")"
test "$(basename "$CONTROL_RUN")" = "$CONTROL_ARM"
test "$(basename "$TREATMENT_RUN")" = "$TREATMENT_ARM"
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
for JOB_ID in "$CONTROL_JOB_ID" "$TREATMENT_JOB_ID"; do
    STATE="$(sacct -n -X -P -j "$JOB_ID" --format=JobIDRaw,State | awk -F'|' -v id="$JOB_ID" '$1==id {sub(/\+.*/, "", $2); print $2}')"
    test "$STATE" = COMPLETED
done
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/eval"

tar --zstd -xf "$P5_ARCHIVE" -C "$TMPDIR" \
    campaign/source/terra campaign/runtime/check_jax_runtime.py
tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
TERRA="$TMPDIR/campaign/source/terra"
BANK="$TMPDIR/bank"
RUNTIME_CHECK="$TMPDIR/campaign/runtime/check_jax_runtime.py"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"

module load stack/2024-06 cuda/12.1.1
PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$TERRA:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 4
scontrol show job "$SLURM_JOB_ID" -o > "$OUTPUT_ROOT/slurm_job.txt"
nvidia-smi -q > "$OUTPUT_ROOT/nvidia_smi.txt"

evaluate_policy() {
    local run_dir="$1"
    local slug="$2"
    local -a checkpoint_args=()
    local update token pattern
    for update in 1000 2000 3000 4000; do
        token="$(printf '%06d' "$update")"
        pattern=("$run_dir"/checkpoints/*_update_"$token".pkl)
        test "${#pattern[@]}" -eq 1
        test -f "${pattern[0]}"
        checkpoint_args+=(--checkpoint "${pattern[0]}")
    done
    for panel in promotion development; do
        "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
            "${checkpoint_args[@]}" --bank-root "$BANK" \
            --accepted-panel "$panel" --terra-revision "$TERRA_REVISION" \
            --horizon 450 --seed "$SEED" \
            --expect-completion-contract exact_visible_dump_v1 \
            --output "$OUTPUT_ROOT/eval/${slug}_${panel}.json"
    done
}

evaluate_policy "$CONTROL_RUN" compact
evaluate_policy "$TREATMENT_RUN" 10m

JAX_PLATFORMS=cpu "$PYTHON" "$BASELINES_ROOT/scripts/v8_full_v8_report.py" \
    --compact-promotion "$OUTPUT_ROOT/eval/compact_promotion.json" \
    --compact-development "$OUTPUT_ROOT/eval/compact_development.json" \
    --10m-promotion "$OUTPUT_ROOT/eval/10m_promotion.json" \
    --10m-development "$OUTPUT_ROOT/eval/10m_development.json" \
    --output-dir "$OUTPUT_ROOT/leaderboard"

printf '%s\n' \
    "status=PASSED" \
    "metric_primary=fixed/development/exact_success_rate" \
    "metric_secondary=fixed/development/macro_completion" \
    "online_metric_role=active_stage_diagnostic_only" \
    "reward_type=DENSE" \
    "reward_transition_launched=false" > "$OUTPUT_ROOT/status.env"
echo "V8_STAGE_A_WHOLE_DISTRIBUTION_EVALUATION_PASSED"
