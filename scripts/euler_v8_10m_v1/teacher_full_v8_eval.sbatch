#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.4h
#SBATCH --time=03:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 002

for REQUIRED in BASELINES_ROOT BASELINES_REVISION SEED BANK_ARCHIVE BANK_SHA \
    BANK_DATASET_SHA TEACHER_CHECKPOINT TEACHER_SHA OUTPUT_ROOT SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done
test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h

TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
P5_SHA=f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa
P5_ARCHIVE=/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-$P5_SHA/campaign.tar.zst
VENV=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426

test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(sha256sum "$P5_ARCHIVE" | awk '{print $1}')" = "$P5_SHA"
test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_SHA"
test "$(sha256sum "$TEACHER_CHECKPOINT" | awk '{print $1}')" = "$TEACHER_SHA"
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

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 1
case "${GPU_NAMES[0]}" in
    "NVIDIA GeForce RTX 3090"|"NVIDIA GeForce RTX 4090") ;;
    *) echo "unsupported GPU: ${GPU_NAMES[0]}" >&2; exit 3 ;;
esac
export JAX_PLATFORMS=cuda
"$PYTHON" "$RUNTIME_CHECK" --min-devices 1
scontrol show job "$SLURM_JOB_ID" -o > "$OUTPUT_ROOT/slurm_job.txt"
nvidia-smi -q > "$OUTPUT_ROOT/nvidia_smi.txt"

for PANEL in promotion development; do
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        --checkpoint "$TEACHER_CHECKPOINT" --bank-root "$BANK" \
        --accepted-panel "$PANEL" --terra-revision "$TERRA_REVISION" \
        --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --output "$OUTPUT_ROOT/eval/main_${PANEL}.json"
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        --checkpoint "$TEACHER_CHECKPOINT" --bank-root "$BANK" \
        --capability-panel "$PANEL" --terra-revision "$TERRA_REVISION" \
        --horizon 450 --seed "$SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --output "$OUTPUT_ROOT/eval/capability_${PANEL}.json"
done

printf '%s\n' \
    "status=PASSED" \
    "teacher_checkpoint_sha256=$TEACHER_SHA" \
    "metric_primary=fixed/development/exact_success_rate" \
    "metric_secondary=fixed/development/macro_completion" \
    "online_metric_role=active_stage_diagnostic_only" > "$OUTPUT_ROOT/status.env"
echo "V8_REFERENCE_TEACHER_WHOLE_DISTRIBUTION_EVALUATION_PASSED"
