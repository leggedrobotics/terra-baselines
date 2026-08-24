#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpuhe.4h
#SBATCH --time=03:45:00
#SBATCH --no-requeue

set -euo pipefail
umask 027

for REQUIRED in \
    OUTPUT_DIR BASELINES_ROOT BASELINES_REVISION \
    GRU_TRAINING_SOURCE GRU_SOURCE_REVISION \
    FF_TRAINING_SOURCE FF_SOURCE_REVISION \
    RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION \
    BANK_ARCHIVE BANK_ARCHIVE_SHA BANK_DATASET_SHA PROMOTION_MANIFEST_SHA \
    PROTOCOL_TERRA_REVISION DISTANCE_SIDECAR_SHA \
    GRU_U40000 GRU_U40000_SHA GRU_U44000 GRU_U44000_SHA \
    FF_U44000 FF_U44000_SHA FF_U86000 FF_U86000_SHA \
    EVAL_SEED VENV TERRA_EULER_USER TERRA_EULER_HOME_ROOT \
    SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h
test "${SLURM_CPUS_PER_TASK:-}" = 8
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$GRU_TRAINING_SOURCE/REVISION")" = "$GRU_SOURCE_REVISION"
test "$(cat "$FF_TRAINING_SOURCE/REVISION")" = "$FF_SOURCE_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test -x "$VENV/bin/python"
test -f "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py"
test -f "$BASELINES_ROOT/scripts/render_v8_fixed_panel_gifs.py"
test ! -e "$OUTPUT_DIR"

HOME_USED_GB="$(lquota | "$BASELINES_ROOT/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT")"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
for CHECKPOINT_AND_SHA in \
    "$GRU_U40000:$GRU_U40000_SHA" \
    "$GRU_U44000:$GRU_U44000_SHA" \
    "$FF_U44000:$FF_U44000_SHA" \
    "$FF_U86000:$FF_U86000_SHA"; do
    CHECKPOINT="${CHECKPOINT_AND_SHA%%:*}"
    EXPECTED_SHA="${CHECKPOINT_AND_SHA##*:}"
    test -f "$CHECKPOINT"
    test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_SHA"
done

tar --zstd -xf "$BANK_ARCHIVE" -C "$TMPDIR" bank
BANK="$TMPDIR/bank"
test "$(sha256sum "$BANK/dataset.json" | awk '{print $1}')" = "$BANK_DATASET_SHA"
test "$(sha256sum "$BANK/evaluation/main/promotion/manifest.jsonl" | awk '{print $1}')" = "$PROMOTION_MANIFEST_SHA"

module load stack/2024-06 cuda/12.1.1
PYTHON="$VENV/bin/python"
SITE_PACKAGES="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$RUNTIME_TERRA_ROOT:$BASELINES_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_enable_cudnn_frontend=false
export JAX_PLATFORMS=cuda,cpu WANDB_MODE=disabled
export XDG_CACHE_HOME="$TMPDIR/xdg-cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
test "${#GPU_NAMES[@]}" -eq 1
test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"

"$PYTHON" - <<'PY'
import jax
import jax.numpy as jnp
from jax import lax
from eval_mcts import EVAL_FORWARD_CHUNK

assert jax.local_device_count() == 1, jax.devices()
assert EVAL_FORWARD_CHUNK == 120, EVAL_FORWARD_CHUNK

def loss(x, kernel):
    value = lax.conv_general_dilated(
        x,
        kernel,
        (1, 1),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return jnp.mean(value * value)

x = jnp.ones((1, 16, 16, 4), dtype=jnp.float32)
kernel = jnp.ones((3, 3, 4, 8), dtype=jnp.float32)
gradient = jax.jit(jax.grad(loss))(x, kernel)
assert gradient.shape == x.shape
assert bool(jnp.all(jnp.isfinite(gradient)))
print(jax.devices(), float(jnp.linalg.norm(gradient)), EVAL_FORWARD_CHUNK)
PY

mkdir -p "$OUTPUT_DIR"
scontrol show job "$SLURM_JOB_ID" -o > "$OUTPUT_DIR/slurm_job.txt"
nvidia-smi -q > "$OUTPUT_DIR/nvidia_smi.txt"
CONTRACT="$OUTPUT_DIR/run_contract.env"
printf '%s\n' \
    "status=RUNNING" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "evaluation_contract=v8_gru_ff_common_runtime_promotion_v1" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "gru_training_source=$GRU_TRAINING_SOURCE" \
    "gru_training_source_revision=$GRU_SOURCE_REVISION" \
    "ff_training_source=$FF_TRAINING_SOURCE" \
    "ff_training_source_revision=$FF_SOURCE_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "bank_dataset_sha256=$BANK_DATASET_SHA" \
    "promotion_manifest_sha256=$PROMOTION_MANIFEST_SHA" \
    "gru_u40000_sha256=$GRU_U40000_SHA" \
    "gru_u44000_sha256=$GRU_U44000_SHA" \
    "ff_u44000_sha256=$FF_U44000_SHA" \
    "ff_u86000_sha256=$FF_U86000_SHA" \
    "dashboard_builder_sha256=$(sha256sum "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py" | awk '{print $1}')" \
    "media_renderer_sha256=$(sha256sum "$BASELINES_ROOT/scripts/render_v8_fixed_panel_gifs.py" | awk '{print $1}')" \
    "media_review_limit=40" \
    "media_frame_stride=10" \
    "policy_mode=deterministic_greedy_unmasked" \
    "horizon=450" \
    "forward_chunk=120" \
    "seed=$EVAL_SEED" \
    "xla_flags=$XLA_FLAGS" > "$CONTRACT"

record_failure() {
    local exit_code="$?"
    if [ "$exit_code" -ne 0 ]; then
        printf '%s\n' \
            "status=FAILED" \
            "exit_code=$exit_code" >> "$CONTRACT"
    fi
}
trap record_failure EXIT

evaluate_treatment() {
    local first_checkpoint="$1"
    local second_checkpoint="$2"
    local output="$3"
    "$PYTHON" "$BASELINES_ROOT/eval_fixed_bank.py" \
        --checkpoint "$first_checkpoint" \
        --checkpoint "$second_checkpoint" \
        --bank-root "$BANK" \
        --accepted-panel promotion \
        --terra-revision "$PROTOCOL_TERRA_REVISION" \
        --horizon 450 \
        --seed "$EVAL_SEED" \
        --expect-completion-contract exact_visible_dump_v1 \
        --expect-reward-protocol-id material_potential_v2 \
        --expect-distance-protocol-id obstacle_geodesic_8_physical_global_v1 \
        --expect-distance-sidecar-sha256 "$DISTANCE_SIDECAR_SHA" \
        --require-productive-workspace-cycles \
        --output "$output"
}

# Keep different architecture fingerprints in separate evaluator invocations.
evaluate_treatment \
    "$GRU_U40000" "$GRU_U44000" "$OUTPUT_DIR/gru_u40000_u44000.json"
evaluate_treatment \
    "$FF_U44000" "$FF_U86000" "$OUTPUT_DIR/ff_u44000_u86000.json"

# Build the primary matched-update selection before rendering. Media inference
# still replays the complete 720-row panel and only retains these selected rows.
SELECTION_DASHBOARD="$OUTPUT_DIR/selection_ff_u44_vs_gru_u44"
"$PYTHON" "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py" \
    --reference-json "$OUTPUT_DIR/ff_u44000_u86000.json" \
    --candidate-json "$OUTPUT_DIR/gru_u40000_u44000.json" \
    --reference-index 0 \
    --candidate-index 1 \
    --reference-label FF-u44 \
    --candidate-label GRU-u44 \
    --review-limit 40 \
    --output-dir "$SELECTION_DASHBOARD"

MEDIA_ROOT="$OUTPUT_DIR/media"
"$PYTHON" "$BASELINES_ROOT/scripts/render_v8_fixed_panel_gifs.py" \
    --checkpoint "$FF_U44000" \
    --fixed-json "$OUTPUT_DIR/ff_u44000_u86000.json" \
    --record-index 0 \
    --bank-root "$BANK" \
    --terra-revision "$PROTOCOL_TERRA_REVISION" \
    --selection "$SELECTION_DASHBOARD/review_selection.json" \
    --label FF-u44 \
    --frame-stride 10 \
    --output-dir "$MEDIA_ROOT"
"$PYTHON" "$BASELINES_ROOT/scripts/render_v8_fixed_panel_gifs.py" \
    --checkpoint "$GRU_U44000" \
    --fixed-json "$OUTPUT_DIR/gru_u40000_u44000.json" \
    --record-index 1 \
    --bank-root "$BANK" \
    --terra-revision "$PROTOCOL_TERRA_REVISION" \
    --selection "$SELECTION_DASHBOARD/review_selection.json" \
    --label GRU-u44 \
    --frame-stride 10 \
    --output-dir "$MEDIA_ROOT"

"$PYTHON" "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py" \
    --reference-json "$OUTPUT_DIR/ff_u44000_u86000.json" \
    --candidate-json "$OUTPUT_DIR/gru_u40000_u44000.json" \
    --reference-index 0 \
    --candidate-index 1 \
    --reference-label FF-u44 \
    --candidate-label GRU-u44 \
    --media-dir "$MEDIA_ROOT" \
    --review-limit 40 \
    --output-dir "$OUTPUT_DIR/dashboard_ff_u44_vs_gru_u44"
"$PYTHON" "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py" \
    --reference-json "$OUTPUT_DIR/gru_u40000_u44000.json" \
    --candidate-json "$OUTPUT_DIR/gru_u40000_u44000.json" \
    --reference-index 0 \
    --candidate-index 1 \
    --reference-label GRU-u40 \
    --candidate-label GRU-u44 \
    --review-limit 40 \
    --output-dir "$OUTPUT_DIR/dashboard_gru_u40_vs_u44"
"$PYTHON" "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py" \
    --reference-json "$OUTPUT_DIR/ff_u44000_u86000.json" \
    --candidate-json "$OUTPUT_DIR/gru_u40000_u44000.json" \
    --reference-index 1 \
    --candidate-index 1 \
    --reference-label FF-u86 \
    --candidate-label GRU-u44 \
    --review-limit 40 \
    --output-dir "$OUTPUT_DIR/dashboard_ff_u86_vs_gru_u44"

GRU_RESULT_SHA="$(sha256sum "$OUTPUT_DIR/gru_u40000_u44000.json" | awk '{print $1}')"
FF_RESULT_SHA="$(sha256sum "$OUTPUT_DIR/ff_u44000_u86000.json" | awk '{print $1}')"
printf '%s\n' \
    "gru_result_sha256=$GRU_RESULT_SHA" \
    "ff_result_sha256=$FF_RESULT_SHA" \
    "status=COMPLETED" >> "$CONTRACT"
find "$OUTPUT_DIR" -type f ! -name hashes.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$OUTPUT_DIR/hashes.sha256"
cat "$OUTPUT_DIR/hashes.sha256"
