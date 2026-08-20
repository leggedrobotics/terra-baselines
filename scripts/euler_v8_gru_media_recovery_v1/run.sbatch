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
    RUNTIME_TERRA_ROOT RUNTIME_TERRA_REVISION \
    BANK_ARCHIVE BANK_ARCHIVE_SHA BANK_DATASET_SHA PROMOTION_MANIFEST_SHA \
    PROTOCOL_TERRA_REVISION \
    PARENT_JOB_ID PARENT_SOURCE_REVISION PARENT_FIXED_ROOT \
    PARENT_CONTRACT_SHA GRU_FIXED_SHA FF_FIXED_SHA SELECTION_SHA \
    GRU_U44000 GRU_U44000_SHA FF_U44000 FF_U44000_SHA \
    VENV TERRA_EULER_USER TERRA_EULER_HOME_ROOT \
    SLURM_JOB_ID TMPDIR; do
    test -n "${!REQUIRED:-}" || { echo "missing $REQUIRED" >&2; exit 2; }
done

test "$(id -un)" = "$TERRA_EULER_USER"
test "$HOME" = "$TERRA_EULER_HOME_ROOT"
test "${SLURM_JOB_PARTITION:-}" = gpuhe.4h
test "${SLURM_CPUS_PER_TASK:-}" = 8
test "$(cat "$BASELINES_ROOT/REVISION")" = "$BASELINES_REVISION"
test "$(cat "$RUNTIME_TERRA_ROOT/REVISION")" = "$RUNTIME_TERRA_REVISION"
test -x "$VENV/bin/python"
test -f "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py"
test -f "$BASELINES_ROOT/scripts/render_v8_fixed_panel_gifs.py"
test ! -e "$OUTPUT_DIR"

PARENT_CONTRACT="$PARENT_FIXED_ROOT/run_contract.env"
GRU_FIXED="$PARENT_FIXED_ROOT/gru_u40000_u44000.json"
FF_FIXED="$PARENT_FIXED_ROOT/ff_u44000_u86000.json"
SELECTION="$PARENT_FIXED_ROOT/selection_ff_u44_vs_gru_u44/review_selection.json"
test "$(sha256sum "$PARENT_CONTRACT" | awk '{print $1}')" = "$PARENT_CONTRACT_SHA"
test "$(sha256sum "$GRU_FIXED" | awk '{print $1}')" = "$GRU_FIXED_SHA"
test "$(sha256sum "$FF_FIXED" | awk '{print $1}')" = "$FF_FIXED_SHA"
test "$(sha256sum "$SELECTION" | awk '{print $1}')" = "$SELECTION_SHA"
grep -Fx "slurm_job_id=$PARENT_JOB_ID" "$PARENT_CONTRACT" >/dev/null
grep -Fx "terra_baselines_revision=$PARENT_SOURCE_REVISION" "$PARENT_CONTRACT" >/dev/null
grep -Fx "status=FAILED" "$PARENT_CONTRACT" >/dev/null

HOME_USED_GB="$(lquota | "$BASELINES_ROOT/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT")"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

test "$(sha256sum "$BANK_ARCHIVE" | awk '{print $1}')" = "$BANK_ARCHIVE_SHA"
for CHECKPOINT_AND_SHA in \
    "$GRU_U44000:$GRU_U44000_SHA" \
    "$FF_U44000:$FF_U44000_SHA"; do
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
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
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

mkdir -p "$OUTPUT_DIR/inputs"
cp "$PARENT_CONTRACT" "$OUTPUT_DIR/inputs/parent_run_contract.env"
cp "$GRU_FIXED" "$OUTPUT_DIR/inputs/gru_u40000_u44000.json"
cp "$FF_FIXED" "$OUTPUT_DIR/inputs/ff_u44000_u86000.json"
cp "$SELECTION" "$OUTPUT_DIR/inputs/review_selection.json"
GRU_FIXED="$OUTPUT_DIR/inputs/gru_u40000_u44000.json"
FF_FIXED="$OUTPUT_DIR/inputs/ff_u44000_u86000.json"
SELECTION="$OUTPUT_DIR/inputs/review_selection.json"

scontrol show job "$SLURM_JOB_ID" -o > "$OUTPUT_DIR/slurm_job.txt"
nvidia-smi -q > "$OUTPUT_DIR/nvidia_smi.txt"
CONTRACT="$OUTPUT_DIR/run_contract.env"
printf '%s\n' \
    "status=RUNNING" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "recovery_contract=v8_gru_media_dashboard_recovery_v1" \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "protocol_terra_revision=$PROTOCOL_TERRA_REVISION" \
    "parent_job_id=$PARENT_JOB_ID" \
    "parent_source_revision=$PARENT_SOURCE_REVISION" \
    "parent_contract_sha256=$PARENT_CONTRACT_SHA" \
    "gru_fixed_json_sha256=$GRU_FIXED_SHA" \
    "ff_fixed_json_sha256=$FF_FIXED_SHA" \
    "selection_sha256=$SELECTION_SHA" \
    "bank_archive_sha256=$BANK_ARCHIVE_SHA" \
    "bank_dataset_sha256=$BANK_DATASET_SHA" \
    "promotion_manifest_sha256=$PROMOTION_MANIFEST_SHA" \
    "gru_u44000_sha256=$GRU_U44000_SHA" \
    "ff_u44000_sha256=$FF_U44000_SHA" \
    "dashboard_builder_sha256=$(sha256sum "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py" | awk '{print $1}')" \
    "media_renderer_sha256=$(sha256sum "$BASELINES_ROOT/scripts/render_v8_fixed_panel_gifs.py" | awk '{print $1}')" \
    "media_review_limit=40" \
    "media_frame_stride=10" \
    "fixed_evaluations_reused=true" \
    "fixed_evaluations_rerun=false" > "$CONTRACT"

record_failure() {
    local exit_code="$?"
    if [ "$exit_code" -ne 0 ]; then
        printf '%s\n' "status=FAILED" "exit_code=$exit_code" >> "$CONTRACT"
    fi
}
trap record_failure EXIT

MEDIA_ROOT="$OUTPUT_DIR/media"
"$PYTHON" "$BASELINES_ROOT/scripts/render_v8_fixed_panel_gifs.py" \
    --checkpoint "$FF_U44000" \
    --fixed-json "$FF_FIXED" \
    --record-index 0 \
    --bank-root "$BANK" \
    --terra-revision "$PROTOCOL_TERRA_REVISION" \
    --selection "$SELECTION" \
    --label FF-u44 \
    --frame-stride 10 \
    --output-dir "$MEDIA_ROOT"
"$PYTHON" "$BASELINES_ROOT/scripts/render_v8_fixed_panel_gifs.py" \
    --checkpoint "$GRU_U44000" \
    --fixed-json "$GRU_FIXED" \
    --record-index 1 \
    --bank-root "$BANK" \
    --terra-revision "$PROTOCOL_TERRA_REVISION" \
    --selection "$SELECTION" \
    --label GRU-u44 \
    --frame-stride 10 \
    --output-dir "$MEDIA_ROOT"

"$PYTHON" "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py" \
    --reference-json "$FF_FIXED" \
    --candidate-json "$GRU_FIXED" \
    --reference-index 0 \
    --candidate-index 1 \
    --reference-label FF-u44 \
    --candidate-label GRU-u44 \
    --media-dir "$MEDIA_ROOT" \
    --review-limit 40 \
    --output-dir "$OUTPUT_DIR/dashboard_ff_u44_vs_gru_u44"
"$PYTHON" "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py" \
    --reference-json "$GRU_FIXED" \
    --candidate-json "$GRU_FIXED" \
    --reference-index 0 \
    --candidate-index 1 \
    --reference-label GRU-u40 \
    --candidate-label GRU-u44 \
    --review-limit 40 \
    --output-dir "$OUTPUT_DIR/dashboard_gru_u40_vs_u44"
"$PYTHON" "$BASELINES_ROOT/scripts/build_v8_benchmark_dashboard.py" \
    --reference-json "$FF_FIXED" \
    --candidate-json "$GRU_FIXED" \
    --reference-index 1 \
    --candidate-index 1 \
    --reference-label FF-u86 \
    --candidate-label GRU-u44 \
    --review-limit 40 \
    --output-dir "$OUTPUT_DIR/dashboard_ff_u86_vs_gru_u44"

"$PYTHON" - "$OUTPUT_DIR" "$GRU_FIXED_SHA" "$FF_FIXED_SHA" "$SELECTION_SHA" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected_hashes = {
    "gru": sys.argv[2],
    "ff": sys.argv[3],
    "selection": sys.argv[4],
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

gru_path = root / "inputs/gru_u40000_u44000.json"
ff_path = root / "inputs/ff_u44000_u86000.json"
selection_path = root / "inputs/review_selection.json"
assert sha256(gru_path) == expected_hashes["gru"]
assert sha256(ff_path) == expected_hashes["ff"]
assert sha256(selection_path) == expected_hashes["selection"]
gru = json.loads(gru_path.read_text())
ff = json.loads(ff_path.read_text())
selection = json.loads(selection_path.read_text())
slots = [row["slot"] for row in selection]
assert len(slots) == len(set(slots)) == 40

for label, fixed, record_index in (("FF-u44", ff, 0), ("GRU-u44", gru, 1)):
    directory = root / "media" / label
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["full_panel_terminal_parity_verified"] is True
    assert receipt["full_panel_no_effect_count_parity_verified"] is True
    assert receipt["selected_slots"] == slots
    assert receipt["reset_verification"] == fixed[record_index]["reset_verification"]
    assert len(receipt["episodes"]) == 40
    assert len(list(directory.glob("slot_*.gif"))) == 40
    assert len(list((directory / "traces").glob("slot_*.json"))) == 40

dashboards = {
    "dashboard_ff_u44_vs_gru_u44": 22,
    "dashboard_gru_u40_vs_u44": -3,
    "dashboard_ff_u86_vs_gru_u44": 4,
}
for name, exact_delta in dashboards.items():
    directory = root / name
    for artifact in ("index.html", "dashboard_data.json", "episodes.csv", "review_selection.json"):
        assert (directory / artifact).is_file(), (name, artifact)
    data = json.loads((directory / "dashboard_data.json").read_text())
    assert data["summary"]["exact_delta"] == exact_delta
primary = root / "dashboard_ff_u44_vs_gru_u44"
assert len(json.loads((primary / "dashboard_data.json").read_text())["media"]) == 40
assert (primary / "review_selection.json").read_bytes() == selection_path.read_bytes()
print("media/dashboard recovery validation passed")
PY

printf '%s\n' "status=COMPLETED" >> "$CONTRACT"
find "$OUTPUT_DIR" -type f ! -name hashes.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$OUTPUT_DIR/hashes.sha256"
cat "$OUTPUT_DIR/hashes.sha256"
