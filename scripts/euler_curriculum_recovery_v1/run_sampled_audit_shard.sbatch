#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-recovery-D2-shard
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/historical_audit/sampled_shards_v1/logs/%x_%j.out

set -euo pipefail
umask 002

: "${SHARD_ID:?submit with SHARD_ID=u1000_s3_7, u4000_s0_3, or u4000_s4_7}"

RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725
HISTORICAL_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724
AUDIT_ROOT="$RUN_ROOT/historical_audit"
SHARD_ROOT="$AUDIT_ROOT/sampled_shards_v1"
SOURCE="$AUDIT_ROOT/source"
BANK="$HISTORICAL_ROOT/bank"
CHECKPOINT_ROOT="$HISTORICAL_ROOT/checkpoints"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py
AUDIT_SCRIPT="$SOURCE/terra-baselines/scripts/analysis/audit_historical_curriculum.py"
OUTPUT="$SHARD_ROOT/eval/${SHARD_ID}.json"

FLAT_U1000="$CHECKPOINT_ROOT/flat/terra-map-v1-flat-euler-2026-07-24-04-53-12_update_001000.pkl"
FLAT_U4000="$CHECKPOINT_ROOT/flat/terra-map-v1-flat-euler-2026-07-24-04-53-12_update_004000.pkl"

case "$SHARD_ID" in
    u1000_s3_7)
        CHECKPOINT_LABEL=flat_u1000
        CHECKPOINT="$FLAT_U1000"
        CHECKPOINT_SHA=ced599e4cca1958f468e794931ee94206dc4e5fdf833e3bb03cb1fb58a865d14
        EXPECTED_RECORDS=5
        SEEDS=(2026072503 2026072504 2026072505 2026072506 2026072507)
        ;;
    u4000_s0_3)
        CHECKPOINT_LABEL=flat_u4000
        CHECKPOINT="$FLAT_U4000"
        CHECKPOINT_SHA=e276bc15f03cb2f4d0f8daf8ca4d2741586ca1d9b7c824779040bd12966caab6
        EXPECTED_RECORDS=4
        SEEDS=(2026072500 2026072501 2026072502 2026072503)
        ;;
    u4000_s4_7)
        CHECKPOINT_LABEL=flat_u4000
        CHECKPOINT="$FLAT_U4000"
        CHECKPOINT_SHA=e276bc15f03cb2f4d0f8daf8ca4d2741586ca1d9b7c824779040bd12966caab6
        EXPECTED_RECORDS=4
        SEEDS=(2026072504 2026072505 2026072506 2026072507)
        ;;
    *)
        echo "Unsupported SHARD_ID=$SHARD_ID" >&2
        exit 2
        ;;
esac

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
export WANDB_MODE=disabled

mkdir -p "$SHARD_ROOT/eval" "$SHARD_ROOT/logs"
test -x "$VENV/bin/python"
test -f "$RUNTIME_CHECK"
test -f "$AUDIT_SCRIPT"
test ! -e "$OUTPUT"

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
test "${#GPU_NAMES[@]}" -eq 1
test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 1

SEED_ARGS=()
for seed in "${SEEDS[@]}"; do
    SEED_ARGS+=(--seed "$seed")
done

"$VENV/bin/python" -u "$AUDIT_SCRIPT" \
    --source-root "$SOURCE" \
    --bank-root "$BANK" \
    --horizon 450 \
    --output "$OUTPUT" \
    --checkpoint "$CHECKPOINT_LABEL=$CHECKPOINT" \
    --expected-checkpoint-sha256 "$CHECKPOINT_LABEL=$CHECKPOINT_SHA" \
    --dataset development/M0 \
    --mode sampled \
    "${SEED_ARGS[@]}"

"$VENV/bin/python" - "$OUTPUT" "$CHECKPOINT_LABEL" "$EXPECTED_RECORDS" "${SEEDS[@]}" <<'PY'
import json
import math
import sys

path, checkpoint_label, expected_records, *expected_seeds = sys.argv[1:]
payload = json.load(open(path))
if payload["schema"] != "terra_historical_curriculum_audit_v1":
    raise SystemExit("unexpected sampled-shard schema")
if payload["mode"] != "sampled":
    raise SystemExit("sampled shard has wrong action-selection mode")
expected_seeds = [int(seed) for seed in expected_seeds]
if payload["seeds"] != expected_seeds:
    raise SystemExit("sampled shard seed receipt mismatch")
if len(payload["records"]) != int(expected_records):
    raise SystemExit("sampled shard record count mismatch")
reward_atol = payload["numerical_tolerances"][
    "terminal_reward_reconstruction_atol"
]
for receipt in payload["reset_integrity"].values():
    if not receipt["passed"] or receipt["env_steps_min"] != 0:
        raise SystemExit("sampled shard reset integrity failed")
for record in payload["records"]:
    if (
        record["checkpoint_label"] != checkpoint_label
        or record["dataset"] != "development/M0"
        or record["mode"] != "sampled"
    ):
        raise SystemExit("sampled shard record identity mismatch")
    for row in record["per_map"]:
        if row["target_mutation"] or row["obstacle_mutation"] or row["nonfinite_state"]:
            raise SystemExit("sampled shard rollout integrity failed")
        reward_error = float(row["terminal_reward_reconstruction_error"])
        if not math.isfinite(reward_error) or reward_error > reward_atol:
            raise SystemExit("sampled shard reward reconstruction failed")
print("SAMPLED_AUDIT_SHARD_PASSED")
PY

touch "$SHARD_ROOT/${SHARD_ID}_PASSED"
