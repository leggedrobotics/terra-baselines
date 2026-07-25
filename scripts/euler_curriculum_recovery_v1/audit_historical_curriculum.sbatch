#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --nodelist=eu-g6-[001-080]
#SBATCH --partition=gpuhe.24h
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=terra-recovery-historical-audit
#SBATCH --output=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/historical_audit/logs/%x_%j.out

set -euo pipefail
umask 002

: "${AUDIT_KIND:?submit with AUDIT_KIND=preflight, deterministic, or sampled}"
case "$AUDIT_KIND" in
    preflight|deterministic|sampled) ;;
    *)
        echo "Unsupported AUDIT_KIND=$AUDIT_KIND" >&2
        exit 2
        ;;
esac

RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725
HISTORICAL_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v1_20260724
AUDIT_ROOT="$RUN_ROOT/historical_audit"
SOURCE="$AUDIT_ROOT/source"
BANK="$HISTORICAL_ROOT/bank"
CHECKPOINT_ROOT="$HISTORICAL_ROOT/checkpoints"
VENV=/cluster/scratch/lterenzi/terra_main_20260717_venv
RUNTIME_CHECK=/cluster/home/lterenzi/codex_terra_edge_validation/scripts/check_jax_runtime.py
AUDIT_SCRIPT="$SOURCE/terra-baselines/scripts/analysis/audit_historical_curriculum.py"
OUTPUT="$AUDIT_ROOT/eval/${AUDIT_KIND}.json"

PARENT="$HISTORICAL_ROOT/parent/E8_FINAL.pkl"
FLAT_U1000="$CHECKPOINT_ROOT/flat/terra-map-v1-flat-euler-2026-07-24-04-53-12_update_001000.pkl"
FLAT_U4000="$CHECKPOINT_ROOT/flat/terra-map-v1-flat-euler-2026-07-24-04-53-12_update_004000.pkl"
STAGED_U1000="$CHECKPOINT_ROOT/staged/terra-map-v1-staged-euler-2026-07-24-04-53-19_update_001000.pkl"
STAGED_U4000="$CHECKPOINT_ROOT/staged/terra-map-v1-staged-euler-2026-07-24-04-53-19_update_004000.pkl"

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

mkdir -p "$AUDIT_ROOT/eval" "$AUDIT_ROOT/logs"
test -x "$VENV/bin/python"
test -f "$RUNTIME_CHECK"
test -f "$AUDIT_SCRIPT"
test -s "$SOURCE/terra/REVISION"
test -s "$SOURCE/terra-baselines/REVISION"
test ! -e "$OUTPUT"

mapfile -t GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
test "${#GPU_NAMES[@]}" -eq 1
test "${GPU_NAMES[0]}" = "NVIDIA GeForce RTX 4090"
"$VENV/bin/python" "$RUNTIME_CHECK" --min-devices 1

COMMON_ARGS=(
    --source-root "$SOURCE"
    --bank-root "$BANK"
    --horizon 450
    --output "$OUTPUT"
)

case "$AUDIT_KIND" in
    preflight)
        "$VENV/bin/python" -u "$AUDIT_SCRIPT" \
            "${COMMON_ARGS[@]}" \
            --checkpoint "flat_u1000=$FLAT_U1000" \
            --expected-checkpoint-sha256 "flat_u1000=ced599e4cca1958f468e794931ee94206dc4e5fdf833e3bb03cb1fb58a865d14" \
            --dataset development/M0 \
            --semantic-label flat_u1000 \
            --semantic-dataset development/M0 \
            --mode deterministic \
            --seed 2026072500
        ;;
    deterministic)
        "$VENV/bin/python" -u "$AUDIT_SCRIPT" \
            "${COMMON_ARGS[@]}" \
            --checkpoint "e8_u20000=$PARENT" \
            --checkpoint "flat_u1000=$FLAT_U1000" \
            --checkpoint "flat_u4000=$FLAT_U4000" \
            --checkpoint "staged_u1000=$STAGED_U1000" \
            --checkpoint "staged_u4000=$STAGED_U4000" \
            --expected-checkpoint-sha256 "e8_u20000=f364a5dbfe3329542317273819b65cf5fc12a7329fef2d9126d8e3f251a9f674" \
            --expected-checkpoint-sha256 "flat_u1000=ced599e4cca1958f468e794931ee94206dc4e5fdf833e3bb03cb1fb58a865d14" \
            --expected-checkpoint-sha256 "flat_u4000=e276bc15f03cb2f4d0f8daf8ca4d2741586ca1d9b7c824779040bd12966caab6" \
            --expected-checkpoint-sha256 "staged_u1000=4c6426eaae89f102837ba8ae2fef8016ad71d4dc5132e0084099ba0c63c604cd" \
            --expected-checkpoint-sha256 "staged_u4000=a4fc0102582266f7a7006c3db2924d4146437bfbc691730af82c1a37e814876e" \
            --dataset development/M0 \
            --dataset development/M1 \
            --dataset development/M2 \
            --dataset train/local_M2_terminal \
            --unique-training-dataset train/local_M2_terminal \
            --semantic-label flat_u1000 \
            --semantic-label flat_u4000 \
            --semantic-label staged_u4000 \
            --semantic-dataset development/M0 \
            --semantic-dataset development/M1 \
            --semantic-dataset development/M2 \
            --mode deterministic \
            --seed 2026072500
        ;;
    sampled)
        "$VENV/bin/python" -u "$AUDIT_SCRIPT" \
            "${COMMON_ARGS[@]}" \
            --checkpoint "flat_u1000=$FLAT_U1000" \
            --checkpoint "flat_u4000=$FLAT_U4000" \
            --expected-checkpoint-sha256 "flat_u1000=ced599e4cca1958f468e794931ee94206dc4e5fdf833e3bb03cb1fb58a865d14" \
            --expected-checkpoint-sha256 "flat_u4000=e276bc15f03cb2f4d0f8daf8ca4d2741586ca1d9b7c824779040bd12966caab6" \
            --dataset development/M0 \
            --mode sampled \
            --seed 2026072500 \
            --seed 2026072501 \
            --seed 2026072502 \
            --seed 2026072503 \
            --seed 2026072504 \
            --seed 2026072505 \
            --seed 2026072506 \
            --seed 2026072507
        ;;
esac

"$VENV/bin/python" - "$OUTPUT" "$AUDIT_KIND" <<'PY'
import json
import math
import sys

path, kind = sys.argv[1:]
payload = json.load(open(path))
if payload["schema"] != "terra_historical_curriculum_audit_v1":
    raise SystemExit("unexpected audit schema")
if not payload["observer_only"]:
    raise SystemExit("historical audit is not marked observer-only")
expected_records = {"preflight": 1, "deterministic": 20, "sampled": 16}[kind]
reward_reconstruction_atol = payload["numerical_tolerances"][
    "terminal_reward_reconstruction_atol"
]
if len(payload["records"]) != expected_records:
    raise SystemExit(
        f"{kind}: expected {expected_records} records, "
        f"got {len(payload['records'])}"
    )
if kind == "deterministic" and payload["d1_maximum_transitions"] != 259200:
    raise SystemExit("deterministic D1 budget receipt is not exactly 259200")
for receipt in payload["reset_integrity"].values():
    if not receipt["passed"] or receipt["env_steps_min"] != 0:
        raise SystemExit(f"reset integrity failed: {receipt}")
for record in payload["records"]:
    for row in record["per_map"]:
        finite_fields = (
            "exact_completion",
            "accepted_buffer_completion",
            "current_terminal_reward",
            "counterfactual_terminal_reward",
            "policy_entropy",
            "action_logit_margin",
        )
        if not all(math.isfinite(float(row[field])) for field in finite_fields):
            raise SystemExit("non-finite audit metric")
        if row["target_mutation"] or row["obstacle_mutation"] or row["nonfinite_state"]:
            raise SystemExit("historical rollout integrity failure")
        if (
            row["terminal_reward_reconstruction_error"]
            > reward_reconstruction_atol
        ):
            raise SystemExit("terminal reward reconstruction mismatch")
        if record["d1_semantic_attribution"] and (
            row["observer_internal_mismatch"]
            or row["observer_transition_mismatch"]
        ):
            raise SystemExit("D1 observer does not reproduce the old transition")
print(f"HISTORICAL_{kind.upper()}_AUDIT_PASSED")
PY

touch "$AUDIT_ROOT/${AUDIT_KIND^^}_PASSED"
