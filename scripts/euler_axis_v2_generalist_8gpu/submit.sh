#!/usr/bin/env bash
# One supported launch ladder for the axis-v2 25-foundation + 15-trench policy.
# SUBMIT=0 is local-only. Later modes stage immutable inputs and require the
# previous finite-update receipt before allocating more GPUs.
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in
    0|stage|canary1|bootstrap|smoke|bootstrap4|smoke4|fallback4|bootstrap4replay|smoke4replay|fallback4replay|expert_bootstrap4|expert_smoke4|expert_fallback4|expert_bootstrap4replay|expert_smoke4replay|expert_fallback4replay|1) ;;
    *) echo "unsupported SUBMIT mode: $SUBMIT" >&2; exit 2 ;;
esac
case "$SUBMIT" in
    expert_*) POLICY_SCOPE=trench_expert ;;
    *) POLICY_SCOPE=generalist ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_GRAPH_REL=configs/axis_v2_continuous_banded_graph_v1.json
git -C "$REPO" ls-files --error-unmatch -- "$RUNTIME_GRAPH_REL" >/dev/null
# shellcheck disable=SC1091
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-alesweber}"

TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_trench_axis_contract_v2_20260825}"
ARTIFACT_ROOT="${AXIS_V2_ARTIFACT_ROOT:-/media/lorenzo/T7/codex/terra_axis_v2_generalist_20260825}"
RELEASE_MANIFEST="${AXIS_V2_RELEASE_MANIFEST:-$ARTIFACT_ROOT/release.env}"
test -f "$RELEASE_MANIFEST" || {
    echo "missing immutable release manifest: $RELEASE_MANIFEST" >&2; exit 2;
}
# shellcheck disable=SC1090
source "$RELEASE_MANIFEST"

for REQUIRED in FULL_BANK_ROOT PARTIAL_BANK_ROOT FULL_BANK_ARCHIVE \
    FULL_BANK_ARCHIVE_SHA FULL_BANK_DATASET_SHA FULL_BANK_BUILD_SHA \
    FULL_BANK_AUDIT_SHA PARTIAL_BANK_ARCHIVE PARTIAL_BANK_ARCHIVE_SHA \
    PARTIAL_BANK_SHA PARTIAL_ALIGNMENT_AUDIT_SHA PROTOCOL_SHA \
    SOURCE_REGISTRY_SHA DISTANCE_ARTIFACT_SHA RUNTIME_TERRA_REVISION_PIN \
    BASELINES_REVISION_PIN \
    EXPECTED_PARTIAL_CONDITIONS EXPECTED_PARTIAL_TRIPLETS \
    EXPECTED_PARTIAL_SIDECARS EXPECTED_TRENCH_AUDIT_SIDECARS \
    EXPERT_PARTIAL_BANK_ROOT EXPERT_PARTIAL_BANK_ARCHIVE \
    EXPERT_PARTIAL_BANK_ARCHIVE_SHA EXPERT_PARTIAL_BANK_SHA \
    EXPERT_PARTIAL_ALIGNMENT_AUDIT_SHA EXPECTED_EXPERT_PARTIAL_CONDITIONS \
    EXPECTED_EXPERT_PARTIAL_TRIPLETS EXPECTED_EXPERT_PARTIAL_SIDECARS \
    EXPECTED_EXPERT_TRENCH_AUDIT_SIDECARS; do
    test -n "${!REQUIRED:-}" || { echo "release manifest missing $REQUIRED" >&2; exit 2; }
done

REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
CAMPAIGN=terra_axis_v2_generalist_8gpu_v1
REMOTE_WORK="$REMOTE_WORK_ROOT/$CAMPAIGN"
REMOTE_RUNS="$REMOTE_RUN_ROOT/$CAMPAIGN/runs"
REMOTE_INPUTS="$REMOTE_RUN_ROOT/$CAMPAIGN/inputs"
WANDB_ENTITY="${TERRA_WANDB_ENTITY:-aless-weber-eth}"
WANDB_PROJECT="${TERRA_WANDB_PROJECT:-mixed-agents}"
LOCAL_PYTHON=/home/lorenzo/moleworks/.venv-terra-uv/bin/python
EXPECTED_PARAMETERS=2311869
SEED=20260825
EXCLUDED_NODES=eu-g6-057,eu-g6-064,eu-g6-065
RELEASE_ID=terra_axis_v2_v6_constraints_v7_foundations_train96_v1
REPLAY_AUTOTUNE_CACHE_SHA=698e856cae464e5fea93e0b2121fc8de4d9cb691135571ca4b5d56f3259d16a3
REPLAY_AUTOTUNE_CACHE="${AXIS_V2_REPLAY_AUTOTUNE_CACHE:-$TERRA_EULER_PROJECT_ROOT/terra_runtime/autotune/terra_trench_align_generalist_partial_v1/rtx4090_cc89_cudnn897_jaxlib0426_45e2dbc_u3501.pbtxt}"
if [ "$POLICY_SCOPE" = trench_expert ]; then
    RUN_PARTIAL_BANK_ARCHIVE="$EXPERT_PARTIAL_BANK_ARCHIVE"
    RUN_PARTIAL_BANK_ARCHIVE_SHA="$EXPERT_PARTIAL_BANK_ARCHIVE_SHA"
    RUN_PARTIAL_BANK_SHA="$EXPERT_PARTIAL_BANK_SHA"
    RUN_PARTIAL_ALIGNMENT_AUDIT_SHA="$EXPERT_PARTIAL_ALIGNMENT_AUDIT_SHA"
    RUN_EXPECTED_PARTIAL_CONDITIONS="$EXPECTED_EXPERT_PARTIAL_CONDITIONS"
    RUN_EXPECTED_PARTIAL_TRIPLETS="$EXPECTED_EXPERT_PARTIAL_TRIPLETS"
    RUN_EXPECTED_PARTIAL_SIDECARS="$EXPECTED_EXPERT_PARTIAL_SIDECARS"
    RUN_EXPECTED_TRENCH_AUDIT_SIDECARS="$EXPECTED_EXPERT_TRENCH_AUDIT_SIDECARS"
else
    RUN_PARTIAL_BANK_ARCHIVE="$PARTIAL_BANK_ARCHIVE"
    RUN_PARTIAL_BANK_ARCHIVE_SHA="$PARTIAL_BANK_ARCHIVE_SHA"
    RUN_PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA"
    RUN_PARTIAL_ALIGNMENT_AUDIT_SHA="$PARTIAL_ALIGNMENT_AUDIT_SHA"
    RUN_EXPECTED_PARTIAL_CONDITIONS="$EXPECTED_PARTIAL_CONDITIONS"
    RUN_EXPECTED_PARTIAL_TRIPLETS="$EXPECTED_PARTIAL_TRIPLETS"
    RUN_EXPECTED_PARTIAL_SIDECARS="$EXPECTED_PARTIAL_SIDECARS"
    RUN_EXPECTED_TRENCH_AUDIT_SIDECARS="$EXPECTED_TRENCH_AUDIT_SIDECARS"
fi

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean" >&2; exit 3;
}
test -z "$(git -C "$TERRA_REPO" status --porcelain)" || {
    echo "Terra runtime must be committed and clean" >&2; exit 3;
}
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_TERRA_REVISION="$(git -C "$TERRA_REPO" rev-parse HEAD)"
test "$RUNTIME_TERRA_REVISION" = "$RUNTIME_TERRA_REVISION_PIN"
test -x "$LOCAL_PYTHON"
test "$(sha256sum "$FULL_BANK_ARCHIVE" | awk '{print $1}')" = "$FULL_BANK_ARCHIVE_SHA"
test "$(sha256sum "$PARTIAL_BANK_ARCHIVE" | awk '{print $1}')" = "$PARTIAL_BANK_ARCHIVE_SHA"
test "$(sha256sum "$EXPERT_PARTIAL_BANK_ARCHIVE" | awk '{print $1}')" = "$EXPERT_PARTIAL_BANK_ARCHIVE_SHA"
test "$(sha256sum "$FULL_BANK_ROOT/dataset.json" | awk '{print $1}')" = "$FULL_BANK_DATASET_SHA"
test "$(sha256sum "$FULL_BANK_ROOT/build_receipt.json" | awk '{print $1}')" = "$FULL_BANK_BUILD_SHA"
test "$(sha256sum "$FULL_BANK_ROOT/audit_receipt.json" | awk '{print $1}')" = "$FULL_BANK_AUDIT_SHA"
test "$(sha256sum "$PARTIAL_BANK_ROOT/trench_alignment_audit.json" | awk '{print $1}')" = "$PARTIAL_ALIGNMENT_AUDIT_SHA"
test "$(sha256sum "$EXPERT_PARTIAL_BANK_ROOT/trench_alignment_audit.json" | awk '{print $1}')" = "$EXPERT_PARTIAL_ALIGNMENT_AUDIT_SHA"

PYTHONPATH="$TERRA_REPO:$REPO" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
FULL_BANK_ROOT="$FULL_BANK_ROOT" PARTIAL_BANK_ROOT="$PARTIAL_BANK_ROOT" \
EXPERT_PARTIAL_BANK_ROOT="$EXPERT_PARTIAL_BANK_ROOT" \
BASELINES_REVISION="$BASELINES_REVISION" RUNTIME_TERRA_REVISION="$RUNTIME_TERRA_REVISION" \
BANK_BUILDER_BASELINES_REVISION="$BASELINES_REVISION_PIN" \
PROTOCOL_SHA="$PROTOCOL_SHA" SOURCE_REGISTRY_SHA="$SOURCE_REGISTRY_SHA" \
DISTANCE_ARTIFACT_SHA="$DISTANCE_ARTIFACT_SHA" PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" \
EXPECTED_PARTIAL_CONDITIONS="$EXPECTED_PARTIAL_CONDITIONS" \
EXPECTED_PARTIAL_TRIPLETS="$EXPECTED_PARTIAL_TRIPLETS" \
EXPECTED_PARTIAL_SIDECARS="$EXPECTED_PARTIAL_SIDECARS" \
EXPECTED_TRENCH_AUDIT_SIDECARS="$EXPECTED_TRENCH_AUDIT_SIDECARS" \
EXPERT_PARTIAL_BANK_SHA="$EXPERT_PARTIAL_BANK_SHA" \
EXPECTED_EXPERT_PARTIAL_CONDITIONS="$EXPECTED_EXPERT_PARTIAL_CONDITIONS" \
EXPECTED_EXPERT_PARTIAL_TRIPLETS="$EXPECTED_EXPERT_PARTIAL_TRIPLETS" \
EXPECTED_EXPERT_PARTIAL_SIDECARS="$EXPECTED_EXPERT_PARTIAL_SIDECARS" \
EXPECTED_EXPERT_TRENCH_AUDIT_SIDECARS="$EXPECTED_EXPERT_TRENCH_AUDIT_SIDECARS" \
RELEASE_ID="$RELEASE_ID" "$LOCAL_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from terra.maps_buffer import partial_reset_bank_sha256
from utils.accepted_bank import load_accepted_bank, validate_staged_training_bank

full = Path(os.environ["FULL_BANK_ROOT"])
partial_root = Path(os.environ["PARTIAL_BANK_ROOT"])
assert validate_staged_training_bank(
    full,
    expected_maps_per_condition=96,
    expected_release_id=os.environ["RELEASE_ID"],
) == 96
bank = load_accepted_bank(
    full,
    "G-UNIFORM",
    os.environ["RUNTIME_TERRA_REVISION"],
    curriculum_stage="full",
    sampler_profile="continuous_banded_v3",
    condition_profile="axis_v2_40_v1",
)
assert len(bank.levels) == 40
assert sum(level.family == "foundation" for level in bank.levels) == 25
assert sum(level.family == "trench" for level in bank.levels) == 15
assert all(panel.condition_count == 38 for panel in bank.evaluation_panels)
assert all(panel.condition_count == 2 for panel in bank.capability_floor_evaluation_panels)
assert tuple(bank.curriculum_depths).count(0) == 2
assert tuple(bank.curriculum_depths).count(1) == 6
assert tuple(bank.curriculum_depths).count(2) == 32
expert = load_accepted_bank(
    full,
    "T-SPECIALIST",
    os.environ["RUNTIME_TERRA_REVISION"],
    curriculum_stage="full",
    sampler_profile="continuous_banded_v3",
    condition_profile="axis_v2_40_v1",
)
assert len(expert.levels) == 15
assert all(level.family == "trench" for level in expert.levels)
assert tuple(expert.curriculum_depths).count(0) == 1
assert tuple(expert.curriculum_depths).count(1) == 0
assert tuple(expert.curriculum_depths).count(2) == 14
assert bank.environment_protocol_sha256 == os.environ["PROTOCOL_SHA"]
assert bank.source_registry_sha256 == os.environ["SOURCE_REGISTRY_SHA"]
index = json.loads((full / "dataset.json").read_text())
assert index["canonical_distance_artifact_sha256"] == os.environ["DISTANCE_ARTIFACT_SHA"]
build = json.loads((full / "build_receipt.json").read_text())
assert build["builder_baselines_revision"] == os.environ["BANK_BUILDER_BASELINES_REVISION"]
assert build["terra_revision"] == os.environ["RUNTIME_TERRA_REVISION"]

assert partial_reset_bank_sha256(partial_root) == os.environ["PARTIAL_BANK_SHA"]
partial = json.loads((partial_root / "partial_reset_bank.json").read_text())
assert partial["canonical_loader_registry_sha256"] == hashlib.sha256(
    (full / "dataset.json").read_bytes()
).hexdigest()
assert len(partial["supported_maps_paths"]) == int(os.environ["EXPECTED_PARTIAL_CONDITIONS"])
sidecars = []
triplets = set()
for maps_path in partial["supported_maps_paths"]:
    manifest = partial_root / maps_path / "partial_completion_manifest.jsonl"
    for line in manifest.read_text().splitlines():
        if line:
            row = json.loads(line)
            sidecars.append(row)
            triplets.add((maps_path, int(row["source_index"])))
assert len(sidecars) == int(os.environ["EXPECTED_PARTIAL_SIDECARS"])
assert len(triplets) == int(os.environ["EXPECTED_PARTIAL_TRIPLETS"])
audit = json.loads((partial_root / "trench_alignment_audit.json").read_text())
assert audit["schema"] == "terra_partial_trench_alignment_audit_v2"
assert audit["accepted"] is True and audit["failed_sidecars"] == 0
assert audit["partial_reset_bank_sha256"] == os.environ["PARTIAL_BANK_SHA"]
assert audit["audited_sidecars"] == int(os.environ["EXPECTED_TRENCH_AUDIT_SIDECARS"])

expert_partial_root = Path(os.environ["EXPERT_PARTIAL_BANK_ROOT"])
assert partial_reset_bank_sha256(expert_partial_root) == os.environ[
    "EXPERT_PARTIAL_BANK_SHA"
]
expert_partial = json.loads(
    (expert_partial_root / "partial_reset_bank.json").read_text()
)
assert expert_partial["subset_family"] == "trench"
assert expert_partial["supported_maps_paths"] == [
    level.maps_path
    for level in expert.levels
    if level.maps_path in set(expert_partial["supported_maps_paths"])
]
assert len(expert_partial["supported_maps_paths"]) == int(
    os.environ["EXPECTED_EXPERT_PARTIAL_CONDITIONS"]
)
expert_rows = []
expert_triplets = set()
for maps_path in expert_partial["supported_maps_paths"]:
    manifest = expert_partial_root / maps_path / "partial_completion_manifest.jsonl"
    for line in manifest.read_text().splitlines():
        if line:
            row = json.loads(line)
            expert_rows.append(row)
            expert_triplets.add((maps_path, int(row["source_index"])))
assert len(expert_rows) == int(os.environ["EXPECTED_EXPERT_PARTIAL_SIDECARS"])
assert len(expert_triplets) == int(os.environ["EXPECTED_EXPERT_PARTIAL_TRIPLETS"])
expert_audit = json.loads(
    (expert_partial_root / "trench_alignment_audit.json").read_text()
)
assert expert_audit["accepted"] is True and expert_audit["failed_sidecars"] == 0
assert expert_audit["partial_reset_bank_sha256"] == os.environ[
    "EXPERT_PARTIAL_BANK_SHA"
]
assert expert_audit["audited_sidecars"] == int(
    os.environ["EXPECTED_EXPERT_TRENCH_AUDIT_SIDECARS"]
)
PY

ALGORITHM_DENYLIST="$REPO/scripts/euler_axis_v2_generalist_8gpu/hlo_algorithm_denylist.pbtxt"
ALGORITHM_DENYLIST_SHA="$(sha256sum "$ALGORITHM_DENYLIST" | awk '{print $1}')"
printf '%s\n' \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "conditions=40 foundation=25 trench=15" \
    "trench_expert_conditions=15 partial_reset_supported_trenches=14" \
    "partial_conditions=$RUN_EXPECTED_PARTIAL_CONDITIONS partial_triplets=$RUN_EXPECTED_PARTIAL_TRIPLETS" \
    "partial_reset_bank_sha256=$RUN_PARTIAL_BANK_SHA" \
    "replay_autotune_cache_sha256=$REPLAY_AUTOTUNE_CACHE_SHA" \
    "ladder=canary1:1gpu-u1,bootstrap:8gpu-u1,smoke:8gpu-u5,phase1:8gpu-u75000,phase2:8gpu-u100000,bootstrap4:4gpu512-u1,smoke4:4gpu512-u5,fallback4:4gpu512-u75000+u100000,bootstrap4replay:4gpu512-u1-cached,smoke4replay:4gpu512-u5-cached,fallback4replay:4gpu512-u75000+u100000-cached"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local contracts passed; no SSH, upload, W&B, or Slurm mutation"
    exit 0
fi

remote() { ssh -o BatchMode=yes "$REMOTE_HOST" "$@"; }
test "$(remote 'id -un')" = "$TERRA_EULER_USER"
remote "test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -w '$TERRA_EULER_SCRATCH_ROOT' && test -x '$REMOTE_VENV/bin/python'"
case "$SUBMIT" in
    bootstrap4replay|smoke4replay|fallback4replay|expert_bootstrap4replay|expert_smoke4replay|expert_fallback4replay)
        remote "test \"\$(sha256sum '$REPLAY_AUTOTUNE_CACHE' | awk '{print \$1}')\" = '$REPLAY_AUTOTUNE_CACHE_SHA'"
        ;;
esac
remote "mkdir -p '$REMOTE_WORK/source' '$REMOTE_WORK/runtime-terra' '$REMOTE_INPUTS' '$REMOTE_RUNS'"

REMOTE_SOURCE="$REMOTE_WORK/source/$BASELINES_REVISION/terra-baselines"
REMOTE_TERRA="$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
if ! remote "test -e '$REMOTE_SOURCE'"; then
    SOURCE_PARTIAL="$REMOTE_WORK/source/.${BASELINES_REVISION}.partial.$$"
    remote "mkdir -p '$SOURCE_PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" | \
        remote "tar -xf - -C '$SOURCE_PARTIAL/terra-baselines'"
    remote "printf '%s\n' '$BASELINES_REVISION' > '$SOURCE_PARTIAL/terra-baselines/REVISION' && mv -T '$SOURCE_PARTIAL' '$REMOTE_WORK/source/$BASELINES_REVISION'"
fi
if ! remote "test -e '$REMOTE_TERRA'"; then
    TERRA_PARTIAL="$REMOTE_WORK/runtime-terra/.${RUNTIME_TERRA_REVISION}.partial.$$"
    remote "mkdir -p '$TERRA_PARTIAL/terra'"
    git -C "$TERRA_REPO" archive --format=tar "$RUNTIME_TERRA_REVISION" | \
        remote "tar -xf - -C '$TERRA_PARTIAL/terra'"
    remote "printf '%s\n' '$RUNTIME_TERRA_REVISION' > '$TERRA_PARTIAL/terra/REVISION' && mv -T '$TERRA_PARTIAL' '$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION'"
fi
remote "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION' && test \"\$(cat '$REMOTE_TERRA/REVISION')\" = '$RUNTIME_TERRA_REVISION'"
remote "test -f '$REMOTE_SOURCE/$RUNTIME_GRAPH_REL'"

upload() {
    local source="$1" destination="$2" expected_sha="$3"
    if ! remote "test -f '$destination'"; then
        local partial="$destination.partial.$$"
        scp -q -o BatchMode=yes "$source" "$REMOTE_HOST:$partial"
        remote "test \"\$(sha256sum '$partial' | awk '{print \$1}')\" = '$expected_sha' && mv -T '$partial' '$destination'"
    fi
    remote "test \"\$(sha256sum '$destination' | awk '{print \$1}')\" = '$expected_sha'"
}

REMOTE_BANK="$REMOTE_INPUTS/full-bank-$FULL_BANK_ARCHIVE_SHA.tar.zst"
REMOTE_PARTIAL="$REMOTE_INPUTS/partial-bank-$RUN_PARTIAL_BANK_ARCHIVE_SHA.tar.zst"
upload "$FULL_BANK_ARCHIVE" "$REMOTE_BANK" "$FULL_BANK_ARCHIVE_SHA"
upload "$RUN_PARTIAL_BANK_ARCHIVE" "$REMOTE_PARTIAL" "$RUN_PARTIAL_BANK_ARCHIVE_SHA"

ASSOCIATIONS="$(remote "sacctmgr -n -P show assoc where user='$TERRA_EULER_USER' format=Account")"
printf '%s\n' "$ASSOCIATIONS" | grep -Eq '^%?es_hutter$'
for PARTITION in gpuhe.4h gpuhe.120h; do
    remote "scontrol show partition '$PARTITION' -o | grep -q 'State=UP'"
    remote "sinfo -h -p '$PARTITION' -N -o '%G' | grep -q 'nvidia_geforce_rtx_4090:8'"
done
remote "df -Pk '$TERRA_EULER_SCRATCH_ROOT' | awk 'NR == 2 {exit !(\$4 > 50000000)}'"
if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: immutable source and inputs staged; scheduler gates passed; no job or W&B mutation"
    exit 0
fi

COMMON_EXPORTS="ALL,POLICY_SCOPE=$POLICY_SCOPE,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,BANK_BUILDER_BASELINES_REVISION=$BASELINES_REVISION_PIN,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,PROTOCOL_SHA=$PROTOCOL_SHA,SEED=$SEED,VENV=$REMOTE_VENV,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,BANK_ARCHIVE=$REMOTE_BANK,BANK_ARCHIVE_SHA=$FULL_BANK_ARCHIVE_SHA,BANK_DATASET_SHA=$FULL_BANK_DATASET_SHA,BANK_BUILD_SHA=$FULL_BANK_BUILD_SHA,BANK_AUDIT_SHA=$FULL_BANK_AUDIT_SHA,SOURCE_REGISTRY_SHA=$SOURCE_REGISTRY_SHA,DISTANCE_ARTIFACT_SHA=$DISTANCE_ARTIFACT_SHA,PARTIAL_ARCHIVE=$REMOTE_PARTIAL,PARTIAL_ARCHIVE_SHA=$RUN_PARTIAL_BANK_ARCHIVE_SHA,PARTIAL_BANK_SHA=$RUN_PARTIAL_BANK_SHA,PARTIAL_ALIGNMENT_AUDIT_SHA=$RUN_PARTIAL_ALIGNMENT_AUDIT_SHA,EXPECTED_PARTIAL_CONDITIONS=$RUN_EXPECTED_PARTIAL_CONDITIONS,EXPECTED_PARTIAL_TRIPLETS=$RUN_EXPECTED_PARTIAL_TRIPLETS,EXPECTED_PARTIAL_SIDECARS=$RUN_EXPECTED_PARTIAL_SIDECARS,EXPECTED_TRENCH_AUDIT_SIDECARS=$RUN_EXPECTED_TRENCH_AUDIT_SIDECARS,EXPECTED_PARAMETERS=$EXPECTED_PARAMETERS,ALGORITHM_DENYLIST_SHA=$ALGORITHM_DENYLIST_SHA"
if [ "$POLICY_SCOPE" = trench_expert ]; then
    BASE_RUN_NAME="axis_v2_trench_expert_${BASELINES_REVISION:0:12}_s${SEED}"
    RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/s$SEED/trench_expert"
else
    BASE_RUN_NAME="axis_v2_generalist_${BASELINES_REVISION:0:12}_s${SEED}"
    RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/s$SEED"
fi

require_complete() {
    local run_dir="$1" run_name="$2" target="$3" devices="$4"
    local envs_per_device=256
    local global_rollout=65536
    if [ "$devices" -eq 1 ]; then global_rollout=8192; fi
    if [ "$devices" -eq 4 ]; then envs_per_device=512; fi
    remote "test -f '$run_dir/completion.env' && grep -qx 'status=COMPLETED' '$run_dir/completion.env' && grep -qx 'target_update=$target' '$run_dir/completion.env' && test -f '$run_dir/checkpoints/${run_name}_FINAL.pkl' && test -f '$run_dir/checkpoint_validation.json' && grep -Eq '\"next_update\": $target(,)?$' '$run_dir/checkpoint_validation.json' && grep -q '\"model_finite\": true' '$run_dir/checkpoint_validation.json' && grep -q '\"optimizer_finite\": true' '$run_dir/checkpoint_validation.json' && grep -qx 'policy_scope=$POLICY_SCOPE' '$run_dir/run_contract.env' && grep -qx 'terra_baselines_revision=$BASELINES_REVISION' '$run_dir/run_contract.env' && grep -qx 'runtime_terra_revision=$RUNTIME_TERRA_REVISION' '$run_dir/run_contract.env' && grep -qx 'partial_reset_bank_sha256=$RUN_PARTIAL_BANK_SHA' '$run_dir/run_contract.env' && grep -qx 'num_devices=$devices' '$run_dir/run_contract.env' && grep -qx 'num_envs_per_device=$envs_per_device' '$run_dir/run_contract.env' && grep -qx 'global_rollout_transitions=$global_rollout' '$run_dir/run_contract.env' && grep -qx 'xla_gpu_autotune_level=4' '$run_dir/run_contract.env' && test -f '$run_dir/cuda_runtime_validation.json' && grep -q '\"status\": \"passed\"' '$run_dir/cuda_runtime_validation.json' && CUDA_SHA=\$(sha256sum '$run_dir/cuda_runtime_validation.json' | awk '{print \$1}') && grep -qx \"cuda_runtime_validation_sha256=\$CUDA_SHA\" '$run_dir/completion.env' && RESULTS_SHA=\$(sha256sum '$run_dir/autotune_results.pbtxt' | awk '{print \$1}') && grep -qx \"autotune_results_sha256=\$RESULTS_SHA\" '$run_dir/completion.env' && JOB_ID=\$(awk -F= '\$1 == \"slurm_job_id\" {print \$2}' '$run_dir/run_contract.env') && test \"\$(sacct -X -n -j \"\$JOB_ID\" --format=State | awk 'NF {print \$1; exit}')\" = COMPLETED"
}

submit_finite() {
    local role="$1" target="$2" devices="$3" run_dir="$4" run_name="$5"
    local cache="$6" cache_sha="$7" job_name="$8"
    local cpus=16
    if [ "$devices" -eq 1 ]; then cpus=4; fi
    if [ "$devices" -eq 4 ]; then cpus=8; fi
    remote "test ! -e '$run_dir' && mkdir -p '$RUN_PARENT' && mkdir '$run_dir'"
    local raw job_id
    raw="$(remote "cat '$REMOTE_SOURCE/scripts/euler_axis_v2_generalist_8gpu/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.4h' --time='01:30:00' --gpus='rtx_4090:$devices' --cpus-per-task='$cpus' --exclude='$EXCLUDED_NODES' --job-name='$job_name' --output='$run_dir/slurm_%j.out' --export='$COMMON_EXPORTS,AUTOTUNE_CACHE=$cache,AUTOTUNE_CACHE_SHA=$cache_sha,RUN_ROLE=$role,TARGET_UPDATE=$target,RESUME_CHECKPOINT=none,RESUME_UPDATE=0,RUN_DIR=$run_dir,RUN_NAME=$run_name'")"
    job_id="${raw%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]]
    remote "printf '%s\n' 'status=SUBMITTED' 'policy_scope=$POLICY_SCOPE' 'role=$role' 'job_id=$job_id' 'devices=$devices' 'target_update=$target' > '$run_dir/submission.env'"
    printf '%s\n' "role=$role" "job_id=$job_id" "devices=$devices" "run_dir=$run_dir"
}

submit_production_chain() {
    local devices="$1" cpus="$2" phase1_role="$3" phase2_role="$4"
    local cache="$5" cache_sha="$6" phase1_dir="$7" phase2_dir="$8"
    local run_name="$9" submission_file="${10}" job_suffix="${11}"
    remote "test ! -e '$phase1_dir' && test ! -e '$phase2_dir' && mkdir -p '$RUN_PARENT' && mkdir '$phase1_dir' '$phase2_dir'"
    local job1_raw job1_id job2_raw job2_id phase1_checkpoint
    job1_raw="$(remote "cat '$REMOTE_SOURCE/scripts/euler_axis_v2_generalist_8gpu/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.120h' --time='119:45:00' --gpus='rtx_4090:$devices' --cpus-per-task='$cpus' --exclude='$EXCLUDED_NODES' --job-name='terra-axis-v2-${job_suffix}-u75' --output='$phase1_dir/slurm_%j.out' --export='$COMMON_EXPORTS,AUTOTUNE_CACHE=$cache,AUTOTUNE_CACHE_SHA=$cache_sha,RUN_ROLE=$phase1_role,TARGET_UPDATE=75000,RESUME_CHECKPOINT=none,RESUME_UPDATE=0,RUN_DIR=$phase1_dir,RUN_NAME=$run_name'")"
    job1_id="${job1_raw%%;*}"
    [[ "$job1_id" =~ ^[0-9]+$ ]]
    phase1_checkpoint="$phase1_dir/checkpoints/${run_name}_FINAL.pkl"
    if ! job2_raw="$(remote "cat '$REMOTE_SOURCE/scripts/euler_axis_v2_generalist_8gpu/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.120h' --time='119:45:00' --gpus='rtx_4090:$devices' --cpus-per-task='$cpus' --exclude='$EXCLUDED_NODES' --dependency='afterok:$job1_id' --kill-on-invalid-dep=yes --job-name='terra-axis-v2-${job_suffix}-u100' --output='$phase2_dir/slurm_%j.out' --export='$COMMON_EXPORTS,AUTOTUNE_CACHE=$cache,AUTOTUNE_CACHE_SHA=$cache_sha,RUN_ROLE=$phase2_role,TARGET_UPDATE=100000,RESUME_CHECKPOINT=$phase1_checkpoint,RESUME_UPDATE=75000,RUN_DIR=$phase2_dir,RUN_NAME=$run_name'")"; then
        remote "scancel -- '$job1_id'"
        exit 3
    fi
    job2_id="${job2_raw%%;*}"
    if [[ ! "$job2_id" =~ ^[0-9]+$ ]]; then
        remote "scancel -- '$job1_id'"
        exit 3
    fi
    remote "printf '%s\n' 'status=SUBMITTED' 'policy_scope=$POLICY_SCOPE' 'phase1_job_id=$job1_id' 'phase2_job_id=$job2_id' 'dependency=afterok:$job1_id' 'devices=$devices' 'phase1_target_update=75000' 'final_target_update=100000' 'terra_baselines_revision=$BASELINES_REVISION' 'runtime_terra_revision=$RUNTIME_TERRA_REVISION' 'partial_reset_bank_sha256=$RUN_PARTIAL_BANK_SHA' 'autotune_cache_sha256=$cache_sha' > '$submission_file'"
    printf '%s\n' \
        "phase1_job_id=$job1_id" \
        "phase2_job_id=$job2_id dependency=afterok:$job1_id" \
        "devices=$devices" \
        "phase1_run_dir=$phase1_dir" \
        "phase2_run_dir=$phase2_dir"
}

CANARY_DIR="$RUN_PARENT/canary1_u1"
CANARY_NAME="${BASE_RUN_NAME}_canary1"
BOOTSTRAP_DIR="$RUN_PARENT/bootstrap_8gpu_u1"
BOOTSTRAP_NAME="${BASE_RUN_NAME}_bootstrap"
SMOKE_DIR="$RUN_PARENT/smoke_8gpu_u5"
SMOKE_NAME="${BASE_RUN_NAME}_smoke"
BOOTSTRAP4_DIR="$RUN_PARENT/bootstrap_4gpu512_u1"
BOOTSTRAP4_NAME="${BASE_RUN_NAME}_bootstrap4"
SMOKE4_DIR="$RUN_PARENT/smoke_4gpu512_u5"
SMOKE4_NAME="${BASE_RUN_NAME}_smoke4"
BOOTSTRAP4REPLAY_DIR="$RUN_PARENT/bootstrap_4gpu512_replay_u1"
BOOTSTRAP4REPLAY_NAME="${BASE_RUN_NAME}_bootstrap4replay"
SMOKE4REPLAY_DIR="$RUN_PARENT/smoke_4gpu512_replay_u5"
SMOKE4REPLAY_NAME="${BASE_RUN_NAME}_smoke4replay"

if [ "$SUBMIT" = bootstrap4 ] || [ "$SUBMIT" = expert_bootstrap4 ]; then
    submit_finite bootstrap4 1 4 "$BOOTSTRAP4_DIR" "$BOOTSTRAP4_NAME" none none terra-axis-v2-bootstrap4
    exit 0
fi
if [ "$SUBMIT" = smoke4 ] || [ "$SUBMIT" = fallback4 ] || \
    [ "$SUBMIT" = expert_smoke4 ] || [ "$SUBMIT" = expert_fallback4 ]; then
    require_complete "$BOOTSTRAP4_DIR" "$BOOTSTRAP4_NAME" 1 4
    BOOTSTRAP4_CACHE="$BOOTSTRAP4_DIR/autotune_results.pbtxt"
    BOOTSTRAP4_CACHE_SHA="$(remote "sha256sum '$BOOTSTRAP4_CACHE' | awk '{print \$1}'")"
    if [ "$SUBMIT" = smoke4 ] || [ "$SUBMIT" = expert_smoke4 ]; then
        submit_finite smoke4 5 4 "$SMOKE4_DIR" "$SMOKE4_NAME" "$BOOTSTRAP4_CACHE" "$BOOTSTRAP4_CACHE_SHA" terra-axis-v2-smoke4
        exit 0
    fi
    require_complete "$SMOKE4_DIR" "$SMOKE4_NAME" 5 4
    remote "test -f '$SMOKE4_DIR/throughput_validation.json' && '$REMOTE_VENV/bin/python' -c 'import json; r=json.load(open(\"$SMOKE4_DIR/throughput_validation.json\")); assert r[\"passed\"] is True; assert r[\"post_compile_median_steps_per_second\"] >= 12000'"
    remote "$REMOTE_VENV/bin/python -c 'import netrc; assert netrc.netrc().authenticators(\"api.wandb.ai\")'"
    SMOKE4_CACHE="$SMOKE4_DIR/autotune_results.pbtxt"
    SMOKE4_CACHE_SHA="$(remote "sha256sum '$SMOKE4_CACHE' | awk '{print \$1}'")"
    submit_production_chain 4 8 phase1_4gpu phase2_4gpu "$SMOKE4_CACHE" "$SMOKE4_CACHE_SHA" \
        "$RUN_PARENT/fallback4_phase1_u75000" "$RUN_PARENT/fallback4_phase2_u100000" \
        "${BASE_RUN_NAME}_4gpu" "$RUN_PARENT/submission_4gpu.env" 4gpu
    exit 0
fi

if [ "$SUBMIT" = bootstrap4replay ] || [ "$SUBMIT" = expert_bootstrap4replay ]; then
    submit_finite bootstrap4replay 1 4 "$BOOTSTRAP4REPLAY_DIR" "$BOOTSTRAP4REPLAY_NAME" "$REPLAY_AUTOTUNE_CACHE" "$REPLAY_AUTOTUNE_CACHE_SHA" "terra-axis-v2-${POLICY_SCOPE}-bootstrap4replay"
    exit 0
fi
if [ "$SUBMIT" = smoke4replay ] || [ "$SUBMIT" = fallback4replay ] || [ "$SUBMIT" = expert_smoke4replay ] || [ "$SUBMIT" = expert_fallback4replay ]; then
    require_complete "$BOOTSTRAP4REPLAY_DIR" "$BOOTSTRAP4REPLAY_NAME" 1 4
    BOOTSTRAP4REPLAY_CACHE="$BOOTSTRAP4REPLAY_DIR/autotune_results.pbtxt"
    BOOTSTRAP4REPLAY_CACHE_SHA="$(remote "sha256sum '$BOOTSTRAP4REPLAY_CACHE' | awk '{print \$1}'")"
    if [ "$SUBMIT" = smoke4replay ] || [ "$SUBMIT" = expert_smoke4replay ]; then
        submit_finite smoke4replay 5 4 "$SMOKE4REPLAY_DIR" "$SMOKE4REPLAY_NAME" "$BOOTSTRAP4REPLAY_CACHE" "$BOOTSTRAP4REPLAY_CACHE_SHA" "terra-axis-v2-${POLICY_SCOPE}-smoke4replay"
        exit 0
    fi
    require_complete "$SMOKE4REPLAY_DIR" "$SMOKE4REPLAY_NAME" 5 4
    remote "test -f '$SMOKE4REPLAY_DIR/throughput_validation.json' && '$REMOTE_VENV/bin/python' -c 'import json; r=json.load(open(\"$SMOKE4REPLAY_DIR/throughput_validation.json\")); assert r[\"passed\"] is True; assert r[\"post_compile_median_steps_per_second\"] >= 12000'"
    remote "$REMOTE_VENV/bin/python -c 'import netrc; assert netrc.netrc().authenticators(\"api.wandb.ai\")'"
    SMOKE4REPLAY_CACHE="$SMOKE4REPLAY_DIR/autotune_results.pbtxt"
    SMOKE4REPLAY_CACHE_SHA="$(remote "sha256sum '$SMOKE4REPLAY_CACHE' | awk '{print \$1}'")"
    submit_production_chain 4 8 phase1_4gpu_replay phase2_4gpu_replay "$SMOKE4REPLAY_CACHE" "$SMOKE4REPLAY_CACHE_SHA" \
        "$RUN_PARENT/fallback4replay_phase1_u75000" "$RUN_PARENT/fallback4replay_phase2_u100000" \
        "${BASE_RUN_NAME}_4gpu_replay" "$RUN_PARENT/submission_4gpu_replay.env" "${POLICY_SCOPE}-4gpu-replay"
    exit 0
fi

if [ "$SUBMIT" = canary1 ]; then
    submit_finite canary1 1 1 "$CANARY_DIR" "$CANARY_NAME" none none terra-axis-v2-1gpu
    exit 0
fi
require_complete "$CANARY_DIR" "$CANARY_NAME" 1 1

if [ "$SUBMIT" = bootstrap ]; then
    submit_finite bootstrap 1 8 "$BOOTSTRAP_DIR" "$BOOTSTRAP_NAME" none none terra-axis-v2-bootstrap
    exit 0
fi
require_complete "$BOOTSTRAP_DIR" "$BOOTSTRAP_NAME" 1 8
BOOTSTRAP_CACHE="$BOOTSTRAP_DIR/autotune_results.pbtxt"
BOOTSTRAP_CACHE_SHA="$(remote "sha256sum '$BOOTSTRAP_CACHE' | awk '{print \$1}'")"

if [ "$SUBMIT" = smoke ]; then
    submit_finite smoke 5 8 "$SMOKE_DIR" "$SMOKE_NAME" "$BOOTSTRAP_CACHE" "$BOOTSTRAP_CACHE_SHA" terra-axis-v2-smoke
    exit 0
fi
require_complete "$SMOKE_DIR" "$SMOKE_NAME" 5 8
remote "test -f '$SMOKE_DIR/throughput_validation.json' && '$REMOTE_VENV/bin/python' -c 'import json; r=json.load(open(\"$SMOKE_DIR/throughput_validation.json\")); assert r[\"passed\"] is True; assert r[\"post_compile_median_steps_per_second\"] >= 12000'"
remote "$REMOTE_VENV/bin/python -c 'import netrc; assert netrc.netrc().authenticators(\"api.wandb.ai\")'"
SMOKE_CACHE="$SMOKE_DIR/autotune_results.pbtxt"
SMOKE_CACHE_SHA="$(remote "sha256sum '$SMOKE_CACHE' | awk '{print \$1}'")"

PHASE1_DIR="$RUN_PARENT/phase1_u75000"
PHASE2_DIR="$RUN_PARENT/phase2_u100000"
submit_production_chain 8 16 phase1 phase2 "$SMOKE_CACHE" "$SMOKE_CACHE_SHA" \
    "$PHASE1_DIR" "$PHASE2_DIR" "$BASE_RUN_NAME" "$RUN_PARENT/submission.env" 8gpu
