#!/usr/bin/env bash
# Stage, smoke, or submit the strict-gate 37-condition generalist recipe.
# SUBMIT=0 is local-only. SUBMIT=stage uploads immutable inputs. SUBMIT=smoke
# runs one real four-GPU PPO update. SUBMIT=1 is gated on that smoke and submits
# u0->u75k plus an afterok u75k->u100k native continuation.
set -euo pipefail

SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in
    0|stage|smoke|1) ;;
    *) echo "SUBMIT must be 0, stage, smoke, or 1" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=cluster/euler_account.sh
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-alesweber}"

TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818}"
ARTIFACT_ROOT=/media/lorenzo/T7/codex/terra_trench_aligned_generalist_partial_20260823
FULL_BANK_ROOT="$ARTIFACT_ROOT/full_bank_a7204ef568f2"
PARTIAL_BANK_ROOT="$ARTIFACT_ROOT/partial_bank_admitted_a7204ef568f2"
FULL_BANK_ARCHIVE="$ARTIFACT_ROOT/trench_aligned_full_bank_a7204ef568f2.tar.zst"
FULL_BANK_ARCHIVE_SHA=7a44ea9477d5d4db8ff1ebf6c5325bd9d8ce1d91b74cb925e7b572a1bd44eaa0
FULL_BANK_DATASET_SHA=874315916ee5a9ffbfe8809dc3a21cb2aeb4e2ec0863c8bf57e1569e7bac3c1e
FULL_BANK_DERIVATION_SHA=cf32ab67a92c163b2448f3ac48cd6ef784710f517218001557aa05bd00a61ef8
PARTIAL_BANK_ARCHIVE="$ARTIFACT_ROOT/trench_aligned_partial_bank_f25398d3_a7204ef568f2.tar.zst"
PARTIAL_BANK_ARCHIVE_SHA=73f3414ae2948be93b3ea03a25a28e570509ccf70221c771a89fdc32915bb4e4
PARTIAL_BANK_SHA=f25398d3debbffe7bb1df1d9c7b4fe491d6835a5180de8ef8dca14235f07dd74
PARTIAL_ALIGNMENT_AUDIT_SHA=8ebd961afe6def4e8bdd6ada8a07525032b9c43c79912c66f7456a9991f40266
PROTOCOL_SHA=511b1b07e43791151d672ae306c87c8222426e8a7fc91ab11cd6fb42c4bcf027
SOURCE_REGISTRY_SHA=5bf5b01b53da186f6a5291a4c26df524608cccbf959e3ce098b180c8a6a03afa
DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
RUNTIME_TERRA_REVISION_PIN=a7204ef568f202f71b2f76943cb8b2f662eb71ff
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
EXPECTED_PARAMETERS=2311869
EXPECTED_PARTIAL_CONDITIONS=35
EXPECTED_PARTIAL_TRIPLETS=238
EXPECTED_PARTIAL_SIDECARS=714
EXPECTED_RELAY_TRIPLETS=85
EXPECTED_IN_ZONE_TRIPLETS=153
EXPECTED_TRENCH_AUDIT_SIDECARS=255
SEED=20260823
FOUNDATION_CURRICULUM_DEPTH_COUNTS=1,6,18
TRENCH_CURRICULUM_DEPTH_COUNTS=1,0,11
EXCLUDED_NODES=eu-g6-064,eu-g6-065

REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
CAMPAIGN=terra_trench_align_generalist_partial_v1
REMOTE_WORK="$REMOTE_WORK_ROOT/$CAMPAIGN"
REMOTE_RUNS="$REMOTE_RUN_ROOT/$CAMPAIGN/runs"
REMOTE_INPUTS="$REMOTE_RUN_ROOT/$CAMPAIGN/inputs"
WANDB_ENTITY="${TERRA_WANDB_ENTITY:-aless-weber-eth}"
WANDB_PROJECT="${TERRA_WANDB_PROJECT:-mixed-agents}"
LOCAL_PYTHON=/home/lorenzo/moleworks/.venv-terra-uv/bin/python

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
test "$(tar --zstd -xOf "$FULL_BANK_ARCHIVE" bank/dataset.json | sha256sum | awk '{print $1}')" = "$FULL_BANK_DATASET_SHA"
test "$(tar --zstd -xOf "$FULL_BANK_ARCHIVE" bank/trench_aligned_runtime_derivation.json | sha256sum | awk '{print $1}')" = "$FULL_BANK_DERIVATION_SHA"
test "$(sha256sum "$PARTIAL_BANK_ARCHIVE" | awk '{print $1}')" = "$PARTIAL_BANK_ARCHIVE_SHA"
test "$(tar --zstd -xOf "$PARTIAL_BANK_ARCHIVE" partial_bank/trench_alignment_audit.json | sha256sum | awk '{print $1}')" = "$PARTIAL_ALIGNMENT_AUDIT_SHA"

PYTHONPATH="$TERRA_REPO:$REPO" JAX_PLATFORMS=cpu \
FULL_BANK_ROOT="$FULL_BANK_ROOT" PARTIAL_BANK_ROOT="$PARTIAL_BANK_ROOT" \
RUNTIME_TERRA_REVISION="$RUNTIME_TERRA_REVISION" PROTOCOL_SHA="$PROTOCOL_SHA" \
SOURCE_REGISTRY_SHA="$SOURCE_REGISTRY_SHA" PARTIAL_BANK_SHA="$PARTIAL_BANK_SHA" \
RELEASE_ID="$RELEASE_ID" SEED="$SEED" \
EXPECTED_PARTIAL_CONDITIONS="$EXPECTED_PARTIAL_CONDITIONS" \
EXPECTED_PARTIAL_TRIPLETS="$EXPECTED_PARTIAL_TRIPLETS" \
EXPECTED_PARTIAL_SIDECARS="$EXPECTED_PARTIAL_SIDECARS" \
EXPECTED_RELAY_TRIPLETS="$EXPECTED_RELAY_TRIPLETS" \
EXPECTED_IN_ZONE_TRIPLETS="$EXPECTED_IN_ZONE_TRIPLETS" \
EXPECTED_TRENCH_AUDIT_SIDECARS="$EXPECTED_TRENCH_AUDIT_SIDECARS" \
"$LOCAL_PYTHON" - <<'PY'
from collections import Counter
import json
import os
from pathlib import Path

from terra.maps_buffer import partial_reset_bank_sha256
from utils.accepted_bank import load_accepted_bank, validate_staged_training_bank
from utils.pooled_sampler import PooledConditionSampler, SamplerSettings

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
    condition_profile="trench_aligned_37_v1",
)
assert len(bank.levels) == 37
assert sum(level.family == "foundation" for level in bank.levels) == 25
assert sum(level.family == "trench" for level in bank.levels) == 12
assert all(panel.condition_count == 35 for panel in bank.evaluation_panels)
assert all(
    panel.condition_count == 2
    for panel in bank.capability_floor_evaluation_panels
)
assert len(bank.curriculum_depths) == len(bank.levels)
labels = {
    level.condition_id: {
        "family": level.family,
        "branch_depth": level.branch_depth,
        "curriculum_depth": bank.curriculum_depths[index],
    }
    for index, level in enumerate(bank.levels)
}
depth_counts = Counter(
    (label["family"], label["curriculum_depth"])
    for label in labels.values()
)
assert tuple(depth_counts[("foundation", depth)] for depth in range(3)) == (
    1,
    6,
    18,
)
assert tuple(depth_counts[("trench", depth)] for depth in range(3)) == (
    1,
    0,
    11,
)
sampler = PooledConditionSampler(
    [level.condition_id for level in bank.levels],
    SamplerSettings(
        rule="continuous_banded_v3",
        update_interval=150,
        mastery_threshold=0.80,
        min_episodes=32,
        competence_ema=0.30,
        max_mass=0.15,
        seed=int(os.environ["SEED"]),
    ),
    maps_per_condition=[level.map_count for level in bank.levels],
    labels=labels,
    allow_sparse_depths=True,
)
assert len(sampler.probabilities) == 37
assert all(sampler.probabilities > 0.0)
assert abs(float(sum(sampler.probabilities)) - 1.0) < 1e-12
assert bank.environment_protocol_sha256 == os.environ["PROTOCOL_SHA"]
assert bank.source_registry_sha256 == os.environ["SOURCE_REGISTRY_SHA"]
assert partial_reset_bank_sha256(partial_root) == os.environ["PARTIAL_BANK_SHA"]
index = json.loads((partial_root / "partial_reset_bank.json").read_text())
assert len(index["supported_maps_paths"]) == int(
    os.environ["EXPECTED_PARTIAL_CONDITIONS"]
)
modes = Counter()
triplets = set()
for maps_path in index["supported_maps_paths"]:
    manifest = partial_root / maps_path / "partial_completion_manifest.jsonl"
    for line in manifest.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        modes[row["pile_mode"]] += 1
        triplets.add((maps_path, int(row["source_index"]), row["pile_mode"]))
assert len(triplets) == int(os.environ["EXPECTED_PARTIAL_TRIPLETS"])
assert sum(modes.values()) == int(os.environ["EXPECTED_PARTIAL_SIDECARS"])
assert modes["relay_corridor"] == 3 * int(
    os.environ["EXPECTED_RELAY_TRIPLETS"]
)
assert modes["in_zone"] == 3 * int(
    os.environ["EXPECTED_IN_ZONE_TRIPLETS"]
)
audit = json.loads((partial_root / "trench_alignment_audit.json").read_text())
assert audit["accepted"] is True and audit["failed_sidecars"] == 0
assert audit["partial_reset_bank_sha256"] == os.environ["PARTIAL_BANK_SHA"]
assert audit["audited_sidecars"] == int(
    os.environ["EXPECTED_TRENCH_AUDIT_SIDECARS"]
)
PY

printf '%s\n' \
    "terra_baselines_revision=$BASELINES_REVISION" \
    "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
    "condition_profile=trench_aligned_37_v1" \
    "conditions=37 foundation=25 trench=12" \
    "curriculum_depths_foundation=$FOUNDATION_CURRICULUM_DEPTH_COUNTS" \
    "curriculum_depths_trench=$TRENCH_CURRICULUM_DEPTH_COUNTS" \
    "sparse_curriculum_depths_allowed=true" \
    "xla_gpu_autotune_level=0 excluded_nodes=$EXCLUDED_NODES" \
    "partial_conditions=$EXPECTED_PARTIAL_CONDITIONS partial_triplets=$EXPECTED_PARTIAL_TRIPLETS" \
    "partial_reset_bank_sha256=$PARTIAL_BANK_SHA" \
    "seed=$SEED targets=smoke:1,phase1:75000,phase2:100000"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local contracts passed; no SSH, upload, W&B, or Slurm mutation"
    exit 0
fi

remote() { ssh -o BatchMode=yes "$REMOTE_HOST" "$@"; }
test "$(remote 'id -un')" = "$TERRA_EULER_USER"
remote "test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -w '$TERRA_EULER_SCRATCH_ROOT' && test -x '$REMOTE_VENV/bin/python'"
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
REMOTE_PARTIAL="$REMOTE_INPUTS/partial-bank-$PARTIAL_BANK_ARCHIVE_SHA.tar.zst"
upload "$FULL_BANK_ARCHIVE" "$REMOTE_BANK" "$FULL_BANK_ARCHIVE_SHA"
upload "$PARTIAL_BANK_ARCHIVE" "$REMOTE_PARTIAL" "$PARTIAL_BANK_ARCHIVE_SHA"

ASSOCIATIONS="$(remote "sacctmgr -n -P show assoc where user='$TERRA_EULER_USER' format=Account")"
printf '%s\n' "$ASSOCIATIONS" | grep -Eq '^%?es_hutter$'
for partition in gpuhe.4h gpuhe.120h; do
    remote "scontrol show partition '$partition' -o | grep -q 'State=UP'"
    remote "sinfo -h -p '$partition' -o '%G' | grep -q 'nvidia_geforce_rtx_4090'"
done
remote "df -Pk '$TERRA_EULER_SCRATCH_ROOT' | awk 'NR == 2 {exit !(\$4 > 50000000)}'"
if [ "$SUBMIT" = stage ]; then
    echo "SUBMIT=stage: exact source and immutable inputs staged; scheduler gates passed; no job or W&B mutation"
    exit 0
fi

COMMON_EXPORTS="ALL,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,PROTOCOL_TERRA_REVISION=$RUNTIME_TERRA_REVISION,PROTOCOL_SHA=$PROTOCOL_SHA,SEED=$SEED,VENV=$REMOTE_VENV,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,BANK_ARCHIVE=$REMOTE_BANK,BANK_ARCHIVE_SHA=$FULL_BANK_ARCHIVE_SHA,BANK_DATASET_SHA=$FULL_BANK_DATASET_SHA,BANK_DERIVATION_SHA=$FULL_BANK_DERIVATION_SHA,SOURCE_REGISTRY_SHA=$SOURCE_REGISTRY_SHA,DISTANCE_SIDECAR_SHA=$DISTANCE_SIDECAR_SHA,PARTIAL_ARCHIVE=$REMOTE_PARTIAL,PARTIAL_ARCHIVE_SHA=$PARTIAL_BANK_ARCHIVE_SHA,PARTIAL_BANK_SHA=$PARTIAL_BANK_SHA,PARTIAL_ALIGNMENT_AUDIT_SHA=$PARTIAL_ALIGNMENT_AUDIT_SHA,EXPECTED_PARTIAL_CONDITIONS=$EXPECTED_PARTIAL_CONDITIONS,EXPECTED_PARTIAL_TRIPLETS=$EXPECTED_PARTIAL_TRIPLETS,EXPECTED_PARTIAL_SIDECARS=$EXPECTED_PARTIAL_SIDECARS,EXPECTED_RELAY_TRIPLETS=$EXPECTED_RELAY_TRIPLETS,EXPECTED_IN_ZONE_TRIPLETS=$EXPECTED_IN_ZONE_TRIPLETS,EXPECTED_TRENCH_AUDIT_SIDECARS=$EXPECTED_TRENCH_AUDIT_SIDECARS,EXPECTED_PARAMETERS=$EXPECTED_PARAMETERS"
BASE_RUN_NAME="trench_generalist_partial_${BASELINES_REVISION:0:12}_s${SEED}"
RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/s$SEED"

if [ "$SUBMIT" = smoke ]; then
    RUN_ROLE=smoke
    RUN_DIR="$RUN_PARENT/smoke"
    RUN_NAME="${BASE_RUN_NAME}_smoke"
    remote "test ! -e '$RUN_DIR' && mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
    JOB_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_trench_align_generalist_partial_v1/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.4h' --time='03:45:00' --gpus='rtx_4090:4' --cpus-per-task='8' --exclude='$EXCLUDED_NODES' --job-name='terra-trench-partial-smoke' --output='$RUN_DIR/slurm_%j.out' --export='$COMMON_EXPORTS,RUN_ROLE=$RUN_ROLE,TARGET_UPDATE=1,RESUME_CHECKPOINT=none,RUN_DIR=$RUN_DIR,RUN_NAME=$RUN_NAME'")"
    JOB_ID="${JOB_RAW%%;*}"
    [[ "$JOB_ID" =~ ^[0-9]+$ ]]
    printf '%s\n' \
        "role=smoke" \
        "job_id=$JOB_ID" \
        "run_dir=$RUN_DIR" \
        "target_update=1"
    exit 0
fi

SMOKE_DIR="$RUN_PARENT/smoke"
SMOKE_NAME="${BASE_RUN_NAME}_smoke"
remote "test -f '$SMOKE_DIR/completion.env' && grep -qx 'status=COMPLETE' '$SMOKE_DIR/completion.env' && grep -qx 'target_update=1' '$SMOKE_DIR/completion.env'"
remote "test -f '$SMOKE_DIR/checkpoint_validation.json' && grep -Eq '\"next_update\": 1(,)?$' '$SMOKE_DIR/checkpoint_validation.json' && grep -q '\"model_finite\": true' '$SMOKE_DIR/checkpoint_validation.json' && grep -q '\"optimizer_finite\": true' '$SMOKE_DIR/checkpoint_validation.json'"
remote "test -f '$SMOKE_DIR/checkpoints/${SMOKE_NAME}_FINAL.pkl' && grep -qx 'terra_baselines_revision=$BASELINES_REVISION' '$SMOKE_DIR/run_contract.env' && grep -qx 'runtime_terra_revision=$RUNTIME_TERRA_REVISION' '$SMOKE_DIR/run_contract.env' && grep -qx 'partial_reset_bank_sha256=$PARTIAL_BANK_SHA' '$SMOKE_DIR/run_contract.env'"
remote "$REMOTE_VENV/bin/python -c 'import netrc; assert netrc.netrc().authenticators(\"api.wandb.ai\")'"

PHASE1_DIR="$RUN_PARENT/phase1_u75000"
PHASE2_DIR="$RUN_PARENT/phase2_u100000"
remote "test ! -e '$PHASE1_DIR' && test ! -e '$PHASE2_DIR' && mkdir -p '$RUN_PARENT' && mkdir '$PHASE1_DIR' '$PHASE2_DIR'"

JOB1_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_trench_align_generalist_partial_v1/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.120h' --time='119:45:00' --gpus='rtx_4090:4' --cpus-per-task='8' --exclude='$EXCLUDED_NODES' --job-name='terra-trench-partial-u75' --output='$PHASE1_DIR/slurm_%j.out' --export='$COMMON_EXPORTS,RUN_ROLE=phase1,TARGET_UPDATE=75000,RESUME_CHECKPOINT=none,RUN_DIR=$PHASE1_DIR,RUN_NAME=$BASE_RUN_NAME'")"
JOB1_ID="${JOB1_RAW%%;*}"
[[ "$JOB1_ID" =~ ^[0-9]+$ ]]

PHASE1_CHECKPOINT="$PHASE1_DIR/checkpoints/${BASE_RUN_NAME}_FINAL.pkl"
if ! JOB2_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_trench_align_generalist_partial_v1/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='gpuhe.120h' --time='119:45:00' --gpus='rtx_4090:4' --cpus-per-task='8' --exclude='$EXCLUDED_NODES' --dependency='afterok:$JOB1_ID' --kill-on-invalid-dep=yes --job-name='terra-trench-partial-u100' --output='$PHASE2_DIR/slurm_%j.out' --export='$COMMON_EXPORTS,RUN_ROLE=phase2,TARGET_UPDATE=100000,RESUME_CHECKPOINT=$PHASE1_CHECKPOINT,RUN_DIR=$PHASE2_DIR,RUN_NAME=$BASE_RUN_NAME'")"; then
    remote "scancel -- '$JOB1_ID'"
    exit 3
fi
JOB2_ID="${JOB2_RAW%%;*}"
if [[ ! "$JOB2_ID" =~ ^[0-9]+$ ]]; then
    remote "scancel -- '$JOB1_ID'"
    exit 3
fi

remote "printf '%s\n' 'phase1_job_id=$JOB1_ID' 'phase2_job_id=$JOB2_ID' 'dependency=afterok:$JOB1_ID' 'phase1_target_update=75000' 'final_target_update=100000' 'terra_baselines_revision=$BASELINES_REVISION' 'runtime_terra_revision=$RUNTIME_TERRA_REVISION' 'partial_reset_bank_sha256=$PARTIAL_BANK_SHA' > '$RUN_PARENT/submission.env'"
printf '%s\n' \
    "phase1_job_id=$JOB1_ID" \
    "phase2_job_id=$JOB2_ID" \
    "dependency=afterok:$JOB1_ID" \
    "phase1_run_dir=$PHASE1_DIR" \
    "phase2_run_dir=$PHASE2_DIR" \
    "final_target_update=100000"
