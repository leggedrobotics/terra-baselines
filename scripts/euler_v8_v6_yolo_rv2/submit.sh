#!/usr/bin/env bash
set -euo pipefail

R2_HORIZON=450
if [ "$#" -ne 1 ]; then
    echo "usage: submit.sh smoke|phase1|phase2  (MASK_VARIANT=mask|nomask|v61|v61_v4|stall_age, default mask)" >&2
    exit 2
fi
PHASE="$1"
case "$PHASE" in
    smoke|phase1|phase2) ;;
    *) echo "invalid phase '$PHASE'" >&2; exit 2 ;;
esac
# The v6.1 stall-age continuation. Phase1 ended at u14000 while still climbing.
# Phase2 changes only the policy observation by adding normalized material stall
# age. It preserves phase1's 2,048 total envs and 65,536 transitions/update by
# reshaping 4x512 to 8x256, and restores the continuous_banded_v2 sampler without
# migration. The production allocation itself performs the first-update smoke.
RESUME_SOURCE_UPDATE=14000
RESUME_TARGET_UPDATE=40000
RESUME_SOURCE_SHA=79312602176e88b696c8c006b3b9af71a4cf121907c7aa8c4865722bd4830609
RESUME_PREPARED_SHA=96600430af3fb0135e0fc94e8f9dd754476067fbfb8635a3db70d6c3519b6971
# Copied lterenzi -> local -> the launching account: scratch is not shared.
RESUME_SOURCE_LOCAL="${TERRA_RESUME_SOURCE_LOCAL:-/home/lorenzo/moleworks/.artifacts/terra_v8_v6_yolo_rv2_continuation_20260813/v8_v6_yolo_rv2_v61_9abf88eb60df_s20260807_update_014000.pkl}"
RESUME_SOURCE_RUN=/cluster/scratch/lterenzi/codex_terra_edge_runs/terra_v8_v6_yolo_rv2/runs/9abf88eb60dfc0eb2395a5cc799b933928b6952c/phase1/s20260807/v6_1_rv2
case "$PHASE" in
    phase2) RESUMING=1 ;;
    *) RESUMING=0 ;;
esac
if [ "$RESUMING" = 1 ] && [ "${MASK_VARIANT:-mask}" != stall_age ]; then
    echo "phase2 is the v6.1 stall-age continuation: set MASK_VARIANT=stall_age" >&2
    exit 2
fi
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in
    0|stage|1) ;;
    *) echo "SUBMIT must be 0, stage, or 1" >&2; exit 2 ;;
esac

# Arm selector. The no-mask arms run on the baseline's exact terra (the mask
# terra computes the mask unconditionally, so flag-off there would still pay
# its per-step cost). v61 is the post-day-2 arm: v6 readout without the
# full-res rebalance, without vf_coef=0.5 and without masking, aux at 0.1.
# v61_v4 is the same scratch architecture/reward contract with only the new
# family-free continuous sampler; it has its own arm and run namespace.
MASK_VARIANT="${MASK_VARIANT:-mask}"
SCRATCH_SAMPLER_PROFILE=continuous_banded_v2
PAIRED_BASELINE_ARM=reward_v2_scratch
case "$MASK_VARIANT" in
    mask)
        EXPECTED_RUNTIME_TERRA_REVISION=04c67bbafce2cb3d1a1de35384dfde477d244349
        TERRA_REPO_DEFAULT=/home/lorenzo/moleworks/.worktrees/terra_v8_v6_yolo_rv2_20260810
        ARM_NAME=v6_3m_yolo_rv2
        ACTION_LOGIT_MASKING=1
        ;;
    nomask)
        EXPECTED_RUNTIME_TERRA_REVISION=3051054bc4c713d95905d3f954e6eabf55d6a85a
        TERRA_REPO_DEFAULT=/home/lorenzo/moleworks/.worktrees/terra_v8_r2_reward_v2_20260810
        ARM_NAME=v6_3m_yolo_rv2_nomask
        ACTION_LOGIT_MASKING=0
        ;;
    v61)
        EXPECTED_RUNTIME_TERRA_REVISION=3051054bc4c713d95905d3f954e6eabf55d6a85a
        TERRA_REPO_DEFAULT=/home/lorenzo/moleworks/.worktrees/terra_v8_r2_reward_v2_20260810
        ARM_NAME=v6_1_rv2
        ACTION_LOGIT_MASKING=0
        ;;
    v61_v4)
        EXPECTED_RUNTIME_TERRA_REVISION=3051054bc4c713d95905d3f954e6eabf55d6a85a
        TERRA_REPO_DEFAULT=/home/lorenzo/moleworks/.worktrees/terra_v8_r2_reward_v2_20260810
        ARM_NAME=v6_1_rv2_v4
        ACTION_LOGIT_MASKING=0
        SCRATCH_SAMPLER_PROFILE=continuous_banded_v4
        PAIRED_BASELINE_ARM=v6_1_rv2
        ;;
    stall_age)
        EXPECTED_RUNTIME_TERRA_REVISION=c2d2a94a124759e9f21c2b37930f717e299f0c46
        TERRA_REPO_DEFAULT=/home/lorenzo/moleworks/.worktrees/terra_v8_v61_stall_age_20260813
        ARM_NAME=v6_1_rv2_stall_age
        ACTION_LOGIT_MASKING=0
        PAIRED_BASELINE_ARM=v6_1_rv2
        ;;
    *) echo "invalid MASK_VARIANT '$MASK_VARIANT'" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=cluster/euler_account.sh
source "$REPO/cluster/euler_account.sh"
terra_euler_configure "${TERRA_EULER_USER:-alesweber}"
TERRA_REPO="${TERRA_REPO:-$TERRA_REPO_DEFAULT}"
ARTIFACT_ROOT=/home/lorenzo/moleworks/.artifacts/terra_v8_r2_training_inputs_20260810
ADMISSION_ROOT=/home/lorenzo/moleworks/.artifacts/terra_v8_r2_admission_20260810
STALL_AGE_ROOT=/home/lorenzo/moleworks/.artifacts/terra_v61_stall_age_continuation_20260813
RESUME_PREPARED_LOCAL="${TERRA_STALL_AGE_PREPARED_LOCAL:-$STALL_AGE_ROOT/v8_v61_stall_age_u14000_prepared.pkl}"
BANK_ARCHIVE="$ARTIFACT_ROOT/treatment_bank.tar.zst"
MATERIALIZATION_RECEIPT="$ARTIFACT_ROOT/treatment_bank_receipt.json"
STATIC_RECEIPT_MANIFEST="$ADMISSION_ROOT/static_v2/receipt_manifest.json"
D4A_RECEIPT="$ADMISSION_ROOT/d4a/d4a_receipt.json"
D4A_MANIFEST="$ADMISSION_ROOT/d4a/receipt_manifest.json"

SEED=20260807
CAMPAIGN=terra_v8_v6_yolo_rv2
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
BANK_SHA=b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725
BANK_DATASET_SHA=5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851
BANK_TREE_SHA=225e13aacd9047e7f241facd3397fd66794e3094a883cc6dc26304decc24d388
DISTANCE_SIDECAR_SHA=f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980
MATERIALIZATION_RECEIPT_SHA=631fac8c3b78ff2c5a9e94ea4032244c9ef05dc6c984b603e4318121a263d3f1
STATIC_RECEIPT_MANIFEST_SHA=9b16c391dbe0c108f4b79833f1940c5fc0ba31903a1e7edbfec1797aa53740d9
EXPECTED_D4A_RECEIPT_SHA=6905300337310456a28ec6177a8c7d74f73892ebe052d11d29e9e0fa5bec7362
EXPECTED_D4A_MANIFEST_SHA=cc969a69810b5ed0d14b85d58a0932ae26659a34686c4eadb760ae24b7cc87a4
REMOTE_HOST="${REMOTE_HOST:-euler-$TERRA_EULER_USER}"
REMOTE_WORK_ROOT="${TERRA_REMOTE_WORK_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation}"
REMOTE_RUN_ROOT="${TERRA_REMOTE_RUN_ROOT:-$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs}"
REMOTE_WORK="$REMOTE_WORK_ROOT/$CAMPAIGN"
REMOTE_RUNS="$REMOTE_RUN_ROOT/$CAMPAIGN/runs"
REMOTE_INPUTS="$REMOTE_RUN_ROOT/$CAMPAIGN/inputs"
REMOTE_VENV="${TERRA_REMOTE_VENV:-/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426}"
WANDB_ENTITY="${TERRA_WANDB_ENTITY:-aless-weber-eth}"
WANDB_PROJECT="${TERRA_WANDB_PROJECT:-mixed-agents}"

case "$REMOTE_HOST" in
    ''|-*|*[!a-zA-Z0-9._@-]*)
        echo "invalid REMOTE_HOST '$REMOTE_HOST'" >&2
        exit 2
        ;;
esac
for REMOTE_PATH in \
    "$REMOTE_WORK_ROOT" "$REMOTE_RUN_ROOT" "$REMOTE_VENV"; do
    case "$REMOTE_PATH" in
        /cluster/*) ;;
        *) echo "remote path must be absolute under /cluster: $REMOTE_PATH" >&2; exit 2 ;;
    esac
    case "$REMOTE_PATH" in
        *[!a-zA-Z0-9_./-]*)
            echo "remote path contains unsupported characters: $REMOTE_PATH" >&2
            exit 2
            ;;
    esac
done
for WANDB_NAME in "$WANDB_ENTITY" "$WANDB_PROJECT"; do
    case "$WANDB_NAME" in
        ''|*[!a-zA-Z0-9_.-]*)
            echo "invalid W&B entity/project '$WANDB_NAME'" >&2
            exit 2
            ;;
    esac
done

test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean" >&2; exit 3;
}
test -z "$(git -C "$TERRA_REPO" status --porcelain)" || {
    echo "runtime Terra must be committed and clean" >&2; exit 3;
}
test "$(git -C "$TERRA_REPO" rev-parse HEAD)" = "$EXPECTED_RUNTIME_TERRA_REVISION"
LOCAL_TERRA_PYTHON=/home/lorenzo/moleworks/.venv-terra-uv/bin/python
test -x "$LOCAL_TERRA_PYTHON"
PYTHONPATH="$TERRA_REPO" JAX_PLATFORMS=cpu PYGAME_HIDE_SUPPORT_PROMPT=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$LOCAL_TERRA_PYTHON" - <<'PY'
from terra.config import (
    REWARD_V2_ALPHA,
    REWARD_V2_BETA,
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_DISTANCE_REF_M,
    REWARD_V2_HORIZON_FAILURE_PENALTY,
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_PROTOCOL_ID,
    REWARD_V2_SHAPING_WEIGHT,
    REWARD_V2_STEP_COST_TOTAL,
    REWARD_V2_SUCCESS_BONUS,
    RewardStage,
)
from terra.env_generation.distance import REWARD_V2_DISTANCE_PROTOCOL_ID
from terra.state import CORRECTED_DENSE_CONTRACT

assert int(RewardStage.REWARD_V2) == 3
assert REWARD_V2_PROTOCOL_ID == "material_potential_v2"
assert REWARD_V2_DISTANCE_PROTOCOL_ID == "obstacle_geodesic_8_physical_global_v1"
assert CORRECTED_DENSE_CONTRACT == "exact_visible_dump_v1"
assert (
    REWARD_V2_DISTANCE_REF_M,
    REWARD_V2_DISTANCE_BOUND,
    REWARD_V2_POTENTIAL_GAMMA,
    REWARD_V2_SUCCESS_BONUS,
    REWARD_V2_HORIZON_FAILURE_PENALTY,
    REWARD_V2_ALPHA,
    REWARD_V2_BETA,
    REWARD_V2_STEP_COST_TOTAL,
    REWARD_V2_SHAPING_WEIGHT,
) == (16.0, 2.5, 0.9984, 6.0, 1.0, 1.0, 1.5, 1.0, 1.0)
PY

for SPEC in \
    "$BANK_ARCHIVE:$BANK_SHA" \
    "$MATERIALIZATION_RECEIPT:$MATERIALIZATION_RECEIPT_SHA" \
    "$STATIC_RECEIPT_MANIFEST:$STATIC_RECEIPT_MANIFEST_SHA"; do
    PATH_LOCAL="${SPEC%:*}"
    EXPECTED_SHA="${SPEC##*:}"
    test "$(sha256sum "$PATH_LOCAL" | awk '{print $1}')" = "$EXPECTED_SHA"
done
test "$(tar --zstd -xOf "$BANK_ARCHIVE" bank/dataset.json | sha256sum | awk '{print $1}')" = "$BANK_DATASET_SHA"
python3 - "$MATERIALIZATION_RECEIPT" <<'PY'
import json, sys

material = json.load(open(sys.argv[1]))
assert material["schema"] == "terra_v8_r2_materialized_distance_bank_v1"
assert material["status"] == "passed"
assert material["base_bank"]["unchanged"] is True
assert material["pair_equivalence"]["scenarios"] == 7520
assert material["pair_equivalence"]["physical_arrays_preserved"] is True
assert material["pair_equivalence"]["metadata_and_pose_sidecars_preserved"] is True
assert material["treatment_bank"]["dataset_json_sha256"] == "5f19861b1ca8feb1ffce909fdf173f2131fb3cb05682b849c948a08573ad7851"
assert material["treatment_bank"]["tree_sha256"] == "225e13aacd9047e7f241facd3397fd66794e3094a883cc6dc26304decc24d388"
assert material["canonical_sidecar"]["dataset_json_sha256"] == "f0c430651d21cced4189a6879eb53187d6abb1607f9a997978ff748506c58980"
PY

test -f "$D4A_RECEIPT" -a -f "$D4A_MANIFEST" || {
    echo "D4a durable replay receipt is missing; reward-v2 launch remains blocked" >&2
    exit 4
}
test "$(sha256sum "$D4A_RECEIPT" | awk '{print $1}')" = "$EXPECTED_D4A_RECEIPT_SHA"
test "$(sha256sum "$D4A_MANIFEST" | awk '{print $1}')" = "$EXPECTED_D4A_MANIFEST_SHA"
D4A_RECEIPT_SHA="$EXPECTED_D4A_RECEIPT_SHA"
D4A_MANIFEST_SHA="$EXPECTED_D4A_MANIFEST_SHA"
python3 - "$D4A_RECEIPT" "$D4A_MANIFEST" "$D4A_RECEIPT_SHA" <<'PY'
import hashlib, json, pathlib, sys

receipt = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2]))
assert receipt["schema"] == "terra_v8_r2_d4a_replay_v1"
assert receipt["status"] == "passed"
assert receipt["all_targeted_traces_match_frozen_rows"] is True
assert receipt["targeted_trace_count"] == 9
targeted = receipt["frozen_parity"]
assert targeted["episodes"] == 720 and targeted["all_nine_equal"] is True
assert all(targeted["targeted_checks"].values())
assert targeted["cross_hardware_policy_drift_is_non_gating"] is True
assert receipt["reset_verification"]["passed"] is True
assert receipt["action_tensor"]["shape"] == [450, 720]
assert receipt["action_tensor"]["dtype"] == "int32"
command = receipt["command_contract"]
assert command["seed"] == 20260807 and command["horizon"] == 450
assert command["deterministic"] is True and command["wandb"] == "disabled"
assert command["baselines_revision"] == "dcc4f955347182e57e6f16e9df81a3f170564d97"
assert command["terra_revision"] == "eb3835c1d17af81e970b973ed5abf687ca6f3a26"
assert command["bank_declared_terra_revision"] == "a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4"
assert receipt["analysis_script_sha256"] == "268f73abfbfea05eff0ee3a41b7995fd6900b5a3268ba5ae91e7153a5d8dc7a4"
assert receipt["analysis_support_sha256"] == "8c182bbcb906d581222b6637ad7d45ca45b196fe748b5a867cee44c46510e3a7"
ledger = receipt["ledger"]
lift_tolerance = ledger["tolerance"]["lift"]
assert lift_tolerance["rule"] == "max(absolute_floor, ulp_multiplier * max_float32_spacing)"
assert lift_tolerance["absolute_floor"] == 1e-6
assert lift_tolerance["ulp_multiplier"] == 4.0
assert ledger["lift_gate_passed"] is True
assert ledger["failed_lift_event_count"] == 0
assert ledger["max_inert_transition_error"] <= ledger["tolerance"]["inert"]
assert ledger["max_dump_progress_telescope_error"] <= ledger["tolerance"]["telescope"]
diagnostics_path = pathlib.Path(sys.argv[1]).parent / receipt["lift_diagnostics"]
diagnostics = json.load(open(diagnostics_path))
assert diagnostics["schema"] == "terra_v8_r2_d4a_lift_diagnostics_v1"
assert diagnostics["status"] == "passed"
assert diagnostics["failed_lift_event_count"] == 0
assert diagnostics["written_before_gate_raise"] is True
diagnostics_sha = hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()
assert diagnostics_sha == receipt["lift_diagnostics_sha256"]
assert manifest["files"][diagnostics_path.name]["sha256"] == diagnostics_sha
assert manifest["files"]["d4a_receipt.json"]["sha256"] == sys.argv[3]
PY

BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
RUNTIME_TERRA_REVISION="$EXPECTED_RUNTIME_TERRA_REVISION"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_TERRA="$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION/terra"
case "$PHASE" in
    smoke) SPAN="0->1" ;;
    phase1) SPAN="0->14000" ;;
    phase2) SPAN="$RESUME_SOURCE_UPDATE->$RESUME_TARGET_UPDATE" ;;
esac
echo "phase=$PHASE arm=$ARM_NAME baseline=$PAIRED_BASELINE_ARM seed=$SEED absolute_updates=$SPAN sampler=$SCRATCH_SAMPLER_PROFILE"
if [ "$RESUMING" = 1 ]; then
    test -f "$RESUME_SOURCE_LOCAL" || {
        echo "resume source checkpoint is not staged locally: $RESUME_SOURCE_LOCAL" >&2
        echo "copy it from $RESUME_SOURCE_RUN/checkpoints/ first" >&2
        exit 3
    }
    test "$(sha256sum "$RESUME_SOURCE_LOCAL" | awk '{print $1}')" = "$RESUME_SOURCE_SHA"
    mkdir -p "$(dirname "$RESUME_PREPARED_LOCAL")"
    if [ ! -f "$RESUME_PREPARED_LOCAL" ]; then
        PREPARED_PARTIAL="$RESUME_PREPARED_LOCAL.partial.$$"
        PYTHONPATH="$TERRA_REPO:$REPO" JAX_PLATFORMS=cpu \
            "$LOCAL_TERRA_PYTHON" "$REPO/scripts/prepare_v61_stall_age_continuation.py" \
            --source "$RESUME_SOURCE_LOCAL" --output "$PREPARED_PARTIAL"
        test "$(sha256sum "$PREPARED_PARTIAL" | awk '{print $1}')" = "$RESUME_PREPARED_SHA"
        mv -T "$PREPARED_PARTIAL" "$RESUME_PREPARED_LOCAL"
    fi
    test "$(sha256sum "$RESUME_PREPARED_LOCAL" | awk '{print $1}')" = "$RESUME_PREPARED_SHA"
    PYTHONPATH="$TERRA_REPO:$REPO" JAX_PLATFORMS=cpu \
        "$LOCAL_TERRA_PYTHON" "$REPO/scripts/prepare_v61_stall_age_continuation.py" \
        --source "$RESUME_SOURCE_LOCAL" --verify "$RESUME_PREPARED_LOCAL"
    echo "resume_source=$RESUME_SOURCE_LOCAL sha256=$RESUME_SOURCE_SHA"
    echo "prepared_resume=$RESUME_PREPARED_LOCAL sha256=$RESUME_PREPARED_SHA"
    echo "resume_shape=8x256x32/32 sampler=continuous_banded_v2 target=$RESUME_TARGET_UPDATE"
fi
echo "terra_baselines_revision=$BASELINES_REVISION runtime_terra_revision=$RUNTIME_TERRA_REVISION"
echo "d4a_receipt_sha256=$D4A_RECEIPT_SHA"
echo "euler_user=$TERRA_EULER_USER remote_host=$REMOTE_HOST"
echo "remote_work=$REMOTE_WORK remote_runs=$REMOTE_RUNS remote_venv=$REMOTE_VENV"
echo "wandb_entity=$WANDB_ENTITY wandb_project=$WANDB_PROJECT"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: local contracts passed; no SSH, upload, W&B, or Slurm mutation"
    exit 0
fi

remote() {
    ssh -o BatchMode=yes "$REMOTE_HOST" "$@"
}

REMOTE_ID="$(remote 'id -un')"
test "$REMOTE_ID" = "$TERRA_EULER_USER" || {
    echo "remote account mismatch: expected $TERRA_EULER_USER, got $REMOTE_ID" >&2
    exit 3
}
remote \
    "test \"\$HOME\" = '$TERRA_EULER_HOME_ROOT' && test -w '$TERRA_EULER_SCRATCH_ROOT' && test -x '$REMOTE_VENV/bin/python'"
HOME_USED_GB="$(remote lquota | "$REPO/cluster/lquota_home_used_gb.sh" "$TERRA_EULER_HOME_ROOT")"
awk -v used="$HOME_USED_GB" 'BEGIN { exit !(used < 45.0) }' || {
    echo "home quota launch gate failed: ${HOME_USED_GB} GB used" >&2
    exit 3
}

if ! remote "test -e '$REMOTE_SOURCE'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" | remote "tar -xf - -C '$PARTIAL/terra-baselines'"
    remote "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv -T '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
if ! remote "test -e '$REMOTE_TERRA'"; then
    PARTIAL="$REMOTE_WORK/runtime-terra/.${RUNTIME_TERRA_REVISION}.partial.$$"
    remote "mkdir -p '$PARTIAL/terra'"
    git -C "$TERRA_REPO" archive --format=tar "$RUNTIME_TERRA_REVISION" | remote "tar -xf - -C '$PARTIAL/terra'"
    remote "printf '%s\n' '$RUNTIME_TERRA_REVISION' > '$PARTIAL/terra/REVISION' && mv -T '$PARTIAL' '$REMOTE_WORK/runtime-terra/$RUNTIME_TERRA_REVISION'"
fi
remote "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION' && \
    test -s '$REMOTE_SOURCE/train_mixed.py' && \
    test -s '$REMOTE_SOURCE/scripts/euler_v8_v6_yolo_rv2/run.sbatch' && \
    test -s '$REMOTE_SOURCE/scripts/run_v8_v6_yolo_rv2.sh' && \
    test \"\$(cat '$REMOTE_TERRA/REVISION')\" = '$RUNTIME_TERRA_REVISION' && \
    test -s '$REMOTE_TERRA/terra/state.py' && test -s '$REMOTE_TERRA/terra/config.py'"
remote "mkdir -p '$REMOTE_INPUTS' '$REMOTE_RUNS'"

upload() {
    local source="$1" destination="$2" expected_sha="$3"
    if ! remote "test -f '$destination'"; then
        local partial="$destination.partial.$$"
        scp -q -o BatchMode=yes "$source" "$REMOTE_HOST:$partial"
        remote "test \"\$(sha256sum '$partial' | awk '{print \$1}')\" = '$expected_sha' && mv -T '$partial' '$destination'"
    fi
    remote "test \"\$(sha256sum '$destination' | awk '{print \$1}')\" = '$expected_sha'"
}

REMOTE_BANK="$REMOTE_INPUTS/treatment-bank-$BANK_SHA.tar.zst"
REMOTE_MATERIALIZATION_RECEIPT="$REMOTE_INPUTS/materialization-$MATERIALIZATION_RECEIPT_SHA.json"
REMOTE_STATIC_MANIFEST="$REMOTE_INPUTS/static-admission-$STATIC_RECEIPT_MANIFEST_SHA.json"
REMOTE_D4A_RECEIPT="$REMOTE_INPUTS/d4a-$D4A_RECEIPT_SHA.json"
REMOTE_D4A_MANIFEST="$REMOTE_INPUTS/d4a-manifest-$D4A_MANIFEST_SHA.json"
upload "$BANK_ARCHIVE" "$REMOTE_BANK" "$BANK_SHA"
upload "$MATERIALIZATION_RECEIPT" "$REMOTE_MATERIALIZATION_RECEIPT" "$MATERIALIZATION_RECEIPT_SHA"
upload "$STATIC_RECEIPT_MANIFEST" "$REMOTE_STATIC_MANIFEST" "$STATIC_RECEIPT_MANIFEST_SHA"
upload "$D4A_RECEIPT" "$REMOTE_D4A_RECEIPT" "$D4A_RECEIPT_SHA"
upload "$D4A_MANIFEST" "$REMOTE_D4A_MANIFEST" "$D4A_MANIFEST_SHA"

# Source and zero-initialized treatment checkpoint are content-addressed inputs.
# Scratch is not shared between accounts, so both are staged from local copies.
REMOTE_RESUME_SOURCE=none
REMOTE_RESUME_CHECKPOINT=none
if [ "$RESUMING" = 1 ]; then
    REMOTE_RESUME_SOURCE="$REMOTE_INPUTS/v6-1-rv2-u$RESUME_SOURCE_UPDATE-$RESUME_SOURCE_SHA.pkl"
    REMOTE_RESUME_CHECKPOINT="$REMOTE_INPUTS/v6-1-rv2-stall-age-u$RESUME_SOURCE_UPDATE-$RESUME_PREPARED_SHA.pkl"
    upload "$RESUME_SOURCE_LOCAL" "$REMOTE_RESUME_SOURCE" "$RESUME_SOURCE_SHA"
    upload "$RESUME_PREPARED_LOCAL" "$REMOTE_RESUME_CHECKPOINT" "$RESUME_PREPARED_SHA"
fi

# Phase2 is one resumable 24-hour production segment. Its absolute u40000 target
# intentionally exceeds the likely segment capacity; a wall-time exit with a
# finite rolling checkpoint is continuable.
case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00; GPU_TYPE=rtx_3090; GPU_COUNT=4; CPUS=4 ;;
    phase1) PARTITION=gpuhe.24h; WALLTIME=23:45:00; GPU_TYPE=rtx_4090; GPU_COUNT=4; CPUS=4 ;;
    phase2) PARTITION=gpuhe.24h; WALLTIME=23:45:00; GPU_TYPE=rtx_4090; GPU_COUNT=8; CPUS=8 ;;
esac
if [ "$SUBMIT" = stage ]; then
    ASSOCIATIONS="$(remote "sacctmgr -n -P show assoc where user='$TERRA_EULER_USER' format=Account")"
    printf '%s\n' "$ASSOCIATIONS" | grep -Eq '^%?es_hutter$'
    remote "scontrol show partition '$PARTITION' -o | grep -q 'State=UP'"
    remote "sinfo -h -p '$PARTITION' -o '%G' | grep -Eq 'gpu:nvidia_geforce_${GPU_TYPE}:([$GPU_COUNT-9]|[1-9][0-9]+)'"
    PARTITION_MAX_TIME="$(remote "scontrol show partition '$PARTITION' -o" | tr ' ' '\n' | awk -F= '$1=="MaxTime" {print $2}')"
    python3 - "$PARTITION_MAX_TIME" "$WALLTIME" <<'PY'
import sys


def seconds(value: str) -> int:
    days, _, rest = value.partition("-")
    if rest:
        fields = [int(x) for x in rest.split(":")] + [0, 0]
        return int(days) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]
    fields = [int(x) for x in value.split(":")]
    while len(fields) < 3:
        fields.insert(0, 0)
    return fields[0] * 3600 + fields[1] * 60 + fields[2]


limit, request = seconds(sys.argv[1]), seconds(sys.argv[2])
if request > limit:
    raise SystemExit(f"requested {sys.argv[2]} exceeds partition MaxTime {sys.argv[1]}")
print(f"partition_max_time={sys.argv[1]} requested_walltime={sys.argv[2]}")
PY
    echo "SUBMIT=stage: code and pinned inputs staged; Slurm association/partition/GPU inventory passed; no job or W&B mutation"
    exit 0
fi

SMOKE_JOB_ID=none
SMOKE_RUN=none
if [ "$PHASE" = phase1 ]; then
    # Preserve the historical scratch arm's update-1 gate. The new phase2 has
    # no separate allocation: its own first full update is the runtime smoke.
    GATING_RECEIPT=smoke_validation.json
    GATING_SAMPLER_PROFILE="$SCRATCH_SAMPLER_PROFILE"
    SMOKE_RUN="$REMOTE_RUNS/$BASELINES_REVISION/smoke/s$SEED/$ARM_NAME"
    remote "test -f '$SMOKE_RUN/$GATING_RECEIPT' -a -f '$SMOKE_RUN/run_contract.env' && \
        test \"\$(stat -c %U '$SMOKE_RUN/$GATING_RECEIPT')\" = '$TERRA_EULER_USER' && \
        test \"\$(stat -c %U '$SMOKE_RUN/run_contract.env')\" = '$TERRA_EULER_USER'"
    remote "python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))[\"passed\"] is True' '$SMOKE_RUN/$GATING_RECEIPT'"
    SMOKE_JOB_ID="$(remote "awk -F= '\$1==\"slurm_job_id\" {print \$2}' '$SMOKE_RUN/run_contract.env'")"
    [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
    SMOKE_STATE="$(remote "sacct -n -X -P -j '$SMOKE_JOB_ID' --format=JobIDRaw,State | awk -F'|' -v id='$SMOKE_JOB_ID' '\$1==id {sub(/\\+.*/, \"\", \$2); print \$2}'")"
    test "$SMOKE_STATE" = COMPLETED
    for EXPECTED in \
        "runtime_terra_revision=$RUNTIME_TERRA_REVISION" \
        "terra_baselines_revision=$BASELINES_REVISION" \
        "distance_artifact_sha256=$DISTANCE_SIDECAR_SHA" \
        "sampler_profile=$GATING_SAMPLER_PROFILE" \
        "euler_user=$TERRA_EULER_USER"; do
        KEY="${EXPECTED%%=*}" VALUE="${EXPECTED#*=}"
        remote "test \"\$(awk -F= -v key='$KEY' '\$1==key {print \$2}' '$SMOKE_RUN/run_contract.env')\" = '$VALUE'"
    done
fi
if [ "$PHASE" = phase1 ] || [ "$PHASE" = phase2 ]; then
    remote "python3 -c 'import netrc; assert netrc.netrc().authenticators(\"api.wandb.ai\")'" || {
        echo "$PHASE requires a W&B api.wandb.ai credential in the selected account's ~/.netrc" >&2
        exit 3
    }
fi

RUN_DIR="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/s$SEED/$ARM_NAME"
JOB_ID=""
cleanup_new_job() {
    local status="$1"
    trap - ERR INT TERM
    set +e
    if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        remote "scancel -- '$JOB_ID'"
    fi
    # rmdir is intentionally the only cleanup: non-empty evidence survives.
    remote "rmdir -- '$RUN_DIR'" || true
    exit "$status"
}
trap 'cleanup_new_job $?' ERR
trap 'cleanup_new_job 130' INT TERM
remote "test ! -e '$RUN_DIR' && mkdir -p '$(dirname "$RUN_DIR")' && mkdir '$RUN_DIR'"

EXPORTS="ALL,TERRA_EULER_USER=$TERRA_EULER_USER,TERRA_EULER_HOME_ROOT=$TERRA_EULER_HOME_ROOT,VENV=$REMOTE_VENV,RUN_BASE=$REMOTE_RUNS,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,PHASE=$PHASE,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,RUNTIME_TERRA_ROOT=$REMOTE_TERRA,RUNTIME_TERRA_REVISION=$RUNTIME_TERRA_REVISION,R2_HORIZON=$R2_HORIZON,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_TREE_SHA=$BANK_TREE_SHA,BANK_RELEASE_ID=$RELEASE_ID,DISTANCE_ARTIFACT_SHA=$DISTANCE_SIDECAR_SHA,MATERIALIZATION_RECEIPT=$REMOTE_MATERIALIZATION_RECEIPT,MATERIALIZATION_RECEIPT_SHA=$MATERIALIZATION_RECEIPT_SHA,STATIC_RECEIPT_MANIFEST=$REMOTE_STATIC_MANIFEST,STATIC_RECEIPT_MANIFEST_SHA=$STATIC_RECEIPT_MANIFEST_SHA,D4A_RECEIPT=$REMOTE_D4A_RECEIPT,D4A_RECEIPT_SHA=$D4A_RECEIPT_SHA,D4A_MANIFEST=$REMOTE_D4A_MANIFEST,D4A_MANIFEST_SHA=$D4A_MANIFEST_SHA,SMOKE_JOB_ID=$SMOKE_JOB_ID,SMOKE_RUN=$SMOKE_RUN,ARM_NAME=$ARM_NAME,ACTION_LOGIT_MASKING=$ACTION_LOGIT_MASKING,RESUME_SOURCE_CHECKPOINT=$REMOTE_RESUME_SOURCE,RESUME_SOURCE_SHA=$([ "$RESUMING" = 1 ] && echo "$RESUME_SOURCE_SHA" || echo none),RESUME_CHECKPOINT=$REMOTE_RESUME_CHECKPOINT,RESUME_CHECKPOINT_SHA=$([ "$RESUMING" = 1 ] && echo "$RESUME_PREPARED_SHA" || echo none)"
JOB_ID_RAW="$(remote "cat '$REMOTE_SOURCE/scripts/euler_v8_v6_yolo_rv2/run.sbatch' | sbatch --parsable --account='es_hutter' --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:$GPU_COUNT' --cpus-per-task='$CPUS' --exclude='eu-g6-064' --job-name='terra-v61-stall-age' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'")"
JOB_ID="${JOB_ID_RAW%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
trap - ERR INT TERM
echo "$PHASE $ARM_NAME $JOB_ID"
