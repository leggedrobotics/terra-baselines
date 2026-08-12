#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 5 ]; then
    echo "usage: submit.sh smoke|screen capability [SEED]" >&2
    echo "       submit.sh smoke|screen nearby SEED DEEP_GATE.json XATTN_GATE.json" >&2
    echo "       submit.sh smoke|screen full SEED DEEP_GATE.json XATTN_GATE.json" >&2
    exit 2
fi
PHASE="$1"
STAGE="${2:-capability}"
SEED="${3:-20260730}"
case "$PHASE" in smoke|screen) ;; *) echo "invalid phase '$PHASE'" >&2; exit 2 ;; esac
case "$STAGE" in
    capability)
        test "$#" -le 3 || { echo "capability does not accept gate receipts" >&2; exit 2; }
        ;;
    nearby)
        test "$#" -eq 5 || { echo "nearby requires both Stage-A gate receipts" >&2; exit 2; }
        PRIOR_STAGE=capability
        ;;
    full)
        test "$#" -eq 5 || { echo "full requires both Stage-B gate receipts" >&2; exit 2; }
        PRIOR_STAGE=nearby
        ;;
    *) echo "stage '$STAGE' is not enabled by this launcher revision" >&2; exit 2 ;;
esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative" >&2; exit 2; }
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TERRA_REPO="${TERRA_REPO:-/home/lorenzo/moleworks/.worktrees/terra_p5c_authority_20260802}"
TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
CAMPAIGN_ID=terra_v8_deep_xattn_v1
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
BANK_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
PARENT=/cluster/scratch/lterenzi/codex_terra_edge_runs/p5c_low_entropy_v1/3478af87950d3d35059344b078209d00785c8481/screen/s20260730/G-DEEP-UNIFORM-WARM/checkpoints/p5c_3478af87950d_screen_g_deep_uniform_warm_s20260730-euler-2026-08-03-00-39-08_update_004000.pkl
PARENT_SHA=4d178c39443009cb4e57d83713421553689f6e3989da0be674184237c14d86cc
REMOTE_HOST="${REMOTE_HOST:-euler}"
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_INPUTS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID/inputs
REMOTE_RUNS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID
ARMS=(G-DEEP-V8-DENSE-WARM G-DEEP-XATTN-V8-DENSE-WARM)
LOCAL_PYTHON="${LOCAL_PYTHON:-/home/lorenzo/moleworks/.venv-terra-uv/bin/python}"
GATE_TOOL="$REPO/scripts/euler_v8_deep_xattn_v1/stage_gate.py"
declare -A PARENTS PARENT_SHAS GATE_PATHS GATE_SHAS REMOTE_GATES

if [ "$STAGE" = capability ]; then
    for ARM in "${ARMS[@]}"; do
        PARENTS[$ARM]="$PARENT"
        PARENT_SHAS[$ARM]="$PARENT_SHA"
        GATE_PATHS[$ARM]=none
        GATE_SHAS[$ARM]=none
        REMOTE_GATES[$ARM]=none
    done
else
    test -x "$LOCAL_PYTHON"
    GATE_ARGUMENTS=("$4" "$5")
    for INDEX in "${!ARMS[@]}"; do
        ARM="${ARMS[$INDEX]}"
        GATE_PATH="${GATE_ARGUMENTS[$INDEX]}"
        test -f "$GATE_PATH"
        INFO="$($LOCAL_PYTHON "$GATE_TOOL" inspect \
            --receipt "$GATE_PATH" --stage "$PRIOR_STAGE" --arm "$ARM")"
        read -r CANDIDATE CANDIDATE_SHA RECEIPT_SHA NEXT_STAGE <<< "$($LOCAL_PYTHON -c \
            'import json,sys; d=json.load(sys.stdin); print(d["candidate_path"], d["candidate_sha256"], d["receipt_sha256"], d["next_stage"])' \
            <<< "$INFO")"
        test "$NEXT_STAGE" = "$STAGE"
        PARENTS[$ARM]="$CANDIDATE"
        PARENT_SHAS[$ARM]="$CANDIDATE_SHA"
        GATE_PATHS[$ARM]="$GATE_PATH"
        GATE_SHAS[$ARM]="$RECEIPT_SHA"
        REMOTE_GATES[$ARM]="$REMOTE_INPUTS/gates/$RECEIPT_SHA.json"
    done
fi

case "$STAGE" in
    capability) SCREEN_UPDATES=2000 ;;
    nearby) SCREEN_UPDATES=4000 ;;
    full) SCREEN_UPDATES=8000 ;;
esac

git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null
test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
test -z "$(git -C "$TERRA_REPO" status --porcelain)" || {
    echo "paired Terra authority must be clean" >&2
    exit 3
}
test "$(git -C "$TERRA_REPO" rev-parse HEAD)" = "$TERRA_REVISION"
test -f "$BANK_LOCAL"
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
test "$(tar --zstd -xOf "$BANK_LOCAL" bank/dataset.json | sha256sum | awk '{print $1}')" = "$BANK_DATASET_SHA"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"

echo "phase=$PHASE stage=$STAGE seed=$SEED updates=$SCREEN_UPDATES"
echo "terra_revision=$TERRA_REVISION"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "release_id=$RELEASE_ID"
echo "bank_archive_sha256=$BANK_SHA"
echo "bank_dataset_sha256=$BANK_DATASET_SHA"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, scratch, W&B, or Slurm mutation"
    for ARM in "${ARMS[@]}"; do
        echo "future sbatch: phase=$PHASE stage=$STAGE arm=$ARM seed=$SEED parent_sha=${PARENT_SHAS[$ARM]} gate_sha=${GATE_SHAS[$ARM]}"
        if [ "$PHASE" = screen ] && [ "$STAGE" = full ]; then
            echo "future dependent tail evaluator: arm=$ARM dependency=afterany:PARENT_JOB_ID"
        fi
    done
    exit 0
fi

for ARM in "${ARMS[@]}"; do
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '${PARENTS[$ARM]}' | awk '{print \$1}')\" = '${PARENT_SHAS[$ARM]}'"
done
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_SOURCE/REVISION'"; then
    PARTIAL="$REMOTE_WORK/.${BASELINES_REVISION}.partial.$$"
    ssh "$REMOTE_HOST" "mkdir -p '$PARTIAL/terra-baselines'"
    git -C "$REPO" archive --format=tar "$BASELINES_REVISION" \
        | ssh "$REMOTE_HOST" "tar -xf - -C '$PARTIAL/terra-baselines'"
    ssh "$REMOTE_HOST" "printf '%s\n' '$BASELINES_REVISION' > '$PARTIAL/terra-baselines/REVISION' && mv '$PARTIAL' '$REMOTE_WORK/$BASELINES_REVISION'"
fi
ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_SOURCE/REVISION')\" = '$BASELINES_REVISION'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS' '$REMOTE_RUNS'"
if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_BANK'"; then
    PARTIAL="$REMOTE_BANK.partial.$$"
    scp -q "$BANK_LOCAL" "$REMOTE_HOST:$PARTIAL"
    ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '$BANK_SHA' && mv '$PARTIAL' '$REMOTE_BANK'"
fi
ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_BANK' | awk '{print \$1}')\" = '$BANK_SHA'"
if [ "$STAGE" != capability ]; then
    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_INPUTS/gates'"
    for ARM in "${ARMS[@]}"; do
        REMOTE_GATE="${REMOTE_GATES[$ARM]}"
        if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_GATE'"; then
            PARTIAL="$REMOTE_GATE.partial.$$"
            scp -q "${GATE_PATHS[$ARM]}" "$REMOTE_HOST:$PARTIAL"
            ssh "$REMOTE_HOST" "test \"\$(sha256sum '$PARTIAL' | awk '{print \$1}')\" = '${GATE_SHAS[$ARM]}' && mv '$PARTIAL' '$REMOTE_GATE'"
        fi
        ssh "$REMOTE_HOST" "test \"\$(sha256sum '$REMOTE_GATE' | awk '{print \$1}')\" = '${GATE_SHAS[$ARM]}'"
    done
fi

if [ "$PHASE" = screen ]; then
    for ARM in "${ARMS[@]}"; do
        SMOKE="$REMOTE_RUNS/$BASELINES_REVISION/smoke/$STAGE/s$SEED/$ARM"
        ssh "$REMOTE_HOST" "test -f '$SMOKE/smoke_validation.json' && python3 -c 'import json; assert json.load(open(\"$SMOKE/smoke_validation.json\"))[\"passed\"] is True' && python3 '$REMOTE_SOURCE/scripts/euler_v8_deep_xattn_v1/stage_gate.py' check-smoke --run-contract '$SMOKE/run_contract.env' --stage '$STAGE' --arm '$ARM' --seed '$SEED' --parent-sha256 '${PARENT_SHAS[$ARM]}' --prior-gate-sha256 '${GATE_SHAS[$ARM]}' >/dev/null"
    done
fi

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00 ;;
    screen) PARTITION=gpuhe.24h; WALLTIME=23:45:00 ;;
esac
for ARM in "${ARMS[@]}"; do
    RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/$STAGE/s$SEED"
    RUN_DIR="$RUN_PARENT/$ARM"
    ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
    EXPORTS="ALL,PHASE=$PHASE,STAGE=$STAGE,ARM=$ARM,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,SCREEN_UPDATES=$SCREEN_UPDATES,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,PARENT_CHECKPOINT=${PARENTS[$ARM]},PARENT_SHA=${PARENT_SHAS[$ARM]},PRIOR_GATE_RECEIPT=${REMOTE_GATES[$ARM]},PRIOR_GATE_SHA=${GATE_SHAS[$ARM]}"
    JOB_ID="$(
        ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_deep_xattn_v1/run.sbatch' | sbatch --parsable --partition='$PARTITION' --time='$WALLTIME' --exclude='eu-g6-064' --job-name='terra-v8-${PHASE}-${STAGE}-${ARM}' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'"
    )"
    echo "$PHASE $STAGE $ARM $JOB_ID"
    if [ "$PHASE" = screen ] && [ "$STAGE" = full ]; then
        SUBMIT=1 REMOTE_HOST="$REMOTE_HOST" TERRA_REPO="$TERRA_REPO" \
            "$REPO/scripts/euler_v8_deep_xattn_v1/submit_tail.sh" \
            "$JOB_ID" "$ARM" "$SEED" "${GATE_PATHS[$ARM]}"
    fi
done
