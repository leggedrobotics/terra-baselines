#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: submit.sh smoke|screen SEED REMOTE_TEACHER.pkl REMOTE_TEACHER_RUN_CONTRACT.env" >&2
    exit 2
fi
PHASE="$1"
SEED="$2"
TEACHER_CHECKPOINT="$3"
TEACHER_RUN_CONTRACT="$4"
case "$PHASE" in smoke|screen) ;; *) echo "invalid phase '$PHASE'" >&2; exit 2 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be nonnegative" >&2; exit 2; }
for PATH_VALUE in "$TEACHER_CHECKPOINT" "$TEACHER_RUN_CONTRACT"; do
    [[ "$PATH_VALUE" = /* ]] || {
        echo "teacher inputs must be absolute Euler paths" >&2
        exit 2
    }
done
SUBMIT="${SUBMIT:-0}"
case "$SUBMIT" in 0|1) ;; *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAMPAIGN_ID=terra_v8_10m_curriculum_v1
STAGE=capability
REMOTE_HOST="${REMOTE_HOST:-euler}"
REMOTE_WORK=/cluster/home/lterenzi/codex_terra_edge_validation/$CAMPAIGN_ID
REMOTE_RUNS=/cluster/scratch/lterenzi/codex_terra_edge_runs/$CAMPAIGN_ID
REMOTE_INPUTS=$REMOTE_RUNS/inputs
BANK_LOCAL=/home/lorenzo/moleworks/.artifacts/terra_v8_combined_accepted_20260803_v5r2.tar.zst
BANK_SHA=dedbbbfcd1aae648094bb7bcb25d7a28e80b96bdf2469bb941c2e321b7aaf82b
BANK_DATASET_SHA=715fa0b25cdb5c96a0f0768b532b29fa754d3c2844cbbff1ecfff5bbcc75e798
RELEASE_ID=terra_v8_v6_constraints_v7_adjacent_train96_v5
ARMS=(G-V8-XATTN-REWARM-CONTROL G-V8-10M-XATTN-WARM)

git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null
test -z "$(git -C "$REPO" status --porcelain)" || {
    echo "terra-baselines must be committed and clean before launch" >&2
    exit 3
}
test -f "$BANK_LOCAL"
test "$(sha256sum "$BANK_LOCAL" | awk '{print $1}')" = "$BANK_SHA"
test "$(tar --zstd -xOf "$BANK_LOCAL" bank/dataset.json | sha256sum | awk '{print $1}')" = "$BANK_DATASET_SHA"
BASELINES_REVISION="$(git -C "$REPO" rev-parse HEAD)"
REMOTE_SOURCE="$REMOTE_WORK/$BASELINES_REVISION/terra-baselines"
REMOTE_BANK="$REMOTE_INPUTS/bank-$BANK_SHA.tar.zst"
SMOKE_REVISION="${SMOKE_REVISION:-$BASELINES_REVISION}"
SMOKE_JOB_IDS_RAW="${SMOKE_JOB_IDS:-}"

echo "phase=$PHASE stage=$STAGE seed=$SEED target_update=$([ "$PHASE" = smoke ] && echo 1 || echo 4000)"
echo "terra_baselines_revision=$BASELINES_REVISION"
echo "release_id=$RELEASE_ID"
echo "teacher_checkpoint=$TEACHER_CHECKPOINT"
echo "map_curriculum=capability_then_nearby_then_full"
echo "reward_curriculum=dense_skill_then_terminal_margin_then_terminal_objective"
if [ "$SUBMIT" = 0 ]; then
    echo "SUBMIT=0: no SSH, scratch, W&B, or Slurm mutation"
    for ARM in "${ARMS[@]}"; do
        echo "future sbatch: phase=$PHASE stage=$STAGE arm=$ARM seed=$SEED"
    done
    exit 0
fi

ssh "$REMOTE_HOST" "test -f '$TEACHER_CHECKPOINT' && test -f '$TEACHER_RUN_CONTRACT'"
TEACHER_SHA="$(ssh "$REMOTE_HOST" "sha256sum '$TEACHER_CHECKPOINT' | awk '{print \$1}'")"
TEACHER_RUN_CONTRACT_SHA="$(ssh "$REMOTE_HOST" "sha256sum '$TEACHER_RUN_CONTRACT' | awk '{print \$1}'")"
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

SMOKE_JOB_ARRAY=()
SMOKE_JOB_STATE_ARRAY=()
if [ "$PHASE" = screen ]; then
    if [ -n "$SMOKE_JOB_IDS_RAW" ]; then
        IFS=, read -r -a SMOKE_JOB_ARRAY <<< "$SMOKE_JOB_IDS_RAW"
        test "${#SMOKE_JOB_ARRAY[@]}" -eq 2
    fi
    for INDEX in "${!ARMS[@]}"; do
        ARM="${ARMS[$INDEX]}"
        SMOKE="$REMOTE_RUNS/$SMOKE_REVISION/smoke/$STAGE/s$SEED/$ARM"
        if [ -z "$SMOKE_JOB_IDS_RAW" ]; then
            ssh "$REMOTE_HOST" "test -f '$SMOKE/run_contract.env'"
            SMOKE_JOB_ARRAY+=("$(ssh "$REMOTE_HOST" "awk -F= '\$1==\"slurm_job_id\" {print \$2}' '$SMOKE/run_contract.env'")")
        fi
        SMOKE_JOB_ID="${SMOKE_JOB_ARRAY[$INDEX]}"
        [[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
        SMOKE_JOB_STATE="$(ssh "$REMOTE_HOST" "sacct -n -X -P -j '$SMOKE_JOB_ID' --format=JobIDRaw,State | awk -F'|' -v id='$SMOKE_JOB_ID' '\$1==id {sub(/\\+.*/, \"\", \$2); print \$2}'")"
        case "$SMOKE_JOB_STATE" in
            PENDING|RUNNING) test -n "$SMOKE_JOB_IDS_RAW" ;;
            COMPLETED)
                ssh "$REMOTE_HOST" "test -f '$SMOKE/smoke_validation.json' && python3 -c 'import json; assert json.load(open(\"$SMOKE/smoke_validation.json\"))[\"passed\"] is True' && test -f '$SMOKE/initialization_diagnostic.json' && python3 -c 'import json; d=json.load(open(\"$SMOKE/initialization_diagnostic.json\")); assert d[\"passed\"] is True; assert d[\"exact_frozen_map_slots\"] == 720; assert d[\"reset_key_contract\"] == \"deterministic_exact_slot_keys_v1\"; assert d[\"teacher_admission\"] == \"provisional_inspection\"' && test \"\$(sha256sum '$SMOKE/initialization_diagnostic.json' | awk '{print \$1}')\" = \"\$(awk -F= '\$1==\"initialization_diagnostic_sha256\" {print \$2}' '$SMOKE/run_contract.env')\" && test \"\$(awk -F= '\$1==\"teacher_checkpoint_sha256\" {print \$2}' '$SMOKE/run_contract.env')\" = '$TEACHER_SHA' && test \"\$(awk -F= '\$1==\"teacher_run_contract_sha256\" {print \$2}' '$SMOKE/run_contract.env')\" = '$TEACHER_RUN_CONTRACT_SHA' && test \"\$(awk -F= '\$1==\"status\" {print \$2}' '$SMOKE/run_contract.env')\" = PASSED"
                ;;
            *) echo "smoke job $SMOKE_JOB_ID has unsupported state '$SMOKE_JOB_STATE'" >&2; exit 3 ;;
        esac
        SMOKE_JOB_STATE_ARRAY+=("$SMOKE_JOB_STATE")
    done
fi

case "$PHASE" in
    smoke) PARTITION=gpuhe.4h; WALLTIME=04:00:00; GPU_TYPE=rtx_3090 ;;
    screen) PARTITION=gpuhe.120h; WALLTIME=119:45:00; GPU_TYPE=rtx_4090 ;;
esac
for INDEX in "${!ARMS[@]}"; do
    ARM="${ARMS[$INDEX]}"
    RUN_PARENT="$REMOTE_RUNS/$BASELINES_REVISION/$PHASE/$STAGE/s$SEED"
    RUN_DIR="$RUN_PARENT/$ARM"
    ssh "$REMOTE_HOST" "mkdir -p '$RUN_PARENT' && mkdir '$RUN_DIR'"
    SMOKE_JOB_ID=none
    SMOKE_RUN=none
    DEPENDENCY_OPTION=
    if [ "$PHASE" = screen ]; then
        SMOKE_JOB_ID="${SMOKE_JOB_ARRAY[$INDEX]}"
        SMOKE_RUN="$REMOTE_RUNS/$SMOKE_REVISION/smoke/$STAGE/s$SEED/$ARM"
        if [ "${SMOKE_JOB_STATE_ARRAY[$INDEX]}" != COMPLETED ]; then
            DEPENDENCY_OPTION="--dependency=afterok:$SMOKE_JOB_ID --kill-on-invalid-dep=yes"
        fi
    fi
    EXPORTS="ALL,PHASE=$PHASE,STAGE=$STAGE,ARM=$ARM,BASELINES_ROOT=$REMOTE_SOURCE,BASELINES_REVISION=$BASELINES_REVISION,SEED=$SEED,BANK_ARCHIVE=$REMOTE_BANK,BANK_SHA=$BANK_SHA,BANK_DATASET_SHA=$BANK_DATASET_SHA,BANK_RELEASE_ID=$RELEASE_ID,TEACHER_CHECKPOINT=$TEACHER_CHECKPOINT,TEACHER_SHA=$TEACHER_SHA,TEACHER_RUN_CONTRACT=$TEACHER_RUN_CONTRACT,TEACHER_RUN_CONTRACT_SHA=$TEACHER_RUN_CONTRACT_SHA,SMOKE_JOB_ID=$SMOKE_JOB_ID,SMOKE_RUN=$SMOKE_RUN"
    JOB_ID="$(
        ssh "$REMOTE_HOST" "cat '$REMOTE_SOURCE/scripts/euler_v8_10m_v1/run.sbatch' | sbatch --parsable --partition='$PARTITION' --time='$WALLTIME' --gpus='$GPU_TYPE:4' $DEPENDENCY_OPTION --exclude='eu-g6-064' --job-name='terra-v8-10m-${PHASE}-${STAGE}-${ARM}' --output='$RUN_DIR/slurm_%j.out' --export='$EXPORTS'"
    )"
    echo "$PHASE $STAGE $ARM $JOB_ID"
done
