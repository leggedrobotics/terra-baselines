#!/usr/bin/env bash
# Sync the M2 dose-curriculum campaign to Euler and submit the three 4-GPU arms.
# Run from THE CONDITION-TELEMETRY CHECKOUT (the M2 instruments live there), on
# this machine, with the `euler` ssh host and both materialized banks present.
#
#   scripts/euler_map_curriculum_v6m/submit_map_curriculum_v6m.sh          # sync + submit
#   SUBMIT=0 scripts/euler_map_curriculum_v6m/submit_map_curriculum_v6m.sh # sync only
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TERRA=/home/lorenzo/moleworks/.worktrees/terra_simple_mapbank_reward_20260730
BANK_V5M="$REPO/terra_data/curriculum_v5m"
BANK_V6M="$REPO/terra_data/curriculum_v6m"
RUN_ROOT=/cluster/scratch/lterenzi/codex_terra_edge_runs/map_curriculum_v6m_reward_v3_20260730
VENV=/home/lorenzo/moleworks/.venv-terra-gpu-uv
SUBMIT="${SUBMIT:-1}"

test -f "$BANK_V5M/curriculum_v5m_provenance.json"
test -f "$BANK_V6M/curriculum_v6m_provenance.json"
test -f "$REPO/logs/smoke_m2_v3/LOCAL_SMOKE_PASSED"
# The M2 diagnostics are the per-condition telemetry and the choke table. A
# checkout without them cannot answer the questions M2 is asking.
test -f "$REPO/scripts/analyze_condition_telemetry.py"
test -f "$REPO/utils/episode_aggregates.py"

# Validate the single relocation reward knob against the same Terra source the
# jobs will use.
PYTHONPATH="$TERRA:$REPO" JAX_PLATFORMS=cpu PYGAME_HIDE_SUPPORT_PROMPT=1 \
    SDL_VIDEODRIVER=dummy "$VENV/bin/python" -m pytest "$REPO/tests/test_m2_presets.py" -q

TERRA_REV="$(git -C "$TERRA" rev-parse HEAD)"
BASELINES_REV="$(git -C "$REPO" rev-parse HEAD)"
TERRA_DIRTY="$(git -C "$TERRA" status --porcelain | wc -l)"
BASELINES_DIRTY="$(git -C "$REPO" status --porcelain | wc -l)"
read_sha() {
    python3 -c "
import json,sys
print(json.load(open(sys.argv[1]))['bank_manifest_csv_sha256'])
" "$1"
}
BANK_V5M_SHA="$(read_sha "$BANK_V5M/curriculum_v5m_provenance.json")"
BANK_V6M_SHA="$(read_sha "$BANK_V6M/curriculum_v6m_provenance.json")"
# Both trees still carry uncommitted work, so pin the files that define the run
# instead of trusting the branch tips alone.
sha() { sha256sum "$1" | awk '{print $1}'; }
SHA_MAPS_BUFFER="$(sha "$TERRA/terra/maps_buffer.py")"
SHA_CURRICULUM="$(sha "$TERRA/terra/curriculum.py")"
SHA_STATE="$(sha "$TERRA/terra/state.py")"
SHA_TERRA_CONFIG="$(sha "$TERRA/terra/config.py")"
SHA_TRAIN_MIXED="$(sha "$REPO/train_mixed.py")"
SHA_MODELS="$(sha "$REPO/utils/models.py")"
SHA_CONFIGS="$(sha "$REPO/configs/training_configs.yaml")"
SHA_AGGREGATES="$(sha "$REPO/utils/episode_aggregates.py")"

echo "terra            = $TERRA_REV (dirty files: $TERRA_DIRTY)"
echo "terra-baselines  = $BASELINES_REV (dirty files: $BASELINES_DIRTY)"
echo "bank v5m         = $BANK_V5M_SHA"
echo "bank v6m         = $BANK_V6M_SHA"

ssh euler "mkdir -p '$RUN_ROOT'/{source,bank,logs,checkpoints}"

rsync -a --delete --exclude '__pycache__' --exclude '.git' --exclude 'data' \
    "$TERRA/" "euler:$RUN_ROOT/source/terra/"
rsync -a --delete --exclude '__pycache__' --exclude '.git' --exclude 'logs' \
    --exclude 'checkpoints' --exclude 'checkpoints_v5m' --exclude 'terra_data' \
    "$REPO/" "euler:$RUN_ROOT/source/terra-baselines/"
# -H keeps the converter's intra-level hardlinks so a bank costs ~6k inodes
# instead of ~21k on the shared scratch file quota.
rsync -aH --delete "$BANK_V5M/" "euler:$RUN_ROOT/bank/v5m/"
rsync -aH --delete "$BANK_V6M/" "euler:$RUN_ROOT/bank/v6m/"

ssh euler "cat > '$RUN_ROOT/source_sync_manifest.txt'" <<EOF
campaign=M2 map curriculum with agent-neutral reward-v3 (CURRICULUM_SPEC_V6.md section 4 maps/scheduler only)
terra_revision=$TERRA_REV
terra_uncommitted_files=$TERRA_DIRTY
terra_baselines_revision=$BASELINES_REV
terra_baselines_uncommitted_files=$BASELINES_DIRTY
terra_maps_buffer_sha256=$SHA_MAPS_BUFFER
terra_curriculum_sha256=$SHA_CURRICULUM
terra_state_sha256=$SHA_STATE
terra_config_sha256=$SHA_TERRA_CONFIG
train_mixed_sha256=$SHA_TRAIN_MIXED
models_sha256=$SHA_MODELS
training_configs_yaml_sha256=$SHA_CONFIGS
episode_aggregates_sha256=$SHA_AGGREGATES
bank_v5m_manifest_csv_sha256=$BANK_V5M_SHA
bank_v6m_manifest_csv_sha256=$BANK_V6M_SHA
holdout_map_indices=3,7,11,14
level_size=864
init=scratch
reward=agent-neutral relocation_progress_mult=1.5
arms=WAVE:m2_wave_long@v5m seed 20260730 | DOSE:m2_dose@v6m seed 20260731 | FAST:m2_dose_fast@v6m seed 20260732
promotion=WAVE,DOSE 3-consecutive-exact | FAST 2-consecutive-exact (graded disjunct DESCOPED)
requeue=forbidden (REVIEW_V6 F5-3 cond_cum resume reset)
synced_at=$(date -Is)
EOF

ssh euler "
    set -e
    cd '$RUN_ROOT'
    test -f source/terra/terra/curriculum.py
    test -f source/terra-baselines/train_mixed.py
    test -f source/terra-baselines/scripts/analyze_condition_telemetry.py
    for L in L0 L1 L2 L3; do
        n=\$(ls bank/v5m/train/\$L/images | wc -l)
        test \"\$n\" -eq 864 || { echo \"v5m level \$L has \$n slots\" >&2; exit 3; }
    done
    for L in L0p L1p L2p L3p; do
        n=\$(ls bank/v6m/train/\$L/images | wc -l)
        test \"\$n\" -eq 864 || { echo \"v6m level \$L has \$n slots\" >&2; exit 3; }
    done
    test \$(ls bank/v5m/held_out/all/images | wc -l) -eq 116
    test \$(ls bank/v6m/held_out/all/images | wc -l) -eq 128
    touch SOURCE_SYNC_VERIFIED
    echo SOURCE_SYNC_VERIFIED
"
scp -q "$REPO/logs/smoke_m2_v3/LOCAL_SMOKE_PASSED" "euler:$RUN_ROOT/LOCAL_SMOKE_PASSED"

if [ "$SUBMIT" != "1" ]; then
    echo "SUBMIT=0, stopping after sync"
    exit 0
fi

ssh euler "
    set -e
    cd '$RUN_ROOT/source/terra-baselines/scripts/euler_map_curriculum_v6m'
    : > '$RUN_ROOT/submitted_jobs.txt'
    for spec in 'WAVE:terra-v6m-wave-long' 'DOSE:terra-v6m-dose' 'FAST:terra-v6m-dose-fast'; do
        arm=\${spec%%:*}
        name=\${spec##*:}
        jid=\$(sbatch --parsable --job-name=\"\$name\" --export=ALL,ARM=\$arm run_map_curriculum_v6m.sbatch)
        echo \"\$arm \$name \$jid\" | tee -a '$RUN_ROOT/submitted_jobs.txt'
    done
    squeue -u lterenzi -o '%.10i %.24j %.9P %.2t %.11M %R' | grep -E 'JOBID|terra-v6m'
"
