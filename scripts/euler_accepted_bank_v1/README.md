# Accepted-bank Euler campaign

This directory implements the immutable Euler path for
[`P5_ACCEPTED_BANK_EXPERIMENTS.md`](../../docs/research/P5_ACCEPTED_BANK_EXPERIMENTS.md).
The canonical research question and the
[`$simple-research-code`](/home/lorenzo/git/codex_skills/skills/simple-research-code/SKILL.md)
constraint remain in that document. These scripts do not define a map bank.

## Inputs and storage

`prepare_submit.sh` accepts only:

```text
prepare_submit.sh PHASE CLEAN_TERRA_REPO ACCEPTED_BANK_ROOT [SEED]
```

Both source repositories must be clean. The bank must be loader-ready, bound
to the exact Terra commit, and declare
`scenario_identity_contract=terra_reset_arrays_sha256_v1`. This depends on the
paired Terra materializer/MapsBuffer commit that implements that contract.
Banks containing `NON_ADMISSION.md` or `REVIEW_ONLY.md` are rejected.

The preparation output is one deterministic `campaign-<sha>.tar.zst` with:

```text
campaign/
  manifest.json
  source/terra/
  source/terra-baselines/
  bank/
  runtime/check_jax_runtime.py
```

Production archives live under
`/cluster/work/rsl/lterenzi/terra_curriculum_campaigns/sha256-<sha>/`.
Each allocated job verifies and expands one archive only in `$TMPDIR`.
Scratch contains only logs, checkpoints, W&B files and receipts under
`/cluster/scratch/lterenzi/codex_terra_edge_runs/accepted_bank_v1/<sha>/<phase>/s<seed>/<arm>/`.

`SUBMIT=0` is the default. It creates the local content-addressed archive and
prints the future remote commands, but performs no SSH, upload, scratch, W&B
or Slurm mutation. A marked pilot bank can be exercised only with
`ALLOW_NON_ADMISSION_FOR_TESTS=1 SUBMIT=0`; that mode cannot submit.

## Phases and gates

| Phase | Jobs | Partition | Updates | Admission |
|---|---:|---|---:|---|
| `smoke` | four arms | `gpuhe.24h` | 1 | finite periodic + FINAL model, optimizer and loss; clean transition integrity |
| `screen` | four arms | `gpuhe.24h` | 2,000 | all four smoke receipts pass |
| future P6 | selected generalist only | `gpuhe.120h` | 20,000 from scratch | separate 256-train-layouts/condition bank plus a passing generalist decision |

Every allocation requires exactly four RTX 4090 GPUs, four CPUs, and the
`4 x 1024 x 32` rollout shape. The dependency-only venv and its complete
artifact ledger are checked before JAX. Non-smoke jobs also require
an `api.wandb.ai` credential in Euler's private `~/.netrc`; W&B is disabled
for smoke.

Screens retain checkpoints every 500 updates and evaluate only 500, 1,000 and
2,000.

Before screen training starts, `select_promotion.py` resets the
ordered promotion and development panels twice from the same checkpoint: once
in a CPU JAX process and once in the allocated GPU process. For a screen this
is the already-validated smoke FINAL checkpoint for that arm. It hashes every
Agent slot with Terra's tested `terra.benchmark_state.agent_state_sha256`
codec and requires the ordered CPU/GPU hashes, episode identities, layer
verification and bank identities to match byte for byte.

`F-ANCHOR` and `T-ANCHOR` are feasibility screens only. The 20k selection
compares `G-UNIFORM` and `G-ADAPTIVE` on the identical full promotion panel:

1. require an update-1,000 to update-2,000 comparison gate pass;
2. rank passing arms by final macro completion, exact successes, then
   worst-condition completion; and
3. choose `G-UNIFORM` on an exact tie.

The sealed panel is never opened by this campaign.

P6 is intentionally not executable here. The screen bank has 64 training
layouts per condition; the canonical plan requires a distinct accepted bank
with 256 training layouts per condition while keeping the evaluation panels
frozen. `prepare_submit.sh promote ...` and `run.sbatch` therefore fail closed
with an explicit message. `select_promotion.py` remains available to produce
the decision receipt after the two generalist screens, but no 20k job can be
launched until the cross-bank P6 contract is implemented and reviewed.

## Commands

Local preparation only:

```bash
SUBMIT=0 scripts/euler_accepted_bank_v1/prepare_submit.sh \
  smoke /path/to/clean/terra /path/to/frozen/accepted-bank 20260730
```

After explicit authorization, set `SUBMIT=1` for `smoke`, then `screen`.
`screen` verifies all four immutable smoke receipts, including campaign, arm
and seed. Each exact run leaf is reserved with one atomic `mkdir`; duplicate
submission fails instead of sharing outputs. Direct `sbatch` use bypasses the
content-addressed admission path and is unsupported.
