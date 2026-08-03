# Terra Legacy-Easy v1 Euler evaluation

This launcher scores an existing policy on the immutable Legacy-Easy
current-runtime capability floor. It does not train, convert checkpoints,
stage code, use the training split, or open the sealed split.

Each policy gets four separate results and logs:

- promotion, deterministic argmax (primary);
- promotion, sampled with seed `20260803` (secondary);
- development, deterministic argmax (primary); and
- development, sampled with seed `20260803` (secondary).

Every call uses the explicit initial states, horizon 450, dense reward,
foundation edge alignment off, trench absolute shaping off, and the
`exact_visible_dump_v1` completion contract frozen by the episode bank. The
result verifier requires all 48 episodes and zero integrity failures. These
scores are a diagnostic capability floor and never enter the constrained-map
macro.

## Policy matrix

Use a three-column TSV with this exact header:

```text
policy_label	checkpoint_path	checkpoint_sha256
deep-p5c-u4000	/cluster/work/rsl/lterenzi/checkpoints/deep_u4000.pkl	<64 lowercase hex>
e8-current-compat	/cluster/work/rsl/lterenzi/checkpoints/e8_current.pkl	<64 lowercase hex>
```

Labels become output directory names and must be unique. Checkpoints and code
roots must be canonical absolute paths. Code roots must be clean Git checkouts
whose `HEAD` equals the supplied full revision.

## Preflight and submit

Run this on Euler after code and checkpoints have been staged deliberately.
The frozen episode bank already lives under `/cluster/work`; this launcher
only reads it.

```bash
export BASELINES_ROOT=/cluster/work/rsl/lterenzi/terra-legacy-eval/terra-baselines
export BASELINES_REVISION=<full revision containing this launcher>
export TERRA_ROOT=/cluster/work/rsl/lterenzi/terra-legacy-eval/terra
export TERRA_REVISION=a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4
export EPISODE_BANK_ROOT=/cluster/work/rsl/lterenzi/terra_legacy_easy_v1_current_episode_bank_v1_20260803_a6e6e5bc
export EPISODE_BANK_JSON_SHA256=ce85dbf2e20b568f6258bef99513fbc30a408dbbf2351e3e12b6c533448b6c64
export EPISODE_BANK_FILES_SHA256=1b64e782da0fb355c3945e95055a213425c453e73961d0bdf83ccf60b2877df1
export PYTHON_BIN=/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426/bin/python
export OUTPUT_BASE=/cluster/scratch/lterenzi/codex_terra_edge_runs/legacy_easy_v1

# Full hash/revision/checkpoint preflight; prints exact commands, submits nothing.
SUBMIT=0 "$BASELINES_ROOT/scripts/euler_legacy_easy_v1/submit.sh" policies.tsv

# Same preflight, then one named Slurm job per TSV row.
SUBMIT=1 "$BASELINES_ROOT/scripts/euler_legacy_easy_v1/submit.sh" policies.tsv
```

`submit.sh` prints a stable index/job-name/policy/output mapping. A job fails if
its output directory already exists. On success, that directory contains the
four JSON files, four logs, `receipt.env`, and `files.sha256`. Inputs are read
in place and never modified; Python bytecode and runtime caches go to the
allocation-local temporary directory.

The submitter exports the canonical installed launcher directory explicitly.
This is required because Slurm executes a copied `run.sbatch` from its spool
directory; helper lookup must never depend on `BASH_SOURCE` inside the job.
