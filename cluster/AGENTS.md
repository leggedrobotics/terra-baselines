# Euler cluster guidance

Use this directory for account-neutral Euler setup and launch support. Read
`cluster/README.md` and resolve storage with `cluster/euler_account.sh` before
editing or invoking a campaign launcher.

## Identity and secrets

- Select the Unix account explicitly with `TERRA_EULER_USER` and use a matching
  named SSH alias. Verify `ssh -o BatchMode=yes HOST id -un` before writes.
- Keep passwords, private keys, W&B API keys, `.netrc`, and environment files
  containing secrets out of the repository, Slurm exports, and logs.
- Give each SSH account its own multiplexing socket (`ControlPath ~/.ssh/cm-%C`)
  so a cached connection cannot cross account boundaries.

## Storage

- Home is only for credentials and small configuration.
- Reproducible code snapshots go under
  `$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_validation`.
- Inputs, logs, W&B files, and checkpoints go under
  `$TERRA_EULER_SCRATCH_ROOT/codex_terra_edge_runs`.
- Long-lived venvs and archives go to writable project/work storage. Scratch is
  purged after inactivity, so it must not contain the only durable copy.
- Dataset ownership and runtime ownership are independent of the execution
  account. Validate read/execute permissions and record exact paths.

## Launcher contract

- Default to a dry contract check (`SUBMIT=0`).
- Provide a staging-only mode that may upload pinned content and make read-only
  Slurm association/partition/GPU-inventory queries, but cannot contact W&B or
  submit a job (`SUBMIT=stage`).
- Require explicit `SUBMIT=1` for `sbatch`.
- Before staging, verify remote identity, `$HOME`, scratch writability, runtime
  executability, repository revisions, and all pinned hashes.
- A migrated account needs its own completed, finite update-1 smoke. Do not
  reuse a smoke receipt stored in another account's private scratch.
- Production jobs require the configured W&B credential, requested and
  allocated GPU-type checks, compiled convolution-backward and NCCL preflights,
  and the campaign's scientific admission gates.
- Never cancel, move, or rewrite an in-flight job merely because the default
  account changed.

Run `cluster/test_euler_account.sh`, `cluster/test_lquota_home_used_gb.sh`, and
`shellcheck` on edited shell launchers.
