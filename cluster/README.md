# Terra on Euler

Terra launchers must keep the login identity separate from code, run, dataset,
and runtime paths. Passwords never belong in this repository, exported Slurm
environments, or launcher arguments; configure a named SSH host with a key.

## Accounts and storage

Resolve the standard roots with:

```bash
cluster/euler_account.sh alesweber
cluster/euler_account.sh lterenzi
```

The selected account owns these defaults:

- home: `/cluster/home/<account>` (credentials and small config only);
- reproducible code snapshots: `/cluster/scratch/<account>/codex_terra_edge_validation`;
- live logs, W&B files, checkpoints, and inputs:
  `/cluster/scratch/<account>/codex_terra_edge_runs`;
- persistent environments/archives: `/cluster/project/rsl/<account>` when
  writable.

Dataset and runtime ownership are independent. For example, jobs running as
`alesweber` may read the dataset in
`/cluster/project/rsl/alesweber/TerraProject/terra/data/terra/train` and a
group-readable pinned venv owned by another RSL account. Record both paths in
the run contract and validate the runtime inside every GPU allocation.

Scratch files not accessed for roughly 15 days are purged. Archive final
checkpoints to persistent project/work storage; do not keep the only copy on
scratch.

## Local SSH aliases

Use distinct aliases and distinct multiplexing sockets for each account:

```sshconfig
Host euler-alesweber
  HostName euler.ethz.ch
  User alesweber
  IdentityFile ~/.ssh/id_ed25519_euler_alesweber
  IdentitiesOnly yes
  ControlPath ~/.ssh/cm-%C

Host euler-lterenzi
  HostName euler.ethz.ch
  User lterenzi
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
  ControlPath ~/.ssh/cm-%C
```

Verify the identity before staging anything:

```bash
ssh -o BatchMode=yes euler-alesweber 'id -un; printf "%s\n" "$HOME"; lquota'
```

## Campaign launchers

Campaign launchers default to a non-mutating local contract check. The active
V8/V6 campaign supports a staging-only step and an explicit submission step:

```bash
# Local validation only: no SSH, W&B, or Slurm mutation.
SUBMIT=0 scripts/euler_v8_v6_yolo_rv2/submit.sh smoke

# Upload immutable code/input snapshots and inspect Slurm eligibility, but submit no job.
TERRA_EULER_USER=alesweber REMOTE_HOST=euler-alesweber \
  SUBMIT=stage scripts/euler_v8_v6_yolo_rv2/submit.sh smoke

# Only after staging, online-auth checks, and explicit run authorization.
TERRA_EULER_USER=alesweber REMOTE_HOST=euler-alesweber \
  SUBMIT=1 scripts/euler_v8_v6_yolo_rv2/submit.sh smoke
```

Useful overrides are:

- `TERRA_REMOTE_WORK_ROOT` for reproducible code snapshots;
- `TERRA_REMOTE_RUN_ROOT` for inputs and run artifacts;
- `TERRA_REMOTE_VENV` for the pinned Python runtime;
- `TERRA_EULER_{HOME,SCRATCH,PROJECT}_ROOT` for unusual storage layouts.

To fall back to the legacy account, set both the account and SSH alias:

```bash
TERRA_EULER_USER=lterenzi REMOTE_HOST=euler-lterenzi ...
```

`SUBMIT=stage` queries the account association, partition state, and matching
GPU inventory with `sacctmgr`, `scontrol`, and `sinfo`; it does not create a job
or a per-run directory.

Do not reuse a smoke receipt across Unix accounts. Run an account-local smoke,
verify the requested/allocated GPU type plus the JAX convolution and NCCL
preflight, and require a completed finite first update before production.

## Monitoring

```bash
ssh euler-alesweber 'squeue -u "$USER" -o "%.18i %.9T %.12M %.30j %.20N %.24R"'
ssh euler-alesweber 'sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES -P'
```

`RUNNING` is not a health result. Inspect the run log, GPU allocation, W&B
history, and the finite update receipt.
