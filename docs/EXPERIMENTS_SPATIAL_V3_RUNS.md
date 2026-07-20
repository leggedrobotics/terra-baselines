# Spatial-v3 E1–E3 launch record (2026-07-20)

Batch spec: `docs/IMPROVEMENTS_SPATIAL_V3_2026-07-20.md`. Code: terra-baselines
`3a21cd6` (branch `agent/spatial-v3-improvements`), terra `34b5d39d` (main). All 54 unit
tests green locally (CPU) before sync.

Success metrics: primary `eval/success_within_horizon_rate`; legacy secondary
`eval/positive_terminations` (baselines at 10k updates: pqtmfmqy 2.81 / 0.131 reward;
nnsksyva 2.02 / 0.095). Compare at matched update counts.

## Snapshot

- Euler snapshot: `/cluster/home/lterenzi/codex_terra_edge_validation/snapshots/spatial-v3-3a21cd6/`
  (`terra-baselines/` = worktree at 3a21cd6, `terra/` = main 34b5d39d; `.git`,
  `__pycache__`, `checkpoints`, `wandb` excluded; `terra-baselines/checkpoints/` created).
- Snapshot sha256 pins (asserted in every sbatch before training):
  - `terra/terra/env.py` `628b897f…` and `terra/terra/state.py` `ef53459f…` — identical to
    the pqtmfmqy template pins (same terra main).
  - `terra-baselines/train_mixed.py` `2a0cd6ed…`, `utils/models.py` `d9c0818b…`.

## Remote sanity gate (Euler login node, CPU, 2026-07-20)

- venv used: `/cluster/scratch/lterenzi/terra_main_20260717_venv` (jax 0.4.26, flax 0.8.2,
  optax 0.2.1, wandb 0.26.1, numpy 1.26.4).
  - **Deviation:** the planned venv `/cluster/scratch/lterenzi/codex_terra_edge_venv` is
    broken — `jax` resolves to an empty namespace package (`jax.__file__ is None`, no pip
    dist-info; consistent with a scratch purge). Switched to the venv that ran pqtmfmqy.
- `tests/test_models.py`: Ran 24 tests, OK (exit 0).
- `tests/test_grow_checkpoint.py`: Ran 6 tests, OK (exit 0).

## E3 student init (grow_checkpoint.py, Euler login node CPU)

- Teacher: `/cluster/home/lterenzi/codex_terra_edge_validation/snapshots/main-34b5d39d-095b261/terra-baselines/checkpoints/terra-foundation-spatial8x8-mb32-10k-euler-2026-07-19-07-19-27_FINAL.pkl`
  (pqtmfmqy v2 spatial base, 10k updates).
- Output: `snapshots/spatial-v3-3a21cd6/terra-baselines/checkpoints/grown_medium_se_from_pqtmfmqy.pkl`
  (9.8 MB), overrides: `resnet_spatial_8x8_se`, `model_size=medium`,
  `critic_hidden_dims=512,256`. Exit 0.
- Per-category report: **copied 26, sliced 65, dense-embed 1, zero-init 1, fresh 33**;
  params source 994,825 → target/grown 2,441,223. Stale optimization state stripped
  (optimizer_state, train_state_step, update, next_update) → fresh restart at update 0.

## sbatch scripts

All in `/cluster/home/lterenzi/codex_terra_edge_validation/scripts/`:

- E1: `terra_sv3_E1_v2algofix_4gpu_20260720.sbatch`
- E2: `terra_sv3_E2_se_bf16_4gpu_20260720.sbatch`
- E3: `terra_sv3_E3_kickstart_med_4gpu_20260720.sbatch`

Common shape (structure cloned from the proven
`terra_foundation_spatial8x8_base_4gpu_20260717.sbatch` template): 4× RTX 4090 via
`--gpus=rtx_4090:4` + `--nodelist=eu-g6-[001-080]`, `--time=48:00:00`, module loads
`eth_proxy stack/2024-06 cuda/12.1.1`, full nvidia LD_LIBRARY_PATH from venv
site-packages, sha256 source pins, strict GPU-name hard guard (exit unless all 4 names ==
"NVIDIA GeForce RTX 4090"), `check_jax_runtime.py --min-devices 4` preflight, then a
W&B-disabled full-shape smoke (131072 steps = exactly 1 update, checkpoint_interval 0,
same flags as production) gating the W&B-online production phase.

Production train shape (all): `--config solo_excavator --num_devices 4
--num_envs_per_device 1024 --num_steps 32 --update_epochs 2 --num_minibatches 32
--total_timesteps 2621440000` (= 20000 updates × 131072) `--log_eval_interval 100
--checkpoint_interval 100 --no_value_clip --flat_minibatch_shuffle`.

Per-experiment:

- **E1** run name `terra-sv3-E1-v2algofix`: `--map_encoder resnet_spatial_8x8
  --model_size base`; ent schedule 0.15 → 0.005 over 19000.
- **E2** run name `terra-sv3-E2-se-bf16`: `--map_encoder resnet_spatial_8x8_se
  --encoder_compute_dtype bfloat16 --critic_hidden_dims 512,256 --model_size base`;
  ent schedule 0.15 → 0.005 over 19000.
- **E3** run name `terra-sv3-E3-kickstart-med`: `--map_encoder resnet_spatial_8x8_se
  --model_size medium --encoder_compute_dtype bfloat16 --critic_hidden_dims 512,256
  --resume_from checkpoints/grown_medium_se_from_pqtmfmqy.pkl
  --no-load-env-from-checkpoint --teacher_checkpoint <pqtmfmqy FINAL>
  --kickstart_kl_coef 1.0 --kickstart_kl_anneal_updates 1500 --kickstart_value_coef 0.5
  --kickstart_value_anneal_updates 500`; ent schedule 0.02 → 0.005 over 10000.

### Deviations / mechanisms of note

1. **Ent schedule has no CLI.** `ent_schedule_{start,end,steps}` are
   `MixedAgentTrainConfig` fields only; `parse_known_args` + hard `SystemExit` on unknown
   args means `--ent_schedule_*` flags would crash. Each sbatch derives a per-run trainer
   (`train_mixed_sv3_e{1,2,3}.py`) from the sha-pinned `train_mixed.py` via `sed` on the
   dataclass default lines only, and hard-fails unless a reverse-sed round-trip `cmp`s
   byte-identical to the original (i.e. proves the only diff is the intended default
   lines). E1/E2: `ent_schedule_steps 9500 → 19000`. E3: also
   `ent_schedule_start 0.15 → 0.02`, steps `9500 → 10000`. `train_mixed.py` itself is
   untouched.
2. **Partition.** Template partition `gpu.24h` contains no `eu-g6` nodes (sinfo
   2026-07-20); all RTX 4090 nodes (`eu-g6-[001-080]`) are in `gpuhe.*`. Submitted with
   `--partition=gpuhe.24h` (TIMELIMIT 2-00:00:00, fits 48:00:00); Euler's router placed
   the jobs in `gpuhe.120h`. Nodelist + typed GRES + name guard unchanged.
3. **Venv.** See sanity-gate deviation above; sbatch scripts use
   `terra_main_20260717_venv` (the venv of the template/pqtmfmqy run).
4. **`--load_env_from_checkpoint false` spelling.** The CLI is a store_true/store_false
   pair; the disable spelling is `--no-load-env-from-checkpoint`, used for E3 (E3 uses
   the current `solo_excavator` EnvConfig, not the checkpoint's).
5. **GPU guard tightened** vs template: template accepted 3090|4090; these scripts accept
   RTX 4090 only (exit 43 otherwise).
6. **W&B tags** are auto-generated by `_wandb_tags_for_config` (config, encoder, dtype,
   critic dims, job id, gpus). No manual `spatial-v3-batch-2026-07-20` tag is injected —
   filter the batch via run names `terra-sv3-E*` or `job:<id>` tags.

## Submissions

Initial batch 2026-07-20T09:46:22Z:

| Exp | Slurm job | sbatch | Outcome |
|-----|-----------|--------|---------|
| E1  | 7862187 | `scripts/terra_sv3_E1_v2algofix_4gpu_20260720.sbatch` | FAILED 10:19:28Z — home disk quota (see incident 1); resubmitted as 7866454 |
| E2  | 7862189 | `scripts/terra_sv3_E2_se_bf16_4gpu_20260720.sbatch` | queued (never started before the quota fix, unaffected) |
| E3  | 7862190 | `scripts/terra_sv3_E3_kickstart_med_4gpu_20260720.sbatch` | queued (never started before the quota fix, unaffected) |

E1 resubmission (same script, unchanged): job **7866454** at 2026-07-20T10:29:05Z.

`sbatch --test-only` (job 7862174, discarded) validated the resource combo and estimated
a worst-case start of 2026-07-21T06:17 on eu-g6-006; actual starts depend on backfill.

## Verification evidence

- **E1 attempt 1 (7862187, FAILED on quota — gates themselves all passed):**
  - Started 09:59:38Z on eu-g6-033. sacct AllocTRES:
    `gres/gpu:nvidia_geforce_rtx_4090=4,gres/gpu=4,mem=32G` — 4090-only confirmed; in-log
    `Allocated GPUs (4):` lists 4× `NVIDIA GeForce RTX 4090`.
  - `ENT_PATCH_VERIFIED: ent_schedule_steps 9500 -> 19000 in train_mixed_sv3_e1.py`.
  - Smoke: `Training: 100%|| 1/1 [08:25<00:00, 505.37s/it]` then `SMOKE_GATE_PASSED`
    (~10:10Z).
  - Production: W&B run **jue4bzqm** (`terra-sv3-E1-v2algofix-euler-2026-07-20-12-10-51`,
    https://wandb.ai/aless-weber-eth/mixed-agents/runs/jue4bzqm), progress bar `0/20000`
    (correct 20000-update accounting). Production **update 1 completed** and logged
    (W&B run summary: `performance/actual_env_steps 131072`, entropy 2.0763,
    explained_variance -0.0038), then the first checkpoint write crashed the job
    (incident 1). Treat jue4bzqm as a crashed 1-update run, not a result.
- E1 attempt 2 (7866454): PENDING (Priority) at handoff.
- E2 (7862189): PENDING (Priority) at handoff.
- E3 (7862190): PENDING (Priority) at handoff.

### Status at handoff (2026-07-20T10:38Z)

All three jobs queued PENDING (Priority) on gpuhe.120h; eu-g6 nodes largely occupied by
mnewton jobs. E1 attempt 1 proved the full pipeline end-to-end on the real allocation
(4× RTX 4090, ent patch, smoke update 1, production update 1, W&B online); the only
failure mode hit was the since-fixed home quota. Next monitor pass should, per job:
confirm sacct AllocTRES `nvidia_geforce_rtx_4090=4`, grep the log for
`SMOKE_GATE_PASSED`, capture the W&B run id from `View run at`, and confirm the
production `Training:` bar advanced past 0/20000 (first checkpoint write at update 1 now
lands on scratch — this was the exact point attempt 1 died).

## Incidents

1. **Home disk quota exceeded killed E1 attempt 1 (7862187).** At 10:19:28Z the first
   production checkpoint write failed:
   `OSError: [Errno 122] Disk quota exceeded` in `helpers.save_pkl_object` →
   `checkpoints/terra-sv3-E1-...pkl` (2.6 MB truncated stub, deleted).
   `/cluster/home/lterenzi` was at the 50 GB hard cap (biggest shares: `orbit` 12G,
   `git` 5.8G, workspace old `terra-baselines/checkpoints` 4.0G, workspace `wandb` 1.5G).
   Remediation (all data preserved, moved to scratch + symlinked back, scratch is at
   407/2500 GB):
   - `$TERRA_WORK/wandb` → `/cluster/scratch/lterenzi/codex_terra_edge_runs/wandb`
   - `$TERRA_WORK/terra-baselines/checkpoints` →
     `…/codex_terra_edge_runs/workspace_terra_baselines_checkpoints`
   - snapshot `spatial-v3-3a21cd6/terra-baselines/checkpoints` →
     `…/codex_terra_edge_runs/checkpoints_spatial_v3_3a21cd6` (holds
     `grown_medium_se_from_pqtmfmqy.pkl` + E1 smoke FINAL pkl)
   Home now 44.2 GB (< 45 GB soft quota); write test OK. All three sbatch scripts write
   checkpoints/W&B through the symlinks, so no script changes were needed. E1 resubmitted
   as 7866454; E2/E3 had not started and are unaffected.
   **Follow-up for Lorenzo:** home stays close to soft quota; `orbit` (12G) and `git`
   (5.8G) are the real hogs if more space is needed.

- Known mitigation on cuDNN autotune failure (not needed so far): add
  `export XLA_FLAGS=--xla_gpu_autotune_level=0` to the affected script and resubmit.

## E3 attempt 1 failure + resubmission (2026-07-20 ~16:40Z)

- Slurm 7862190 FAILED at the END of its smoke phase: training itself completed
  (1/1 smoke updates, kickstart medium + bf16 healthy on 4x4090), but the final smoke
  checkpoint save raised `_pickle.PicklingError: Can't pickle <class
  '__main__.MixedAgentTrainConfig'>: it's not the same object`. Root cause: the teacher
  loader's `register_checkpoint_config_classes()` overwrote `__main__`'s config class in
  the sed-derived per-run trainer. Fixed on the branch by `d1765d7`
  (register fills only MISSING names); the 3a21cd6 snapshot predated the fix.
- Remediation: new snapshot `snapshots/spatial-v3-d1765d7` (git archive of 009445d;
  terra hard-linked from the old snapshot; grown checkpoint copied in), derived sbatch
  `terra_sv3_E3_kickstart_med_4gpu_20260720b.sbatch` (only TERRA_SOURCE changed; all four
  sha pins re-verified), resubmitted as Slurm **7886677**.
- E1 (7866454, still pending) and E2 (7862189, running) are unaffected: neither invokes the
  teacher loader, and their own-class pickling is self-consistent. They stay on the
  3a21cd6 snapshot.
