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

## Addendum 2026-07-20 (evening)

- **Speed probe** `terra-sv3-speed1g` (Slurm 7882583, COMPLETED): E2 config on 1x4090,
  steady 11.3-11.5k steps/s vs E2's 10.8k/GPU → ~94% multi-GPU scaling efficiency.
  Also first end-to-end bf16 GPU training evidence (200 updates, clean).
- **E3 attempt 1 (7862190) FAILED** at the smoke's final checkpoint save:
  `register_checkpoint_config_classes()` overwrote `__main__.MixedAgentTrainConfig` in the
  sed-derived trainer → pickle identity error. Kickstart smoke itself trained fine.
  Fixed in `d1765d7` (register only missing names + regression test), helpers.py synced to
  the snapshot, **resubmitted as 7886071**.
- **E4 single-GPU learning pilot** `terra-sv3-E4-1g` (Slurm 7887940): exact E2 config on
  1x4090, same total data budget as the original A/B (1.31B env steps = 40k updates),
  ent schedule rescaled to 38000 updates so entropy-vs-env-steps matches pqtmfmqy.
  Purpose: if learning-per-env-step matches E2, single-GPU runs become the default
  screening tool (4 experiments per node instead of 1; ~94% aggregate throughput).

## E4 — cross-attention readout (submitted 2026-07-20 ~17:05Z)

- Encoder v4 `resnet_spatial_8x8_se_xattn` (F13, commit 0cf7f4a): agent-conditioned
  cross-attention readout (64 tokens @8x8, agent query + 4 latents, 4 heads) fused with the
  unchanged flatten path. E4 vs E2 isolates the readout; all other flags identical to E2
  (bf16, critic 512,256, no value clip, flat shuffle, ent 0.15->0.005 over 19k, 20k updates).
- Local GPU probe (4090, batch 256): fwd+bwd 19.1 ms f32 / 9.9 ms bf16 (1.93x), grads finite,
  attention branch cost within noise of v3. Params: v4 base 1,088,733 (+86k over v3).
- Remote CPU gate on snapshot `snapshots/spatial-v4-0cf7f4a`: test_models 31/31 OK.
- Slurm **7889031**, sbatch `terra_sv4_E4_xattn_bf16_4gpu_20260720.sbatch` (derived from E2's;
  models.py sha pin recomputed; all four pins verified).

## Coordination note (2026-07-20 ~19:05Z, main session)

- E3 resubmit **7886071** reached RUNNING on 4x RTX 4090 (teacher loaded, kickstart coefs
  active, smoke compiling); pending duplicate **7886677** cancelled — resubmit
  `terra_sv3_E3_kickstart_med_4gpu_20260720b.sbatch` if 7886071 fails.
- E4 xattn config verified locally end-to-end before its cluster slot: 62/62 unit tests at
  948c788 + 3-update CPU training smoke with the full E4 flag set (bf16, critic 512,256,
  no value clip, flat shuffle) including per-update + FINAL checkpoint saves and a
  foreign-entrypoint reload of the saved pkl.

## E5 + E6 — harder maps & step efficiency (submitted 2026-07-22 ~00:20 CEST)

Both kickstart from the E3 FINAL checkpoint (init AND teacher =
`snapshots/spatial-v3-3a21cd6/.../terra-sv3-E3-kickstart-med-euler-2026-07-20-20-45-27_FINAL.pkl`),
submitted with `--dependency=afterok:7886071` so they auto-start when E3 completes.
Snapshot `spatial-v3-c57a4d9` (adds solo_excavator_short300 preset). KL anneal 1000,
ent 0.02→0.005/10k, medium se + bf16 + critic 512,256 + algo fixes, 20k updates, 4×4090.

- **E5** Slurm **8063942** — `--config solo_excavator_rectangles_dumpzone`
  (foundations_rectangles_real_dumpzone, 600-step horizon): cross-task transfer to
  dump-zone maps with the simple-maps teacher KL-anchoring the start.
- **E6** Slurm **8063944** — `--config solo_excavator_short300` (same foundations maps as
  E1-E4, horizon 450→300): direct pressure on time-to-completion; primary added metric =
  successful-episode length (deployability).
- Fallback if E3 ends non-zero (DependencyNeverSatisfied): repoint STUDENT_INIT/TEACHER to
  the periodic `...-20-45-27.pkl` and resubmit.

## Finals + E5/E6 resubmission (2026-07-22 morning)

- **E1 FINAL (3buorfp3): 2.833 / 0.132, swhr 0.997, ep_len 59.2** — beats teacher 2.810/0.131.
- **E3 FINAL (j0bs2fkl): 3.054 / 0.142, swhr 0.997, ep_len 55.2** — best policy to date.
- E4 @~17k: 2.496 / 0.117 (recovered; already above E2's from-scratch final).
- E5/E6 first attempt (8063942/8063944) FAILED at the gate: NFS attribute-cache race —
  dependency released them seconds after E3 wrote its FINAL through the scratch symlink and
  `test -f` on a fresh node missed it. Fixed with a 5×20s retry guard + corrected job names;
  resubmitted as **E5 8105958, E6 8105959** (no dependency needed; checkpoint verified).

## E5/E6 correction (2026-07-22 ~07:40 CEST)

The NFS-race diagnosis for the first E5/E6 failure was WRONG. Real root cause (xtrace probe
8107155): the sbatch-derivation sed `s|terra-sv3-E3-kickstart-med|<new name>|g` also rewrote
the E3 checkpoint FILENAME inside TEACHER_CKPT/STUDENT_INIT, so the gate tested a
nonexistent `terra-sv3-E5-dumpzone-ks-euler-..._FINAL.pkl`. Both attempts failed on that.
Fixed the paths (verified resolving to the real E3 final), resubmitted:
**E5 = 8107668, E6 = 8107669** (held 8105959 cancelled — Slurm snapshots scripts at submit
time, so releasing it would have run the broken copy). Lesson for derived sbatches: verify
every substituted path RESOLVES (test -f on the expanded value), not just that pins match.

## E5/E6 attempt 4 — IN PRODUCTION (2026-07-22 ~08:30 CEST)

Third failure root cause: raw E3 FINAL carries exact-resume metadata (next_update=20000,
optimizer_state) → --resume_from rejected continuing a completed run. Fix: stripped
warm-start copy `..._FINAL_warmstart.pkl` (model/train_config/env_config only) as
STUDENT_INIT; teacher stays the raw final. Smokes passed on attempt 4:
- **E5 = 8108911**, W&B **gud7cbwg** (dumpzone transfer)
- **E6 = 8108912**, W&B **3y60iiwn** (300-step horizon)
Cumulative E5/E6 lessons: (1) verify every substituted sbatch path RESOLVES;
(2) raw finished-run checkpoints need metadata stripping before warm-start reuse —
consider a train_mixed flag `--warm_start_from` that does this implicitly.

## E4 final + early E5/E6 (2026-07-22 ~09:30 CEST)

- **E4 FINAL (k8vnwp5u): 2.586 / 0.121, swhr 0.993, ep_len 63.0.** From-scratch ladder:
  E2 se 2.33 < E4 xattn 2.59 < teacher 2.81 < E1 2.83 < E3 3.05. The attention readout
  beats SE-only by +11% at matched from-scratch conditions — genuine architecture win;
  still trails warm-started runs → E4' (xattn kickstarted from E3) is the definitive test.
- **E6 @~1.5k: 3.07/0.143, ep_len 55.2** — warm start held perfectly under the 300-step
  horizon; already at E3-final level. CAVEAT: episodes are ~55 steps, so a 300-step horizon
  barely binds — E6 is effectively "E3 continued + slack horizon". For real time-pressure,
  consider E6b at horizon ~120 or an explicit early-finish bonus.
- **E5 @early: 0.00 pos_term, reward −0.006** — expected: dumpzone relocation is a
  different completion criterion; teacher KL anchors the motor skills while the task is
  relearned. Judge at 3-5k updates; if still ~0, consider the 2-stage curriculum preset
  (ring → dumpzone) instead of cold dumpzone.

## E5 stopped early (2026-07-22, user decision — save compute)

Cancelled 8108911 at 850/20000 updates: zero completions AND near-zero dense reward
throughout (existence-penalty level) — the transferred ring-digging policy never found the
reward stream on dumpzone maps, and the signal was flat well past the KL release (1000-update
anneal nearly done, no inflection). Verdict recorded as: **naive cross-task kickstart
(teacher + init only) does not bootstrap a new task family**; the E5b design (2-stage
curriculum ring→dumpzone, higher entropy start ~0.10, dump-bonus boost) is the canonical
next attempt when dumpzone becomes a priority again. GPUs freed for E4'/E7.

## E7 + E8 launched (2026-07-22 afternoon)

- **E7 = 8128309** (`terra-sv5-E7-saxattn-ks`): v5 encoder (identity-init token self-attn ×2
  + cross-attn readout, commit a271c9d — identity at init exactly 0.0), grown from E3 final
  (2.44M → 2.77M params), kickstarted. E7 vs E4' isolates the self-attention increment.
- **E8 = 8128315** (`terra-sv3-E8-multitask-ks`): multi-task foundations + trenches/double +
  trenches/double_diagonal (all 64×64, same value semantics, depth-1 targets — verified),
  E3 warm-start (same arch), light teacher (kl 0.5, anneal 500). Tests whether one policy
  digs both families; unlike E5, the dense dig-reward stream exists on trenches from step 1.
- Snapshot `spatial-v5-bbacfde` (remote gate: test_models 37/37 OK). All checkpoint paths
  and sha pins verified resolving before submission.
Fleet: E6 (running), E4' 8123147 (pending), E7, E8 (pending). E5 cancelled earlier.

## v5 landed + E7 submitted + dataset ready (2026-07-22 ~10:45 CEST)

- v5 `resnet_spatial_8x8_se_sa_xattn` committed (a271c9d): identity-init token mixer
  (2 pre-norm blocks over 64 tokens), exact identity at init (0.0 diff vs v4), 68 tests
  green. Flax MHA lacks out-kernel init → explicit zeros Dense projection (documented).
- **E7 = Slurm 8128390**: E3 grown → v5 medium (2.767M incl. wide critic; grow validated),
  kickstart from E3, E4p flags otherwise. Snapshot spatial-v5-a271c9d. E7 vs E4p isolates
  the self-attention increment at the ceiling.
- foundations_128 dataset v2 verified + uploaded (correct `foundations` source family —
  v1 had scaled the wrong family; caught via E6's W&B curriculum "Level 0: foundations").
  Structural proof: 2x2-block-downsample of every 128 map equals its 64 source exactly.
- F15 code agent implementing (env equivalence overrides, stage growth, teacher obs
  downsample). E8 gate: F15 review + memory smoke.

## E9 — 128×128 resolution pilot SUBMITTED (2026-07-22 ~12:30 CEST)

- **Slurm 8131923**, snapshot `res128-585a29a`, sbatch `terra_sv3_E9_res128_ks_4gpu_20260722.sbatch`.
- Student: E3 grown to 5-stage medium se @128 (`grown_medium_se_5stage_128.pkl`, 2,793,655
  params; first grow attempt at 1.9M was missing `--maps_edge_length 128` — wrong flatten
  grid; caught by param-count check, regrown correctly: 23 added-stage leaves, embeddings
  sliced 64→128 rows, flatten/readout/heads copied exactly).
- Teacher: E3 raw final with `--teacher_obs_downsample 2` (F15). Known pilot caveat: the
  transform does not rescale the `loaded` feature (teacher sees ~4× its trained load range
  while student carries); fast-follow = integer-divide loaded by 4 if kickstart looks weak
  during hauling.
- Equivalence knobs: `--config solo_excavator_128` (foundations_128, max_steps 450),
  `--agent_move_tiles 10 --dig_radius_tiles 10`, `--reward_normalizer 280`, footprint
  auto-scales (meters/tile_size: 5×9 → 11×19). 512 envs/device (memory), 10k updates
  (655.4M steps). In-job full-shape smoke doubles as the memory gate.
- Equivalence checks to read off the run: episode step counts ≈ 64² runs (~55 for
  successes); reward magnitudes comparable after normalizer 280.

## E9 failure RCA + local gate fix (2026-07-22 ~15:20 CEST)

- Slurm **8131923 FAILED** during the W&B-disabled first-update smoke before production
  started. Failure signature:
  `flax.errors.ScopeParamShapeError: Initializer expected to generate shape (64, 8) but got shape (128, 8) instead for parameter "embedding" in "/agent_state_net/embedding_1"`.
- Root cause: `--teacher_obs_downsample 2` downsampled the teacher inputs, but
  `train_mixed.py` still instantiated the teacher module with the student 128×128 env object.
  `get_model_ready` derives `AgentStateNet` position-embedding width from
  `env.batch_cfg.maps_dims.maps_edge_length`, so the frozen 64-world teacher checkpoint was
  wrapped in a 128-position teacher module.
- Fix on local branch: build the frozen teacher model using a minimal env-like object with
  the teacher checkpoint's native `maps_edge_length`, while the student rollout/env remains
  128×128. Added regression
  `TeacherObsDownsampleTest.test_checkpoint_teacher_model_uses_native_embedding_width`.
- Verified locally before any relaunch:
  - `py_compile` for `train_mixed.py`, `train.py`, `utils/models.py`, `scripts/grow_checkpoint.py`,
    and the touched tests.
  - Focused CPU tests: `tests/test_training_utils.py TeacherObsDownsampleTest
    ResolutionScalingConfigTest` (10 tests OK) and `tests/test_models.py
    SpatialV4XAttnEncoderTest SpatialV5SelfAttnEncoderTest ResolutionScaling128Test`
    (16 tests OK).
  - Real-checkpoint local CPU smoke with the E9 flags reduced to `1` CPU, `2` envs,
    `2` steps, and one update: loaded real `grown_medium_se_5stage_128.pkl` plus the real
    E3 FINAL teacher, printed student `num_embeddings_agent=128`, teacher
    `num_embeddings_agent=64`, completed the PPO update, and saved a FINAL checkpoint.

## F16 implemented locally — attention hardening + probes (2026-07-22)

No Slurm job launched from this change yet. The code now exposes two gated architecture
knobs for the next attention ablation batch:

- `--attention_compute_dtype {encoder,float32,bfloat16}`: default `encoder` preserves the
  current E4'/E7 semantics. `float32` keeps v4/v5 cross-attention and v5 token-mixer
  attention math in f32 while the spatial-ResNet conv trunk can remain bf16. Non-default
  values fail fast on non-attention encoders.
- `--token_mixer_residual_init_scale FLOAT`: default `0.0` preserves exact v5
  identity-at-init and param counts. Small positive values (`1e-3`/`1e-2` candidates) make
  the token-mixer residual projections nonzero so gradients reach the inner MHA/MLP params
  immediately. Nonzero values fail fast unless the encoder is v5.

Added `scripts/analysis/ablate_attention_checkpoint.py` for post-training probes:
`--mode xattn` zeros the direct cross-attention branch, `--mode token_mixer` zeros v5 mixer
blocks into identity behavior, and `--mode all_attention` does both. Always run `--dry_run`
first to audit matched Flax paths before saving an ablated checkpoint.

Local verification:

- `py_compile`: `train_mixed.py`, `train.py`, `utils/models.py`, `scripts/grow_checkpoint.py`,
  `scripts/analysis/ablate_attention_checkpoint.py`, and touched tests.
- `tests/test_models.py`: 42 tests OK (default v3/v4/v5 param counts unchanged).
- `tests/test_training_utils.py`: 33 tests OK (including E9 teacher-env regression and F16
  config no-op guards).
- Ablation helper dry-run on a temporary v5 checkpoint: `--mode xattn` matched 14 leaves;
  `--mode token_mixer` matched 36 leaves; no checkpoint written.
- Quiet local first-update smoke with W&B disabled and real trainer loop:
  `--map_encoder resnet_spatial_v5 --encoder_compute_dtype bfloat16
  --attention_compute_dtype float32 --token_mixer_residual_init_scale 0.001
  --no_value_clip --flat_minibatch_shuffle`, `1` CPU device, `2` envs, `2` steps, `1`
  update. It completed the PPO update and saved
  `checkpoints/f16-attn-local-smoke-local-2026-07-22-17-01-24_FINAL.pkl`.

Launch discipline from E9 is now explicit in the F16 spec: before any new attention Slurm
job, run a quiet local W&B-off first-update smoke with the exact checkpoint/teacher/model
flags and confirm update 1 plus checkpoint save/reload. Model init alone is not a sufficient
gate.

## E9b + E10 submitted together (2026-07-22 ~17:40 CEST)

- Fresh Euler snapshot: `f16-e9e10-20260722` from local branch
  `agent/spatial-v3-improvements` at base commit `88ccaba` plus dirty F16/E9 fixes. Remote
  integrity checks passed before submission:
  - `terra/terra/env.py`
    `628b897fcbd372e86e7ee5c2c3cb41f7b9c9300539c388452085bf66acea9685`
  - `terra/terra/state.py`
    `ef53459f7177f994b1dd54375a7a9be479e3391d334bc278e6409c9395f2fafa`
  - `terra-baselines/train_mixed.py`
    `2efc18f88862738984852f2d55aaeff1d37f95a3f6e704692d432636560e420b`
  - `terra-baselines/utils/models.py`
    `2205cab36694f90d56bf65a5511449eaccf1d75d2749a23507ec1f0b62c39f18`
  - `terra-baselines/scripts/grow_checkpoint.py`
    `32e8a3e4fadcafd5286e8adcd69ab7ac4d47995df2ba7e2f6c6dfb828ebd7847`
- **E9b 128-res relaunch**: Slurm **8183718**, script
  `terra_sv3_E9b_res128_ks_4gpu_20260722.sbatch`. Same experiment as E9 but on the fixed
  teacher-env code snapshot: `solo_excavator_128`, 5-stage medium SE student,
  `--teacher_obs_downsample 2`, 512 envs/GPU, 10k updates. Student init
  `grown_medium_se_5stage_128.pkl`
  (`56900f03e44ce14aabb57e57a71d72856cf4b909533aad70991188fea5e7bd75`). It preserves
  the original pilot's loaded-feature caveat for comparability.
- **E10 v5 f32-attn eps-mixer**: Slurm **8183719**, script
  `terra_sv5_E10_f32attn_epsmix_ks_4gpu_20260722.sbatch`. This is the combined attention
  follow-up: E3-grown v5 student, bf16 conv trunk, `--attention_compute_dtype float32`,
  `--token_mixer_residual_init_scale 0.001`, 1024 envs/GPU, 20k updates. Student init
  `grown_medium_v5_f32attn_eps001_from_e3.pkl`
  (`aecaf469b4a04d2fd49340d3905fc8a3357a78e375f4dda2c36f665ce50882e7`) was grown after
  patching `grow_checkpoint.py` so the mixer epsilon is baked into the checkpoint rather
  than overwritten by `--resume_from`.
- Both jobs were submitted simultaneously and were **PENDING (Priority)** immediately after
  `sbatch`. Do not mark either healthy until the in-job W&B-disabled first-update smoke logs
  `SMOKE_GATE_PASSED`; production only starts after that gate.

## Live refresh (2026-07-23 ~10:10 CEST)

- **E9b failed during the first-update smoke**, so no W&B production run was created. It passed
  the 4×4090 allocation guard, CUDA/cuDNN/NCCL runtime preflight, and ent-schedule patch, then
  aborted inside the 128×128 PPO update:
  `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 11489681704 bytes`.
  XLA reported `preallocated temp allocation: 10.70GiB`; peak buffers included
  `bf16[512,128,128,24]` conv activations and rollout map gathers shaped
  `s8[1,16384,128,128]`. This validates the teacher-env fix but invalidates the
  512-env/GPU memory assumption. Next E9 gate should keep the fixed snapshot and try
  `--num_minibatches 64` first (minibatch 512→256) before reducing envs/GPU or rollout length.
- **E9c memory-fit relaunch submitted** as Slurm **8261393** (2026-07-23 ~10:21 CEST), script
  `terra_sv3_E9c_res128_mb64_ks_4gpu_20260723.sbatch`. This is the narrow follow-up to the
  E9b OOM. It keeps snapshot `f16-e9e10-20260722`, the same fixed 128×128 teacher-env code,
  the same 5-stage medium SE student and E3 teacher checkpoints, 4×RTX 4090 guard,
  CUDA/cuDNN/NCCL runtime preflight, 512 envs/GPU, 32 steps, smoke
  `--total_timesteps 65536`, and production `--total_timesteps 655360000`; the only memory-fit
  shape change is `--num_minibatches 64` in both smoke and production.
- **E9c passed the in-job W&B-disabled smoke** at 10:31 CEST and started production W&B
  `0ixsswn4`. The smoke's slow cuDNN autotune line explicitly showed the intended reduced
  minibatch activation shape `bf16[256,128,128,24]` (E9b OOMed on 512). Production update 1
  completed, so the Slurm job is healthy by the launch gate. At ~11:17 CEST it was @153/10000
  with first eval still zero-success (`pos_term 0`, `reward -0.007`, `swhr 0.000`) and ~6.3k
  global steps/s. Memory fit is solved; learning/equivalence is not proven yet.
- **E10 passed its smoke and is training** as W&B `lc0s6wke`. It resumed after the
  update-1000 eval/checkpoint block. At log progress `1691/20000`, the latest W&B summary was
  `pos_term 3.088`, `reward 0.144`, `swhr 1.000`, `ep_len 55.0`. Normal train throughput is
  ~24.8k steps/s; the closest control E7 was ~25.5k steps/s on this sample, so the current
  f32-attention/epsilon-mixer cost is small. Against E4′ xattn (~27.3k), E10 is ~9% slower,
  but E4′ does not include the v5 token mixer.
- Other active runs are healthy:
  - E6 `3y60iiwn`: @15486, `3.126 / 0.146`, `swhr 0.999`, `ep_len 54.3`, ~27.9k steps/s.
  - E4′ `hafipl4q`: @14000, `3.121 / 0.145`, `swhr 0.998`, `ep_len 54.4`, ~27.3k steps/s.
  - E7 `n3t8cy9a`: @13000, `3.117 / 0.145`, `swhr 0.998`, `ep_len 54.5`, ~25.5k steps/s.
  - E8 multitask `smn1lc4b`: @12125, `4.387 / 0.172`, `swhr 0.997`, `ep_len 39.8`,
    ~26.8k steps/s; this is not directly comparable to single-task foundations because the eval
    task mix changed.

## Live refresh (2026-07-23 ~14:18 CEST)

All six tracked jobs are still Slurm RUNNING, with no active error markers in the latest log
tails. W&B summaries and logs now read:

- **E6 `3y60iiwn`**: @~17.2k/20k, `3.136 / 0.146`, `swhr 1.000`, `ep_len 54.3`,
  ~27.8k steps/s. Still healthy; basically E3 continued above the old E3 final.
- **E4′ `hafipl4q`**: @~15.9k/20k, `3.154 / 0.147`, `swhr 1.000`, `ep_len 54.0`,
  ~27.3k steps/s. Currently the best single-task foundations run on the board.
- **E7 `n3t8cy9a`**: @~14.7k/20k, `3.114 / 0.145`, `swhr 0.999`, `ep_len 54.5`,
  ~25.8k steps/s. Healthy, but not beating E4′ so far; the identity-init mixer is not yet
  paying for itself.
- **E8 multitask `smn1lc4b`**: @~13.4k/20k, `4.431 / 0.174`, `swhr 0.997`,
  `ep_len 39.7`. Good multitask signal, but not apples-to-apples with the single-task
  foundations runs because the eval task mix changed.
- **E10 `lc0s6wke`**: @~3.3k/20k, `3.059 / 0.143`, `swhr 0.997`, `ep_len 55.0`,
  ~24.8k steps/s. Healthy, but early and currently below E4′/E6/E7. The f32-attention plus
  epsilon-mixer cost remains small versus E7 (~4-5% slower), but there is no performance win
  visible yet.
- **E9c `0ixsswn4`**: @~1.1k/10k, still `0.000 / -0.007`, `swhr 0.000`, `FORWARD=1.0`,
  `DO=0.0`, ~6.6k steps/s. Runtime/memory fit is solved, but behaviorally this is not
  learning yet. The likely next diagnosis is cross-resolution kickstart semantics rather
  than PPO memory: teacher loaded-feature scaling, action/logit spatial transfer, and why
  the initialized policy collapses to forward-only.

## Live refresh (2026-07-23 ~14:36 CEST)

All six tracked jobs are still RUNNING. Latest Slurm/W&B read:

- **E6 `3y60iiwn`**: @~17.5k/20k, `3.159 / 0.147`, `swhr 0.999`, `ep_len 53.9`,
  ~28.0k steps/s. Now slightly ahead of E4′ on the scalar summary, but both are within
  late-stage eval noise.
- **E4′ `hafipl4q`**: @~16.0k/20k, `3.149 / 0.146`, `swhr 0.999`, `ep_len 54.1`,
  ~27.4k steps/s. Still a strong single-task ceiling run; checkpoint/eval at update 16000
  just ran.
- **E7 `n3t8cy9a`**: @~14.9k/20k, `3.119 / 0.145`, `swhr 0.999`, `ep_len 54.5`,
  ~25.9k steps/s. Healthy but remains behind E4′/E6.
- **E8 multitask `smn1lc4b`**: @~13.5k/20k, `4.400 / 0.173`, `swhr 0.997`,
  `ep_len 40.0`. Still healthy; fps sample is noisy around eval/checkpoint, with normal
  steps returning to mid/high-20k.
- **E10 `lc0s6wke`**: @~3.5k/20k, `3.089 / 0.144`, `swhr 0.999`, `ep_len 55.0`,
  ~24.8k steps/s. Healthy and back around the E3-final scalar, but no sign yet that f32
  attention plus epsilon mixer beats E4′/E7.
- **E9c `0ixsswn4`**: @~1.2k/10k, still `0.000 / -0.007`, `swhr 0.000`, `FORWARD=1.0`,
  `DO=0.0`, ~6.6k steps/s. This is now a stronger negative signal: the 128×128 memory fix
  works, but the policy remains behaviorally collapsed to forward-only. Next controlled work
  should debug cross-resolution transfer before spending more GPU on this arm.

## Live refresh (2026-07-23 ~15:03 CEST)

All six tracked jobs are still Slurm RUNNING. No active error/traceback/OOM markers in the
last 300 log lines.

- **E6 `3y60iiwn`**: @~17.8k/20k, `3.141 / 0.146`, `swhr 0.999`, `ep_len 54.2`,
  ~27.9k steps/s. Healthy; near the E4′/E6 ceiling band.
- **E4′ `hafipl4q`**: @~16.2k/20k, `3.161 / 0.147`, `swhr 0.999`, `ep_len 53.9`,
  ~27.4k steps/s. Currently the strongest single-task scalar summary.
- **E7 `n3t8cy9a`**: @~15.0k/20k, `3.113 / 0.145`, `swhr 0.999`, `ep_len 54.5`.
  It is in the update-15000 checkpoint/eval block; normal steps remain ~25.8k steps/s.
- **E8 multitask `smn1lc4b`**: @~13.6k/20k, `4.458 / 0.175`, `swhr 0.997`,
  `ep_len 39.5`. Multitask metrics are strong, but recent log steps are intermittently very
  slow despite no error markers. Watch whether this clears after the current eval/checkpoint
  vicinity.
- **E10 `lc0s6wke`**: @~3.8k/20k, `3.061 / 0.143`, `swhr 0.998`, `ep_len 55.3`,
  ~24.7k steps/s. Healthy, but still no clear advantage from f32 attention plus epsilon mixer.
- **E9c `0ixsswn4`**: @~1.3k/10k, still `0.000 / -0.007`, `swhr 0.000`, `FORWARD=1.0`,
  `DO=0.0`, ~6.6k steps/s. The negative read is now robust enough that continuing mainly
  spends GPU to characterize the failure curve; the next useful work is a local/cheap
  cross-resolution transfer debug, not another blind 128×128 PPO run.

## E9c cancellation + NaN RCA (2026-07-23 ~15:15 CEST)

E9c was cancelled manually after the collapse was traced to NaNs rather than ordinary
learning failure. Slurm **8261393** ended `CANCELLED` after `04:53:13`; the other runs were
not touched.

Confirmed evidence:

- The W&B curve was forward-only from the first eval through cancellation:
  `eval/FORWARD %=1.0`, `eval/DO=0.0`, `eval/positive_terminations=0`,
  `eval/success_within_horizon_rate=0` at updates 100, 200, ..., 1300.
- The saved production checkpoint
  `checkpoints/terra-sv3-E9c-res128-mb64-ks-euler-2026-07-23-10-31-51.pkl`
  at `next_update=1401` had **2,793,655 / 2,793,655 model params NaN** and NaNs in
  the optimizer state.
- More importantly, the W&B-disabled smoke checkpoint
  `checkpoints/terra-sv3-E9c-res128-mb64-ks-smoke-euler-2026-07-23-10-22-10_FINAL.pkl`
  at `next_update=1` already had **all model params NaN** and NaN `total_loss`,
  `actor_loss`, `value_loss`, `entropy`, `kickstart/kl`, and `kickstart/value_mse`.
  Therefore the bad behavior began on the first PPO update; production never had a
  finite model.
- The grown 128 init checkpoint `grown_medium_se_5stage_128.pkl` is finite, so the NaNs
  are introduced by the first training update, not by the checkpoint grow itself.
- E9c's W&B train history logged kickstart coefficients and rewards, but the actual loss
  scalars (`total_loss`, `actor_loss`, `value_loss`, `entropy`, `kickstart/kl`,
  `kickstart/value_mse`) were missing/NaN during the active kickstart window. In the healthy
  64×64 kickstart controls (E3/E4′), those scalars are finite from update 0 and the policy is
  already competent by update 100.
- Dataset arrays checked so far are finite; 128 `images`, `occupancy`, `dumpability`, and
  `distance` arrays have no NaNs/Infs. The first 20 128 image arrays downsample exactly to
  their 64 source by both `a[::2, ::2]` and 2×2 block mean. A bool-mask comparison probe
  tripped on NumPy bool subtraction before finishing mask equality, but no data finiteness
  issue was found.

Most likely root causes to debug before any relaunch:

1. **Finite smoke gate missing.** The immediate operational bug is that `SMOKE_GATE_PASSED`
   only meant "update 0 returned"; it did not assert finite loss, params, or optimizer state.
   This allowed an all-NaN checkpoint into production.
2. **First-update numerical fault in the 128/kickstart loss path.** Because the finite grown
   init becomes all-NaN after one PPO update, the next probe should isolate whether the NaN
   enters through student logits/entropy, teacher logits/KL, value MSE, advantages/targets, or
   optimizer gradients.
3. **Teacher obs transform remains semantically incomplete.** `downsample_teacher_obs`
   downsamples global maps and integer-divides agent x/y plus width/height, but leaves local
   1D maps and `agent_states[..., loaded]` unchanged. That can feed the 64-world teacher a
   mixed-resolution local/global observation and wrong load phase. This is a correctness bug
   even if it is not yet proven to be the NaN trigger.
4. **128 policy equivalence still unproven.** Before training, evaluate the finite
   `grown_medium_se_5stage_128.pkl` itself and print student vs teacher action-probability
   histograms on the same reset batch. A function-preserving 128 grow should not start
   forward-only if the E3 behavior transferred.

Do not relaunch E9 blindly. Required next gates:

- Add a cheap local/GPU debug probe for one reset/rollout batch: finite obs, finite student
  logits/value/entropy, finite teacher logits/value/entropy after `teacher_obs_downsample=2`,
  KL/value-MSE values, action argmax/prob histograms, advantage/target ranges, gradient
  finiteness before optimizer apply, and param finiteness after apply.
- Extend the Slurm smoke gate to load the smoke FINAL checkpoint and fail unless model params,
  optimizer state, and all loss scalars are finite.
- Fix or replace teacher obs construction so the teacher input is a coherent 64-world
  observation, including local maps and loaded/capacity scaling, before another 128×128
  production run.

## E9c final RCA + E9d relaunch gate (2026-07-23 ~16:55 CEST)

The E9c NaN/collapse was reproduced locally under the exact per-GPU production shape
(`512` envs, `32` steps, `num_minibatches=64`, 5-stage medium SE 128 student, E3 teacher
with `teacher_obs_downsample=2`). The first diagnostics split the failure into two bugs:

1. `LocalMapWrapper._wrap_with_masks` summed 128-resolution workspace maps and cast the
   sums to `IntLowDim` (`int8`). 128-equivalent workspaces can exceed 127 (a full 12×12
   probe produced 144), which wrapped local maps before the policy saw them. Fixed by
   storing local workspace sums as `IntMap`, and by scaling model-side local maps with
   `--local_map_area_scale 4` to preserve the 64-resolution magnitude.
2. Even after the local-map fix, the rollout had non-finite policy values/log-probs before
   PPO. `flax.linen.Embed` returns NaNs for high out-of-range indices; 128 edge positions
   and the 128→64 teacher transform can expose exact-edge indices such as 64 for a 64-row
   position table. Fixed by clipping `AgentStateNet` discrete embedding inputs
   (positions, angles, agent type) at the model boundary.

Related correctness fixes now in the E9d source:

- `downsample_teacher_obs` also divides loaded dirt by `downsample**2`; E9d uses
  `--loaded_max_override 400` so the student loaded scale matches the 128 tile-count volume.
- PPO finite diagnostics now check rollout obs/value/reward/log-prob, raw and normalized
  advantages/targets, student value/logits/log-prob/ratio/entropy, teacher value/logits,
  gradient norm, model params, and optimizer state. `--fail_on_nonfinite` is enabled in
  both the W&B-disabled smoke and production phases.
- Foundation dataset audit remains clean: 600 finite 128 maps, 388 metadata files,
  128 `images`/`occupancy`/`dumpability` stride-match their 64 foundations source; `distance`
  is the correct 128 recomputation (`realistic_max_distance=48`, source 64 uses 24).

Local gate passed after the fixes:

- Exact 1-GPU full-shape command completed one PPO update and saved
  `checkpoints/e9-local-exact-bf16-embedclip-local-2026-07-23-16-44-45_FINAL.pkl`.
- Checkpoint inspection: model finite fraction `1.0`, optimizer finite fraction `1.0`.
- Loss/diagnostics: `total_loss=3.5877`, `value_loss=0.5256`, `actor_loss=0.1495`,
  `entropy=1.3661`, `kickstart/kl=2.0685`, `kickstart/value_mse=0.6917`,
  rollout/value/log-prob/raw advantages/normalized advantages/student logits/teacher logits
  finite fractions all `1.0`, finite grad norm `40.895`.
- Targeted tests passed: local-map dtype regression, embedding-index clipping regression,
  and `test_training_utils.py` (39 tests).

E9d submitted as Slurm **8323457** using the same snapshot path
`f16-e9e10-20260722`, with the fixed files synced and hash-pinned. At submission it was
`PENDING (Priority)`. The script still performs runtime checks, dataset audit, and a
W&B-disabled finite one-update smoke before starting production W&B.
