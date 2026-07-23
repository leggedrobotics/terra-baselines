# Experiments — currently running (updated 2026-07-23 ~16:55 CEST)

Branch `agent/spatial-v3-improvements`. Detailed incident/launch trail:
`docs/EXPERIMENTS_SPATIAL_V3_RUNS.md`. Research overview:
`docs/research/terra_excavation_policy/`. All jobs: Euler 4×RTX 4090, 20k updates,
solo excavator, kickstart flags per run. Primary metric
`eval/success_within_horizon_rate`; legacy comparator `eval/positive_terminations`.

| Exp | Slurm | W&B | Snapshot | What it tests | State |
|---|---|---|---|---|---|
| E6 short300 | 8108912 | 3y60iiwn | spatial-v3-c57a4d9 | E3 continued under 300-step horizon (efficiency; horizon barely binds — effectively pushes E3's ceiling) | RUNNING @~17.8k/20k; eval summary = 3.141 / 0.146, swhr 0.999, ep_len 54.2; ~27.9k steps/s |
| E4′ xattn-ks | 8123147 | hafipl4q | spatial-v3-c57a4d9 | cross-attention readout at the ceiling (grown E3→v4-medium + kickstart) | RUNNING @~16.2k/20k; eval summary = 3.161 / 0.147, swhr 0.999, ep_len 53.9; ~27.4k steps/s |
| E7 v5-ks | 8128309 | n3t8cy9a | spatial-v5-bbacfde | + identity-init token self-attention (E7 vs E4′ isolates the mixing increment) | RUNNING @~15.0k/20k; eval summary = 3.113 / 0.145, swhr 0.999, ep_len 54.5; in checkpoint/eval block, normal ~25.8k steps/s |
| E8 multitask | 8128315 | smn1lc4b | spatial-v5-bbacfde | one policy for foundations + trenches/double + double_diagonal (E3 warm-start, teacher kl 0.5/500) | RUNNING @~13.6k/20k; eval summary = 4.458 / 0.175, swhr 0.997, ep_len 39.5; recent log steps are slow/noisy, no error markers; not directly comparable to single-task foundations |
| E9 128-res | 8131923 | — | res128-585a29a | 128×128 foundations with 5-stage SE student + 64 teacher obs downsample x2 | FAILED in smoke: teacher model used 128-row position embeddings for a 64-row teacher checkpoint; fixed locally with regression + CPU smoke before any relaunch |
| E9b 128-res relaunch | 8183718 | — | f16-e9e10-20260722 | E9 rerun on the fixed teacher-env code snapshot; same 5-stage SE 128 student and local-smoked flags | FAILED in smoke after runtime preflight: 512 env/GPU OOM in 128×128 PPO update (`RESOURCE_EXHAUSTED`, 10.70 GiB temp + 11.49 GB allocation attempt); next gate should try `num_minibatches=64` before reducing envs |
| E9c 128-res mb64 | 8261393 | 0ixsswn4 | f16-e9e10-20260722 | narrow E9b memory-fit relaunch: same 512 env/GPU, 32 steps, fixed 128 snapshot/checkpoints, but PPO `num_minibatches=64` | CANCELLED after RCA: first-update smoke checkpoint already had all model params NaN; production stayed forward-only (`FORWARD=1.0`, `DO=0.0`) through ~1.4k updates |
| E9d 128-res mb64 fixed | 8323457 | — | f16-e9e10-20260722 + synced fixes | E9c relaunch after local full-shape gate: embedding index clamp, local-map `IntMap`, local-map area scale 4, loaded/downsample fix, finite smoke guard | PENDING `(Priority)`; accepted 2026-07-23 ~16:55 CEST; in-job dataset audit + W&B-disabled finite smoke must pass before production |
| E10 v5 f32-attn eps-mixer | 8183719 | lc0s6wke | f16-e9e10-20260722 | E3-grown v5 student with bf16 trunk, f32 attention math, and token mixer residual init scale 0.001 baked into the grown checkpoint | RUNNING; smoke passed; @~3.8k/20000, latest eval summary 3.061 / 0.143, swhr 0.998, ep_len 55.3; current normal train ~24.7k steps/s |

Baselines to beat: E3 final 3.054 / 0.142 (ep_len 55.2); E1 final 2.833 / 0.132;
teacher pqtmfmqy 2.810 / 0.131.

Notes:
- Duplicate E7 8128390 (parallel session, snapshot spatial-v5-a271c9d) cancelled
  2026-07-22 ~11:10 — 8128309 kept (submitted first; snapshot bbacfde = a271c9d + E8
  preset, identical v5 code).
- **Naming: E8 = multitask trench run (8128315). E9 = original 128×128-resolution pilot
  (8131923, failed before production). E9b = fixed relaunch that OOMed. E9c = mb64 memory-fit
  relaunch that exposed all-NaN first update. E9d = fixed mb64 relaunch. E10 = v5
  attention follow-up with f32 attention + epsilon mixer.**
- E10 briefly sat in the update-1000 eval/checkpoint block but resumed. Current fps control is
  E7 because both are 4×4090, 1024 env/GPU v5 kickstart runs; E10 adds f32 attention math and
  epsilon mixer.
- E9c proved that the `num_minibatches=64` memory shape fits, but the existing smoke gate
  was too weak: it completed update 0 while writing an all-NaN model/loss checkpoint. Any
  future smoke must assert finite `total_loss`, `entropy`, `kickstart/*`, model params, and
  optimizer state before production starts.
- E9d's local gate passed the exact 1-GPU per-device shape used by the Slurm job
  (512 env/GPU, 32 steps, mb64, bf16, grown 5-stage 128 student, teacher downsample x2):
  model/optimizer finite fractions 1.0; rollout/logits/teacher logits/advantages finite
  fractions 1.0; finite grad norm 40.895.
- E4-1g pilot 7887940 (parallel session's single-GPU parity probe) may still be running —
  not tracked here.
