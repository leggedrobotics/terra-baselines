# Experiments — currently running (updated 2026-07-22 ~11:15 CEST)

Branch `agent/spatial-v3-improvements`. Detailed incident/launch trail:
`docs/EXPERIMENTS_SPATIAL_V3_RUNS.md`. Research overview:
`docs/research/terra_excavation_policy/`. All jobs: Euler 4×RTX 4090, 20k updates,
solo excavator, kickstart flags per run. Primary metric
`eval/success_within_horizon_rate`; legacy comparator `eval/positive_terminations`.

| Exp | Slurm | W&B | Snapshot | What it tests | State |
|---|---|---|---|---|---|
| E6 short300 | 8108912 | 3y60iiwn | spatial-v3-c57a4d9 | E3 continued under 300-step horizon (efficiency; horizon barely binds — effectively pushes E3's ceiling) | RUNNING, ~3.08 early |
| E4′ xattn-ks | 8123147 | hafipl4q | spatial-v3-c57a4d9 | cross-attention readout at the ceiling (grown E3→v4-medium + kickstart) | RUNNING (early production) |
| E7 v5-ks | 8128309 | — | spatial-v5-bbacfde | + identity-init token self-attention (E7 vs E4′ isolates the mixing increment) | PENDING |
| E8 multitask | 8128315 | — | spatial-v5-bbacfde | one policy for foundations + trenches/double + double_diagonal (E3 warm-start, teacher kl 0.5/500) | PENDING |

Baselines to beat: E3 final 3.054 / 0.142 (ep_len 55.2); E1 final 2.833 / 0.132;
teacher pqtmfmqy 2.810 / 0.131.

Notes:
- Duplicate E7 8128390 (parallel session, snapshot spatial-v5-a271c9d) cancelled
  2026-07-22 ~11:10 — 8128309 kept (submitted first; snapshot bbacfde = a271c9d + E8
  preset, identical v5 code).
- **Naming: E8 = multitask trench run (8128315). The 128×128-resolution run (F15 work in
  the parallel session, gated on its review + memory smoke) should take E9.**
- E4-1g pilot 7887940 (parallel session's single-GPU parity probe) may still be running —
  not tracked here.
