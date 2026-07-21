# Terra Excavation Policy — Running Notes

- Local canonical: `docs/research/terra_excavation_policy/` (terra-baselines,
  branch `agent/spatial-v3-improvements`, worktree `terra-baselines_improvements`)
- Notion mirror: https://app.notion.com/p/3a456453c75781718406cbaff40335e1
  (research.md `3a456453c757819384b3ef49367e64dc`, notes.md `3a456453c7578178ba10c491e3d9e6f7`)
- Detailed docs: `docs/IMPROVEMENTS_SPATIAL_V3_2026-07-20.md` (spec),
  `docs/EXPERIMENTS_SPATIAL_V3_RUNS.md` (launch ledger)

## Status (2026-07-21)

E3 kickstart-medium is the best Terra policy to date: **2.99 pos_term / 0.139 reward /
swhr 1.000 at 15.4k/20k updates** — surpassed its teacher's final (2.81/0.131) at ~30% of
budget. E1 (algo fixes on v2 encoder) caught the teacher at 18.4k (2.80/0.130) with the
floor phase remaining. E2 (same stack as E3 but from scratch) finaled BELOW teacher
(2.33/0.108) → kickstart-not-from-scratch is now the data-backed playbook. E4 (xattn from
scratch) recovered from an early flatline to 1.15 @11.9k; verdict deferred to a kickstarted
rerun. E1 final ~2h, E3 ~6h, E4 ~7h from 22:45 CEST 2026-07-20+1.

## Leg log

| Date | Leg | Outcome |
|---|---|---|
| 2026-07-17 | PR #15 review (2e3b544 global-pool resnet) | global-pool readout bottleneck diagnosed; flatten readout + fixes designed |
| 2026-07-19/20 | Matched 10k A/B spatial_8x8 vs atari (pqtmfmqy / nnsksyva) | spatial +39% pos_term per-sample at 3.9× wall-clock |
| 2026-07-20 | spatial-v3 batch built (spec → Opus agents → 8-angle review → fixes; 54→62 tests) | encoder se (11ch+SE), bf16, critic width, no-value-clip, flat shuffle, kickstart, grow script |
| 2026-07-20 | E1–E3 launched (Euler 4×4090, 20k updates each) | incidents: home-quota kill (E1a), pickle identity (E3a) — both fixed same day |
| 2026-07-20 | F13 xattn encoder + E4 launched | cross-attention readout; probes pass; bf16 1.93× kept |
| 2026-07-21 | E2 final 2.33/0.108; E3 passes teacher at 5.9k updates | from-scratch drag confirmed; kickstart validated |

## Open items / next actions

1. Collect E1/E3/E4 finals; write comparison section into research.md.
2. E5: dumpzone-curriculum kickstart from E3 final (`solo_excavator_rectangles_dumpzone`
   preset; teacher+init = E3 checkpoint; sbatch ready to derive from E3's).
3. E6 step-efficiency: shorter horizon (max_steps 300) kickstart; add successful-episode
   length to the primary metric set.
4. E4′: xattn kickstarted from E3 (grow medium-se → medium-se-xattn).
5. Later: timeout bootstrap port, invalid-action masking, Muon-on-trunk A/B.

## Decisions & gotchas

- Entropy-schedule stretch (19k vs 9.5k) makes mid-run pos_term comparisons vs old runs
  invalid — compare at matched entropy phase or at finals only.
- `positive_terminations` is unbounded/episode-length-sensitive; primary metric is
  `success_within_horizon_rate` (AGENTS.md contract).
- Euler storage contract: run artifacts (checkpoints, WANDB_DIR) on scratch, never home
  (45/50 GB); scratch purges ~15 days; archives → work/rsl (big) or project/rsl (venvs).
- Derived per-run trainers (sed-patched `train_mixed_sv3_*.py`) pickle their own
  `__main__` config class — registration shims must fill only missing names (d1765d7).
- 4090 cuDNN autotune "Results mismatch" E-lines during compile are benign noise; the
  known real failure is `CUDNN_STATUS_INTERNAL_ERROR` (mitigate with autotune_level=0).
- grow_checkpoint: flax numbers ResidualMapBlocks sequentially ACROSS stages — block
  matching must be stage-aware or non-final-stage growth scrambles weights silently.
- Two sessions drive this branch — coordinate via `docs/EXPERIMENTS_SPATIAL_V3_RUNS.md`
  before touching the queue or worktree.
