# Terra Excavation Policy — Running Notes

- Local canonical: `docs/research/terra_excavation_policy/` (terra-baselines,
  branch `agent/spatial-v3-improvements`, worktree `terra-baselines_improvements`)
- Notion mirror: https://app.notion.com/p/3a456453c75781718406cbaff40335e1
  (research.md `3a456453c757819384b3ef49367e64dc`, notes.md `3a456453c7578178ba10c491e3d9e6f7`)
- Detailed docs: `docs/IMPROVEMENTS_SPATIAL_V3_2026-07-20.md` (spec),
  `docs/EXPERIMENTS_SPATIAL_V3_RUNS.md` (launch ledger)

## Status (2026-07-22)

FINALS: **E3 3.054 / 0.142 (swhr 0.997, ep_len 55.2) — best Terra policy to date, +8.7%
over teacher and fastest episodes.** E1 2.833/0.132 beats teacher on its own encoder →
PPO fixes confirmed net-positive. E2 (from scratch) had finaled below teacher (2.33/0.108).
E4 (xattn from scratch) recovered to 2.50 @~17k, already above E2's final — weak positive
for the attention readout; final pending. E5 (dumpzone transfer, Slurm 8105958) and E6
(300-step horizon efficiency, 8105959) launched, both kickstarted from the E3 final
(first attempt lost to an NFS attr-cache race at the gate; retry guard added).

## Leg log

| Date | Leg | Outcome |
|---|---|---|
| 2026-07-17 | PR #15 review (2e3b544 global-pool resnet) | global-pool readout bottleneck diagnosed; flatten readout + fixes designed |
| 2026-07-19/20 | Matched 10k A/B spatial_8x8 vs atari (pqtmfmqy / nnsksyva) | spatial +39% pos_term per-sample at 3.9× wall-clock |
| 2026-07-20 | spatial-v3 batch built (spec → Opus agents → 8-angle review → fixes; 54→62 tests) | encoder se (11ch+SE), bf16, critic width, no-value-clip, flat shuffle, kickstart, grow script |
| 2026-07-20 | E1–E3 launched (Euler 4×4090, 20k updates each) | incidents: home-quota kill (E1a), pickle identity (E3a) — both fixed same day |
| 2026-07-20 | F13 xattn encoder + E4 launched | cross-attention readout; probes pass; bf16 1.93× kept |
| 2026-07-21 | E2 final 2.33/0.108; E3 passes teacher at 5.9k updates | from-scratch drag confirmed; kickstart validated |
| 2026-07-22 | E1/E3 finals; E5+E6 launched from E3 ckpt | E3 3.05/0.142 record; E1 2.83/0.132 > teacher; NFS gate race fixed |

## Open items / next actions

1. Collect E4 final; then E4′ decision (xattn kickstarted from E3 if E4 < E1).
2. Watch E5 (dumpzone) / E6 (short300) — compare E6 ep_len vs E3's 55.2 baseline.
3. Later: relocations_harder + trenches ladder, timeout bootstrap port, invalid-action
   masking, Muon-on-trunk A/B.

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
