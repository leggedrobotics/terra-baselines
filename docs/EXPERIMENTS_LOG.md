# Experiments — completed log (spatial-encoder line)

Chronological record of finished runs. Metrics: eval/positive_terminations /
eval/rewards (final), swhr = eval/success_within_horizon_rate, ep_len =
eval/avg_positive_episode_length. Full incident trail:
`docs/EXPERIMENTS_SPATIAL_V3_RUNS.md`.

| Date | Run | W&B | Config | Final | Verdict |
|---|---|---|---|---|---|
| 07-19/20 | spatial A/B | pqtmfmqy | resnet_spatial_8x8 base, 10k updates, mb32 | 2.810 / 0.131 | spatial beats atari +39% per-sample at 3.9× wall-clock; became the E3 teacher |
| 07-20 | atari control | nnsksyva | atari base, 10k, mb32 | 2.020 / 0.095 | control; 113k steps/s |
| 07-21 | E2 | nr032qs7 | se+bf16+critic512 FROM SCRATCH, 20k | 2.329 / 0.108, swhr 0.976 | below teacher — from-scratch drag of heavier bundles (bf16 confirmed +50% throughput) |
| 07-22 | E1 | 3buorfp3 | spatial_8x8 + algo fixes (no value clip, flat shuffle, 19k ent), 20k | 2.833 / 0.132, swhr 0.997, ep_len 59.2 | beats teacher on its own encoder → PPO fixes net-positive |
| 07-22 | E3 | j0bs2fkl | medium se, grown init + kickstart from pqtmfmqy, 20k | **3.054 / 0.142, swhr 0.997, ep_len 55.2** | **best policy to date (+8.7%); kickstart playbook validated — surpassed teacher at 30% budget** |
| 07-22 | E4 | k8vnwp5u | se_xattn FROM SCRATCH, 20k | 2.586 / 0.121, swhr 0.993, ep_len 63.0 | xattn beats SE from scratch +11% (2.59 vs 2.33) — real architecture win; still below warm-started runs |
| 07-22 | E5 | gud7cbwg | dumpzone transfer (E3 warm-start + teacher), CANCELLED @850/20k | 0.000 / −0.005 | naive cross-task kickstart does NOT bootstrap a new task family (no reward stream found); E5b = 2-stage curriculum + higher entropy when dumpzone becomes priority |

Cross-run findings:
- Warm-start (grow + kickstart) >> from-scratch for introducing architecture/capacity
  changes (E3 vs E2; E4' line continues this).
- bf16 encoder compute: +50% production throughput (43k vs 28.8k steps/s), numerics clean.
- pmap scaling ~95% (single-GPU probe 11.4k vs 10.8k/GPU ×4).
- Entropy-schedule stretch makes mid-run comparisons vs 9.5k-schedule runs invalid —
  compare finals or matched entropy phase only.
- Episode length: E3 55.2 vs E1 59.2 steps — bigger warm-started net is also faster per
  episode; 300-step horizon (E6) exerts no pressure since episodes are ~55 steps.
