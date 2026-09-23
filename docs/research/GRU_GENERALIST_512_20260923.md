# GRU generalist on a 512-map bank

September 23, 2026. Branch `gru-bigbank-20260923` in both repositories.

## Why this run

The u110000 feed-forward generalist scores 599/608 on the development panel,
but greedy decoding completes only 14/32 starts of one held-out four-section
road (straight trench: 29/32). All 18 road failures follow one pattern: the
policy finishes the near half of the road in about 30 steps (7 of 16 digs),
then makes no task progress for about 416 steps while repeating 2–11-action
cycles, up to 342 no-effect actions and repeated relifting of dumped soil.
The straight-trench failures stall the same way at 92–95%.

A CPU replay with recorded logits shows what happens inside the cycles:

| States | Median top-action probability | Mean entropy | Mean value |
| --- | ---: | ---: | ---: |
| Successful episodes | 0.998 | 0.17 nats | 5.1 |
| Failures before the stall | 0.990 | 0.35 | 3.8 |
| Inside the stall | 0.42 | 1.49 | −0.2 |

The network recognizes that it is lost, but a memoryless policy under argmax
turns an uncertain state into a fixed cycle. Sampling escapes (16 samples
solve all 32 road starts). The greedy outcome is numerically fragile: the CPU
replay scores 12/32, and nearby checkpoints score u100000 24/32, u109250 0/32,
u110000 14/32. The start pose matters only through the intermediate state it
leads to.

The training bank is small for a policy trained on about 6.5 billion
transitions: 3,840 slots, but 1,821 distinct dig targets (865 parent layouts),
96 per condition. The historical V8 comparison favored recurrence: GRU u40000
677/720 versus feed-forward u86000 670/720 on the V8 promotion panel.

This run changes two things together: recurrent memory and 5.3 times more
distinct maps per condition. It is a combined capability run, not an ablation.

## Recipe

- **Student:** random initialization. Same residual spatial encoder and critic
  as the incumbent (`resnet_spatial_8x8_se_sa_xattn`, stages 24/48/64/96 ×
  2/2/3/3, critic 512/256), actor Dense(160) → GRU(64) with the concat skip,
  2,359,445 parameters. Observations: the September set (carry work,
  relocation distance, admissible and executable digging, trench alignment)
  plus remaining time. The residual actor head and retained-work context are
  feed-forward-only and are not used.
- **Teacher:** u110000 (`3fd74795…`) for both families through the task-teacher
  path, which rebuilds the teacher's native inputs from raw observations.
  Policy KL starts at 1 and follows a cosine to zero over 655.36M transitions
  (10,000 updates at this batch), as in the 2026-09-15 restart. No value
  distillation. Commit `f21c526` adds recurrent-student support: the
  `[sequence, time]` minibatch is flattened for the teacher forward and KL;
  a shared checkpoint is evaluated once.
- **PPO:** as the 2026-09-15 restart: Adam 3e-4, 2 epochs × 32 minibatches,
  entropy 0.02 constant, global-minibatch advantage normalization, value
  coefficient 2, no value clipping, clip 0.2, γ 0.9984, λ 0.95. Reward v2 with
  all added behavior costs at zero.
- **Batch:** 4 × RTX 4090, 512 environments per GPU, 32 steps: 65,536
  transitions and 64 Adam steps per update; 16 intact 32-step sequences per
  local minibatch.
- **Bank:** `train_v3_generalist_512`, the 40 generalist conditions with 512
  training maps each (the original 96 plus 416 new maps from unused generator
  indices), source-disjoint from every evaluation panel. Build details below.

## Evaluation plan

- Development panel (`gate_main`, 608 maps, greedy, 450 actions) at u2500,
  u5000, u10000 (end of teacher KL) and every 10,000 updates after. References:
  the 2026-09-15 restart at u2500 348/608 and u5000 386/608; u110000 599/608.
- Robustness: greedy on the 32-start road, straight and tee panels (u110000:
  road 14/32, straight 29/32), longest no-progress stall, and start search
  (best plan over several starts, since initial positioning is nearly free on
  the machine).
- Continue while the development panel and the multi-start panels improve;
  stop on a plateau across several checkpoints, not on online success.

## Bank build

`train_v3_generalist_512`: 20,480 slots, 15,112 distinct maps. Archive
SHA-256 `3ddaaf53…`, built under `.artifacts/terra_gru_bigbank_20260923/`
(README there has the full receipts).

- 24 conditions (all trenches, procedural and strip foundations, V7
  foundations): 512 distinct maps, the original 96 plus 416 new.
- 15 slab-footprint conditions: 178 distinct maps (96 + 82); large slab 154.
  The building-outline source pool is exhausted at generator index 541. Slots
  are equalized at 512 per condition by repeating these maps 2–4 times, so
  condition exposure matches the incumbent recipe (62.5% foundation resets).
- New maps come from the generator revision of the original pool (60d01307),
  at never-used indices; 814 existing maps were first regenerated
  byte-identically. Old maps are byte-identical to the September bank.
- No new map shares an identity or an exact target raster with any of 32,983
  evaluation maps. R2 distance recomputed exactly on 300 sampled new maps.
  The trainer loads all 20,480 slots with verified finite trench metadata.

## Status

September 24, 00:30. Euler smoke 15005789 (one RTX 4090, 512 environments,
full bank, baselines f8b1094, Terra 83e6f630) completed three updates in
18 minutes: runtime lock, cuDNN/NCCL loaders, convolution backward and
parameter count pass; 20,480 maps load with verified trench metadata; the
shared u110000 teacher is evaluated once. First update including compilation
638 s, then about 3,850 environment steps/s per GPU with teacher KL active.
The final checkpoint (Adam step 192) has finite parameters and optimizer
state, zero transition-integrity counts, teacher KL 1.24 (foundation 10,144
and trench 6,240 rows) and no memory failure.

Production job 15005791 (4 × RTX 4090, gpuhe.120h, u100000, W&B online) is
queued; Slurm estimates a start between September 28 and 30. At about 4.5 s
per update, one 120-hour segment covers most of the 100,000 updates.
