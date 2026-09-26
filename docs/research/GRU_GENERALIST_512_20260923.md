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

## Results

Same evaluator and panels for every row; greedy, 450 actions. u110000 on one
Euler RTX 4090, GRU checkpoints on the local RTX 4090 (JAX 0.4.33 on both).

| Policy | Full 608 | Foundations | Trenches | Roads | Stalled episodes | Road 32 starts | Straight 32 | Tee 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FF u110000 (teacher) | 598 | 381/384 | 217/224 | 32/32 | 19 | 14 (18 stalls) | 29 | 32 |
| GRU u2500 (KL weight 0.85) | 597 | 380/384 | 217/224 | 30/32 | 15 | 12 (22 stalls) | 30 | 32 |
| GRU u5000 (KL weight 0.50) | 597 | 378/384 | 219/224 | 32/32 | 18 | 18 (14 stalls) | 32 | 32 |
| GRU u7500 (KL weight 0.15) | 597 | 377/384 | 220/224 | 30/32 | 17 | 11 (21 stalls) | 31 | 32 |
| GRU u10000 (KL weight 0) | 586 | 368/384 | 218/224 | 32/32 | 29 | **28 (4 stalls)** | 31 | 32 |
| GRU u12500 | 593 | 373/384 | 220/224 | 31/32 | 24 | 23 (9 stalls) | 29 | 32 |
| GRU u15000 | **599** | 379/384 | 220/224 | 31/32 | 18 | 29 (5 stalls) | 32 | 32 |

At u2500 the student has cloned the teacher (policy KL 0.12): it gains seven
maps and loses eight against u110000, and repeats the teacher's road stalls.
The same number of transitions (164M) brought the 2026-09-15 feed-forward
restart, which had weaker teachers, to 386/608. Whether memory and the larger
bank remove the stalls is only testable after the KL has faded (u10000).
At u5000 (KL weight 0.5) the road starts rise to 18/32 and straight to 32/32,
above the teacher's 14/32 and 29/32, while the full panel holds at 597/608
(seven gains, eight losses versus u110000).
Training speed on 4 × RTX 3090 is about 9 s per update (7.3–7.9k environment
steps/s), about half the RTX 4090 rate; the teacher forward costs about 10%.

The road-start gain at u5000 did not persist at u7500 (18 → 11 of 32). This
panel moves as a block between nearby checkpoints (the teacher lineage went
24, 0 and 14 of 32 at u100000, u109250 and u110000), so single-checkpoint
changes on it are weak evidence; the development panel holds at 597/608 with
trenches rising 217 → 220.

At u10000, where teacher guidance reaches zero, the two effects separate. The
32-start road panel reaches 28/32 with 4 stalls (teacher 14/32 with 18), but
19 development maps are lost since u7500 (8 gained), 13 of them foundations.
Every one of the 19 losses is the same loop signature: 200–430 steps without
task progress, a repeating 1–8-action pattern, and often hundreds of no-effect
actions; several foundations stop with the dig complete and the soil still
undisposed. The decision point is u15000: whether PPO recovers these without
the teacher.

By u15000 the foundations recovered (368 → 379/384) and the GRU passes its
teacher on the development panel (599 vs 598) while keeping the road gain
(29/32 vs 14/32).

## Dump reach 5.5 m (switched in at about u20000)

The machine digs out to 6.5 m but dumps reliably only within about 5.5 m.
Terra used one cone for both, and a dump's soil lands within 2 tiles of the
free cells' centroid, i.e. out to 6.5 m. Terra `da5bd656` adds
`agent.dump_max_radius_m`: excavator dumps (and truck transfer and the
free-space check) use only cone cells within that radius; the minimum stays at
3.64 m. Soil is released within the reach; relaxation lets the pile edge
slide under one tile further (the unit test measures a 4.6 m mean and 5.83 m
maximum at 5.5). The default 0 keeps the frozen v1 benchmark; the trainer and
evaluator take `--dump_max_radius_m` / `--dump-max-radius-m`.

Admission (`tools/dump_reach_admission.py`, base positions where the footprint
fits, dumping from the digging pose, relays through neutral cells allowed):
every map of the development panel and of the 20,480-slot training bank stays
fully serviceable at 5.5 m. Direct one-hop service drops mainly for
`fnd-slab-apron-d16` (193 → 30 of 512 slots) and `fnd-proc-side1-road`
(455 → 380), which now need relays.

Existing policies depend on the far dumps:

| Policy | Dumps to 6.5 m | Dumps to 5.5 m | Foundations at 5.5 | Trenches at 5.5 | Stalled episodes at 5.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FF u110000 | 598 | 537 | 336/384 | 201/224 | 180 |
| GRU u15000 | 599 | 540 | 332/384 | 208/224 | 108 |

The run continues from its latest checkpoint under the 5.5 m reach (same run
directory and W&B run name; the first resumed segment marks the switch).

After the switch (resumed at u21000, 4 × RTX 4090 from 07:27 on September 26,
4.2 s/update), scored with 5.5 m dumps:

| Policy | Full 608 | Foundations | Trenches | Stalled episodes | Road 32 starts | Straight 32 | Tee 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU u22500 | 590 | 374/384 | 216/224 | 26 | 24 | 32 | 32 |
| GRU u25000 | 597 | 378/384 | 219/224 | 21 | 29 | 32 | 32 |

Four thousand updates under the new reach recover the old-rule score (599 at
u15000) to within two maps, 60 above the teacher under the same rule.
