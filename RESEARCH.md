# Terra Baselines Research Context

## Objective

Train and evaluate a reproducible global excavation policy for foundations and
trenches, including navigation, dirt relocation, precise finishing, and legal
dumping. The sibling `terra` repository owns environment dynamics and map
generation; this repository owns optimization, models, evaluation,
checkpoints, and cluster execution.

## Current V8 curriculum

The current method is
[Continuous Banded v3](docs/research/CONTINUOUS_BANDED_V3_DESIGN_20260812.md).
It is one uninterrupted process over all 47 V8 conditions:

- 80% of assignment mass is distributed globally over open conditions with
  immutable depth weights `4:2:1`;
- 20% is uniform replay over mastered conditions;
- no foundation/trench quota affects sampling;
- a 15% per-condition water-fill cap prevents a single-condition monopoly;
- exact-success EMA with minimum exposure controls mastery and demotion; and
- source-disjoint fixed panels audit checkpoints but never update the sampler.

Earlier family-balanced samplers are historical experiment provenance, not
selectable training modes. The selected v6.1 update-14,000 source is converted
offline into a native v3 checkpoint; runtime training has no compatibility or
sampler-migration mode.

Here, depth is immutable map-difficulty metadata and band is a changing sampler
role. Online success is weighted by the live sampler distribution; it is not a
whole-V8 benchmark result. Map allocation and reward design remain separate
causal variables.

The completed primary experiment held reward dense and trained two random-start
all-47 controls: the 2.856M compact deep+xattn policy and the original 480k
Atari-base policy. They shared the map sampler, transition budget, PPO shape,
seed, horizon, and fixed evaluations. The Atari policy is a deliberately small
system control, not a pure encoder ablation, because its actor, critic, and
local-map heads are also smaller.

The active capability run continues the selected v6.1 reward-v2 checkpoint
with two declared changes: Continuous Banded v3 and one normalized material
stall-age observation. Reward-v2, the v6.1 spatial architecture, action-mask
setting, PPO shape, learning rate, horizon, bank, and seed remain fixed. This
is a practical combined treatment, not a causal sampler or observation
ablation. Fixed source-disjoint panels—not reward return or online success—
decide whether to continue it.

The mechanism is informed by, but does not copy constants from:

- [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html),
  for persistent replay in procedural environments;
- [Self-Paced Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html),
  for competence-paced movement toward a target distribution;
- [Replay-Guided Adversarial Environment Design](https://proceedings.neurips.cc/paper/2021/hash/0e915db6326b6fb6a3c56546980a8c93-Abstract.html)
  and [ACCEL](https://proceedings.mlr.press/v162/parker-holder22a.html), for
  replay plus capability-frontier environment design; and
- [C-Procgen](https://proceedings.neurips.cc/paper_files/paper/2024/hash/24662461d2194d1bc70a47b6b6771026-Abstract-Conference.html),
  for evidence that broad multi-context training can retain an implicit
  easy-to-hard curriculum.

These sources motivate the mechanisms only. Fixed source-disjoint Terra
evaluations decide whether the chosen masses and thresholds work.

## Research hygiene

- Preserve Terra revision, dataset and graph hashes, reward, horizon, reset
  distribution, architecture, PPO shape, seed, and checkpoint hash.
- Treat online metrics as training diagnostics and fixed panels as behavioral
  evidence.
- Record live jobs and checkpoint decisions in the dated experiment ledger;
  do not infer promotion from `RUNNING`, update 0, or online completion.
