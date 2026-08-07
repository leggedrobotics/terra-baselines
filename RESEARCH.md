# Terra Baselines Research Context

## Objective

Train and evaluate a reproducible global excavation policy for foundations and
trenches, including navigation, dirt relocation, precise finishing, and legal
dumping. The sibling `terra` repository owns environment dynamics and map
generation; this repository owns optimization, models, evaluation,
checkpoints, and cluster execution.

## Current V8 curriculum

The authoritative design, evidence, decisions, and operational contract are in
[V8 10M scale-up and curriculum](docs/research/V8_10M_SCALEUP.md). The accepted
map treatment is one uninterrupted `continuous_banded_v1` process per arm, not
separate Stage A/B/C jobs:

- all 47 V8 conditions have positive probability from update 0;
- foundation and trench each receive 50% of target assignment probability;
- within each family, 10% is uniform over every condition, 75% is uniform over
  the entire active depth, and 15% is uniform over the next depth;
- exact completed training episodes update the mastery EMA; and
- source-disjoint fixed panels audit and select checkpoints but never control
  the online sampler.

Here, depth is immutable map-difficulty metadata and band is a changing sampler
role. Online success is weighted by the live sampler distribution; it is not a
whole-V8 benchmark result. Map allocation and reward design remain separate
causal variables.

The current reward screen therefore holds that map treatment fixed and pairs
two random-start compact deep+xattn policies: one remains dense, while the
other starts an irreversible 5,000-update dense-to-terminal fade after both
families reach active depth 2. Both arms use one common Terra/baselines binary;
fixed source-disjoint panels, not reward return or online success, decide the
comparison. The exact launch and evaluation contract is recorded in the
authoritative design document linked above.

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
