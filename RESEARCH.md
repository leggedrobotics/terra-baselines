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

The current primary experiment holds reward dense and trains two random-start
all-47 controls: the 2.856M compact deep+xattn policy and the original 480k
Atari-base policy. They share the map sampler, transition budget, PPO shape,
seed, horizon, and fixed evaluations. The Atari policy is a deliberately small
system control, not a pure encoder ablation, because its actor, critic, and
local-map heads are also smaller.

Reward fading is a later fork of the compact dense trunk, not a second
random-start map-curriculum run. Once both sampler families reach depth 2 (or
are fully mastered) and the nearest retained checkpoint has fixed promotion
and development evidence, two children resume the same model, optimizer,
update count, and sampler state. One remains dense; the other irreversibly
fades to the terminal objective over 5,000 updates and then trains for 1,000
updates at the exact terminal objective. Fixed source-disjoint panels, not
reward return or online success, decide both comparisons. The exact launch and
evaluation contract is recorded in the authoritative design document linked
above.

The active reward and termination audit is
[V8 reward and termination audit](docs/research/V8_REWARD_TERMINATION_AUDIT.md).
It preserves exact success, separates strict completion from continuous
material progress, and records the failure diagnostics required before the
matched dense-to-terminal screen. Its explicit fixed-checkpoint experiment
proposal supersedes the older use of sampler depth as a reward-admission gate
only after that amendment is implemented and recorded in the authoritative V8
design.

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
