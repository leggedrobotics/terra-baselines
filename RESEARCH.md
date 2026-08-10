# Terra Baselines Research Context

## Objective

Train and evaluate a reproducible global excavation policy for foundations and
trenches, including navigation, dirt relocation, precise finishing, and legal
dumping. The sibling `terra` repository owns environment dynamics and map
generation; this repository owns optimization, models, evaluation,
checkpoints, and cluster execution.

## Current V8 curriculum

The authoritative design, evidence, decisions, and operational contract are in
[V8 improvement set](docs/research/V8_IMPROVEMENT_SET_20260810.md) and the
[reward and termination audit](docs/research/V8_REWARD_TERMINATION_AUDIT.md).
The accepted follow-up map treatment is one uninterrupted
`continuous_banded_v2` process per arm, not separate Stage A/B/C jobs:

- all 47 V8 conditions have positive probability from update 0;
- foundation and trench each receive 50% of target assignment probability;
- within each family, 10% is uniform over every condition and 90% is spread
  over every unmastered condition with shallow-to-deep weights `4:2:1`;
- any eligible condition can graduate independently from exact completed
  training episodes, so one straggler cannot pin its family; and
- source-disjoint fixed panels audit and select checkpoints but never control
  the online sampler.

The completed compact/Atari experiment used `continuous_banded_v1`; that fact
is historical evidence, not the sampler contract for the next reward screen.
An explicit one-way migration validates and preserves its sampler state before
both matched R2 arms recompute probabilities under v2.

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

The next mainline reward question is R2: compare the current dense recipe with
one normalized material-potential reward-v2 bundle after both arms take the
same v1-to-v2 sampler migration, output-preserving carry-input expansion,
fresh optimizer, and LR warmup. Only if reward-v2 wins does R3 fork the selected
v2 checkpoint into fixed-shaping and episode-latched dense-to-sparse children.
The former whole-objective anneal is retired from the mainline because it
changes several reward semantics at once. Fixed source-disjoint panels, not
reward return or online success, decide both comparisons.

The active reward and termination audit is
[V8 reward and termination audit](docs/research/V8_REWARD_TERMINATION_AUDIT.md).
It preserves exact success, separates strict completion from continuous
material progress, and records the diagnostics and analytic admission gates
required before R2. Its fixed-checkpoint experiment proposal supersedes the
older sampler-depth reward trigger and whole-objective fade.

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
