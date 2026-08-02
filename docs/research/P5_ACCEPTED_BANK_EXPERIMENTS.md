# P5 Accepted-Bank Experiment Implementation

- Status: P5 complete; `G-ADAPTIVE` selected; matched P5b follow-up in progress
- Date: 2026-07-30
- Canonical authority:
  [`D5_D7_IMPLEMENTATION_PLAN.md`](/home/lorenzo/moleworks/.worktrees/terra_simple_mapbank_reward_20260730/D5_D7_IMPLEMENTATION_PLAN.md)
  §P5a
- Research-code constraints:
  [`$simple-research-code`](/home/lorenzo/git/codex_skills/skills/simple-research-code/SKILL.md)
- Terra branch: `experiment/simple-mapbank-reward-v3`
- terra-baselines branch: `experiment/simple-mapbank-reward-v3`

This document is the executable terra-baselines side of the canonical P5a
plan. The canonical plan should link here rather than duplicate these commands
or implementation details.

Final P5 evidence, the standardized condition leaderboard, and the frozen
three-arm warm-start depth/sampler screen are recorded in
[`P5_RESULTS_ANALYSIS.md`](P5_RESULTS_ANALYSIS.md) and
[`P5_FOLLOWUP_GOAL.md`](P5_FOLLOWUP_GOAL.md). Those documents do not change the
P5 treatment after the fact.

## 1. Question and minimal matrix

The initial screen has six scratch-trained arms:

| Arm | Training conditions | Condition sampler | Question |
|---|---|---|---|
| `F-ANCHOR` | accepted foundation `Anchor` conditions | uniform | are the new foundation anchors learnable? |
| `F-SPECIALIST` | every accepted foundation condition | uniform | can a foundation-only policy learn the full foundation distribution? |
| `T-ANCHOR` | accepted trench `Anchor` conditions | uniform | are the new trench anchors learnable? |
| `T-SPECIALIST` | every accepted trench condition | uniform | can a trench-only policy learn the full trench distribution? |
| `G-UNIFORM` | every accepted condition | uniform | can one generalist learn the target distribution directly? |
| `G-ADAPTIVE` | the identical all-condition bank | adaptive progressive | does progressive exposure improve the generalist? |

The anchor controls and family specialists are feasibility diagnostics. They
do not enter the causal curriculum ablation or promotion decision, which
compares only `G-UNIFORM` and `G-ADAPTIVE`.

They are not clean negative-transfer A/B comparisons with the generalist.
With the frozen 18-foundation/14-trench/32-total condition set, uniform sampling
gives each `F-SPECIALIST` condition `1/18` of assignments and each
`T-SPECIALIST` condition `1/14`, versus `1/32` in `G-UNIFORM`: respectively
`1.78x` and `2.29x` more dose per condition. The specialists answer whether a
family is learnable under concentrated family-only compute. A specialist is
invalid when its accepted family contains only anchors, because that would
duplicate the anchor control; the loader rejects this before JAX starts.

No strict stage unlock, per-environment demotion, partial reset, reward
curriculum, teacher, or old 12-map bank is part of this comparison.

All six arms freeze:

- agent-neutral `relocation_progress_mult = 1.5`;
- horizon 450, `DENSE`, trench absolute shaping off;
- tracked single-excavator actions and observations;
- scratch initialization;
- medium E8 MLP with `resnet_spatial_8x8_se`, bf16 encoder and
  `512,256` critic;
- PPO and entropy settings supplied by the one shared entrypoint; and
- one accepted-bank protocol hash.

## 2. Accepted-bank input contract

`--accepted-bank-root` is required. Its `dataset.json` must use
`terra_curriculum_loader_bank_v1` and contain:

- `environment_protocol = "environment_protocol.json"` plus its canonical
  SHA-256;
- `reset_prng = {jax_default_prng_impl: threefry2x32,
  jax_threefry_partitionable: true}` inside that hashed protocol;
- `scenario_identity_contract = "terra_reset_arrays_sha256_v1"` at the bank
  root, as emitted and enforced by the paired Terra commit;
- `source_registry = "source_registry.jsonl"` plus its file SHA-256;
- `train[]` entries with `condition_id`, `family`, `branch_depth`,
  `maps_path`, and exactly `map_count = 64`;
- exact MapsBuffer datasets under every training `maps_path`; and
- `promotion`, `development`, and `sealed` evaluation panels.

The caller must pass the exact immutable Terra revision recorded in the source
archive manifest. Before JAX initialization, the loader:

1. verifies the protocol receipt hash and equality with the protocol derived
   from the imported Terra code plus that explicit revision, without consulting
   `.git`;
2. rejects a missing or different reset-PRNG contract before constructing an
   environment;
3. verifies the source-registry hash;
4. verifies every staged training path declares exactly 64 maps, has one
   condition with contiguous manifest slots `1..64`, requires local
   `slot_count = 64`, and contains exactly `img_1.npy..img_64.npy` in each of
   the five reset-array directories;
5. verifies every evaluation panel path, count, condition count and contiguous
   manifest; and
6. recomputes each evaluation `episode_id` from `scenario_id`, `reset_seed`,
   and the frozen protocol hash.

This makes review-only banks, the legacy 12-map bank, stale Terra revisions,
and inherited `DATASET_PATH` values invalid inputs.

## 3. Adaptive sampler

There is one small host-side sampler. Every map condition owns one Terra level;
the trainer writes an explicit categorical level assignment for every
environment.

The uniform arm keeps `q = Uniform(conditions)`.

Every 150 PPO updates, the adaptive arm computes a completion EMA for each
condition with enough completed episodes, then:

```text
q = 0.20 * Uniform + 0.80 * Frontier
```

`Frontier` is a temperature-softmax over completion of conditions below the
0.75 mastery threshold. A condition stuck at zero therefore cannot monopolize
training; the most advanced unsolved conditions receive more exposure. A
mastered condition remains in the uniform floor and re-enters automatically if
its completion EMA falls below the threshold. Per-condition mass is capped at
0.15.

The receipt keeps three quantities separate: sampled level assignments, maps
actually instantiated by reset, and completed episodes used to estimate
competence. It reports both the accumulating current window and the last
closed window; completed-episode mass is never labelled as realized exposure.

## 4. Fixed evaluation and comparison gate

Use `eval_fixed_bank.py --accepted-panel` for the frozen `promotion`,
`development`, or `sealed` panel. It consumes each row's frozen `reset_seed`
and verifies that it selects the declared exact slot. Seed-to-slot mapping is
part of the executable protocol: the evaluator asserts partitionable
`threefry2x32` before deriving reset keys, so an import-order or JAX-config
change fails rather than silently shuffling or duplicating panel slots.

Every checkpoint reports:

- exact successes and rate;
- condition-macro terminal absolute completion;
- micro completion mean, median, p10 and p25;
- worst condition and its completion; and
- per-family and per-condition results.

For each checkpoint after the first, the evaluator writes a comparison to the
previous checkpoint. The gate passes only when integrity is clean and:

```text
progress =
    at least one additional exact-success map
    OR condition-macro completion gain >= 0.01

guards =
    micro p10 delta >= -0.05
    AND worst-condition delta >= -0.05
```

The exact-rate quantum is recorded as `1 / panel_size`. There are no
`26/32`, `6/8`, or other panel-size-specific counts.

The two specialist policies still evaluate the complete frozen panel so their
cross-family behavior remains visible, but their primary diagnostic fields are
the trained family's values:

- exact successes in `summary.by_family[family]`;
- macro completion, micro p10 and worst-condition completion in
  `summary.graded.by_family[family]`; and
- clean global `summary.integrity`.

The untrained family's scores are transfer diagnostics. The evaluator's global
`comparison_to_previous` gate is not a specialist feasibility gate: its macro
and worst-condition terms include the untrained family. The bounded specialist
screen is interpreted as progress when trained-family exact success increases
by at least one map or trained-family macro completion increases by at least
`0.01`, while that family's p10 and worst-condition completion each regress by
no more than `0.05`. These diagnostics never select the generalist promotion.

## 5. Entrypoints

The six YAML config names are the arm names themselves. A single non-Slurm
entrypoint keeps architecture and PPO arguments shared:

```bash
export TERRA_ROOT=/path/to/source-archive/terra
export TERRA_REVISION="$(cat /path/to/source-archive/terra/REVISION)"
export RUN_ROOT=/path/to/run-artifacts
export PYTHON_BIN=/path/to/python
export SEED=0

scripts/run_accepted_bank_screen.sh \
  G-UNIFORM /path/to/accepted-bank p5-g-uniform-s0 2000
```

`NUM_DEVICES`, `NUM_ENVS_PER_DEVICE`, and `NUM_STEPS` are operational
parameters; their defaults are `1`, `1024`, and `32`. The script computes
`total_timesteps` from those values and the required update count. `SEED` is
required and must match between paired `G-UNIFORM` and `G-ADAPTIVE` runs.
The production PPO shape is frozen at 2 update epochs, 32 minibatches and
learning rate `3e-4`; inline evaluation is disabled, and numbered checkpoints
are retained every 500 updates. `FINITE_CHECK_INTERVAL` defaults to 10 for
2k/20k runs; set it to 1 for an update-1 smoke. The script neither submits a
job nor chooses an Euler partition. It accepts no trailing training arguments,
so callers cannot override the frozen treatment.

Example fixed evaluation:

```bash
python eval_fixed_bank.py \
  --checkpoint /path/to/checkpoint.pkl \
  --bank-root /path/to/accepted-bank \
  --accepted-panel development \
  --terra-revision "$TERRA_REVISION" \
  --expect-completion-contract exact_visible_dump_v1 \
  --output /path/to/development_eval.json
```

The immutable Euler preparation, smoke, screen, reset-parity and fail-closed
future-20k boundary are documented in
[`scripts/euler_accepted_bank_v1/README.md`](../../scripts/euler_accepted_bank_v1/README.md).
`SUBMIT=0` is local-only and performs no remote mutation.

## 6. Implementation checklist

- [x] reject legacy/review-only/stale bank roots before JAX;
- [x] select foundation anchors, all foundations, trench anchors, all trenches,
  or all accepted conditions from descriptor fields rather than directory-name
  conventions;
- [x] port only the global uniform/adaptive sampler, not the old M3 bank or
  strict staged scheduler;
- [x] freeze reward and horizon across condition levels;
- [x] disable the per-environment Terra ratchet;
- [x] log intended mass, sampled assignments, reset exposure and completed
  episodes without conflating them;
- [x] add the six parameterized configs and one shared local/allocation
  entrypoint;
- [x] freeze and test the production PPO shape, explicit seed, checkpoint
  cadence, and finite-check cadence in that entrypoint;
- [x] add condition-macro, micro-p10 and worst-condition evaluation;
- [x] replace fixed panel counts with a continuous, map-count-aware comparison
  gate;
- [x] use frozen evaluation reset seeds and episode identities;
- [x] bind the JAX reset-PRNG mode into the protocol and reject runtime/config
  disagreement before fixed-panel reset;
- [x] validate archive runs from an explicit frozen Terra revision without
  requiring `.git`;
- [x] require reset-array scenario identity before JAX;
- [x] separate assignments, reset exposure and completed-episode competence;
- [x] reject mixed-treatment or duplicate-update fixed-evaluation lists;
- [x] implement content-addressed Euler smoke and screen launch;
- [x] fail closed on P6 until the separate 256-train-layouts/condition bank
  exists; retain the generalist selection receipt as the future P6 gate;
- [x] pass the full terra-baselines CPU suite against the paired Terra commit
  (`249 passed`);
- [x] complete one local or allocated-GPU first-update smoke per arm;
- [x] submit the bounded screens only after the accepted bank is frozen.

The unchecked items are execution gates, not evidence supplied by this
implementation commit alone.

## 7. 2026-08-01 allocated-gate finding

The first content-addressed six-arm smoke completed update 1 and passed its
checkpoint receipts. The following screen attempt stopped before W&B or PPO in
the CPU/GPU reset-parity gate: the loader had generated scalar reset seeds with
legacy `jax_threefry_partitionable=false`, while `train.py` and
`train_mixed.py` explicitly run with `true`. Under the real runtime the
512-slot promotion panel selected only 319 unique slots.

The accepted correction is to preserve all reviewed maps and source-disjoint
splits, pin partitionable Threefry in the hashed environment protocol,
regenerate only evaluation reset seeds and episode IDs, and rerun the six
smokes and screens. Forcing the evaluator back to the legacy mode is rejected
because it would create a different stochastic environment contract from the
trainer.

## 8. Final P5 execution result

The PRNG-corrected six-arm campaign
`f8aac348d64c7f71ee65273e6729ad142828731598ce383b2ac0331e225ebaaa`
completed on 2026-08-02. Every job produced a passing receipt, 2,000 updates,
262,144,000 transitions, and exact deterministic promotion/development
evaluation. The formal selector chose `G-ADAPTIVE` because it alone passed the
update-1,000 to update-2,000 promotion-retention gate. This is one paired seed,
not a general scheduler-superiority claim. P6 remains fail-closed until the
separate 256-training-layouts-per-condition bank exists.
