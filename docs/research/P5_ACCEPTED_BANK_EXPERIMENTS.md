# P5 Accepted-Bank Experiment Implementation

- Status: local/Euler implementation complete; validation pending
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

## 1. Question and minimal matrix

The initial screen has four scratch-trained arms:

| Arm | Training conditions | Condition sampler | Question |
|---|---|---|---|
| `F-ANCHOR` | accepted foundation `Anchor` conditions | uniform | are the new foundation anchors learnable? |
| `T-ANCHOR` | accepted trench `Anchor` conditions | uniform | are the new trench anchors learnable? |
| `G-UNIFORM` | every accepted condition | uniform | can one generalist learn the target distribution directly? |
| `G-ADAPTIVE` | the identical all-condition bank | adaptive progressive | does progressive exposure improve the generalist? |

No strict stage unlock, per-environment demotion, partial reset, reward
curriculum, teacher, or old 12-map bank is part of this comparison.

All arms freeze:

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
2. verifies the source-registry hash;
3. verifies each training manifest is one condition with contiguous slots and
   exactly 64 maps per condition;
4. verifies every evaluation panel path, count, condition count and contiguous
   manifest; and
5. recomputes each evaluation `episode_id` from `scenario_id`, `reset_seed`,
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
and verifies that it selects the declared exact slot.

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

## 5. Entrypoints

The four YAML config names are the arm names themselves. A single non-Slurm
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
- [x] select foundation anchors, trench anchors, or all accepted conditions
  from descriptor fields rather than directory-name conventions;
- [x] port only the global uniform/adaptive sampler, not the old M3 bank or
  strict staged scheduler;
- [x] freeze reward and horizon across condition levels;
- [x] disable the per-environment Terra ratchet;
- [x] log intended mass, sampled assignments, reset exposure and completed
  episodes without conflating them;
- [x] add the four parameterized configs and one shared local/allocation
  entrypoint;
- [x] freeze and test the production PPO shape, explicit seed, checkpoint
  cadence, and finite-check cadence in that entrypoint;
- [x] add condition-macro, micro-p10 and worst-condition evaluation;
- [x] replace fixed panel counts with a continuous, map-count-aware comparison
  gate;
- [x] use frozen evaluation reset seeds and episode identities;
- [x] validate archive runs from an explicit frozen Terra revision without
  requiring `.git`;
- [x] require reset-array scenario identity before JAX;
- [x] separate assignments, reset exposure and completed-episode competence;
- [x] reject mixed-treatment or duplicate-update fixed-evaluation lists;
- [x] implement content-addressed Euler smoke and screen launch;
- [x] fail closed on P6 until the separate 256-train-layouts/condition bank
  exists; retain the generalist selection receipt as the future P6 gate;
- [x] pass the full terra-baselines CPU suite against the paired Terra commit
  (`218 passed`);
- [ ] complete one local or allocated-GPU first-update smoke per arm;
- [ ] submit the bounded screens only after the accepted bank is frozen.

The unchecked items are execution gates, not evidence supplied by this
implementation commit alone.
