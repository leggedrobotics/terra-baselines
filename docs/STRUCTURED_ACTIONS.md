# Structured Terra actions (opt-in)

`train_structured.py` is the direct experimental training entrypoint for the
solo tracked excavator's `terra_structured_v1` action space. It consumes a
saved native initial-state bank (`initial` list, optionally `episodes` metadata),
including the cone/perpendicular-edge game bank. `train_mixed.py` remains the
legacy eight-way training path. This initial implementation is one-device,
vectorized across environments; it does not submit jobs or enable W&B.

The runner preserves the bank's pull-length and other native environment
settings. Native digging has no bucket-width constraint: trenches are widened
to bucket size during postprocessing. The temporary rectangular strip gate was
removed; old saved values of `dig_working_strip_width_m` are ignored by Terra.
See the paired checkout's `docs/PULL_DIRECTION_ALIGNMENT.md`. The manual game
uses a 1 m minimum pull; full-bank finishability remains a separate evaluation.

## Action and model contract

The existing type IDs are retained: forward/backward (0/1), clockwise and
counterclockwise base turns (2/3), DO (6), wait (7). Cabin-only types 4/5 are
masked out in the policy; the manual game still supports them. Type-conditioned
argument heads output move distances 1–5 cells, turn amounts 1–6 native
30-degree steps, and DO headings 0–11 (absolute cabin index relative to base).
DO performs one dig **or** one unload according to load; they are separate
policy decisions. The environment checks intermediate base orientations and
native movement clearance.

On the integer pose grid, one-cell moves are available only at cardinal base
headings. Oblique headings use amounts 2–5; requests that would clip to a
one-cell displacement are blocked. This prevents repeated rounded one-cell
moves from drifting along a different direction from the base heading.

The policy masks come from actual native outcomes, including relifting staged
soil and full-load unload checks. Wait is only the all-invalid fallback unless
`--allow-wait` enables it. Infeasible argument rows receive an internal dummy
categorical to keep unused-row gradients finite, but their type is masked out.

`utils/models.py` adds opt-in argument heads to the existing encoder/GRU.
Zero-initialized context projections feed both actor and critic remaining
machine time, remaining decision budget, timing visit state, and history of
amounts/headings/executed durations. Existing action-type history is retained.
Recurrent PPO replays complete trajectories with resets after terminal steps.

Rollouts store type, arguments, native masks, joint log probability and executed
duration. PPO clips the ratio of joint selected-action probabilities. Entropy
is the exact type entropy plus conditional entropy weighted by the probability
of each type. Four coefficients can be tuned independently. Unused argument
heads never contribute to a selected action's log probability.

## Time and return objective

The provisional defaults are 14,400 modeled seconds and a 450-decision guard.
The observation contains both remaining budgets because zero-duration actions
still consume the guard. `StructuredClock` charges setup once per actual visit;
movement away and back opens another visit. `--timing-json` accepts fields from
`terra.structured_actions.StructuredTimeConfig`; timing estimates need physical
calibration. Changing these values changes the training problem.

The native `material_time_v1` reward is material-potential difference plus the
configured behavior costs, minus `3.6 * executed_seconds / time_budget_seconds`.
Exact completion within both budgets earns +6 once; timeout earns -1 once. A
completed action that overruns time is a timeout. Separate native dig/unload
events preserve material and handling accounting. This path does not call the
legacy 450-step reward-v2 formula.

`gamma=1` is deliberate: potential differences telescope over a finite episode,
and modeled time is penalized explicitly. An explicit `--gamma` below one uses
`gamma ** (executed_seconds / discount_reference_seconds)` and is a different
objective: the undiscounted shaping deltas no longer telescope. GAE uses
`lambda ** (executed_seconds / discount_reference_seconds)`; default lambda=.95
and reference=30 seconds. Lambda, reference duration, episode budget and entropy
are initial settings, not behaviorally tuned conclusions. Terminal episodes,
including timeouts and the decision guard, never bootstrap into a reset state.

## Bounded smoke

From this `terra-baselines` worktree:

```bash
export PYTHONPATH="../terra:.:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
/home/lorenzo/moleworks/.venv-terra-uv/bin/python train_structured.py \
  --initial-states /home/lorenzo/moleworks/.artifacts/terra_manual_edge_20261009/initial_states_cone30_perp.pkl \
  --bank-indices 0,1 \
  --output /home/lorenzo/moleworks/.artifacts/terra_structured_20261009/ppo_cpu_smoke \
  --num-envs 2 --num-steps 2 --epochs 1 --updates 1
```

The initial native masks/observation and transition kernels take time to compile.
This runner uses a Python rollout loop around vectorized JIT kernels to keep the
first implementation inspectable. It is not yet a throughput-qualified replacement
for the production `pmap` trainer. `metrics.jsonl` records finite update evidence;
`config.json` records the resolved protocol. `checkpoint.pkl` is written atomically.

For CUDA, use `.venv-terra-gpu-uv`, export its NVIDIA library directories, disable
JAX preallocation on a shared GPU, and run the Terra RL `check_jax_runtime.py`
preflight before the same bounded command. A model update or a native first
update proves implementation startup, not policy quality or map finishability.

## Checkpoints and evaluation

`--warm-start-from LEGACY.pkl` reuses a shape-compatible encoder, recurrent core
and type head. It initializes new argument/context parameters, resets the critic
and optimizer, and starts fresh environments. Move arguments initially favor five
cells; base turns favor one step. Heading-conditioned DO and removed cabin types
mean the complete action distribution cannot exactly preserve legacy behavior.
Architecture mismatches fail clearly. Optional `--model-config FILE.json`
overrides architecture/observation settings for scratch; warm starts inherit the
source model configuration. Legacy config classes are registered before unpickling.

`--resume-from STRUCTURED.pkl --updates N` resumes the native optimizer, update
index, RNG, live terrain, recurrent carry, action history and time/visit clocks.
Repeat the original protocol arguments; `N` is the absolute final update target.
Changes to recorded timing, bank identity metadata, model, batch or PPO settings
reject native resume. Source code must retain the same dynamics and action
protocol; the loader does not detect arbitrary code edits. Output location and
checkpoint cadence may change.

`--teacher-checkpoint` rejects explicitly: there is no approved probability
mapping from an eight-way teacher's cabin actions to heading-conditioned DO.
Teacher guidance should be added only with an explicit mapped target and test.

Before selecting a training recipe, compare fixed initial maps on completion,
accepted disposal, modeled time and decisions. The small saved game bank is a
runtime diagnostic, not a source-disjoint generalization benchmark.
