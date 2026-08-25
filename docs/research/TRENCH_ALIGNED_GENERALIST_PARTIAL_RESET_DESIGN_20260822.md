# Draft design — trench-aligned generalist with partial resets

Status: immutable artifacts and update-1 smoke validated; the first production
allocation exposed a node-sensitive cuDNN autotuning failure before update 1;
runtime-only recovery candidate pending, 2026-08-24

## Decision

Add one named training recipe, `trench_align_generalist_partial_v1`. Partial
resets are on by default inside this recipe, but remain opt-in for the generic
`train_mixed.py` entry point. This makes the supported capability easy to use
without silently changing unrelated experiments.

The recipe combines three existing treatments:

1. the strict fresh-trench dig-alignment rule and its three observations;
2. the full-support `continuous_banded_v3` curriculum; and
3. a short, sparse partial-reset acquisition phase whose samples never update
   full-start mastery statistics.

It is a fresh MLP run. GRU and movement-feedback changes are deliberately out
of scope so that the first result answers whether the environment rule plus a
generalized reset curriculum is trainable.

## Environment and map scope

The immutable V8 bank remains the source of truth. A named
`trench_aligned_37_v1` condition profile selects the 37 conditions that satisfy
the strict gate's metadata and static-feasibility requirements:

- 25 foundation conditions, whose transition rules are unchanged;
- 12 finite-metadata trench conditions, with strict yaw/standoff admission;
- no `trn-net4-*` conditions, because the current layouts are statically
  infeasible under the gate; and
- no `v7-trn-*` conditions, because their frozen bank lacks the required
  finite-section metadata.

The view preserves the canonical curriculum depths rather than relabeling the
surviving trench conditions: foundation has 1/6/18 conditions at depths
0/1/2, while trench has 1/0/11. The missing trench depth-1 bin is therefore an
explicit property of this narrowed profile. Only this named profile enables
the sampler's sparse-depth path; the generic constructor still rejects a
continuous graph missing any family-depth bin. `continuous_banded_v3` remains
global over the conditions that exist, gives every selected condition positive
support, and retains the frozen depth weights, mastery rule, replay mass, and
per-condition cap. No probability mass or synthetic condition is invented for
the absent bin.

The profile is a view over the canonical 47-condition registry, not a new
release pretending that the excluded directories never existed. The root
source registry, review admission, and array identities stay intact. A derived
runtime artifact may rebind only the environment-protocol revision when the
protocol payload is byte-equivalent apart from its revision/hash fields; that
derivation must have its own receipt.

Evaluation uses only full starts. The main evaluation family must contain the
same 35 constrained/core conditions admitted by the profile; the two
capability-floor conditions use the existing capability-floor panels.

## Partial-reset bank

The partial bank is an action-only sidecar over canonical map slots. It may be
sparse in two ways:

- a condition with no accepted sidecars remains a normal full-start condition;
- within a supported condition, only canonical sources with a complete nested
  50/75/90 triplet are eligible for partial resets.

Every source triplet uses one deterministic geometry policy. The ordered
policy for this recipe is:

1. `relay_corridor`, which stages the excavated material along a source-to-dump
   service corridor;
2. `in_zone`, used only when the complete relay triplet cannot be generated.

Modes cannot be mixed across tiers of one source. The bank index and each leaf
record the ordered policy, every accepted manifest row records the chosen
mode, and rejection rows record failed modes. This makes fallback visible in
the artifact rather than an unreported generator behavior.

For trench sources, an accepted triplet also needs a strict-alignment service
witness after staged soil and completed holes are applied. For each remaining
fresh cell, the witness requires at least one persistent, footprint-clear dig
station with:

- base yaw within 15 degrees of one of that cell's finite trench sections;
- standoff in the closed 3.5–7.0 m interval;
- an immediately valid backward continuation sharing a section; and
- a monotone set of admissible dig cones covering every remaining cell.

This witness is a necessary action-chain coverage check, not a proof of a full
episode plan. A bank is launchable only if every admitted trench sidecar passes
the check. A failed source triplet is excluded; it must not be silently treated
as aligned.

## Reset schedule

The schedule is the existing 10,000-update Backplay-style acquisition window:

| absolute update | partial share | eligible tiers |
| ---: | ---: | --- |
| 0–2,499 | 25% | 90% complete |
| 2,500–4,999 | 25% | 75%, 90% complete |
| 5,000–7,499 | 25% | 50%, 75%, 90% complete |
| 7,500–9,999 | linear 25% to 0% | 50%, 75%, 90% complete |
| 10,000 onward | 0% | full starts only |

Thus at least 75% of lanes are full starts at every update. A partial lane is
sampled only from the common support of all scheduled tiers. Conditions
without partial support remain eligible on full-start lanes.

Only completed full-start episodes feed `continuous_banded_v3` competence,
mastery, and replay statistics. Partial episodes affect PPO gradients, but do
not let a 90%-complete reset promote a condition. The schedule uses absolute
update indices and the partial-bank digest is part of native-resume identity.

## Observation, objective, and model contract

The recipe requires:

- `reward_v2` with timing variant 0;
- carry-work observation;
- reward-v2 reset context `[Q_reset, H_reset / V0]`;
- material stall-age observation;
- trench alignment observation;
- strict trench dig alignment and finite-metadata preflight;
- unmasked actions; and
- fresh initialization or a native resume from the identical architecture and
  treatment fingerprint.

The earlier partial-reset causal comparison intentionally omitted stall age to
isolate one intervention. That experimental exclusion is not an environment
incompatibility. This capability recipe permits the two observations together
and adds a focused configuration/model test for their combined feature shape.

## Artifact and launch contract

The run consumes four pinned inputs:

1. committed `terra-baselines` source;
2. committed Terra runtime source;
3. a canonical accepted-bank archive plus runtime-protocol derivation receipt;
4. a partial-reset archive plus bank digest and trench-alignment audit receipt.

The launcher records the condition profile, selected condition/slot counts,
full and partial bank digests, selected pile-mode counts, gate tolerances,
model parameter count, sampler rule, reset schedule, revisions, seed, and Slurm
hardware in `run_contract.env`.

The previous smoke job `11515185` is evidence only for configuration loading:
it loaded the 37 hard-coded map levels and gate metadata, then failed before
update 1 because pooled sampling is supported only through
`--accepted-bank-root`. It is not a training smoke and cannot authorize a
production launch.

The first immutable-artifact smoke, job `11528804`, passed archive, profile,
strict-alignment, CUDA, `pmap`, model-shape, and partial-bank preflights, then
failed before update 1 because the sampler constructor still required a trench
depth-1 condition. That failure is also not training evidence. The correction
preserves canonical depths, opts only `trench_aligned_37_v1` into sparse-depth
construction, and makes both the local launcher and the Slurm job instantiate
the exact sampler before training starts.

The corrected smoke, job `11529665` on `eu-g6-045`, completed update 1 and
wrote a finite model/optimizer checkpoint. Its production successor,
`11529891` on `eu-g6-065`, reached the same model and sampler construction but
failed before update 1 on all four replicas with
`CUDNN_STATUS_EXECUTION_FAILED` in bf16 convolution-backward-filter execution.
It wrote no checkpoint and zero W&B training updates; dependent job `11529893`
was cancelled by `afterok` as designed. This is runtime evidence, not a policy
or curriculum result.

The first recovery froze `XLA_FLAGS=--xla_gpu_autotune_level=0`, retained the
existing `eu-g6-064` exclusion, and added `eu-g6-065`. Jobs `11626135` and
`11626137` then started successfully, but the 2026-08-25 audit measured only
3,124.6 steps/s in phase 1. The exact same 4 x RTX 4090 transition shape
sustained 16,771.1 steps/s in C0 `11152229`, 16,503.0 in T1 `11152230`, and
15,800.5 in GRU control `11364188`. Current first/last-window throughput was
flat, the GRU control was fast during the same 25% partial-start window, and
C0/T1 differ by less than 2%; the compiler flag, not partial reset or trench
alignment, caused the 5.0--5.4x regression.

Pinned jaxlib 0.4.26 [uses XLA
`4e8e23f16bc925b6f27817de098a8e1e81296bb5`](https://github.com/jax-ml/jax/blob/jaxlib-v0.4.26/third_party/xla/workspace.bzl#L17-L31).
The failing `cuda_dnn.cc:7927` call is
[`cudnnBackendExecute` in the cuDNN frontend execution-plan runner](https://github.com/openxla/xla/blob/4e8e23f16bc925b6f27817de098a8e1e81296bb5/xla/stream_executor/cuda/cuda_dnn.cc#L7910-L7937).
The replacement therefore keeps level-4 profiling, sets
`--xla_gpu_enable_cudnn_frontend=false` to route convolutions through the
[legacy cuDNN runners](https://github.com/openxla/xla/blob/4e8e23f16bc925b6f27817de098a8e1e81296bb5/xla/stream_executor/cuda/cuda_dnn.cc#L8215-L8304),
and sets `--xla_gpu_deterministic_ops=true`. That last flag matters because
this pinned XLA [retains wrong-result profiles in the candidate
list](https://github.com/openxla/xla/blob/4e8e23f16bc925b6f27817de098a8e1e81296bb5/xla/service/gpu/stream_executor_util.cc#L601-L613)
and otherwise [sorts the list by measured
runtime](https://github.com/openxla/xla/blob/4e8e23f16bc925b6f27817de098a8e1e81296bb5/xla/service/gpu/stream_executor_util.cc#L677-L707).
This is a runtime-only repair: seed, maps, reset schedule, sampler,
observations, rewards, gate, parameter tree, bf16 compute, and PPO settings
remain fixed. The launcher also exercises the exact 512 x 16 x 16 x 64 and
512 x 8 x 8 x 96 3x3 backward filters plus the 8 x 8 x 96 -> 32 1x1
flatten-reduction filter, and compares their all-ones gradients with
closed-form values before W&B is contacted. Recovery submission first resumes
five updates on one Euler RTX 4090 with the same per-device batch, then runs the
same five-update canary on four GPUs. Only the four-GPU job enforces the
12,000-steps/s throughput threshold and unlocks production.

The recovery source is the immutable native u3,500 checkpoint
`trench_generalist_partial_a1488abeab2f_s20260823_update_003500.pkl`, SHA-256
`f84a6cdfcb4aba0ca55abf1a658e4d57d21c6dffff9c4c2f61263733cd4f4790`,
with optimizer step 224,000. A native stall-age run has no offline-migration
receipt, so resume now accepts a checkpoint whose saved training config already
contains `stall_age_observation=true`; legacy checkpoints still require the
preparation receipt. The old slow jobs stay live until the exact replacement
passes the Euler gate.

## Gates before production

Production submission follows only after all of the following pass:

1. focused CPU tests for ordered mode fallback, sparse support, loader
   validation, condition selection, schedule, and combined observations;
2. exact validation of every selected full-start level and every partial
   sidecar;
3. complete strict-alignment audit of all admitted trench sidecars;
4. one-GPU Euler compiler execution, followed by four visible RTX 4090 devices,
   analytic exact-shape bf16 convolution backward checks, and `pmap`/NCCL
   preflight in the submitted allocation;
5. five actual native-resume PPO updates with finite losses, parameters,
   optimizer state, and readable checkpoints;
6. a post-compile median of at least 12,000 steps/s across updates 3--5; and
7. a completion receipt with `status=COMPLETE` at absolute target u3,505.

Replacement production resumes u3,500 to an absolute target of 75,000 only
after the throughput smoke succeeds, followed by an `afterok` native
continuation to 100,000. The 12,000-steps/s floor leaves the first segment
inside a 120-hour allocation; the matched expectation is 15,800--16,800
steps/s. The continuation consumes the phase-1 `FINAL` checkpoint, uses a new
linked W&B run identity, and validates the absolute update, checkpoint digest,
partial-bank digest, architecture, and finite model/optimizer state before it
starts. The u10k partial-reset window is an acquisition aid, not an evaluation
shortcut; comparisons and checkpoint ranking remain full-start and
condition-attributed.

## Frozen implementation artifacts

The launch candidate was generated from committed Terra revision
`a7204ef568f202f71b2f76943cb8b2f662eb71ff`. The derived full bank preserves
the canonical 47-condition release and selects the 37-condition view at load
time. Its three main panels contain 35 conditions and 560/560/1,120 slots;
the separate two-condition capability-floor panels contain 32/32/64 slots.

The generalized partial bank contains 238 complete source triplets, or 714
sidecars with exactly 238 examples at each of 50%, 75%, and 90% completion.
It supports 35 of the 37 training conditions:

- 85 triplets use `relay_corridor`;
- 153 triplets use the explicit `in_zone` fallback; and
- `fnd-slab-allfree` and `trn-straight-allfree` remain full-start-only.

The first complete trench audit found three failed sidecars across three
source triplets. The derivation removed each complete triplet, not only the
failing tier. The final audit passes all 255 admitted trench sidecars with zero
failures. Frozen identities are:

- full `dataset.json` SHA-256:
  `874315916ee5a9ffbfe8809dc3a21cb2aeb4e2ec0863c8bf57e1569e7bac3c1e`;
- environment protocol SHA-256:
  `511b1b07e43791151d672ae306c87c8222426e8a7fc91ab11cd6fb42c4bcf027`;
- partial-bank semantic digest:
  `f25398d3debbffe7bb1df1d9c7b4fe491d6835a5180de8ef8dca14235f07dd74`;
  and
- final alignment-audit SHA-256:
  `8ebd961afe6def4e8bdd6ada8a07525032b9c43c79912c66f7456a9991f40266`.

These counts show broad reset coverage, not that every condition can be
partially reset. Sparse-support conditions stay in the 37-way full-start
sampler and are never substituted with another condition's partial state.

### Reproduction sequence

The checked-in path list is
`configs/trench_aligned_37_maps_paths.txt`. With `BASELINES` set to this
checkout, `TERRA` set to the pinned Terra checkout, `PYTHON` set to
`/home/lorenzo/moleworks/.venv-terra-uv/bin/python`, and `ART` set to
`/media/lorenzo/T7/codex/terra_trench_aligned_generalist_partial_20260823`,
the semantic build sequence is:

```bash
PYTHONPATH="$TERRA:$BASELINES" "$PYTHON" \
  "$BASELINES/scripts/build_trench_aligned_runtime_bank.py" \
  --input-root /home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819 \
  --output-root "$ART/full_bank_a7204ef568f2" \
  --terra-root "$TERRA" \
  --terra-revision a7204ef568f202f71b2f76943cb8b2f662eb71ff

PYTHONPATH="$TERRA" "$PYTHON" "$TERRA/tools/materialize_partial_reset_bank.py" \
  --input-root "$ART/full_bank_a7204ef568f2" \
  --output-root "$ART/partial_bank_candidate_a7204ef568f2" \
  --seed 20260823 \
  --max-attempts-per-variant 10 \
  --max-source-triplets-per-condition 8 \
  --max-sources-scanned-per-condition 12 \
  --pile-mode relay_corridor \
  --pile-mode in_zone \
  --include-maps-path-file "$BASELINES/configs/trench_aligned_37_maps_paths.txt"

PYTHONPATH="$TERRA" "$PYTHON" \
  "$TERRA/tools/audit_partial_reset_trench_alignment.py" \
  --canonical-root "$ART/full_bank_a7204ef568f2" \
  --partial-root "$ART/partial_bank_candidate_a7204ef568f2" \
  --output "$ART/partial_bank_candidate_a7204ef568f2/trench_alignment_audit.json" \
  --workers 12

PYTHONPATH="$TERRA:$BASELINES" "$PYTHON" \
  "$BASELINES/scripts/filter_partial_reset_alignment_audit.py" \
  --input-root "$ART/partial_bank_candidate_a7204ef568f2" \
  --output-root "$ART/partial_bank_admitted_a7204ef568f2" \
  --audit "$ART/partial_bank_candidate_a7204ef568f2/trench_alignment_audit.json"

PYTHONPATH="$TERRA" "$PYTHON" \
  "$TERRA/tools/audit_partial_reset_trench_alignment.py" \
  --canonical-root "$ART/full_bank_a7204ef568f2" \
  --partial-root "$ART/partial_bank_admitted_a7204ef568f2" \
  --output "$ART/partial_bank_admitted_a7204ef568f2/trench_alignment_audit.json" \
  --workers 12
```

The launch archives are
`trench_aligned_full_bank_a7204ef568f2.tar.zst` (SHA-256 `7a44ea9477d5d4db8ff1ebf6c5325bd9d8ce1d91b74cb925e7b572a1bd44eaa0`)
and `trench_aligned_partial_bank_f25398d3_a7204ef568f2.tar.zst` (SHA-256
`73f3414ae2948be93b3ea03a25a28e570509ccf70221c771a89fdc32915bb4e4`).
The launcher exposes four explicit modes:

```bash
SUBMIT=0 scripts/euler_trench_align_generalist_partial_v1/submit.sh
SUBMIT=stage scripts/euler_trench_align_generalist_partial_v1/submit.sh
SUBMIT=smoke scripts/euler_trench_align_generalist_partial_v1/submit.sh
# Only after the smoke's update-1 checkpoint and completion receipt validate:
SUBMIT=1 scripts/euler_trench_align_generalist_partial_v1/submit.sh
```

## Acceptance and claim limits

Implementation acceptance means the named recipe is reproducible, the bank
and schedule contracts fail closed, and the update-1 smoke passes. It does not
mean partial resets improve final performance.

The current evidence supports feasibility, not causality: capped generation
found relay triplets in 5/12 trench families, explicit in-zone fallback in six
of the seven relay failures, and strict-alignment coverage for the 15 probed
sidecars. `trn-straight-allfree` remained full-start-only in that probe. A
matched partial-on/partial-off study would still be required to isolate the
reset curriculum's contribution.
