# Terra network capacity and representation follow-up

The combined continuation keeps the trained spatial encoder and adds a parallel
actor head. This is the most direct capacity increase for the current compute
budget: the actor is much smaller than the critic, and extra fully connected
layers cost little next to processing the 64×64 map. The user explicitly chose
one combined training run, accepting that its behavioral outcome will not
isolate the effect of time input, teacher release, and actor growth.

This is a candidate improvement, not evidence that parameter count caused the
remaining failures. Exact completion and continuous material metrics still
decide whether the combined recipe helps. Added efficiency costs stay off
until competence is established.

## Current checkpoint and chosen growth

The broad u5000 checkpoint has 2,311,701 parameters. Its relevant branches are:

| Branch | Structure | Parameters |
| --- | --- | ---: |
| Spatial encoder | 24/48/64/96 channels; 2/2/3/3 residual blocks; 64 spatial tokens; two mixers; flatten plus nine-query attention readout | 1,629,852 |
| Actor | 704→160→48→8 | 120,920 |
| Critic | 704→512→256→1 | 492,545 |

`actor_residual_head=True` adds a separate **704→512→512→8** MLP from the
actor's fused map, pose, local-workspace, history, and optional time features.
Its output is added to the existing logits. The two hidden layers use ReLU;
the final kernel and bias start at exactly zero. The old actor remains present
and trainable. The critic already has the wider head and is retained.

The new head adds **627,720 parameters**, taking actor capacity from 120,920 to
748,640 parameters. The two time embeddings add 1,408 parameters, for a total
of **2,940,829**. Model weights plus the two fp32 Adam moment arrays increase
by about **7.20 MiB per GPU**. Activations, gradients, compiled executable
buffers and allocator overhead are separate from that figure.

JAX 0.4.33 CPU lowering on the actual checkpoint reports **529.934 million →
531.193 million forward FLOPs per sample**, an increase of **0.237%**. Parameter
count rises 27.2% because the added dense weights are used once per sample,
whereas convolution weights are reused at many map locations. This arithmetic
estimate does not establish GPU latency, backward cost, memory fit or training
throughput. The combined finite CUDA update and synchronized warmed timing
remain the runtime check.

Actual checkpoint migration on the saved slot 503 snapshot gave exactly zero
logit and value differences before any updates. The CPU tests additionally
verify all existing model leaves, Adam moments, Adam count, train-state step and
native update clock remain unchanged. New Adam moments are zero. The zero
output projection receives a gradient immediately; after its first update,
gradients reach both hidden layers. The old parameters remain trainable.

Because the shared Adam count is mature, the first update of a new leaf with
zero moments can have a normalized step around 3.16 times the nominal learning
rate. Initial function preservation does not imply unchanged gradients or
unchanged first-step behavior. Inspect the combined smoke's residual logits,
PPO KL, clip fraction and finite checks before considering it runnable.

The topology is explicitly recorded as `actor_residual_head`; no map-encoder
canonical name or existing parameter meaning changes. The one-way migration
helper rejects a changed existing tree, a nonzero new output projection, or
missing native optimizer state. Ordinary subsequent resume uses the saved
new tree and native clock.

## Measured finite startup

The combined one-GPU smoke completed two updates with finite model, optimizer,
rollout and teacher state. It used 128 environments and 32 rollout steps,
with two epochs and 32 minibatches: an eight-times smaller global rollout than
the production 4×256 configuration. Adam continued 320,000→320,128 while the
checkpoint clock advanced u5000→u5002. This is runtime evidence; the smoke
checkpoints are rejected as parents by the production launcher.

| Same 1×128 smoke layout | u5001 approximate PPO KL | u5002 approximate PPO KL |
| --- | ---: | ---: |
| Prior control | 0.08476 | 0.11994 |
| Prior foundation-release smoke | 0.08492 | 0.18397 |
| Combined time, actor growth, release and cache | 0.09393 | 0.11106 |

These short checks do not establish growth-specific startup instability. The
production parent's 0.01494 KL used a different batch size and is not a direct
comparator for these smoke updates. They also do not establish improved or
preserved broad-map completion.

The learned residual branch was separately inspected on saved failure snapshots
4, 296, 503 and 564. Its maximum absolute added logit was 0.108 at u5001 and
0.082 at u5002. Removing only that branch from the same updated policy changed
the action distribution by mean total variation 0.0146 and 0.00658 respectively;
none of the four greedy actions changed at either checkpoint. The branch is
learning and does not dominate these examples. This is a small function probe,
not a rollout assessment or an estimate over the broad distribution.

Measurements are in `.artifacts/terra_oracle_combined_20260916/`:
`smoke_metrics.json`, `smoke.log`, and `actor_residual_probe.json`. Warmed GPU
throughput is measured separately; the second smoke update alone is not a
stable performance comparison.

## Why this growth before a larger encoder

Three alternatives were considered:

| Change | Capacity/compute implication | Decision |
| --- | --- | --- |
| Parallel actor head | +627,720 parameters, ~0.63 million matrix MACs per sample; preserves old output | Prepare for the combined run |
| Two more identity-initialized 96-channel token mixers | +168,192 parameters, ~12.2 million matrix/attention MACs per sample; adds spatial interaction depth | Keep as a later option if spatial relation diagnostics justify it |
| Increase all convolution widths / `model_size=large` | Many old leaves change shape; convolution work scales roughly with products of adjacent widths | Do not pay this cost without evidence of an encoder bottleneck |

The 160-dimensional map output still limits what reaches either actor head.
The new actor cannot reconstruct information discarded by the observations or
encoder. More actor capacity also provides no explicit search, planning state
or long-term memory. It can represent richer decisions from the current
features; learning useful excavation sequences still depends on experience,
credit assignment, and teacher guidance.

## Cheap probes on real failure states

`scripts/analysis/terra_representation_diagnostics.py` uses stored physical
snapshots and their original previous-action histories. It refreshes Terra's
observation wrappers after a counterfactual cell edit, then compares the
preprocessed inputs, map embedding, policy probabilities, greedy action and
critic value. It can also be run on later time-aware checkpoints.

The cell edits are representation probes, **not legal mass-conserving soil
transitions**. Sensitivity means a signal reaches the network; it does not
establish that the resulting action is correct or that a legal suffix exists.
The 12 selected failures are deliberately diverse and are not a prevalence
sample.

The frozen u5000 checkpoint produced the following results across 64 probes:

| Probe | Cases | Identical model inputs | Mean policy total variation | Mean absolute value change |
| --- | ---: | ---: | ---: | ---: |
| Remove nearest single residual cell | 10 | 0 | 0.0523 | 0.0892 |
| Remove furthest single residual cell | 9 | 0 | 0.0752 | 0.2574 |
| Increase furthest positive pile by one unit | 9 | 9 | 0 | 0 |
| Change episode age only to 50, 440 or 449 | 36 | 36 | 0 | 0 |

One of the 19 residual probes changed the greedy action. These cases do not
show a general inability of the encoder to notice a one-cell residual. They do
show exact aliasing of the tested distant pile amounts and episode ages. Added
capacity cannot separate identical inputs. The time input addresses the age
aliasing. The pile result remains a separate observation issue; this continuation
does not silently change global height preprocessing or native teacher inputs.

Artifacts, outside the source tree:

- `.artifacts/terra_oracle_followup_20260916/representation/u5000.json`
- `.artifacts/terra_oracle_followup_20260916/representation/capacity_preview.json`

Example invocation from this checkout, with the paired Terra on `PYTHONPATH`:

```bash
JAX_PLATFORMS=cpu python scripts/analysis/terra_representation_diagnostics.py \
  --checkpoint "$CHECKPOINT" \
  --audit-dir "$FAILURE_AUDIT" \
  --output "$OUTPUT_JSON"
```

## Task-gradient interference diagnostic

`utils/task_gradient_diagnostics.py` accepts a **real production minibatch**:
preprocessed model inputs, sampled actions, old log probabilities and values,
targets, production-normalized advantages, family labels and selected frozen
teacher logits. It measures foundation/trench PPO and KL gradients only through
the shared `maps_net` subtree. The PPO objective includes the current clipped
actor loss, optional clipped value loss and entropy. The helper is restricted
to the current unmasked feedforward recipe without auxiliary losses or value
distillation.

It reports conditional gradient norms, transition-weighted contribution norms,
teacher-coefficient-weighted KL norms, and family/teacher gradient cosines.
With `axis_name`, it reduces actual gradients across devices before computing
norms and cosines. Advantages must already have been normalized by the
production update; re-normalizing separately by family would change the
question. Missing families produce unavailable conditional norms/cosines,
rather than a fabricated zero-interference result.

This helper is not called on every update: four reverse passes would undermine
the throughput work. Invoke it on a saved complete minibatch or at an explicitly
rare diagnostic checkpoint. Tests on four virtual CPU devices verify equivalence
to the complete global minibatch despite 6:2 family imbalance and devices with
no trench rows. **No real training-gradient measurement has been collected yet**;
the tests only establish the reduction and weighting behavior.

## Throughput review

Caching selected frozen teacher logits and values once per rollout preserves
native teacher observations, action history and family labels. Chunking uses
the existing PPO minibatch size, and cached leaves follow ordinary shuffling.
The cache costs about 1.125 MiB per global rollout at 32,768 samples, including
the value retained for diagnostics. It saves repeated inference across the two
PPO epochs; it does not imply doubling end-to-end throughput.

After all teacher coefficients permanently reach zero, a separate static
teacher-free update graph removes teacher-only observation construction as
well as the already-skipped loss inference. This should cause one intentional
additional compile at release, not a compile every update. Throughput reports
must include that transition and distinguish warmed execution from startup.

Family-bucketed teacher inference, reset compaction, geometry reuse and static
reward specialization remain profiling candidates. The source alone does not
justify rewriting their semantics. Batch layout, precision, environment count,
PPO epochs and rollout length remain learning settings rather than free runtime
optimizations.

## Warmed local GPU measurements

The RTX 4090 benchmark uses one saved physical state repeated across each
batch, JAX 0.4.33, three warmups and 15 synchronized repetitions per kernel.
Compilation is excluded. Initial native/grown outputs and selected live/cached
teacher outputs match exactly on GPU.

| Kernel | Batch 128 parent / grown | Batch 256 parent / grown |
| --- | ---: | ---: |
| Forward | 2.979 / 2.998 ms | 5.695 / 5.677 ms |
| PPO-like forward and gradient | 11.699 / 11.967 ms | 22.196 / 22.533 ms |

At batch 256, the added capacity costs about 1.5% in the measured gradient
kernel; inference is effectively unchanged. These are local kernel timings,
not GH200 scaling, full PPO-update throughput or evidence of better behavior.
Teacher caching retains exact outputs and halves the number of teacher
inferences across two PPO epochs; it does not imply twice the training speed.

Executable-observation refresh takes 3.960 ms for 128 states and 6.006 ms for
256 states, with reachability refresh disabled as in this probe. That makes
geometry reuse worth investigating, but this measurement does not isolate
which predicates can be reused, and does not include reset batching. A warmed
XProf/Perfetto trace is saved under the combined artifact directory. No
speculative reset/RNG rewrite or geometry shortcut was included.

The complete combined CUDA smoke reaches u5002 / Adam320128 and native resume
reaches u5003 / Adam320192 with original migration/release origins preserved.
All parameters, optimizer slots, sampled rollout/teacher tensors and checked
losses are finite. The smoke's 1x128 population is smaller than production and
must never become a production parent. Production remains 4x256.

Evidence: `.artifacts/terra_oracle_combined_20260916/benchmark_b128.json`,
`benchmark_b256.json`, `smoke_metrics.json`, `resume/result.json`, and `profile/`.
