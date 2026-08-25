# V8 GRU fixed benchmark

This evaluation-only campaign compares the concat-skip GRU and feed-forward
relay policies on one common frozen environment and promotion panel. It does
not train, resume, modify a checkpoint, or write to W&B.

The four pinned checkpoints answer two practical questions:

- GRU u40k to u44k: did the recurrent policy improve over the preceding 4k
  updates before the latest durable u44k milestone?
- GRU u44k versus FF u44k and FF u86k: how does the recurrent pilot compare at
  matched environment transitions and against the feed-forward frontier? Both
  treatments collect 65,536 transitions per update, despite using different
  GPU counts, so the two u44k checkpoints are also transition-matched.

The GRU and FF checkpoints are deliberately passed to `eval_fixed_bank.py` in
two separate invocations. The evaluator requires every checkpoint within one
invocation to share a treatment fingerprint; combining the architectures in a
single invocation would correctly fail that gate. Both invocations still run
sequentially in the same Slurm job, on the same GPU, evaluator, Terra runtime,
bank, horizon, seed, and forward chunk.

Contract:

- 1 NVIDIA GeForce RTX 4090;
- greedy, unmasked PPO at horizon 450;
- fixed 720-map promotion panel and forward chunk 120;
- evaluator source: the committed launcher checkout;
- common Terra runtime: `25f855db3d913fd638c4e56b1740437a2b7122ca`;
- accepted-bank archive:
  `b04513ffd1d6a33721802538f76b521bddc81fac492e0ad923ce790d0edec725`;
- promotion manifest:
  `dbfbe56307a5c3a10eaad3d9fa3d4b2a90fb13a3f3593de4fa1dd551e1d8a826`;
- exact completion and reward-v2 protocol gates enabled; and
- W&B disabled.

Dry-run locally without SSH or Slurm mutation:

```bash
scripts/euler_v8_gru_fixed_benchmark_v1/submit.sh
```

Stage the committed evaluator source without submitting:

```bash
SUBMIT=stage scripts/euler_v8_gru_fixed_benchmark_v1/submit.sh
```

Submit the pinned evaluation:

```bash
SUBMIT=1 scripts/euler_v8_gru_fixed_benchmark_v1/submit.sh
```

This is a capability comparison, not a recurrence-only ablation. Training
seed, GPU count, and the feed-forward run's effective training runtime differ.
The common evaluation runtime removes inference-time environment differences,
but it cannot remove those training confounds. The promotion panel is also a
reused development instrument, not an untouched paper test.

The job additionally emits three offline dashboards:

- matched-update `FF-u44` versus `GRU-u44` with 40 deterministic review GIFs;
- GRU acquisition from u40 to u44; and
- `GRU-u44` versus the `FF-u86` frontier.

The media renderer repeats the full 720-row inference for each matched policy,
uses the canonical chunk size, freezes completed states on a uniform ten-step
frame grid, and rejects the output unless exact outcome, episode length,
terminal material, and aggregate no-effect counts reproduce the fixed record.
