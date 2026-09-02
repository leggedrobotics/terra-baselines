# V8 GRU recurrence probe

This evaluation-only job runs the paired carry intervention on the u44k
concat-skip GRU checkpoint. It uses two identical 120-row chunks in one
process: normal recurrent carry versus carry zeroed before every decision.
The canonical fixed u44k result is an immutable input and the normal target
rows must reproduce it before the probe writes a passed receipt.

Contract: one RTX 4090, horizon 450, seed 20260807, forward chunk 120, greedy
unmasked actions, common Terra runtime `25f855d`, accepted V8 promotion bank,
and W&B disabled. `SUBMIT=0` is local/read-only; `SUBMIT=stage` stages committed
source and validates remote inputs without Slurm submission; `SUBMIT=1`
submits the one-GPU job.

```bash
scripts/euler_v8_gru_recurrence_probe_v1/submit.sh
SUBMIT=stage scripts/euler_v8_gru_recurrence_probe_v1/submit.sh
SUBMIT=1 scripts/euler_v8_gru_recurrence_probe_v1/submit.sh
```

The job is intentionally not chained to training and does not submit itself.
`SUBMIT=1` creates a durable, atomically claimed output reservation before the
Slurm RPC. An ambiguous launcher failure must be reconciled against that claim
and the scheduler; the launcher never deletes it automatically and therefore
cannot silently duplicate the job.
