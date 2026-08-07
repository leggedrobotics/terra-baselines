# V8 compact curriculum retraining from random initialization

This is the first runnable step of the fixed banded-curriculum experiment in
`docs/research/V8_10M_SCALEUP.md`.

The first job deliberately trains only Stage A:

- random compact deep+xattn initialization;
- the two all-free capability conditions, 50/50 by family;
- dense reward and full 450-step resets;
- no external teacher, KL loss, partial resets, or adaptive sampling; and
- 6,000 updates with checkpoints every 500 updates.

Stage B is not launched by this job. It may start from the selected Stage-A
checkpoint only after both source-disjoint capability panels pass `12/16` per
condition at two consecutive checkpoints. Stage B then uses the fixed
`banded_preview15_v1` population: 10% capability replay, 75% nearby frontier,
and 15% constraint preview.

Dry-run and submit:

```bash
SUBMIT=0 scripts/euler_v8_banded_scratch_v1/submit.sh smoke 20260807
SUBMIT=1 scripts/euler_v8_banded_scratch_v1/submit.sh smoke 20260807
```

The 6,000-update screen is accepted only after the update-1 smoke completes:

```bash
SUBMIT=1 SMOKE_REVISION=<revision> \
  scripts/euler_v8_banded_scratch_v1/submit.sh screen 20260807
```
