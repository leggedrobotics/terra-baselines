# V8 10M nearby-stage long run

This is the paired Stage-B continuation of the compact and 10M deep+xattn
policies. Parents are selected from corrected whole-V8 promotion evaluation;
development is reported but never promotes a checkpoint.

The frozen training population is:

- 15 conditions and 96 layouts per condition: 1,440 distinct training maps;
- 25% random full resets on the two mastered all-free anchors;
- 75% random full resets on the 13 nearby geometry conditions;
- 50/50 foundation/trench mass within each slice; and
- no per-environment promotion, demotion, partial reset, or reward transition.

Both arms use the independently trained full-V8 compact update-7,500 checkpoint
as one frozen teacher on their current Stage-B rollout observations. Its
source contract, checkpoint, and four fixed whole-V8 evaluation files are
hash-pinned and revalidated in every job. It reaches `538/720` exact and
`0.868` macro completion on development plus `31/32` on the separate
capability panel. Policy KL `1.0 -> 0` over 3,000 updates and value imitation
`0.5 -> 0` over 1,000 updates are identical across arms and are not a reward
fade.

Both arms receive 20,000 updates (`1,310,720,000` transitions), retain every
1,000-update checkpoint, and are evaluated on the fixed 720-map main and
32-map capability promotion/development panels. Two consecutive anchor
retention failures trigger rollback in the gate receipt. Stage C and reward
fading remain locked.

```bash
SUBMIT=0 scripts/euler_v8_10m_stage_b_v1/submit.sh \
  smoke 20260730 /absolute/path/to/stage_b_selection.json
```

Set `SUBMIT=1` only after the selection receipt and dry run validate. Submit
the screen only after both update-1 smokes pass.
