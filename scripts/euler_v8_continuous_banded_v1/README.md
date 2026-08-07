# V8 continuous all-47 banded curriculum from scratch

This is the single runnable compact deep+xattn treatment specified in
`docs/research/V8_10M_SCALEUP.md`. It starts from random parameters and keeps
all 47 accepted V8 conditions at positive probability for the entire run.

The `continuous_banded_v1` sampler operates independently within the
foundation and trench families, with 50% target assignment probability on
each family. Actual active, reset, transition, and completed-episode exposures
are logged separately because different episode lengths can move them away
from 50/50. Within a family its distribution is:

```text
0.10 * Uniform(all family conditions)
+ 0.75 * Uniform(the entire active depth band)
+ 0.15 * Uniform(the next depth band)
```

At update 0 the two anchors receive approximately 75.43% of total target
assignments, the 13 nearby conditions receive 17.79%, and the 32 constraint
conditions receive 6.78%. Thus every condition is present without turning the
initial distribution into flat all-map training.

Both families start with active depth 0. The active band is the shallowest
depth not fully mastered and includes mastered siblings at that depth. At the
last depth, the active-band term becomes 0.90; once every depth is mastered,
the family is uniform over all of its conditions. These are continuously
updated bands, not checkpoint-bounded curriculum stages. The CLI value
`--accepted-bank-scope full` only selects all 47 bank conditions.
There is no per-condition probability cap; uniform weighting inside a depth
keeps a single weak condition from absorbing the active-band mass.

The sampler refreshes every 150 updates from exact `task_done` outcomes. It
uses an EMA with alpha `0.30`, requires at least 32 completed episodes, promotes
the active depth at `>=0.80`, and demotes any depth below `0.65`. Promotion and
demotion have no consecutive-window streak requirement. Fixed held-out panels
audit and select checkpoints; they never drive the live sampler. Online
training success is weighted by the current dynamic `q` and is not a whole-V8
benchmark score. Only the fixed held-out all-47 panels provide
whole-distribution audit and model-selection evidence.

Everything else remains fixed: dense reward, full 450-step resets, no teacher,
no warm start, and the existing compact deep+xattn model and PPO settings. The
absolute training target is 20,000 updates, with checkpoints every 500 updates
for later fixed-panel evaluation at 1,000-update spacing. Separate fixed
evaluations should be submitted from retained checkpoints during or after an
allocation; they do not run synchronously every 1,000 training updates. If a
training job reaches update 20,000, its launcher also attempts the complete
post-target sweep.

Dry-run and submit a new update-1 smoke from the exact committed source:

```bash
SUBMIT=0 scripts/euler_v8_continuous_banded_v1/submit.sh smoke 20260807
SUBMIT=1 scripts/euler_v8_continuous_banded_v1/submit.sh smoke 20260807
```

Only that same revision's completed smoke can admit the first 24-hour
allocation toward the absolute 20,000-update target:

```bash
SUBMIT=1 scripts/euler_v8_continuous_banded_v1/submit.sh screen 20260807
```

Legacy smoke job `10004034` predates this all-47 continuous contract and is not
valid admission evidence.

If the first allocation stops before update 20,000 and the fixed held-out
trajectory is still improving, resume from an explicit retained checkpoint.
The continuation restores model, optimizer, update/schedule, and the complete
continuous-sampler state; environment RNG and action history restart, so it is
a statistical continuation rather than a bit-exact one.

```bash
SUBMIT=0 scripts/euler_v8_continuous_banded_v1/submit.sh \
  continuation 20260807 \
  /cluster/work/rsl/lterenzi/terra_v8_continuous_banded_v1/<revision>/screen/full/s20260807/G-V8-XATTN-CONTINUOUS-BANDED/checkpoints/<checkpoint>.pkl
```

Use `SUBMIT=1` only after that held-out decision; continuation requests the
120-hour queue but keeps the same absolute update-20,000 target.

Any extension beyond update 20,000 is a separately declared treatment with a
new absolute target; this launcher never extends the experiment silently.
