# Foundation and trench reward training

## Current recipe: learn completion before adding costs

The September 12 decision supersedes applying behavior costs from scratch.
Continue the corrected zero-cost foundation and trench controls separately,
preserving their native model, Adam state and absolute update/entropy clocks.
Keep the current banks, PPO settings, executable fresh-dig observation and
Terra 46738cde environment, including the hard trench-alignment rule and soil
protections. This is a continuation of the two controls, with no new scratch
or generalist run. Do not renew the early cost arms that suppress excavation.

Introduce costs only after two successive retained fixed-panel evaluations,
at least 2,500 updates apart, meet 90% exact completion: at least 58/64
foundation episodes or 202/224 trench episodes. Foundation uses the same
64-map validation panel; trench still evaluates the full 608 development
episodes and uses its 224 trench rows for this decision. Keep greedy decoding,
the 450-step horizon, reset identities and recorded integrity checks. Online
training success and excavation percentage do not substitute for exact
fixed-panel completion. At the latest evaluated u10000, the controls achieve
1/64 and 20/224 respectively, so neither is eligible yet.

Freeze the latest qualifying zero-cost checkpoint and evaluation as the
reference for every later stage. Both evaluations must also stay within three
percentage points of that reference (at most one fewer foundation success or
six fewer trench successes), while remaining above 90%. This limit always uses
the original zero-cost reference, so successive stages cannot accumulate losses.

| Fraction of previous combined 2x | Lateral dig cost | Travel cost / m | Turn cost / rad |
| --- | ---: | ---: | ---: |
| 0%: learn completion | 0 | 0 | 0 |
| 25% | 0.125 | 0.0025 | 0.01 |
| 50% | 0.25 | 0.005 | 0.02 |
| 100% | 0.5 | 0.01 | 0.04 |

Each cost stage holds its costs constant for 5,000 additional updates, with
evaluations after +2,500 and +5,000. Increase only after both pass the same
completion criterion; if they fail, do not increase. The two task families
qualify independently. Retain a zero-cost sibling from each accepted parent
and compare at equal additional updates before claiming an efficiency benefit.
Check native resume for two finite updates and complete the CSCS runtime check
before the first full stage. The trainer's PPO loop is unchanged.

`finetune_after_completion.py` checks the reports, the native parent checkpoint,
its actual optimizer count and the existing training bank. Without `--execute`
it prints the decision and, if eligible, the native training arguments; it
performs no training or Slurm submission. Exit status 3 means the completion
criterion was not met. Use the Terra Python environment and an explicit bank
root with the original distance metadata:

```bash
python scripts/foundation_reward_sweep/finetune_after_completion.py \
  --family foundation \
  --previous-evaluation /path/to/previous_validation.json \
  --evaluation /path/to/latest_validation.json \
  --reference-evaluation /path/to/latest_validation.json \
  --checkpoint /path/to/latest_evaluated_checkpoint.pkl \
  --dataset-root /path/to/existing_foundation_bank \
  --run-dir /path/to/new_foundation_p25_run
```

The initial reference is the latest qualifying zero-cost evaluation. For later
increases, keep that same `--reference-evaluation`, supply the two evaluations
from the current cost stage, and add
`--parent-stage /path/to/current_stage_run/penalty_stage.json`. Run the checked
command with `--execute` only inside the assigned GPU allocation. It creates
the fresh output directory and `penalty_stage.json` before invoking the normal
trainer. Copy that file with the checkpoint when moving or continuing a stage.
The launcher sets `INITIALIZATION=resume`, `BANK_TRANSFER=0` and
`BEHAVIOR_FINETUNE=1`; the launcher keeps the digging observation fixed and
permits the cost change through the existing native continuation interface. A same-stage walltime
continuation uses `train.sh` with the same costs and actual checkpoint update.

Judge completion first, then compare productive base poses, unique required
area per setup, retained pose-to-pose distance, workspace edge adjacency,
lateral fresh-dig volume and loose-soil relifts on common successful episodes.
Raw navigation action count is secondary because deployment delegates travel
between retained work poses to the navigation stack; workspace continuity still
matters. No stage is accepted on reduced movement alone. These development and
validation panels are used for tuning; they do not establish test performance.

The recipe is prepared; no delayed-cost job has been submitted. CSCS access
must be restored and newer control checkpoints evaluated before selecting a
parent. See the [research decision](../../docs/research/FOUNDATION_BEHAVIOR_20260907.md)
and [current experiment state](../../docs/EXPERIMENTS_RUNNING.md).

## Historical September 11 scratch cost screen

The launcher runs four independent scratch policies on one CSCS Daint node,
one GPU each. Set `COST_SUITE` explicitly in the container environment:

| Suite | Foundation arms | Trench arms |
| --- | --- | --- |
| `paired` | control and combined 2x costs | control and combined 2x costs |
| `components` | lateral-only and relocation-only | lateral-only and relocation-only |

Control uses zero behavior costs. Lateral-only uses 0.5 lateral cost and zero
travel/turn costs. Relocation-only uses zero lateral cost, 0.01 per metre and
0.04 per radian. Combined uses all three nonzero costs. All arms use seed
20260909 and a 24-hour allocation: at most 96 GPU-hours per suite, or 192 for
the eight-arm comparison including startup. Node differences limit factorial
interaction claims.

The September 11 replacement restarts both suites with the corrected straight
movement environment. The old ba9cc214 control/2x allocation timed out; its
checkpoints remain historical evidence. Its never-started component job was
cancelled so it cannot train on the known movement bug. The fixed environment,
executable fresh-dig observation, data and PPO settings are shared within the
new comparison. No generalist training is included. The
[restart plan](../../../../../.artifacts/terra_movement_restart_cscs_20260911/PLAN.md)
records the source pair, smoke checks and submissions.

`train.sh` supports explicit `INITIALIZATION=scratch`. It requires update zero,
no `RESUME_FROM`, and `BANK_TRANSFER=0`; model, Adam and exploration start fresh.
`INITIALIZATION=resume` preserves the existing native transfer/continuation
interface. After the first two scratch updates, ordinary continuation restores
that new checkpoint with `BANK_TRANSFER=0`. It imports no historical policy.

`TASK_FAMILY=foundation` (default) uses `foundation_reward_sweep`, 256 maps in
`train/all`. `TASK_FAMILY=trench` uses `trench_align_v2_specialist_spec`, 1,440
maps in `train_v2_pooled_trench15`. Each family receives its own bank root and
distance protocol receipt. The trench bank includes junctions and constrained
dumping; it is trench-only but is not an unconstrained easy-map bank.

`cscs_scratch.sbatch CAMPAIGN_ROOT` checks the four-GPU runtime, then starts
`run_cscs_scratch_arm.sh` as four Slurm tasks, each bound to one GH200. Each
checks its own two fresh PPO updates before starting production. The job fails
if a rank fails. Logs, checkpoints, episode receipts and offline W&B histories
are separated by Slurm job and arm. Checkpoints are retained every 500 updates.
The 500,000-update target exceeds one allocation; walltime bounds the segment.
`verify_scratch_smoke.py` takes the three expected costs explicitly through
`--lateral-dig-cost`, `--base-travel-cost`, and `--base-turn-cost` and checks them
in both scratch checkpoints, together with the actual Adam counts and finite
model, optimizer, loss and transition-integrity state.

For later segments, use each latest complete checkpoint, the same task/seed/costs,
`INITIALIZATION=resume`, `BANK_TRANSFER=0`, the actual `START_UPDATE` and a fresh
output directory. Do not resubmit `cscs_scratch.sbatch` to continue a trained run;
that script intentionally initializes all four arms from zero.

Evaluate retrieved checkpoints separately with
`BANK_ROOT=... bash eval.sh CHECKPOINT OUTPUT.json validation`. Keep the same
64 initial episodes, greedy decoding and 450-step horizon; compare equal update
counts. Report square/rectangle/L through `by_primary_cell`. Success and task
progress come first; compare work-pose efficiency and adjacency on common
successes. The easy bank cannot establish generalist or constrained-dumping
performance. For trenches, use `../excavation_reliability/eval.sh`, preserving
the full 608 development panel and its original reset keys, and report the
224 trench rows (14 conditions) separately. The train-only straight-allfree
condition is not in this development panel. Do not pool outcomes across task
families or tune on a sealed/test panel. Each task has one paired seed; this
screen does not measure variance across training seeds.
