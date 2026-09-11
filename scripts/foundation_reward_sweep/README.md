# Foundation and trench reward training

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
