# Foundation reward training

The current experiment starts four independent policies from scratch on one
CSCS Daint node: control and 2x costs at seeds 20260909 and 20260910. Allocation
is 24 hours, at most 96 GPU-hours including startup checks. The repaired Terra
environment and executable fresh-dig observation are shared by all four arms.
The [experiment plan](../../../../../.artifacts/terra_foundation_scratch_cscs_20260909/PLAN.md)
records the data, evaluation rules, local evidence and launch status.

`train.sh` supports explicit `INITIALIZATION=scratch`. It requires update zero,
no `RESUME_FROM`, and `BANK_TRANSFER=0`; model, Adam and exploration start fresh.
`INITIALIZATION=resume` preserves the existing native transfer/continuation
interface. After the first two scratch updates, ordinary continuation restores
that new checkpoint with `BANK_TRANSFER=0`. It imports no historical policy.

`cscs_scratch.sbatch CAMPAIGN_ROOT` checks the four-GPU runtime, then starts
`run_cscs_scratch_arm.sh` as four Slurm tasks, each bound to one GH200. Each
checks its own two fresh PPO updates before starting production. The job fails
if a rank fails. Logs, checkpoints, episode receipts and offline W&B histories
are separated by Slurm job and arm. Checkpoints are retained every 500 updates.
The 500,000-update target exceeds one allocation; walltime bounds the segment.

For later segments, use each latest complete checkpoint, the same seed/costs,
`INITIALIZATION=resume`, `BANK_TRANSFER=0`, the actual `START_UPDATE` and a fresh
output directory. Do not resubmit `cscs_scratch.sbatch` to continue a trained run;
that script intentionally initializes all four arms from zero.

Evaluate retrieved checkpoints separately with
`BANK_ROOT=... bash eval.sh CHECKPOINT OUTPUT.json validation`. Keep the same
64 initial episodes, greedy decoding and 450-step horizon; compare equal update
counts. Report square/rectangle/L through `by_primary_cell`. Success and task
progress come first; compare work-pose efficiency and adjacency on common
successes. The easy bank cannot establish generalist or constrained-dumping
performance. Both costs and two seeds are paired; this is a bounded sensitivity
screen, not strong statistical confidence.
