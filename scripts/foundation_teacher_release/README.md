# Foundation teacher release

One bounded comparison tests whether obsolete foundation guidance restricts
adaptation to the corrected physics. Both arms resume the broad generalist's
native u5000 checkpoint. The control keeps the original teacher cosine. The
treatment fades only the foundation coefficient linearly from 0.85355339 to
zero over 1,250 updates; trench guidance keeps the original cosine to u20000.

PPO, four GPUs x 256 environments, 32 rollout steps, two epochs, 32 minibatches,
model, observations, map bank, reward-v2 and zero behavior costs stay unchanged.
Each update is 32,768 transitions and 64 Adam steps. The first stage costs
40.96M additional transitions per arm. A qualified 1,250-update hold brings the
maximum to 81.92M per arm. The two arms run sequentially on one four-GPU node.
This control tests teacher release; it does not repeat the early-penalty study.

## Run

Use the same Terra revision, container, original 3,840-map training bank and
two teacher files as `terra_generalist_broad_teachers_20260915`. Keep the parent
checkpoint and these inputs read-only. Stage this committed baselines checkout
to a new source directory; leave the original campaign source unchanged.

The cluster container environment needs these absolute paths:

```text
TERRA_ROOT=<original frozen generalist source>/terra
INPUTS_ROOT=<original generalist campaign>/inputs
PARENT_CHECKPOINT=<original segment>/training/checkpoints/generalist-broad-teachers-4672272_update_005000.pkl
BANK_ROOT=<finite-enriched evaluation bank>
EXPERIMENT_ROOT=<new foundation-teacher-release campaign>
```

Use the original `terra-jax+jax24.10-v1.sqsh` container and its `/capstor`,
`/iopsstor` and `/users` mounts. Run:

```bash
sbatch --chdir="$EXPERIMENT_ROOT" \
  --output="$EXPERIMENT_ROOT/slurm-%j.log" \
  scripts/foundation_teacher_release/run_pair.sbatch \
  "$EXPERIMENT_ROOT/release.edf.toml" "$STAGED_BASELINES_ROOT"
```

The Slurm entry point requests one node/four GH200 GPUs for at most six hours.
Check the current maintenance/admission window before submission. It validates
CUDA convolution backward and collective communication in the allocation.
Each trainer fails on nonfinite state and saves regular native checkpoints.
Outputs go to `EXPERIMENT_ROOT/segments/SLURM_JOB_ID`.

The direct trainer copies the actual parent's resolved configuration, changes
only experiment/output/budget fields and prints the differences. To inspect it
before training, use `run.py --dry-run` with the arguments in `run_pair.sh`.
Configuration checks require four visible devices; CPU inspection can use
`JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4`.
`run.py --smoke` is only a 1-GPU x 128-environment, at-most-two-update runtime
check. Its smaller batch does not establish production learning or throughput.
The production path rejects these diagnostic checkpoints as parents.

## Decision

Evaluate the complete unchanged 608-map development panel, greedy450, at
u6250 and u7500. Report 384 foundations, 224 trenches and the included 32-road
subset, all 38 conditions, material progress, paired gains/losses, and behavior
on maps both arms solve. A failed evaluation must stop the experiment rather
than silently permit more training.

The hold proceeds only if:

- Foundation completion is at least the control's.
- At most two control trench successes and one control road success are lost.
- Mean foundation excavation and accepted disposal each lose at most 0.5
  percentage points versus control.
- No evaluated condition loses more than two net completions versus control.

These are explicit engineering criteria for this screen, not significance
tests. Eight additional foundation completions with these retention checks
passing is a useful signal. Another seed is a later decision. The script never
extends past u7500 or promotes efficiency penalties automatically.

Native checkpoint u6250 contains the optimization at index6249, whose
foundation coefficient is about 0.000683. The first hold rollout at index6250
(producing u6251) uses exactly zero. Resuming u6250 restores the original release
origin; it does not start a second fade. Trench KL is about 0.7778 at6250 and
0.6913 at7500.

The existing delayed-cost experiment and unchanged broad continuation to50k
remain separate decisions. Recover their latest checkpoints before allocating
more training. This pilot requires only its already preserved u5000 parent.
