# Combined broad continuation

Lorenzo selected one combined intervention instead of separate learning controls
on September 16. Native model, Adam moments and clocks come from the broad
generalist; environments, action history and RNG restart as in ordinary resume.

- Remaining episode fraction enters both actor and critic through initially
  zero embeddings. The constant-input variant is only a cheap local parity test.
- Add a parallel `704 -> 512 -> 512 -> 8` action branch with zero final output.
  It preserves the initial policy and adds 627,720 parameters; existing weights
  and Adam slots are copied exactly, new moments start at zero.
- Fade foundation guidance from its actual coefficient over 40.96M transitions
  (1,250 updates at the unchanged global batch). Preserve the trench cosine.
- Cache frozen teacher outputs once per rollout. Remove teacher-only geometry
  after both teachers are permanently released.
- Preserve corrected physics, reward-v2, full broad resets, all PPO settings,
  and **zero added behavior costs**.

`run.py --checkpoint PARENT --inputs INPUTS --output OUTPUT --updates 2500`
performs at most 81.92M new transitions, with four GPUs, 256 environments per
GPU, 32 rollout steps and 64 Adam steps per update. `--dry-run` prints the
resolved differences; `--smoke --updates 2` is a local 1x128 runtime check and
its checkpoint cannot seed production. Ordinary resume of a grown checkpoint
automatically retains the time input, capacity and original release clock.

`run.sbatch CONTAINER.edf.toml BASELINES_ROOT` runs one four-GH200 job as
`lterenzi`; exported roots name existing inputs, a paired Terra snapshot, the
native parent, evaluation bank and output directory. It trains once, then
evaluates saved halfway/final checkpoints on the same 608-case greedy450 panel.
No architecture sweep, second learning control, automatic efficiency stage or
automatic extension follows. The combined result cannot identify the separate
causal contribution of time, capacity or teacher release.

Do not use the two-control `foundation_teacher_release/run_pair.sbatch` launcher
for this decision. Recovery of existing delayed-cost outputs remains separate.
