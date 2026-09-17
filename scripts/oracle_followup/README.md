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

On September 17 Lorenzo authorized continuing the evaluated u7500 checkpoint
toward **u100000**. The earlier 2,500-update pilot cap is removed.

`run.py --checkpoint PARENT --inputs INPUTS --output OUTPUT --target-update 100000`
uses an absolute target, with four GPUs, 256 environments per GPU, 32 rollout
steps and 64 Adam steps per update. From u7500 this adds at most 3.03104 billion
transitions. `--dry-run` prints the resolved differences. `--smoke` permits only
one or two new updates at 1x128; those checkpoints cannot seed production.
Ordinary resume retains the time input, capacity, native Adam and release clock.
The LR remains 3e-4, entropy coefficient 0.02, foundation KL zero, and trench
KL keeps its original absolute cosine to zero at u20000. All added costs stay zero.

`run.sbatch CONTAINER.edf.toml BASELINES_ROOT` runs one four-GH200 job as
`lterenzi`; exported roots name existing inputs, a paired Terra snapshot, the
native parent, evaluation bank and output directory. Each allocation lasts up
to 24 hours. Queue three sequential allocations with `afterany:PREVIOUS_JOB_ID`
and export that same ID as `PREVIOUS_JOB_ID`. A successor runs only after a
COMPLETED or TIMEOUT predecessor, selects the newest atomically saved checkpoint
in this campaign, and resumes toward the same absolute u100000 target. It exits
without training if u100000 is already saved, after recovering its evaluation
if needed. Failures or cancellation stop the
chain; no extra allocation is automatically submitted.

Checkpoints are saved every 250 updates. A host callback pauses PPO at u10000,
u20000, u35000, u50000, u75000 and u100000 for the existing fixed 608-case greedy450
panel. The evaluator uses GPU0 in a separate process with a 30-minute timeout;
the parent retains its training state and compiled functions. Results and
explicit success/failure status files go under `EXPERIMENT_ROOT/evaluation`.
Evaluation failures do not terminate training or enable efficiency costs.
An evaluation interrupted at the saved parent milestone is retried on resume.
At u20000 the trainer intentionally compiles its teacher-free path once.

No second learning control or automatic efficiency stage follows. The combined
result cannot identify the separate contribution of time, capacity or release.

Do not use the two-control `foundation_teacher_release/run_pair.sbatch` launcher
for this decision. Recovery of existing delayed-cost outputs remains separate.
