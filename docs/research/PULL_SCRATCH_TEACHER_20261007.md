# Scratch student with compatible bulk teacher guidance

This is a separate, user-authorized experiment. The warm-start comparison is
CSCS job 4994593 / W&B `qbh06dnp`; its source snapshot and running process are
unchanged. The new experiment uses branch `pull-direction-scratch-teacher` in
`.worktrees/terra_pull_scratch_teacher_20261007/{terra,terra-baselines}`.

Both the actor and critic start randomly. GRU110000 is loaded only as a frozen
teacher, with its own observation preprocessing and recurrent carry. Before
the first optimizer update, the trainer compares same-shaped actor and critic
arrays with the teacher and writes `scratch_initialization.json`. A larger
input layer alone is not accepted as proof of scratch initialization.

The seed 20261006, architecture, 20,480-slot training bank, reward coefficients,
450-action horizon and machine settings match the warm-start experiment.
Half of each device's lanes remain precision episodes, sampling the same 95
geometrically screened broad foundations. Bulk lanes sample the full bank.
All lanes use 2.5 m cutting space. Precision adds a 0.6 m band and 25-degree tangent
tolerance. The student sees the precision-required band explicitly.

Teacher policy KL starts at 1 and fades to 0 over 3,000 updates. Value imitation
is 0. Its candidate pool contains all 12,800 training foundation slots, covering
4,428 excavation sources and 25 conditions. This is a geometry-based candidate
pool, not a claim of teacher competence on every map. Trenches and precision
episodes never receive teacher KL.

At each pre-action state, the teacher sees the same terrain, pose and history
through its saved environment settings and the original generator ABC axes.
For an empty bucket, the old and current DO kernels must select exactly the
same cells, volume, material type and admission. Two allowed actions with
different excavation footprints are excluded. Loaded DO remains compatible
after startup checks that machine and dump settings match. This establishes
immediate action compatibility; future routing and spoil placement can still
be poor. The teacher's carry advances on every lane and resets only when that
episode ends, including while its KL is masked off.

Cached logits and the eligibility label follow recurrent PPO sequence
shuffling. The KL mean uses the globally selected transition count across
devices; no eligible transitions gives exactly zero loss and gradient.
`kickstart/eligible_transition_fraction` reports exposure over all transitions;
`kickstart/bulk_compatible_fraction` reports compatibility among candidate bulk
transitions. Bulk and precision task completion remain separate metrics.

The launch uses one seed on four GH200s, 512 environments per device, 32 rollout
steps, two PPO epochs and 32 minibatches, a 24-hour limit, and checkpoints every 100
updates. It is a comparison of two complete training recipes: initialization
and teacher guidance both change, so a difference cannot be attributed solely
to initialization. Compare matched environment transitions on the same fixed
bulk and feasible precision evaluation panels; training success alone is not
held-out evidence. No offline demonstrations are used.

Before launch, focused native tests cover changed-footprint rejection, legacy
teacher observation restoration, carry/reset behavior, selected-count loss
normalization and true scratch initialization. A 32-episode GPU teacher probe
uses eight predeclared, distinct training sources with four fixed starts each.
It reports compatibility exposure and progress; it does not require full-map
completion to admit the teacher as a temporary prior. A separate actual PPO
update must save and reload finite model, optimizer and rollout state, with
positive eligible teacher exposure. Compilation alone is not a passed smoke.

The existing precision limitations remain: 95 maps have static coverage, one
synthetic two-row trench has a complete native tape, and the broad foundation
manual tape is incomplete. Physical ramp heights and bucket swept volume are
not simulated. Do not describe the precision pool as fully native-qualified.

Scripts: `scripts/pull_scratch_teacher/`. Local evidence and exact launch
files: `.artifacts/terra_pull_scratch_teacher_20261007/` in Moleworks. Subsequent
segments must resume saved optimizer/update state, not restart from scratch.
