# Jointly controlled teams and single machines from a generalist parent

Plan, method and results: [docs/research/MULTI_AGENT_JOINT_POLICY_20260923.md](../../docs/research/MULTI_AGENT_JOINT_POLICY_20260923.md).

- `run.py` trains a team of `--agents` tracked excavators, or the machines in
  `--types` (0 excavator, 2 skid steer; all tracked). It takes the parent
  generalist's full recipe (encoder, observations, reward-v2, PPO layout,
  entropy 0.02, LR 3e-4) and changes only the machines. A team warm start makes
  every agent the parent policy; shared parameters keep the parent's Adam
  moments; the intent decoder starts at zero. The behavior costs in effect at
  the parent checkpoint are held fixed. Teachers and demonstrations stay off.
  `--scratch` trains the same recipe from a fresh initialization,
  `--maps-path` selects another bank directory (its `dataset.json` then stands
  for the distance sidecar) and `--ent-schedule START END UPDATES` replaces the
  entropy schedule. `--resume CHECKPOINT` continues toward the absolute
  `--updates` target.
- `evaluate.py` reports first-episode success, steps (rounds) to success,
  final completion and the executed plan per machine (travel between work
  poses, scooped units) for `--agents 1..4` or `--types`, on reset maps chosen
  by `--seed`; the same seed gives the same maps for any team size.
  `--maps-path` evaluates on another bank directory (e.g. a held-out split). A
  single-agent checkpoint with `--agents > 1` is the zero-shot team baseline.
- `compare.py` pairs evaluation results on the same maps: success gained or
  lost, round speedup and executed-plan speedup (0.5 m/s travel, 30 s per
  0.3 m³ scoop by default).
- `render.py` draws the first episodes of a 2 x 2 grid of environments as a GIF.

Cluster entry points (all resume from the newest checkpoint of their
experiment, so a later allocation continues the same run):

- CSCS, one four-GH200 node: `run.sbatch` (one run on four GPUs) or
  `run_pair.sbatch` (two independent two-GPU runs). The EDF environment sets
  `TERRA_ROOT`, `BASELINES_ROOT` (code snapshot), `EXPERIMENT_ROOT`,
  `PARENT_CHECKPOINT`, `TRAIN_BANK_ROOT` (directory holding the bank
  directories), `TARGET_UPDATES`, and optionally `TEAM_AGENTS`, `RUN_NAME`,
  `DEVICES`, `ENVS_PER_DEVICE` and `EXTRA_ARGS` (passed to `run.py`).
  `evaluate_panel.sbatch` and `render_panel.sbatch` run panels on the debug
  partition (see the variables in `evaluate_panel.sh` / `render_panel.sh`).
- Euler, one GPU: `run_euler.sbatch ARM.env` runs one training segment
  (chain 4 h segments with `--dependency=afterany`) and
  `evaluate_euler.sbatch EVAL.env` runs an evaluation panel sequentially. The
  env files add `VENV`, `BANK_ARCHIVE` (a `bank/` tar.zst unpacked into node
  TMPDIR) and `GPU_MODEL` to the variables above.
