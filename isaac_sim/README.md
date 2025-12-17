# `isaac_sim/` tools: extract plans and visualize rollouts

This folder contains helper scripts to run a trained policy on a **single map folder** (e.g. a MoleWorks exported site) and produce:
- a **plan** (`.pkl` + optional `.json`) extracted from **DO** actions (dig/dump waypoints)
- a lightweight **plan GIF** (DO-waypoints only)
- a full **rollout GIF** (every step, movement + DO actions)

## Map folder format

All tools here expect `MAP_DIR` to be a directory containing (at minimum):

```
MAP_DIR/
  images/img_1.npy
  occupancy/img_1.npy
  dumpability/img_1.npy
  distance/img_1.npy      # required by Terra rewards (can be generated)
  actions/img_1.npy       # optional (defaults to zeros if missing in Terra loader)
  metadata/map.json       # optional (trenches-style metadata)
```

## Quickstart: one command (“run policy on a map”)

Use `moleworks_plan_on_map.sh` to:
- pick a checkpoint (latest `checkpoints/*.pkl`, unless you pass one explicitly)
- (optionally) generate `distance/img_1.npy` if missing
- run `extract_map.py`
- write artifacts **into the map folder**

### Example

```bash
cd <path_to_terra_baselines>
./isaac_sim/moleworks_plan_on_map.sh <map_dir>
```

### With an explicit checkpoint

```bash
./isaac_sim/moleworks_plan_on_map.sh <map_dir> <checkpoint_pkl>
```

### Useful environment overrides

```bash
N_STEPS=800 SEED=7 ROLLOUT_GIF_EVERY=2 ./isaac_sim/moleworks_plan_on_map.sh <map_dir>
```

Artifacts written to `MAP_DIR/`:
- `terra_plan.pkl`
- `terra_plan.json`
- `terra_plan.gif`
- `terra_plan_rollout.gif`

Notes:
- The script uses `conda run -n terra ...`, so you do **not** need to `conda activate terra` in your shell.
- If `distance/img_1.npy` is missing, the script tries to auto-run `../terra/tools/generate_distance_maps.py --dataset MAP_DIR`.

## Advanced: use `extract_map.py` directly

`extract_map.py` is the underlying Python entrypoint. It runs a rollout on a **single map** and extracts a waypoint plan from DO actions.

### Example

```bash
cd <path_to_terra_baselines>
conda run -n terra python -u -s isaac_sim/extract_map.py \
  --policy_path <checkpoint_pkl> \
  --map_path <map_dir> \
  --n_steps 400 \
  --seed 0 \
  --output_path <output_plan_pkl> \
  --serialize \
  --render_plan_gif \
  --render_rollout_gif \
  --rollout_gif_every 2
```

### Key flags

- `--policy_path`: checkpoint `.pkl` produced by training (`train_mixed.py` / `train.py`).
- `--map_path`: map directory (`MAP_DIR`) with `images/img_1.npy`, etc.
- `--n_steps`: max rollout length.
- `--seed`: RNG seed for action sampling.
- `--output_path`: where to write the plan `.pkl` (the `.json`/`.gif` names are derived from this).
- `--serialize`: also write JSON (`.json`) next to the `.pkl`.
- `--render_plan_gif`: write a lightweight DO-waypoints plan GIF.
- `--render_rollout_gif`: write a full rollout GIF using Terra’s renderer.
- `--rollout_gif_every`: subsample rollout frames for smaller GIFs.

### Output naming

If `--output_path /some/path/foo.pkl`:
- plan pkl: `/some/path/foo.pkl`
- plan json: `/some/path/foo.json` (with `--serialize`)
- plan gif: `/some/path/foo.gif` (with `--render_plan_gif`)
- rollout gif: `/some/path/foo_rollout.gif` (with `--render_rollout_gif`)

### If the policy never “does” (0 DO actions)

Some rollouts produce 0 DO actions (no dig/dump). In that case:
- the plan `.pkl` will be tiny / empty (0 waypoints)
- the plan GIF will still be created as a single base-map frame
- the rollout GIF will still be created

## Coordinate convention (important)

Terra uses `pos_base = [x, y]` where:
- `x` is the **first** map index (axis 0, “rows”, increases **down** in images)
- `y` is the **second** map index (axis 1, “cols”, increases **right** in images)

So `(x=0, y=0)` is the **top-left** tile.
