Single-map inference utilities.

Use `inference_single_map.py` from `terra-baselines/` root:

`python inference/inference_single_map.py --policy <checkpoint.pkl> --config trench_excavator --map_name map --n_steps 500 --seed 0`

Notes:
- Default map root is `inference/maps/`; `--map_name` is appended to that path.
- Use `--map_path` to pass an explicit map file/folder and override `--map_name`.
- Use `--config` (same as `visualize_mixed.py`) to enforce the same agent/action setup.
- Add `--use-mcts -sim 32` to plan each inference action with MCTS; without it, inference uses the original PPO action path.
- If `--out_path` is omitted, the GIF name is auto-generated from checkpoint + map + timestamp.
