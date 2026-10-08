# Inference and visualization

Policy inference belongs in this repository; saved-recording rendering belongs
in Terra. The usual 3D workflow is to capture a native episode, optionally
postprocess its plan, then render the saved recording:

```bash
terra-postprocess render episode.json.gz --out episode.html
terra-postprocess render episode.json.gz --out episode.mp4
```

The [Terra media workflow](https://github.com/leggedrobotics/terra/blob/main/terra/postprocess/README.md)
owns installation, 3D HTML/MP4/GIF export, postprocessing and galleries. Its
[recording guide](https://github.com/leggedrobotics/terra/blob/main/terra/viewer3d/README.md#record-from-a-rollout)
documents `ReplayRecorder.append` for sequential steps and `append_joint` for
explicit joint transitions. Both use states from the native evaluator.

There is no generic checkpoint-to-3D command here. Use the adapter matched to
the checkpoint's model, recurrent state, joint/sequential action semantics,
environment settings, reset cohort and RNG. Recording helpers do not infer
those settings. A fresh CPU capture is a new episode, not proof that the exact
historical GPU evaluation was replayed. Rendering an existing recording does
not run inference.

## Available pipelines

| Purpose | Entry point | Output / scope |
| --- | --- | --- |
| Render an existing native or processed recording | `terra-postprocess render` | 3D HTML, MP4 or GIF; no inference |
| Batch policy inspection | [`visualize_mixed.py`](../visualize_mixed.py) | 2D GIF showing a grid of rollouts from compatible checkpoints |
| Single-map policy inspection | [`inference_single_map.py`](inference_single_map.py) | 2D rollout GIF; sampled/greedy PPO or optional MCTS |
| Agent path inspection | [`visualize_paths.py`](../visualize_paths.py) | Single-episode GIF with colored paths and movement statistics |
| V8 fixed-panel review | [`scripts/render_v8_fixed_panel_gifs.py`](../scripts/render_v8_fixed_panel_gifs.py) | Selected episode GIFs and traces from the deterministic, unmasked promotion panel |
| Export a plan for ROS conversion | [`isaac_sim/extract_map.py`](../isaac_sim/extract_map.py) | Single-map PKL and schema-v2 JSON; optional DO-waypoint GIF (`--render_plan_gif`) and rollout GIF (`--render_rollout_gif`) |
| Current joint-fleet capture | Runtime-specific evaluator plus `ReplayRecorder.append_joint` | Native endpoint recording; exact fleet postprocessing additionally uses `NativeFleetRecorder` substeps |

The fixed-panel renderer runs the complete 720-row cohort with canonical
120-row inference chunks, matching checkpoint, manifest, reset/RNG protocol and
full-panel terminal/material/no-effect results before exporting selected rows.
Keep that workflow intact when selecting a few illustrative episodes; the
other diagnostic tools do not establish that parity. The ROS export tool lives
in this baselines repository; its [guide](../isaac_sim/README.md) describes the
map and waypoint contract. Joint capture adapters remain specific to the
checkpoint and native runtime, rather than a generic inference command.

## Single-map 2D diagnostic

Run from the `terra-baselines` root:

```bash
python inference/inference_single_map.py --policy checkpoint.pkl --config trench_excavator --map_name map --n_steps 500 --seed 0
```

- The default map root is `inference/maps/`; `--map_name` is appended to it.
- Use `--map_path` for an explicit map file/folder, overriding `--map_name`.
- `--config` selects a named map/agent preset, as in `visualize_mixed.py`.
- `--deterministic 1` uses argmax actions; the default samples the policy.
- Add `--use-mcts -sim 32` to select MCTS; otherwise it uses the PPO action path.
- `--out_path` selects the output GIF. If omitted, its name includes the
  checkpoint, map and timestamp.

This command runs a new policy episode and produces a Pygame GIF. For a grid of
2D policy rollouts, use [`visualize_mixed.py`](../visualize_mixed.py); for saved
native or postprocessed 3D recordings, use the shared renderer above.
