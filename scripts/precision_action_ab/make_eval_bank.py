"""Create the common held-out native reset bank; run once on the staged runtime."""
import argparse
import json
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from terra.state import static_rules
from terra.wrappers import TraversabilityMaskWrapper
from train_structured import (
    LOCAL_MAP_FIELDS, _cast_like, default_model_config, load_initial_bank, load_map_starts, rule_switches,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    split = json.loads(args.split_manifest.read_text())
    slots = split["eval_representative_slots"]
    metadata = {record["slot"]: record for record in split["slots"]}
    manifest_path = Path(os.environ["DATASET_PATH"]) / split["dataset_level"] / "manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    for slot in metadata:
        for field in ("map_id", "source_id", "dig_sha256", "source_scenario_id"):
            if manifest[slot][field] != metadata[slot][field]:
                raise ValueError(f"staged dataset differs from the split at slot {slot}: {field}")
    group_ids = {slot: group["group"] for group in split["groups"] for slot in group["slots"]}
    model_config = default_model_config()
    model_config.update(json.loads((Path(__file__).resolve().parents[1] / "structured/campaign_model.json").read_text()))
    model_config.update(structured_actions=True, precision_observation_v2=True, action_logit_masking=False)
    template_bank, static_cfg = load_initial_bank(args.template)
    static_cfg = static_cfg._replace(precision_safety=True, enforce_foundation_border_alignment=True)
    template = jax.tree.map(lambda value: np.asarray(value[0]), template_bank)
    template = template._replace(env_cfg=template.env_cfg._replace(
        precision_safety=np.asarray(True), enforce_foundation_border_alignment=np.asarray(True)))
    env, data = load_map_starts(split["dataset_level"], static_cfg, model_config,
                                distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
                                precision_slots=slots)
    row_by_slot = {int(slot): row for row, slot in enumerate(np.asarray(env.maps_buffer.slot_indices[0]))}

    @jax.jit
    def reset(arrays, key, state):
        fresh = state._replace(key=key)._reset(static_cfg, *arrays[:-1], distance_map_override=arrays[-1])
        fresh = TraversabilityMaskWrapper.wrap(fresh)
        fresh = fresh._replace(env_cfg=state.env_cfg, world=fresh.world._replace(
            **{name: getattr(state.world, name) for name in LOCAL_MAP_FIELDS}))
        return _cast_like(fresh, state)

    initial, episodes = [], []
    with static_rules(**rule_switches(static_cfg)):
        for slot in slots:
            row = row_by_slot[slot]
            arrays = tuple(jnp.asarray(np.asarray(array)[row]) for array in data[:-1])
            poses = set()
            for attempt in range(32):
                seed = 20261013 + slot * 32 + attempt
                state = jax.device_get(reset(arrays, jax.random.PRNGKey(seed), jax.tree.map(jnp.asarray, template)))
                agent = state._get_current_agent_state()
                position = tuple(int(value) for value in np.asarray(agent.pos_base).reshape(-1))
                base_heading = int(np.asarray(agent.angle_base).item())
                cabin_heading = int(np.asarray(agent.angle_cabin).item())
                pose = (*position, base_heading, cabin_heading)
                if pose in poses:
                    continue
                poses.add(pose)
                initial.append(state)
                target = np.asarray(state.world.target_map.map)
                episodes.append(dict(source="bank", source_slot=slot, group=group_ids[slot],
                                     **metadata[slot], mode="precision", precision=True, reset_seed=seed,
                                     position=list(position), base_heading=base_heading, cabin_heading=cabin_heading,
                                     decoder="greedy", dig_units=int(np.clip(-target, 0, None).sum()),
                                     dig_cells=int((target < 0).sum())))
                if len(poses) == 4:
                    break
            if len(poses) != 4:
                raise ValueError(f"slot {slot}: only {len(poses)} distinct poses after 32 reset seeds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(dict(initial=initial, episodes=episodes, template=str(args.template),
                         split_manifest=str(args.split_manifest), dataset_path=os.environ["DATASET_PATH"]),
                    stream, protocol=pickle.HIGHEST_PROTOCOL)
    load_initial_bank(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(episodes, indent=2) + "\n")
    print(f"Saved {len(initial)} native precision starts from {len(slots)} held-out geometry groups to {args.output}")


if __name__ == "__main__":
    main()
