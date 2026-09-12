"""A linear behavior-cost ramp indexed by completed native PPO updates.

TrainConfig and the R2 receipt declare target costs. The saved environment
contains the effective costs used by the last completed rollout. This small
state makes those two quantities verifiable and preserves progress on resume.
"""
import copy
import math


COST_KEYS = ("lateral_dig_cost", "base_travel_cost", "base_turn_cost")
SCHEMA = "terra_behavior_cost_ramp_v1"


def field(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def declared_costs(config):
    return {key: float(field(config, key, 0.0)) for key in COST_KEYS}


def ramp_progress(state, completed_updates):
    return min(1.0, max(0.0, (completed_updates - state["start_update"]) / state["duration_updates"]))


def ramp_costs(state, completed_updates):
    progress = ramp_progress(state, completed_updates)
    # Preserve exact endpoints, including zero, for checkpoint comparisons.
    if progress == 0.0:
        return dict(state["start_costs"])
    if progress == 1.0:
        return dict(state["target_costs"])
    return {key: state["start_costs"][key] + progress * (
        state["target_costs"][key] - state["start_costs"][key]) for key in COST_KEYS}


def validate_ramp_state(state, completed_updates, target_costs):
    if not isinstance(state, dict) or set(state) != {
        "schema", "start_update", "duration_updates", "start_costs", "target_costs"
    } or state["schema"] != SCHEMA:
        raise ValueError("Invalid behavior_cost_ramp_state schema")
    for key, minimum in (("start_update", 0), ("duration_updates", 1)):
        if type(state[key]) is not int or state[key] < minimum:
            raise ValueError(f"Behavior ramp {key} must be an integer >= {minimum}")
    if completed_updates < state["start_update"]:
        raise ValueError("Behavior ramp starts after checkpoint next_update")
    for vector in (state["start_costs"], state["target_costs"]):
        if not isinstance(vector, dict) or set(vector) != set(COST_KEYS):
            raise ValueError("Behavior ramp must contain all three cost fields")
        if any(not math.isfinite(value) or value < 0 for value in vector.values()):
            raise ValueError("Behavior ramp costs must be finite and nonnegative")
    if state["target_costs"] != target_costs:
        raise ValueError("Behavior ramp target differs from declared training costs")
    if any(state["target_costs"][key] < state["start_costs"][key] for key in COST_KEYS) or (
        state["target_costs"] == state["start_costs"]
    ):
        raise ValueError("Behavior ramp must increase costs from its accepted parent")


def restore_behavior_cost_ramp(config, checkpoint, checkpoint_mode, resume_update, parent_behavior):
    """Restore an existing ramp, or explicitly start the next ramp at a parent."""
    requested = field(config, "behavior_cost_ramp_updates", 0)
    if checkpoint_mode == "warm_start" and not requested:
        # Parameters-only initialization deliberately starts a fresh objective,
        # optimizer, clock and environment, just as for fixed-cost checkpoints.
        return None
    saved = checkpoint.get("behavior_cost_ramp_state") if checkpoint else None
    if not requested and saved is None:
        return None
    if (checkpoint_mode != "resume" or checkpoint is None
            or checkpoint.get("next_update") != resume_update
            or field(config, "resume_update") is not None
            or field(config, "reward_stage") != "reward_v2"
            or not field(config, "load_env_from_checkpoint", True)
            or field(config, "finetune_task_bank", False)):
        raise ValueError("Behavior ramp requires native reward_v2 continuation on the same bank and clock")
    target = declared_costs(config)
    if field(config, "executable_dig_observation", False) != parent_behavior["executable_dig_observation"]:
        raise ValueError("A behavior-cost ramp cannot also change digging observations")
    if saved is not None:
        validate_ramp_state(saved, resume_update, declared_costs(checkpoint["train_config"]))
        if target == saved["target_costs"]:
            if requested not in (0, saved["duration_updates"]):
                raise ValueError("Behavior ramp duration changed across resume")
            return copy.deepcopy(saved)
        if resume_update < saved["start_update"] + saved["duration_updates"]:
            raise ValueError("Cannot replace an unfinished behavior ramp")
    if not requested or not field(config, "finetune_foundation_behavior", False):
        raise ValueError("Starting a behavior ramp requires an explicit duration and behavior fine-tune")
    state = {
        "schema": SCHEMA, "start_update": int(resume_update),
        "duration_updates": requested,
        "start_costs": {key: parent_behavior[key] for key in COST_KEYS},
        "target_costs": target,
    }
    validate_ramp_state(state, resume_update, target)
    return state
