#!/usr/bin/env python3
"""Fast isolated checks for edge-digging action-mask and speedup changes."""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        return (int(parsed),)
    if isinstance(parsed, (tuple, list)):
        return tuple(int(x) for x in parsed)
    raise ValueError(f"expected int, tuple, or list, got {type(parsed).__name__}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_reward_components(info: dict, prefix: str) -> None:
    _assert("reward_components" in info, f"{prefix} info has no reward_components")
    components = info["reward_components"]
    expected_keys = {
        "agent_rewards",
        "terminal",
        "trench",
        "existence",
        "agent_active",
        "num_agents",
    }
    _assert(set(components) == expected_keys, f"{prefix} reward component keys differ")
    _assert(np.asarray(components["agent_rewards"]).shape == (4,), f"{prefix} agent_rewards shape differs")
    _assert(np.asarray(components["agent_active"]).shape == (4,), f"{prefix} agent_active shape differs")
    for key in ("terminal", "trench", "existence", "num_agents"):
        _assert(np.asarray(components[key]).shape == (), f"{prefix} {key} shape differs")


def test_ppo_mask() -> None:
    import jax.numpy as jnp

    from utils.action_masking import apply_action_mask

    logits = jnp.arange(24, dtype=jnp.float32).reshape(3, 8)
    action_mask = jnp.array(
        [
            [True, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, True],
            [False, False, False, False, False, False, False, False],
        ],
        dtype=jnp.bool_,
    )

    masked = np.asarray(apply_action_mask(logits, action_mask))
    logits_np = np.asarray(logits)

    _assert(masked[0, 0] == logits_np[0, 0], "valid action 0 was changed")
    _assert(masked[0, 1] < -1e8, "invalid action 1 was not masked")
    _assert(masked[0, 6] < -1e8, "invalid DO action was not masked")
    _assert(masked[0, 7] < -1e8, "unavailable DO_NOTHING leaked into masked logits")
    _assert(masked[1, 7] == logits_np[1, 7], "DO_NOTHING action was masked")
    _assert(np.all(masked[1, :7] < -1e8), "masked row leaked a non-DO_NOTHING action")
    _assert(np.all(masked[2] < -1e8), "all-false row was silently repaired")
    print("PASS ppo-mask: invalid logits are blocked and malformed rows are not repaired")


def test_training_accounting() -> None:
    import jax
    import jax.numpy as jnp

    from terra.config import EnvConfig
    from train import TrainConfig
    from train_mixed import (
        MixedAgentTrainConfig,
        _first_checkpoint_env_config,
        _num_agents_for_prev_actions,
    )

    total_timesteps = 50_000_000_000
    num_devices = 4
    num_envs_per_device = 1024
    num_steps = 32
    expected_num_envs = num_devices * num_envs_per_device
    expected_steps_per_update = num_steps * expected_num_envs
    expected_num_updates = total_timesteps // expected_steps_per_update
    expected_actual_steps = expected_num_updates * expected_steps_per_update

    for config in (
        TrainConfig(
            name="accounting-test",
            num_devices=num_devices,
            num_envs_per_device=num_envs_per_device,
            num_steps=num_steps,
            total_timesteps=total_timesteps,
        ),
        MixedAgentTrainConfig(
            name="accounting-test",
            num_devices=num_devices,
            num_envs_per_device=num_envs_per_device,
            num_steps=num_steps,
            total_timesteps=total_timesteps,
        ),
    ):
        _assert(config.num_envs == expected_num_envs, "num_envs should include all local devices once")
        _assert(
            config.env_steps_per_update == expected_steps_per_update,
            "env_steps_per_update should be num_steps * num_envs_per_device * num_devices",
        )
        _assert(
            config.num_updates == expected_num_updates,
            "num_updates should not divide by num_devices after num_envs already includes devices",
        )
        _assert(
            config.actual_total_timesteps == expected_actual_steps,
            "actual_total_timesteps should match num_updates * env_steps_per_update",
        )

    steps_per_second = 2.0 * expected_steps_per_update
    _assert(
        steps_per_second == 262144,
        "2 updates/s at 4*1024 envs and 32 steps should be 262144 env steps/s",
    )
    env_config = EnvConfig()._replace(
        agent_types=(0, 2),
        action_types=(0, 1),
        tile_size=1.0,
        max_steps_in_episode=550,
    )
    batched_env_config = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x)[None, None].repeat(2, 0).repeat(3, 1),
        env_config,
    )
    scalar_env_config = _first_checkpoint_env_config(batched_env_config)
    _assert(np.asarray(scalar_env_config.tile_size).shape == (), "resume env tile_size stayed batched")
    _assert(
        tuple(int(np.asarray(x)) for x in scalar_env_config.agent_types) == (0, 2),
        "resume env agent_types were not preserved while unbatching",
    )
    _assert(
        tuple(int(np.asarray(x)) for x in scalar_env_config.action_types) == (0, 1),
        "resume env action_types were not preserved while unbatching",
    )
    resume_config = MixedAgentTrainConfig(
        name="resume-prev-actions-test",
        num_devices=1,
        num_envs_per_device=1,
        agent_types_override=(0,),
    )
    _assert(
        _num_agents_for_prev_actions(
            config=resume_config,
            env_params=batched_env_config,
            env=None,
            env_params_override=scalar_env_config,
        ) == 2,
        "resume num_prev_actions inference should prefer checkpoint env_config over CLI default",
    )

    repo_root = Path(__file__).resolve().parents[1]
    for script_name in ("train.py", "train_mixed.py"):
        tree = ast.parse((repo_root / script_name).read_text())
        reset_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "reset_fn_p"
        ]
        _assert(len(reset_calls) == 1, f"{script_name} should call reset_fn_p exactly once")
        _assert(
            len(reset_calls[0].args) == 11,
            f"{script_name} reset_fn_p call must pass foundation-border maps and distance maps",
        )

    train_mixed_source = (repo_root / "train_mixed.py").read_text()
    _assert(
        "config.edge_features_dim = infer_edge_features_dim_from_model_params(" in train_mixed_source,
        "train_mixed.py resume path must infer checkpoint edge feature width before model init",
    )
    _assert(
        "checkpoint_use_action_mask = infer_use_action_mask_from_train_config(" in train_mixed_source,
        "train_mixed.py resume path must restore checkpoint action-mask setting before model init",
    )
    _assert(
        "if config.action_mask_cli_override is None:" in train_mixed_source,
        "train_mixed.py resume path must let explicit CLI action-mask overrides win",
    )
    for script_name in ("visualize_mixed.py", "inference/inference_single_map.py"):
        source = (repo_root / script_name).read_text()
        _assert(
            "return jax.tree_util.tree_map(_take0, nt)" in source,
            f"{script_name} must unbatch tuple agent/action types as pytrees",
        )

    print("PASS training-accounting: multi-device accounting and reset_prepared wiring are correct")


def test_model_policy() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from terra.actions import TrackedAction
    from utils.models import SimplifiedCoupledCategoricalNet
    from utils.utils_ppo import obs_to_model_input, policy

    batch_size = 2
    map_size = 32
    angles_cabin = 12
    prev_actions = jnp.zeros((batch_size, 5), dtype=jnp.int32)
    obs_dict = {
        "agent_states": jnp.zeros((batch_size, 4, 8), dtype=jnp.float32),
        "agent_active": jnp.array(
            [[1, 0, 0, 0], [1, 0, 0, 0]],
            dtype=jnp.int8,
        ),
        "num_agents": jnp.ones((batch_size,), dtype=jnp.int32),
        "local_map_action_neg": jnp.zeros((batch_size, angles_cabin), dtype=jnp.float32),
        "local_map_action_pos": jnp.zeros((batch_size, angles_cabin), dtype=jnp.float32),
        "local_map_target_neg": jnp.zeros((batch_size, angles_cabin), dtype=jnp.float32),
        "local_map_target_pos": jnp.zeros((batch_size, angles_cabin), dtype=jnp.float32),
        "local_map_dumpability": jnp.zeros((batch_size, angles_cabin), dtype=jnp.float32),
        "local_map_obstacles": jnp.zeros((batch_size, angles_cabin), dtype=jnp.float32),
        "local_map_border_workspace": jnp.zeros((batch_size, angles_cabin), dtype=jnp.float32),
        "local_map_edge_alignment_error": jnp.zeros((batch_size, angles_cabin), dtype=jnp.float32),
        "local_map_border_diggable": jnp.zeros((batch_size, angles_cabin), dtype=jnp.float32),
        "traversability_mask": jnp.zeros((batch_size, map_size, map_size), dtype=jnp.float32),
        "reachability_mask": jnp.zeros((batch_size, map_size, map_size), dtype=jnp.float32),
        "action_map": jnp.zeros((batch_size, map_size, map_size), dtype=jnp.float32),
        "target_map": jnp.zeros((batch_size, map_size, map_size), dtype=jnp.float32),
        "agent_width": jnp.full((batch_size,), 3, dtype=jnp.int32),
        "agent_height": jnp.full((batch_size,), 3, dtype=jnp.int32),
        "padding_mask": jnp.zeros((batch_size, map_size, map_size), dtype=jnp.float32),
        "dumpability_mask": jnp.ones((batch_size, map_size, map_size), dtype=jnp.float32),
        "interaction_mask": jnp.zeros((batch_size, map_size, map_size), dtype=jnp.float32),
        "action_mask": jnp.array(
            [
                [True, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, True],
            ],
            dtype=jnp.bool_,
        ),
        "edge_features": jnp.zeros((batch_size, 10), dtype=jnp.float32),
    }

    class TrainCfg(dict):
        clip_action_maps = True

    legacy_cfg = TrainCfg(
        {
            "num_prev_actions": 5,
            "clip_action_maps": True,
            "loaded_max": 100,
        }
    )
    legacy_obs_dict = dict(obs_dict)
    legacy_obs_dict.pop("action_mask")
    legacy_obs_dict.pop("edge_features")
    legacy_model_obs = obs_to_model_input(legacy_obs_dict, prev_actions, legacy_cfg)
    _assert(len(legacy_model_obs) == 24, f"expected 24 model obs entries, got {len(legacy_model_obs)}")
    _assert(legacy_model_obs[22].shape == (batch_size, 8), "legacy action_mask fallback shape differs")
    _assert(np.all(np.asarray(legacy_model_obs[22])), "legacy no-mask path must use an all-true mask")
    _assert(legacy_model_obs[23].shape == (batch_size, 0), "legacy no-mask path must not add edge features")

    legacy_model = SimplifiedCoupledCategoricalNet(
        num_prev_actions=5,
        num_embeddings_agent=map_size,
        map_min_max=(-1, 1),
        local_map_min_max=(-16, 16),
        loaded_max=100,
        action_type=TrackedAction,
    )
    legacy_params = legacy_model.init(jax.random.PRNGKey(0), legacy_model_obs)
    _, legacy_pi = policy(legacy_model.apply, legacy_params, legacy_model_obs)
    legacy_logits = np.asarray(legacy_pi.logits_parameter())
    _assert(np.all(legacy_logits > -1e8), "policy default unexpectedly masked legacy logits")

    train_cfg = TrainCfg(
        {
            "num_prev_actions": 5,
            "clip_action_maps": True,
            "loaded_max": 100,
            "use_action_mask": True,
            "edge_features_dim": 10,
        }
    )
    model_obs = obs_to_model_input(obs_dict, prev_actions, train_cfg)
    _assert(len(model_obs) == 24, f"expected 24 model obs entries, got {len(model_obs)}")
    _assert(model_obs[22].shape == (batch_size, 8), "action_mask was not at model obs index 22")
    _assert(model_obs[23].shape == (batch_size, 10), "edge_features was not at model obs index 23")

    model = SimplifiedCoupledCategoricalNet(
        num_prev_actions=5,
        num_embeddings_agent=map_size,
        map_min_max=(-1, 1),
        local_map_min_max=(-16, 16),
        loaded_max=100,
        action_type=TrackedAction,
    )
    params = model.init(jax.random.PRNGKey(0), model_obs)
    value, pi = policy(model.apply, params, model_obs, use_action_mask=True)
    logits = np.asarray(pi.logits_parameter())
    _, unmasked_pi = policy(model.apply, params, model_obs, use_action_mask=False)
    unmasked_logits = np.asarray(unmasked_pi.logits_parameter())

    _assert(np.asarray(value).shape == (batch_size, 1), f"unexpected value shape {np.asarray(value).shape}")
    _assert(logits.shape == (batch_size, 8), f"unexpected policy logits shape {logits.shape}")
    _assert(logits[0, 1] < -1e8, "policy did not mask invalid action 1")
    _assert(logits[0, 7] < -1e8, "policy leaked unavailable DO_NOTHING")
    _assert(logits[1, 7] > -1e8, "policy masked available DO_NOTHING")
    _assert(np.all(logits[1, :7] < -1e8), "policy leaked unavailable actions")
    _assert(
        np.all(unmasked_logits > -1e8),
        "policy use_action_mask=False still masked logits",
    )

    print("PASS model-policy: no-mask legacy input and explicit masked edge-feature input both work")


def _batched_env_config(env_cfg, num_envs: int):
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda x: jnp.array(x)[None].repeat(num_envs, axis=0),
        env_cfg,
    )


def test_env_action_mask(
    map_path: str,
    agent_types: tuple[int, ...],
    action_types: tuple[int, ...],
    num_envs: int,
    seed: int,
    expect_cabin_disabled: bool,
    disable_jit: bool,
) -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_disable_jit", disable_jit)

    from terra.config import BatchConfig, CurriculumGlobalConfig, EnvConfig, RewardsType
    from terra.env import TerraEnvBatch

    class SmokeCurriculumGlobalConfig(CurriculumGlobalConfig):
        levels = [
            {
                "maps_path": map_path,
                "max_steps_in_episode": 16,
                "rewards_type": RewardsType.DENSE,
                "apply_trench_rewards": False,
            }
        ]

    batch_cfg = BatchConfig(curriculum_global=SmokeCurriculumGlobalConfig())
    env = TerraEnvBatch(batch_cfg=batch_cfg, shuffle_maps=False)
    env_cfg = EnvConfig()._replace(
        agent_types=agent_types,
        action_types=action_types,
    )
    env_cfgs = _batched_env_config(env_cfg, num_envs)

    reset_keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    timestep = env.reset(env_cfgs, reset_keys)
    _assert("action_mask" in timestep.observation, "reset observation has no action_mask")
    _assert("action_mask" in timestep.info, "reset info has no action_mask")

    obs_mask = np.asarray(timestep.observation["action_mask"])
    info_mask = np.asarray(timestep.info["action_mask"])
    _assert(obs_mask.shape == (num_envs, 8), f"unexpected reset obs mask shape {obs_mask.shape}")
    _assert(info_mask.shape == (num_envs, 8), f"unexpected reset info mask shape {info_mask.shape}")
    _assert(obs_mask.dtype == np.bool_, f"obs mask dtype is {obs_mask.dtype}, expected bool")
    _assert(np.array_equal(obs_mask, info_mask), "reset obs/info masks differ")
    _assert(np.all(obs_mask[:, 7]), "reset DO_NOTHING is not always available")
    _assert(np.all(np.any(obs_mask, axis=-1)), "reset contains an all-false action mask")
    if expect_cabin_disabled:
        _assert(not np.any(obs_mask[:, 4:6]), "cabin actions should be disabled for this agent type")

    step_keys = jax.random.split(jax.random.PRNGKey(seed + 1), num_envs)
    actions = env.batch_cfg.action_type.new(
        jnp.full((num_envs, 1), 7, dtype=jnp.int8)
    )
    next_timestep = env.step(timestep, actions, step_keys)
    next_obs_mask = np.asarray(next_timestep.observation["action_mask"])
    next_info_mask = np.asarray(next_timestep.info["action_mask"])
    _assert(next_obs_mask.shape == (num_envs, 8), f"unexpected step obs mask shape {next_obs_mask.shape}")
    _assert(np.array_equal(next_obs_mask, next_info_mask), "step obs/info masks differ")
    _assert(np.all(next_obs_mask[:, 7]), "step DO_NOTHING is not always available")
    _assert(np.all(np.any(next_obs_mask, axis=-1)), "step contains an all-false action mask")
    if expect_cabin_disabled:
        _assert(not np.any(next_obs_mask[:, 4:6]), "step cabin actions should be disabled")
    _assert(
        np.asarray(next_timestep.observation["edge_features"]).shape == (num_envs, 10),
        "batched env edge_features shape is wrong",
    )

    print(
        "PASS env-action-mask: "
        f"map={map_path} agent_types={agent_types} action_types={action_types} "
        f"num_envs={num_envs}"
    )


def _assert_pytree_equal(left, right, label: str) -> None:
    import jax

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    _assert(left_tree == right_tree, f"{label}: pytree structures differ")
    for idx, (left_leaf, right_leaf) in enumerate(zip(left_leaves, right_leaves)):
        left_np = np.asarray(left_leaf)
        right_np = np.asarray(right_leaf)
        _assert(
            np.array_equal(left_np, right_np),
            f"{label}: leaf {idx} differs; shapes {left_np.shape} vs {right_np.shape}",
        )


def test_synthetic_batch_step_fast_reset_parity(
    agent_types: tuple[int, ...],
    action_types: tuple[int, ...],
    num_envs: int,
    seed: int,
    disable_jit: bool,
    map_size: int = 32,
) -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_disable_jit", disable_jit)

    from terra.config import AgentConfig, BatchConfig, EnvConfig
    from terra.env import TerraEnv

    agent_cfg = AgentConfig(width=3, height=3, move_tiles=3)
    env_cfg = EnvConfig()._replace(
        agent=agent_cfg,
        maps=EnvConfig().maps._replace(edge_length_px=map_size),
        tile_size=1.0,
        max_steps_in_episode=16,
        agent_types=agent_types,
        action_types=action_types,
    )
    target_map = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    target_map = target_map.at[map_size // 2, map_size // 2].set(jnp.int8(-1))
    padding_mask = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    action_map = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    dumpability = jnp.ones((map_size, map_size), dtype=jnp.bool_)
    trench_axes = jnp.zeros((3, 3), dtype=jnp.float32)
    foundation_border_axes = jnp.zeros((64, 3), dtype=jnp.float32)
    distance_map = jnp.zeros((map_size, map_size), dtype=jnp.float32)

    env = TerraEnv.new(maps_size_px=map_size, rendering=False)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    env_cfgs = _batched_env_config(env_cfg, num_envs)
    target_maps = jnp.repeat(target_map[None], num_envs, axis=0)
    padding_masks = jnp.repeat(padding_mask[None], num_envs, axis=0)
    action_maps = jnp.repeat(action_map[None], num_envs, axis=0)
    dumpability_masks = jnp.repeat(dumpability[None], num_envs, axis=0)
    trench_axes_batch = jnp.repeat(trench_axes[None], num_envs, axis=0)
    trench_types = jnp.zeros((num_envs,), dtype=jnp.int32)
    foundation_border_axes_batch = jnp.repeat(
        foundation_border_axes[None], num_envs, axis=0
    )
    foundation_border_types = jnp.zeros((num_envs,), dtype=jnp.int32)
    distance_maps = jnp.repeat(distance_map[None], num_envs, axis=0)
    timestep = jax.vmap(env.reset)(
        keys,
        target_maps,
        padding_masks,
        trench_axes_batch,
        trench_types,
        foundation_border_axes_batch,
        foundation_border_types,
        dumpability_masks,
        action_maps,
        distance_maps,
        env_cfgs,
    )
    actions = BatchConfig().action_type.new(
        jnp.full((num_envs, 1), 7, dtype=jnp.int8)
    )

    def fast_batch(ts):
        ts_no_reset = jax.vmap(env.step_no_reset)(
            ts.state,
            actions,
            ts.env_cfg,
        )

        def _reset_done_envs(ts_inner):
            def _reset_one(
                ts_one,
                target,
                padding,
                trench_axis,
                trench_kind,
                foundation_axis,
                foundation_kind,
                dumpability_mask,
                action,
                distance,
            ):
                def _reset_branch(item):
                    state_reset, obs_reset = env._reset_existent(
                        item.state,
                        target,
                        padding,
                        trench_axis,
                        trench_kind,
                        foundation_axis,
                        foundation_kind,
                        dumpability_mask,
                        action,
                        distance,
                        item.env_cfg,
                    )
                    infos = {
                        **item.info,
                        "action_mask": obs_reset["action_mask"],
                        "edge_features": obs_reset["edge_features"],
                        "target_tiles": state_reset.world.interaction_mask.map.reshape(-1),
                    }
                    return item._replace(
                        state=state_reset,
                        observation=obs_reset,
                        info=infos,
                    )

                return jax.lax.cond(
                    ts_one.done,
                    _reset_branch,
                    lambda item: item,
                    ts_one,
                )

            return jax.vmap(_reset_one)(
                ts_inner,
                target_maps,
                padding_masks,
                trench_axes_batch,
                trench_types,
                foundation_border_axes_batch,
                foundation_border_types,
                dumpability_masks,
                action_maps,
                distance_maps,
            )

        return jax.lax.cond(
            jnp.any(ts_no_reset.done),
            _reset_done_envs,
            lambda item: item,
            ts_no_reset,
        )

    def slow_batch(ts):
        return jax.vmap(env.step)(
            ts.state,
            actions,
            target_maps,
            padding_masks,
            trench_axes_batch,
            trench_types,
            foundation_border_axes_batch,
            foundation_border_types,
            dumpability_masks,
            action_maps,
            distance_maps,
            ts.env_cfg,
        )

    slow = slow_batch(timestep)
    fast = fast_batch(timestep)
    _assert(not np.any(np.asarray(fast.done)), "synthetic no-reset setup unexpectedly ended")
    _assert_pytree_equal(fast, slow, "synthetic no-reset batch step")

    def with_forced_timeout(ts, done_mask):
        env_steps = jnp.where(
            done_mask,
            ts.env_cfg.max_steps_in_episode,
            ts.state.env_steps,
        )
        return ts._replace(
            state=ts.state._replace(env_steps=env_steps),
            done=jnp.zeros_like(ts.done),
        )

    all_done_mask = jnp.ones((num_envs,), dtype=jnp.bool_)
    all_done_timestep = with_forced_timeout(timestep, all_done_mask)
    slow_all_done = slow_batch(all_done_timestep)
    fast_all_done = fast_batch(all_done_timestep)
    _assert(np.all(np.asarray(fast_all_done.done)), "synthetic all-done setup did not end")
    _assert_pytree_equal(fast_all_done, slow_all_done, "synthetic all-done batch step")

    if num_envs > 1:
        partial_done_mask = (jnp.arange(num_envs) % 2) == 0
        partial_done_timestep = with_forced_timeout(timestep, partial_done_mask)
        slow_partial_done = slow_batch(partial_done_timestep)
        fast_partial_done = fast_batch(partial_done_timestep)
        _assert(
            np.array_equal(np.asarray(fast_partial_done.done), np.asarray(partial_done_mask)),
            "synthetic partial-done setup did not preserve the expected done mask",
        )
        _assert_pytree_equal(
            fast_partial_done,
            slow_partial_done,
            "synthetic partial-done batch step",
        )

    print(
        "PASS synthetic-batch-step-fast-reset: fast batched reset gating matches "
        f"old single-env step composition for no/all/partial done, num_envs={num_envs}"
    )


def test_synthetic_step_fast_reset_parity(
    agent_types: tuple[int, ...],
    action_types: tuple[int, ...],
    seed: int,
    disable_jit: bool,
    map_size: int = 32,
) -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_disable_jit", disable_jit)

    from terra.config import AgentConfig, BatchConfig, EnvConfig
    from terra.env import TerraEnv

    agent_cfg = AgentConfig(width=3, height=3, move_tiles=3)
    env_cfg = EnvConfig()._replace(
        agent=agent_cfg,
        maps=EnvConfig().maps._replace(edge_length_px=map_size),
        tile_size=1.0,
        max_steps_in_episode=16,
        agent_types=agent_types,
        action_types=action_types,
    )
    target_map = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    target_map = target_map.at[map_size // 2, map_size // 2].set(jnp.int8(-1))
    padding_mask = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    action_map = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    dumpability = jnp.ones((map_size, map_size), dtype=jnp.bool_)
    trench_axes = jnp.zeros((3, 3), dtype=jnp.float32)
    foundation_border_axes = jnp.zeros((64, 3), dtype=jnp.float32)
    distance_map = jnp.zeros((map_size, map_size), dtype=jnp.float32)

    env = TerraEnv.new(maps_size_px=map_size, rendering=False)
    timestep = env.reset(
        jax.random.PRNGKey(seed),
        target_map,
        padding_mask,
        trench_axes,
        jnp.int32(0),
        foundation_border_axes,
        jnp.int32(0),
        dumpability,
        action_map,
        distance_map,
        env_cfg,
    )
    action = BatchConfig().action_type.do_nothing()

    slow = env.step(
        timestep.state,
        action,
        target_map,
        padding_mask,
        trench_axes,
        jnp.int32(0),
        foundation_border_axes,
        jnp.int32(0),
        dumpability,
        action_map,
        distance_map,
        env_cfg,
    )
    fast = env.step_no_reset(timestep.state, action, env_cfg)
    _assert(not bool(np.asarray(fast.done)), "synthetic no-reset setup unexpectedly ended")
    _assert_pytree_equal(fast, slow, "synthetic single no-reset step")

    done_timestep = timestep._replace(
        state=timestep.state._replace(env_steps=timestep.env_cfg.max_steps_in_episode)
    )
    slow_done = env.step(
        done_timestep.state,
        action,
        target_map,
        padding_mask,
        trench_axes,
        jnp.int32(0),
        foundation_border_axes,
        jnp.int32(0),
        dumpability,
        action_map,
        distance_map,
        env_cfg,
    )
    fast_done = env.step_no_reset(done_timestep.state, action, env_cfg)
    state_reset, obs_reset = env._reset_existent(
        fast_done.state,
        target_map,
        padding_mask,
        trench_axes,
        jnp.int32(0),
        foundation_border_axes,
        jnp.int32(0),
        dumpability,
        action_map,
        distance_map,
        fast_done.env_cfg,
    )
    infos = {
        **fast_done.info,
        "action_mask": obs_reset["action_mask"],
        "edge_features": obs_reset["edge_features"],
        "target_tiles": state_reset.world.interaction_mask.map.reshape(-1),
    }
    fast_done = fast_done._replace(
        state=state_reset,
        observation=obs_reset,
        info=infos,
    )
    _assert(bool(np.asarray(fast_done.done)), "synthetic forced-reset setup did not end")
    _assert_pytree_equal(fast_done, slow_done, "synthetic single forced-reset step")

    print("PASS synthetic-step-fast-reset: step_no_reset factoring preserves TerraEnv.step")


def _make_agent_state(
    pos: tuple[int, int],
    agent_type: int,
    action_type: int,
    loaded: int = 0,
    shovel_lifted: int = 0,
):
    import jax.numpy as jnp

    from terra.agent import AgentState
    from terra.settings import IntLowDim, IntMap

    return AgentState(
        pos_base=jnp.array(pos, dtype=IntMap),
        angle_base=jnp.array([0], dtype=IntLowDim),
        angle_cabin=jnp.array([0], dtype=IntLowDim),
        wheel_angle=jnp.array([0], dtype=IntLowDim),
        loaded=jnp.array([loaded], dtype=IntLowDim),
        agent_type=jnp.array([agent_type], dtype=IntLowDim),
        action_type=jnp.array([action_type], dtype=IntLowDim),
        shovel_lifted=jnp.array([shovel_lifted], dtype=IntLowDim),
    )


def _make_state(agent_states, current_agent: int, num_agents: int, map_size: int = 32):
    import jax
    import jax.numpy as jnp

    from terra.agent import Agent
    from terra.config import AgentConfig, EnvConfig
    from terra.map import GridWorld
    from terra.state import State

    agent_cfg = AgentConfig(width=3, height=3, move_tiles=3)
    env_cfg = EnvConfig()._replace(
        agent=agent_cfg,
        maps=EnvConfig().maps._replace(edge_length_px=map_size),
        tile_size=1.0,
        max_steps_in_episode=16,
    )
    target_map = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    padding_mask = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    action_map = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    dumpability = jnp.ones((map_size, map_size), dtype=jnp.bool_)
    trench_axes = jnp.zeros((3, 3), dtype=jnp.float32)
    foundation_border_axes = jnp.zeros((64, 3), dtype=jnp.float32)
    world = GridWorld.new(
        target_map=target_map,
        padding_mask=padding_mask,
        trench_axes=trench_axes,
        trench_type=jnp.int32(0),
        foundation_border_axes=foundation_border_axes,
        foundation_border_type=jnp.int32(0),
        dumpability_mask_init=dumpability,
        action_map=action_map,
        relocation_distance_map_override=jnp.zeros((map_size, map_size), dtype=jnp.float32),
    )
    dummy = _make_agent_state((2, 2), 0, 0)
    padded_states = tuple(list(agent_states) + [dummy] * (4 - len(agent_states)))
    agent = Agent(
        width=agent_cfg.width,
        height=agent_cfg.height,
        moving_dumped_dirt=False,
        agent_states=padded_states,
        agent_active=jnp.array([1 if i < num_agents else 0 for i in range(4)], dtype=jnp.int8),
        num_agents=num_agents,
        current_agent=jnp.int32(current_agent),
    )
    return State(
        key=jax.random.PRNGKey(0),
        env_cfg=env_cfg,
        world=world,
        agent=agent,
        env_steps=0,
    )


def test_state_action_mask(disable_jit: bool) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_disable_jit", disable_jit)

    from terra.config import BatchConfig
    from terra.settings import IntLowDim

    loaded_excavator = _make_agent_state((16, 16), agent_type=0, action_type=0, loaded=1)
    previous_agent = _make_agent_state((5, 5), agent_type=0, action_type=0)
    state = _make_state((loaded_excavator, previous_agent), current_agent=0, num_agents=2)
    mask, _ = state._get_action_mask_and_edge_features(BatchConfig().action_type.do_nothing())
    mask = np.asarray(mask)
    _assert(mask.shape == (8,), f"unexpected state mask shape {mask.shape}")
    _assert(mask.dtype == np.bool_, f"state mask dtype is {mask.dtype}, expected bool")
    _assert(mask[7], "DO_NOTHING is not available")
    _assert(not np.any(mask[:4]), f"loaded excavator movement/base-turn actions leaked valid bits: {mask}")

    skidsteer = _make_agent_state((16, 16), agent_type=2, action_type=0, loaded=0, shovel_lifted=0)
    skid_state = _make_state((skidsteer,), current_agent=0, num_agents=1)
    skid_mask, _ = skid_state._get_action_mask_and_edge_features(
        BatchConfig().action_type.do_nothing()
    )
    skid_mask = np.asarray(skid_mask)
    _assert(skid_mask[6], f"skidsteer DO should be valid when it toggles the shovel: {skid_mask}")
    _assert(not np.any(skid_mask[4:6]), f"skidsteer cabin actions should be disabled: {skid_mask}")
    _assert(skid_mask[7], "skidsteer DO_NOTHING is not available")

    loaded_wheeled_truck = _make_agent_state(
        (16, 16), agent_type=1, action_type=1, loaded=1
    )
    loaded_truck_state = _make_state((loaded_wheeled_truck,), current_agent=0, num_agents=1)
    loaded_truck_mask, _ = loaded_truck_state._get_action_mask_and_edge_features(
        BatchConfig().action_type.do_nothing()
    )
    loaded_truck_mask = np.asarray(loaded_truck_mask)
    _assert(
        not np.any(loaded_truck_mask[:2]),
        f"loaded wheeled truck movement should be masked as an easy no-op: {loaded_truck_mask}",
    )
    _assert(not np.any(loaded_truck_mask[4:6]), f"truck cabin actions should be disabled: {loaded_truck_mask}")
    _assert(loaded_truck_mask[6], f"loaded truck DO should remain policy-visible: {loaded_truck_mask}")

    empty_wheeled_truck = _make_agent_state(
        (16, 16), agent_type=1, action_type=1, loaded=0
    )
    empty_truck_state = _make_state((empty_wheeled_truck,), current_agent=0, num_agents=1)
    empty_truck_mask, _ = empty_truck_state._get_action_mask_and_edge_features(
        BatchConfig().action_type.do_nothing()
    )
    empty_truck_mask = np.asarray(empty_truck_mask)
    _assert(not empty_truck_mask[6], f"empty truck DO should be masked: {empty_truck_mask}")

    wheel_limit_agent = _make_agent_state((16, 16), agent_type=0, action_type=1)
    wheel_limit_agent = wheel_limit_agent._replace(
        wheel_angle=jnp.array([2], dtype=IntLowDim)
    )
    wheel_limit_state = _make_state((wheel_limit_agent,), current_agent=0, num_agents=1)
    wheel_limit_mask, _ = wheel_limit_state._get_action_mask_and_edge_features(
        BatchConfig().action_type.do_nothing()
    )
    wheel_limit_mask = np.asarray(wheel_limit_mask)
    _assert(not wheel_limit_mask[2], f"wheels-left at max angle should be masked: {wheel_limit_mask}")
    _assert(wheel_limit_mask[3], f"wheels-right from max angle should remain valid: {wheel_limit_mask}")

    info_mask = np.asarray(skid_state._get_infos(BatchConfig().action_type.do_nothing(), False)["action_mask"])
    _assert(np.array_equal(skid_mask, info_mask), "state info action_mask differs from direct mask")
    _, edge_features_jax = skid_state._get_action_mask_and_edge_features(
        BatchConfig().action_type.do_nothing()
    )
    edge_features = np.asarray(edge_features_jax)
    _assert(edge_features.shape == (10,), f"unexpected edge feature shape {edge_features.shape}")
    _assert(edge_features[0] == float(skid_mask[6]), "edge_features valid_do does not match action mask")
    _assert(0.0 <= edge_features[8] <= 1.0, "edge_features remaining-edge share is not normalized")
    print("PASS state-action-mask: current-agent comparisons, DO availability, info mask, and DO_NOTHING are correct")


def test_state_step_dispatch(disable_jit: bool) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_disable_jit", disable_jit)

    from terra.config import BatchConfig

    turn_left = BatchConfig().action_type.new(jnp.array([2], dtype=jnp.int8))

    tracked_agent = _make_agent_state((16, 16), agent_type=0, action_type=0)
    tracked_state = _make_state((tracked_agent,), current_agent=0, num_agents=1)
    tracked_next = tracked_state._step(turn_left, turn=False)
    _assert(
        not np.array_equal(
            np.asarray(tracked_next._get_current_agent_state().angle_base),
            np.asarray(tracked_state._get_current_agent_state().angle_base),
        ),
        "tracked action 2 did not turn the base",
    )
    _assert(
        np.array_equal(
            np.asarray(tracked_next._get_current_agent_state().wheel_angle),
            np.asarray(tracked_state._get_current_agent_state().wheel_angle),
        ),
        "tracked action 2 unexpectedly changed wheel angle",
    )

    wheeled_agent = _make_agent_state((16, 16), agent_type=0, action_type=1)
    wheeled_state = _make_state((wheeled_agent,), current_agent=0, num_agents=1)
    wheeled_next = wheeled_state._step(turn_left, turn=False)
    _assert(
        not np.array_equal(
            np.asarray(wheeled_next._get_current_agent_state().wheel_angle),
            np.asarray(wheeled_state._get_current_agent_state().wheel_angle),
        ),
        "wheeled action 2 did not turn the wheels",
    )
    _assert(
        np.array_equal(
            np.asarray(wheeled_next._get_current_agent_state().angle_base),
            np.asarray(wheeled_state._get_current_agent_state().angle_base),
        ),
        "wheeled action 2 unexpectedly changed base angle",
    )

    print("PASS state-step-dispatch: tracked and wheeled primitive routing is preserved")


def test_synthetic_env_action_mask(
    agent_types: tuple[int, ...],
    action_types: tuple[int, ...],
    seed: int,
    expect_cabin_disabled: bool,
    disable_jit: bool,
    map_size: int = 32,
) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_disable_jit", disable_jit)

    from terra.config import AgentConfig, EnvConfig
    from terra.env import TerraEnv

    agent_cfg = AgentConfig(width=3, height=3, move_tiles=3)
    env_cfg = EnvConfig()._replace(
        agent=agent_cfg,
        maps=EnvConfig().maps._replace(edge_length_px=map_size),
        tile_size=1.0,
        max_steps_in_episode=16,
        agent_types=agent_types,
        action_types=action_types,
    )
    target_map = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    padding_mask = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    action_map = jnp.zeros((map_size, map_size), dtype=jnp.int8)
    dumpability = jnp.ones((map_size, map_size), dtype=jnp.bool_)
    trench_axes = jnp.zeros((3, 3), dtype=jnp.float32)
    foundation_border_axes = jnp.zeros((64, 3), dtype=jnp.float32)
    distance_map = jnp.zeros((map_size, map_size), dtype=jnp.float32)

    env = TerraEnv.new(maps_size_px=map_size, rendering=False)
    timestep = env.reset(
        jax.random.PRNGKey(seed),
        target_map,
        padding_mask,
        trench_axes,
        jnp.int32(0),
        foundation_border_axes,
        jnp.int32(0),
        dumpability,
        action_map,
        distance_map,
        env_cfg,
    )
    reset_obs_mask = np.asarray(timestep.observation["action_mask"])
    reset_info_mask = np.asarray(timestep.info["action_mask"])
    reset_edge_features = np.asarray(timestep.observation["edge_features"])
    _assert(reset_obs_mask.shape == (8,), f"unexpected synthetic reset mask shape {reset_obs_mask.shape}")
    _assert(np.array_equal(reset_obs_mask, reset_info_mask), "synthetic reset obs/info masks differ")
    _assert(reset_obs_mask[7], "synthetic reset DO_NOTHING is not available")
    _assert(np.any(reset_obs_mask), "synthetic reset contains an all-false mask")
    if expect_cabin_disabled:
        _assert(not np.any(reset_obs_mask[4:6]), f"synthetic reset cabin actions should be disabled: {reset_obs_mask}")
    _assert(reset_edge_features.shape == (10,), f"unexpected synthetic reset edge feature shape {reset_edge_features.shape}")
    _assert(np.array_equal(reset_edge_features, np.asarray(timestep.info["edge_features"])), "synthetic reset obs/info edge features differ")
    _assert_reward_components(timestep.info, "synthetic reset")

    from terra.config import BatchConfig

    action = BatchConfig().action_type.do_nothing()
    next_timestep = env.step(
        timestep.state,
        action,
        target_map,
        padding_mask,
        trench_axes,
        jnp.int32(0),
        foundation_border_axes,
        jnp.int32(0),
        dumpability,
        action_map,
        distance_map,
        env_cfg,
    )
    step_obs_mask = np.asarray(next_timestep.observation["action_mask"])
    step_info_mask = np.asarray(next_timestep.info["action_mask"])
    step_edge_features = np.asarray(next_timestep.observation["edge_features"])
    _assert(step_obs_mask.shape == (8,), f"unexpected synthetic step mask shape {step_obs_mask.shape}")
    _assert(np.array_equal(step_obs_mask, step_info_mask), "synthetic step obs/info masks differ")
    _assert(step_obs_mask[7], "synthetic step DO_NOTHING is not available")
    _assert(np.any(step_obs_mask), "synthetic step contains an all-false mask")
    if expect_cabin_disabled:
        _assert(not np.any(step_obs_mask[4:6]), f"synthetic step cabin actions should be disabled: {step_obs_mask}")
    _assert(step_edge_features.shape == (10,), f"unexpected synthetic step edge feature shape {step_edge_features.shape}")
    _assert(np.array_equal(step_edge_features, np.asarray(next_timestep.info["edge_features"])), "synthetic step obs/info edge features differ")
    _assert_reward_components(next_timestep.info, "synthetic step")

    boundary_state = timestep.state._replace(
        env_steps=timestep.env_cfg.max_steps_in_episode
    )
    boundary_timestep = env.step(
        boundary_state,
        action,
        target_map,
        padding_mask,
        trench_axes,
        jnp.int32(0),
        foundation_border_axes,
        jnp.int32(0),
        dumpability,
        action_map,
        distance_map,
        env_cfg,
    )
    _assert(bool(np.asarray(boundary_timestep.done)), "pre-step max boundary did not terminate")
    _assert_reward_components(boundary_timestep.info, "synthetic boundary step")

    print(
        "PASS synthetic-env-action-mask: "
        f"TerraEnv reset/step action_mask wiring works for agent_types={agent_types} "
        f"action_types={action_types}"
    )


def _run_cases(cases: Iterable[str], args: argparse.Namespace) -> None:
    for case in cases:
        if case == "ppo-mask":
            test_ppo_mask()
        elif case == "training-accounting":
            test_training_accounting()
        elif case == "model-policy":
            test_model_policy()
        elif case == "state-action-mask":
            test_state_action_mask(disable_jit=args.disable_jit)
        elif case == "state-step-dispatch":
            test_state_step_dispatch(disable_jit=args.disable_jit)
        elif case == "synthetic-env-action-mask":
            test_synthetic_env_action_mask(
                agent_types=args.agent_types,
                action_types=args.action_types,
                seed=args.seed,
                expect_cabin_disabled=args.expect_cabin_disabled,
                disable_jit=args.disable_jit,
                map_size=args.synthetic_map_size,
            )
        elif case == "env-action-mask":
            test_env_action_mask(
                map_path=args.map_path,
                agent_types=args.agent_types,
                action_types=args.action_types,
                num_envs=args.num_envs,
                seed=args.seed,
                expect_cabin_disabled=args.expect_cabin_disabled,
                disable_jit=args.disable_jit,
            )
        elif case == "synthetic-batch-step-fast-reset":
            test_synthetic_batch_step_fast_reset_parity(
                agent_types=args.agent_types,
                action_types=args.action_types,
                num_envs=args.num_envs,
                seed=args.seed,
                disable_jit=args.disable_jit,
                map_size=args.synthetic_map_size,
            )
        elif case == "synthetic-step-fast-reset":
            test_synthetic_step_fast_reset_parity(
                agent_types=args.agent_types,
                action_types=args.action_types,
                seed=args.seed,
                disable_jit=args.disable_jit,
                map_size=args.synthetic_map_size,
            )
        else:
            raise ValueError(f"unknown case: {case}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "ppo-mask",
            "training-accounting",
            "model-policy",
            "state-action-mask",
            "state-step-dispatch",
            "synthetic-env-action-mask",
            "env-action-mask",
            "synthetic-batch-step-fast-reset",
            "synthetic-step-fast-reset",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--jax-platforms", default="cpu")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-size", type=int, default=1)
    parser.add_argument("--map-path", default="foundations_real_ring")
    parser.add_argument("--agent-types", type=_parse_int_tuple, default=(0,))
    parser.add_argument("--action-types", type=_parse_int_tuple, default=(0,))
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expect-cabin-disabled", action="store_true")
    parser.add_argument("--disable-jit", action="store_true")
    parser.add_argument("--synthetic-map-size", type=int, default=32)
    args = parser.parse_args()

    os.environ["JAX_PLATFORMS"] = args.jax_platforms
    if args.dataset_path is not None:
        os.environ["DATASET_PATH"] = args.dataset_path
    os.environ["DATASET_SIZE"] = str(args.dataset_size)

    if args.case in ("all", "env-action-mask"):
        _assert(os.environ.get("DATASET_PATH", "") != "", "DATASET_PATH must be set for env-action-mask")

    if args.case == "all":
        cases = (
            "ppo-mask",
            "training-accounting",
            "model-policy",
            "state-action-mask",
            "state-step-dispatch",
            "synthetic-env-action-mask",
            "env-action-mask",
            "synthetic-batch-step-fast-reset",
            "synthetic-step-fast-reset",
        )
    else:
        cases = (args.case,)
    _run_cases(cases, args)


if __name__ == "__main__":
    main()
