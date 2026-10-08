"""Prevent a teacher checkpoint from silently becoming a scratch student."""

import copy
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from train_mixed import _checkpoint_load_mode
from utils.scratch_teacher import verify_scratch_student_initialization


def config(**changes):
    values = dict(warm_start_from=None, resume_from=None, resume_update=None,
                  teacher_checkpoint="gru110000.pkl", teacher_bulk_compatibility=True)
    values.update(changes)
    return SimpleNamespace(**values)


def parameters(value, inputs):
    return {"params": {
        "actor_pre_gru": {"kernel": jnp.full((3, 4), value, jnp.float32)},
        "actor_post_gru": {"kernel": jnp.full((4, 8), value, jnp.float32)},
        "mlp_v": {"kernel": jnp.full((3, 1), value, jnp.float32)},
        "maps_net": {"cnn": {"Conv_0": {
            "kernel": jnp.full((3, 3, inputs, 2), value, jnp.float32),
        }}},
    }}


def test_teacher_source_alone_is_not_a_student_checkpoint():
    cfg = config()
    assert _checkpoint_load_mode(cfg) is None
    student, teacher = parameters(2, 13), parameters(1, 12)
    before = {path: np.asarray(value).copy() for path, value in flatten_dict(teacher).items()}
    report = verify_scratch_student_initialization(
        cfg, SimpleNamespace(params=student, step=jnp.int32(0)), teacher,
    )
    assert report["initialization"] == "scratch"
    assert report["actor_and_critic_distinct_from_teacher"]
    for path, value in flatten_dict(teacher).items():
        np.testing.assert_array_equal(value, before[path])


@pytest.mark.parametrize("copied_group", ["actor", "critic", "both"])
def test_new_precision_channel_cannot_hide_copied_teacher_heads(copied_group):
    teacher = parameters(1, 12)
    student = parameters(2, 13)
    if copied_group in ("actor", "both"):
        for name in ("actor_pre_gru", "actor_post_gru"):
            student["params"][name] = teacher["params"][name]
    if copied_group in ("critic", "both"):
        student["params"]["mlp_v"] = teacher["params"]["mlp_v"]
    with pytest.raises(ValueError, match="parameters equal the teacher"):
        verify_scratch_student_initialization(
            config(), SimpleNamespace(params=student, step=0), teacher,
        )


@pytest.mark.parametrize("changes", [
    {"warm_start_from": "teacher.pkl"}, {"resume_from": "student.pkl"}, {"resume_update": 7},
])
def test_restore_configuration_cannot_be_reported_as_scratch(changes):
    with pytest.raises(ValueError, match="cannot restore"):
        verify_scratch_student_initialization(
            config(**changes), SimpleNamespace(params=parameters(2, 13), step=0), parameters(1, 12),
        )


def test_nonzero_optimizer_step_cannot_be_reported_as_initialization():
    with pytest.raises(ValueError, match="before optimizer updates"):
        verify_scratch_student_initialization(
            config(), SimpleNamespace(params=parameters(2, 13), step=1), parameters(1, 12),
        )


def test_actual_gru110000_actor_and_critic_are_not_loaded_into_scratch_student():
    checkpoint_path = os.environ.get("TERRA_PRECISION_SOURCE_CHECKPOINT")
    if checkpoint_path is None:
        pytest.skip("set TERRA_PRECISION_SOURCE_CHECKPOINT for the actual GRU110000 check")
    from terra.config import BatchConfig, MapsDimsConfig
    from utils.helpers import checkpoint_evaluation_config, load_pkl_object, register_checkpoint_config_classes
    from utils.models import get_model_ready

    register_checkpoint_config_classes()
    teacher = load_pkl_object(checkpoint_path)
    cfg = copy.deepcopy(checkpoint_evaluation_config(teacher))
    cfg.warm_start_from = cfg.resume_from = cfg.resume_update = None
    cfg.teacher_checkpoint = checkpoint_path
    cfg.precision_required_band_observation = True
    env = SimpleNamespace(
        batch_cfg=BatchConfig(maps_dims=MapsDimsConfig(maps_edge_length=64)),
        executable_dig_observation=True,
    )
    _, initialized = get_model_ready(jax.random.PRNGKey(20261007), cfg, env)
    report = verify_scratch_student_initialization(
        cfg, SimpleNamespace(params=initialized, step=0), teacher["model"],
    )
    assert report["actor_and_critic_distinct_from_teacher"]
    assert report["comparisons"]["actor"]["parameter_arrays_distinct_from_teacher"] > 0
    assert report["comparisons"]["critic"]["parameter_arrays_distinct_from_teacher"] > 0
    print("ACTUAL_SCRATCH_INITIALIZATION", report, flush=True)
