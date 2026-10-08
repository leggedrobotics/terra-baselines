"""Verify the initial student independently of the frozen teacher's weights."""

import numpy as np
from flax.traverse_util import flatten_dict


def verify_scratch_student_initialization(config, train_state, teacher_params):
    """Return a small startup report; reject a copied actor or critic.

    Called only for a fresh scratch run, after loading its separate teacher and
    before optimization. Common parameter shapes are compared explicitly; the
    student's extra precision input cannot make this check pass by itself.
    Resumed runs retain their original initialization history and skip this.
    """
    option = (lambda name: config.get(name)) if isinstance(config, dict) else (
        lambda name: getattr(config, name, None)
    )
    if option("warm_start_from") is not None or option("resume_from") is not None:
        raise ValueError("scratch initialization cannot restore student parameters")
    if option("resume_update") is not None:
        raise ValueError("scratch initialization cannot restore an update counter")
    step = int(np.asarray(train_state.step).reshape(()))
    if step != 0:
        raise ValueError("scratch initialization must be checked before optimizer updates")
    if teacher_params is None:
        raise ValueError("scratch initialization check requires the separately loaded teacher")

    student = flatten_dict(train_state.params)
    teacher = flatten_dict(teacher_params)
    groups = {
        "actor": {"actor_pre_gru", "actor_gru", "actor_post_gru", "mlp_pi"},
        "critic": {"mlp_v"},
    }
    comparisons = {}
    for label, names in groups.items():
        compared = distinct = 0
        for path, value in student.items():
            if not any(part in names for part in path) or path not in teacher:
                continue
            current, reference = np.asarray(value), np.asarray(teacher[path])
            if current.shape != reference.shape:
                continue
            if not np.isfinite(current).all() or not np.isfinite(reference).all():
                raise ValueError(f"{label} initialization contains nonfinite parameters")
            compared += 1
            distinct += int(not np.array_equal(current, reference))
        if compared == 0:
            raise ValueError(f"no same-shape {label} parameters available for scratch verification")
        if distinct == 0:
            raise ValueError(f"scratch student's {label} parameters equal the teacher")
        comparisons[label] = {
            "same_shape_parameter_arrays_compared": compared,
            "parameter_arrays_distinct_from_teacher": distinct,
        }
    return {
        "initialization": "scratch",
        "student_checkpoint": None,
        "optimizer_step": step,
        "teacher_checkpoint": option("teacher_checkpoint"),
        "actor_and_critic_distinct_from_teacher": True,
        "comparisons": comparisons,
    }
