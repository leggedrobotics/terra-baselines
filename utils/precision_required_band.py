"""One-channel parameter migration for a precision-mask warm start."""

import jax.numpy as jnp
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict


STEM_KERNEL_PATH = ("params", "maps_net", "cnn", "Conv_0", "kernel")


def migrate_precision_required_band_params(source_params, target_params):
    """Copy every old weight and append an exactly zero CNN input slice.

    This is a parameters-only warm-start migration. The trainer separately
    verifies that only precision_required_band_observation was enabled and
    constructs a fresh optimizer. No unrelated shape/key change is accepted.
    """
    source = flatten_dict(source_params)
    target = flatten_dict(target_params)
    if source.keys() != target.keys() or STEM_KERNEL_PATH not in source:
        raise ValueError("precision-mask migration requires identical parameter keys and a CNN stem")
    for path in source:
        if path != STEM_KERNEL_PATH and jnp.shape(source[path]) != jnp.shape(target[path]):
            raise ValueError(f"precision-mask migration cannot change {'/'.join(path)}")
        if source[path].dtype != target[path].dtype:
            raise ValueError(f"precision-mask migration cannot change dtype of {'/'.join(path)}")

    kernel = source[STEM_KERNEL_PATH]
    wanted = jnp.shape(target[STEM_KERNEL_PATH])
    if kernel.ndim != 4 or wanted != (*kernel.shape[:2], kernel.shape[2] + 1, kernel.shape[3]):
        raise ValueError("precision-mask migration must append exactly one CNN input channel")
    source[STEM_KERNEL_PATH] = jnp.concatenate(
        (kernel, jnp.zeros((*kernel.shape[:2], 1, kernel.shape[3]), dtype=kernel.dtype)), axis=2
    )
    migrated = unflatten_dict(source)
    return FrozenDict(migrated) if isinstance(target_params, FrozenDict) else migrated
