import jax.numpy as jnp


def _masked_logit_value(dtype):
    if dtype == jnp.float16:
        return jnp.asarray(-1e4, dtype=dtype)
    return jnp.asarray(-1e9, dtype=dtype)


def apply_action_mask(logits, action_mask):
    """Mask unavailable categorical actions before sampling/log-prob/entropy."""
    if action_mask.shape != logits.shape:
        raise ValueError(
            f"action_mask shape {action_mask.shape} must match logits shape {logits.shape}"
        )
    action_mask = action_mask.astype(jnp.bool_)
    return jnp.where(
        action_mask,
        logits,
        jnp.full_like(logits, _masked_logit_value(logits.dtype)),
    )
