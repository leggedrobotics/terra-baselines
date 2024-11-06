import jax
import jax.numpy as jnp

seed = 43
rng = jax.random.PRNGKey(seed)
rng, rng_step = jax.random.split(rng, 2)
print(rng_step)
print(rng)

def custom_dot(x,y):
    return x+y

x = jnp.asarray([
    1,
    2
])

y = jnp.asarray([
    1,
    2
])

print(jax.vmap(custom_dot)(x, y))

def init_timestep_mcts(timestep, n_env_per_step):
    # Apply a custom function to each element in the nested structure of timestep
    timestep_expanded = jax.tree_map(
        lambda x: jnp.concatenate([x[[i], ...].repeat(n_env_per_step, axis=0) for i in range(x.shape[0])], axis=0),
        timestep
    )
    return timestep_expanded

# Example usage with a nested dictionary containing arrays of different dimensions
ex = init_timestep_mcts({"A": {"A1": jnp.array([1, 2])}, "B": jnp.array([1, 2])}, 10)
print(ex)