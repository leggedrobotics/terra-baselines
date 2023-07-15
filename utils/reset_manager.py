import jax
import jax.numpy as jnp

class ResetManager:
    def __init__(self, rl_config, observation_shapes, eval=False) -> None:
        self.activate_reset_manager = rl_config["activate_reset_manager"]
        self.context_length = rl_config["context_length"]
        if eval:
            self.num_envs = rl_config["num_test_rollouts"]
        else:
            self.num_envs = rl_config["num_train_envs"]
        self.reset(observation_shapes)

    def reset(self, observation_shapes):
        w, h = observation_shapes["action_map"]
        self.context_dict = {
            "action_maps": jnp.empty((self.context_length, self.num_envs, w, h)),
            "actions": jnp.empty((self.context_length, self.num_envs)),
        }
        self.step_idx = 0

    def _update_buffer(
            self,
            action_map,
            actions
    ):
        actions = actions.action[..., 0]
        self.context_dict["action_maps"] = jnp.concatenate(
            (
                self.context_dict["action_maps"][1:],
                action_map[None]
            ),
            axis=0
        )
        self.context_dict["actions"] = jnp.concatenate(
            (
                self.context_dict["actions"][1:],
                actions[None]
            ),
            axis=0
        )

        self.step_idx += 1


    def update(self, obs, actions):
        """
        Checks for:
        1. Unsolvable maps
        2. Infinite loops of actions
        """
        if not self.activate_reset_manager:
            return self.dummy()

        # Update buffer
        self._update_buffer(obs["action_map"], actions)

        # Get reset array
        if self.step_idx <= self.context_length:
            return jnp.zeros((self.num_envs,), dtype=jnp.bool_)

        # 1. Unsolvable maps
        # TODO
        
        # 2. Infinite loops of actions
        # Always the same action
        a = self.context_dict["actions"]
        same_action = ((a - a[..., [0]]).sum(-1) == 0).astype(jnp.bool_)
        # Two recurring actions
        a1 = self.context_dict["actions"][..., ::2]
        a2 = self.context_dict["actions"][..., 1::2]
        recurring_action1 = (a1 - a1[..., [0]]).sum(-1) == 0
        recurring_action2 = (a2 - a2[..., [0]]).sum(-1) == 0
        recurring_action = (recurring_action1 * recurring_action2).astype(jnp.bool_)
        infinite_action_loop = same_action | recurring_action

        jax.debug.print("infinite_action_loop = {x}", x=infinite_action_loop)
        assert infinite_action_loop.shape == (self.num_envs,)
        return infinite_action_loop

    def dummy(self,):
        return jnp.zeros((self.num_envs,), dtype=jnp.bool_)
