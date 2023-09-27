import jax
import math
import wandb
from jax import Array
from typing import Any
from terra.config import EnvConfig
import numpy as np
from terra.config import MapType
from terra.config import RewardsType

class Curriculum:

    def __init__(self, rl_config, n_devices) -> None:
        self.n_devices = n_devices
        self.num_dof_random = int(rl_config["random_dof_ratio"] * rl_config["num_train_envs"])
        print(f"Number of random dofs = {self.num_dof_random} / {rl_config['num_train_envs']}")
        self.num_dof = rl_config["num_train_envs"] - self.num_dof_random
        self.dofs_main = np.zeros((self.num_dof,), dtype=np.int8)
        self.dofs_random = np.zeros((self.num_dof_random,), dtype=np.int8)
        self.dofs = np.concatenate((self.dofs_main, self.dofs_random), axis=0)

        self.rl_config = rl_config
        # self.increase_dof_threshold = rl_config["increase_dof_threshold"]
        self.increase_dof_consecutive_episodes = rl_config["increase_dof_consecutive_episodes"]

        self.max_increase_dof_ratio = rl_config["max_increase_dof_ratio"]
        self.max_decrease_dof_ratio = rl_config["max_decrease_dof_ratio"]

        self.curriculum_dicts = [
            {
                "map_width": -1,  # in meters
                "map_height": -1,  # in meters
                "max_steps_in_episode": 300,
                "map_type": MapType.TRENCHES,
                "rewards_type": RewardsType.DENSE,
            },
        ]

        # Set selection of random dofs here (only active through 'last_dof_type' in rl config)
        self.dof_selection = np.array([12, 13, 14], dtype=np.int8)

        self.curriculum_len = len(self.curriculum_dicts)
        self.idx_sparse_rewards_levels = [i for i, level in enumerate(self.curriculum_dicts) if level["rewards_type"] == RewardsType.SPARSE]
        self.n_sparse_levels = len(self.idx_sparse_rewards_levels)
        print(f"{self.idx_sparse_rewards_levels=}")

        # Get even eval dofs
        n_eval_each = rl_config["num_test_rollouts"] // self.curriculum_len
        n_mod = rl_config["num_test_rollouts"] % self.curriculum_len
        eval_dofs =  [i for _ in range(n_eval_each) for i in range(self.curriculum_len)]
        last = eval_dofs[-1]
        eval_dofs += [last for _ in range(n_mod)]
        self.dofs_eval = np.array(eval_dofs, dtype=np.int8)

        # Compute the minimum number of embedding length for the one-hot features
        self.num_embeddings_agent_min = max([max([el["map_width"], el["map_height"]]) for el in self.curriculum_dicts])
        print(f"{self.num_embeddings_agent_min=}")

        # For each main env, the number of updates since the last done
        self.max_episodes_no_dones = rl_config["max_episodes_no_dones"]
        self.n_gae_steps = rl_config["n_steps"]
        self.updates_no_dones = np.zeros(self.dofs_main.shape, dtype=np.int16)
        self.updates_dones = np.zeros(self.dofs_main.shape, dtype=np.int16)

    def _evaluate_progress(self, metrics_dict: dict[str, Any], terminated: Array, timeouts: Array) -> int:
        """
        Goes from the training metrics to a DoF (degrees of freedom) value,
        considering the current DoF.
        """
        terminated = terminated.reshape(-1)
        timeouts = timeouts.reshape(-1)
        terminated = terminated[:self.num_dof]
        timeouts = timeouts[:self.num_dof]

        # values_individual = metrics_dict["values_individual"]
        # values_individual = values_individual[:self.num_dof]
        # targets_individual = metrics_dict["targets_individual"]
        # targets_individual = targets_individual[:self.num_dof]
        # value_losses_individual = np.square(values_individual - targets_individual)

        # Increase dof
        # [method 1: using value loss]
        # increase_dof = (value_losses_individual / targets_individual < self.increase_dof_threshold) * (targets_individual > 0)
        # increase_dof *= dones_after_update  # only update dof if it completed the previous level
        # [method 2: using consecutive successful terminations]
        self.updates_dones += terminated
        self.updates_dones = np.where(
            timeouts,
            0,
            self.updates_dones
        )
        increase_dof = self.updates_dones > self.increase_dof_consecutive_episodes

        # Decrease dof
        self.updates_no_dones += timeouts
        self.updates_no_dones = np.where(
            terminated,
            0,
            self.updates_no_dones
        )
        decrease_dof = self.updates_no_dones > self.max_episodes_no_dones
        decrease_dof *= self.dofs_main > 0  # make sure the dofs are not decreased to negative numbers

        # Limit the numbre of configs that can increase at a given step
        max_increase_dof_ratio_abs = int(self.max_increase_dof_ratio * increase_dof.shape[0])
        increase_dof_cumsum = np.cumsum(increase_dof)
        max_increase_dof_ratio_mask = increase_dof_cumsum < max_increase_dof_ratio_abs
        increase_dof *= max_increase_dof_ratio_mask

        # Limit the numbre of configs that can decrease at a given step
        max_decrease_dof_ratio_abs = int(self.max_decrease_dof_ratio * decrease_dof.shape[0])
        decrease_dof_cumsum = np.cumsum(decrease_dof)
        max_decrease_dof_ratio_mask = decrease_dof_cumsum < max_decrease_dof_ratio_abs
        decrease_dof *= max_decrease_dof_ratio_mask
        
        # assert((increase_dof @ decrease_dof).item() == 0, f"{(increase_dof @ decrease_dof).item()=}")

        dofs = self.dofs_main + increase_dof.astype(np.int8) - decrease_dof.astype(np.int8)

        if self.rl_config["last_dof_type"] == "random":
            # last dof level at random
            random_dofs = np.random.randint(0, self.curriculum_len, (dofs.shape[0],), dtype=np.int8)
            dofs = np.where(dofs < self.curriculum_len, dofs, random_dofs)
        elif self.rl_config["last_dof_type"] == "none":
            # cap dof to last level
            dofs = np.where(dofs < self.curriculum_len, dofs, self.curriculum_len - 1)
        elif self.rl_config["last_dof_type"] == "sparse":
            # last dof level at random from all sparse reward levels
            random_dofs = np.random.randint(0, self.n_sparse_levels, (dofs.shape[0],), dtype=np.int8)
            random_dofs_sparse_rewards = np.array([self.idx_sparse_rewards_levels[i] for i in random_dofs.tolist()])
            dofs = np.where(dofs < self.curriculum_len, dofs, random_dofs_sparse_rewards)
        elif self.rl_config["last_dof_type"] == "random_from_selection":
            random_dofs = np.random.choice(self.dof_selection, (self.dofs.shape[0],))
            dofs = np.where(dofs < self.curriculum_len, dofs, random_dofs)
        else:
            raise(ValueError(f"{self.rl_config['last_dof_type']=} does not exist."))
        
        self.dofs_main = dofs.astype(np.int8)
        
        # Random dofs
        unlocked_dofs = np.arange(self.dofs.max() + 1)
        self.dofs_random = np.random.choice(unlocked_dofs, (self.num_dof_random,)).astype(np.int8)

        # Combine the two
        self.dofs = np.concatenate((self.dofs_main, self.dofs_random), axis=0)

        # Update vars for next iteration
        dof_changed = increase_dof | decrease_dof
        self.updates_dones *= (~dof_changed)  # set to 0 if dof has just changed
        self.updates_no_dones *= (~dof_changed)  # set to 0 if dof has just changed

        # Logging
        wandb.log(
            {
                "increase_dof": increase_dof.sum(),
                "decrease_dof": decrease_dof.sum(),
            }
        )
    
    def _get_dofs_count_dict(self,):
        dof_counts = [sum(1 for dof in self.dofs if dof == j) for j in range(self.curriculum_len)]
        dofs_count_dict = {
            f"curriculum dof count {dof}": count
            for dof, count in zip(range(self.curriculum_len), dof_counts)
        }
        return dofs_count_dict
    
    def _get_dofs_count_dict_eval(self,):
        dof_counts = [sum(1 for dof in self.dofs_eval if dof == j) for j in range(self.curriculum_len)]
        dofs_count_dict = {
            f"curriculum dof count {dof}": count
            for dof, count in zip(range(self.curriculum_len), dof_counts)
        }
        return dofs_count_dict
    
    def get_cfgs(self, metrics_dict: dict[str, Any], terminated: Array, timeouts: Array):
        self._evaluate_progress(metrics_dict, terminated, timeouts)
        
        map_widths = [self.curriculum_dicts[dof]["map_width"] for dof in self.dofs]
        map_heights = [self.curriculum_dicts[dof]["map_height"] for dof in self.dofs]
        max_steps_in_episodes = [self.curriculum_dicts[dof]["max_steps_in_episode"] for dof in self.dofs]
        map_types = [self.curriculum_dicts[dof]["map_type"] for dof in self.dofs]
        rewards_type = [self.curriculum_dicts[dof]["rewards_type"] for dof in self.dofs]
        apply_trench_rewards = [self.curriculum_dicts[dof]["map_type"] == MapType.TRENCHES for dof in self.dofs]

        env_cfgs = jax.vmap(EnvConfig.parametrized)(
            np.array(map_widths),
            np.array(map_heights),
            np.array(max_steps_in_episodes),
            np.array(self.dofs),
            np.array(map_types),
            np.array(rewards_type),
            np.array(apply_trench_rewards),
            )
        env_cfgs = jax.tree_map(
            lambda x: jax.numpy.reshape(x, (self.n_devices, x.shape[0] // self.n_devices, *x.shape[1:])), env_cfgs
        )
        dofs_count_dict = self._get_dofs_count_dict()
        return env_cfgs, dofs_count_dict

    def get_cfgs_init(self,):
        map_widths = [self.curriculum_dicts[0]["map_width"] for _ in self.dofs]
        map_heights = [self.curriculum_dicts[0]["map_height"] for _ in self.dofs]
        max_steps_in_episodes = [self.curriculum_dicts[0]["max_steps_in_episode"] for _ in self.dofs]
        map_types = [self.curriculum_dicts[0]["map_type"] for _ in self.dofs]
        rewards_type = [self.curriculum_dicts[0]["rewards_type"] for _ in self.dofs]
        apply_trench_rewards = [self.curriculum_dicts[0]["map_type"] == MapType.TRENCHES for _ in self.dofs]
        

        env_cfgs = jax.vmap(EnvConfig.parametrized)(
            np.array(map_widths),
            np.array(map_heights),
            np.array(max_steps_in_episodes),
            np.array(self.dofs),
            np.array(map_types),
            np.array(rewards_type),
            np.array(apply_trench_rewards),
            )
        env_cfgs = jax.tree_map(
            lambda x: jax.numpy.reshape(x, (self.n_devices, x.shape[0] // self.n_devices, *x.shape[1:])), env_cfgs
        )
        dofs_count_dict = self._get_dofs_count_dict()
        return env_cfgs, dofs_count_dict
    
    def get_cfgs_eval(self,):
        map_widths = [self.curriculum_dicts[dof]["map_width"] for dof in self.dofs_eval]
        map_heights = [self.curriculum_dicts[dof]["map_height"] for dof in self.dofs_eval]
        max_steps_in_episodes = [self.curriculum_dicts[dof]["max_steps_in_episode"] for dof in self.dofs_eval]
        map_types = [self.curriculum_dicts[dof]["map_type"] for dof in self.dofs_eval]
        rewards_type = [self.curriculum_dicts[dof]["rewards_type"] for dof in self.dofs_eval]
        apply_trench_rewards = [self.curriculum_dicts[dof]["map_type"] == MapType.TRENCHES for dof in self.dofs_eval]

        env_cfgs = jax.vmap(EnvConfig.parametrized)(
            np.array(map_widths),
            np.array(map_heights),
            np.array(max_steps_in_episodes),
            np.array(self.dofs_eval),
            np.array(map_types),
            np.array(rewards_type),
            np.array(apply_trench_rewards)
            )
        
        dofs_count_dict = self._get_dofs_count_dict_eval()
        
        return env_cfgs, dofs_count_dict

    def _get_curriculum(self, idx: int):
        return self.curriculum_dicts[idx]

    def get_num_embeddings_agent_min(self,) -> int:
        return self.num_embeddings_agent_min
