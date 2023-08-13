import jax
from jax import Array
from typing import Any
from terra.config import EnvConfig
import numpy as np
from terra.config import MapType
from terra.config import RewardsType

class Curriculum:

    def __init__(self, rl_config) -> None:
        self.dofs = np.zeros((rl_config["num_train_envs"],), dtype=np.int8)
        self.rl_config = rl_config
        self.change_dof_threshold = rl_config["change_dof_threshold"]
        self.max_change_ratio = rl_config["max_change_ratio"]

        self.curriculum_dicts = [
            # TRENCHES
            # {
            #     "map_width": -1,  # in meters
            #     "map_height": -1,  # in meters
            #     "max_steps_in_episode": 150,
            #     "map_type": MapType.TRENCHES,
            #     "rewards_type": RewardsType.DENSE,
            # },
            # {
            #     "map_width": -1,  # in meters
            #     "map_height": -1,  # in meters
            #     "max_steps_in_episode": 150,
            #     "map_type": MapType.TRENCHES,
            #     "rewards_type": RewardsType.SPARSE,
            # },
            # {
            #     "map_width": -1,  # in meters
            #     "map_height": -1,  # in meters
            #     "max_steps_in_episode": 150,
            #     "map_type": MapType.TRENCHES,
            #     "rewards_type": RewardsType.DENSE,
            # },
            # {
            #     "map_width": -1,  # in meters
            #     "map_height": -1,  # in meters
            #     "max_steps_in_episode": 150,
            #     "map_type": MapType.TRENCHES,
            #     "rewards_type": RewardsType.SPARSE,
            # },
            # {
            #     "map_width": -1,  # in meters
            #     "map_height": -1,  # in meters
            #     "max_steps_in_episode": 200,
            #     "map_type": MapType.TRENCHES,
            #     "rewards_type": RewardsType.DENSE,
            # },
            # {
            #     "map_width": -1,  # in meters
            #     "map_height": -1,  # in meters
            #     "max_steps_in_episode": 200,
            #     "map_type": MapType.TRENCHES,
            #     "rewards_type": RewardsType.SPARSE,
            # },

            # FOUNDATIONS
            {
                "map_width": -1,  # in meters
                "map_height": -1,  # in meters
                "max_steps_in_episode": 300,
                "map_type": MapType.FOUNDATIONS,
                "rewards_type": RewardsType.DENSE,
            },
            {
                "map_width": -1,  # in meters
                "map_height": -1,  # in meters
                "max_steps_in_episode": 300,
                "map_type": MapType.FOUNDATIONS,
                "rewards_type": RewardsType.SPARSE,
            },
        ]

        self.curriculum_len = len(self.curriculum_dicts)

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

    def _evaluate_progress(self, metrics_dict: dict[str, Any], dones_after_update: Array) -> int:
        """
        Goes from the training metrics to a DoF (degrees of freedom) value,
        considering the current DoF.
        """

        values_individual = metrics_dict["values_individual"]
        targets_individual = metrics_dict["targets_individual"]
        value_losses_individual = np.square(values_individual - targets_individual)

        increase_dof = (value_losses_individual / targets_individual < self.change_dof_threshold) * (targets_individual > 0)
        increase_dof *= dones_after_update  # only update dof if it completed the previous level

        # Limit the numbre of configs that can change at a given step
        max_change_ratio_abs = int(self.max_change_ratio * increase_dof.shape[0])
        increase_dof_cumsum = np.cumsum(increase_dof)
        max_change_ratio_mask = increase_dof_cumsum < max_change_ratio_abs
        increase_dof *= max_change_ratio_mask
        
        dofs = self.dofs + increase_dof.astype(np.int8)

        if self.rl_config["last_dof_random"]:
            # last dof level at random
            random_dofs = np.random.randint(0, self.curriculum_len, (dofs.shape[0],), dtype=np.int8)
            dofs = np.where(dofs < self.curriculum_len, dofs, random_dofs)
        else:
            # cap dof to last level
            dofs = np.where(dofs < self.curriculum_len, dofs, self.curriculum_len - 1)
        
        self.dofs = dofs.astype(np.int8)
    
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
    
    def get_cfgs(self, metrics_dict: dict[str, Any], dones_after_update: Array):
        self._evaluate_progress(metrics_dict, dones_after_update)
        
        map_widths = [self.curriculum_dicts[dof]["map_width"] for dof in self.dofs]
        map_heights = [self.curriculum_dicts[dof]["map_height"] for dof in self.dofs]
        max_steps_in_episodes = [self.curriculum_dicts[dof]["max_steps_in_episode"] for dof in self.dofs]
        map_types = [self.curriculum_dicts[dof]["map_type"] for dof in self.dofs]
        rewards_type = [self.curriculum_dicts[dof]["rewards_type"] for dof in self.dofs]

        env_cfgs = jax.vmap(EnvConfig.parametrized)(
            np.array(map_widths),
            np.array(map_heights),
            np.array(max_steps_in_episodes),
            np.array(self.dofs),
            np.array(map_types),
            np.array(rewards_type),
            )
        
        dofs_count_dict = self._get_dofs_count_dict()
        return env_cfgs, dofs_count_dict

    def get_cfgs_init(self,):
        map_widths = [self.curriculum_dicts[0]["map_width"] for _ in self.dofs]
        map_heights = [self.curriculum_dicts[0]["map_height"] for _ in self.dofs]
        max_steps_in_episodes = [self.curriculum_dicts[0]["max_steps_in_episode"] for _ in self.dofs]
        map_types = [self.curriculum_dicts[0]["map_type"] for _ in self.dofs]
        rewards_type = [self.curriculum_dicts[0]["rewards_type"] for _ in self.dofs]
        

        env_cfgs = jax.vmap(EnvConfig.parametrized)(
            np.array(map_widths),
            np.array(map_heights),
            np.array(max_steps_in_episodes),
            np.array(self.dofs),
            np.array(map_types),
            np.array(rewards_type),
            )
        dofs_count_dict = self._get_dofs_count_dict()
        return env_cfgs, dofs_count_dict
    
    def get_cfgs_eval(self,):
        map_widths = [self.curriculum_dicts[dof]["map_width"] for dof in self.dofs_eval]
        map_heights = [self.curriculum_dicts[dof]["map_height"] for dof in self.dofs_eval]
        max_steps_in_episodes = [self.curriculum_dicts[dof]["max_steps_in_episode"] for dof in self.dofs_eval]
        map_types = [self.curriculum_dicts[dof]["map_type"] for dof in self.dofs_eval]
        rewards_type = [self.curriculum_dicts[dof]["rewards_type"] for dof in self.dofs_eval]

        env_cfgs = jax.vmap(EnvConfig.parametrized)(
            np.array(map_widths),
            np.array(map_heights),
            np.array(max_steps_in_episodes),
            np.array(self.dofs_eval),
            np.array(map_types),
            np.array(rewards_type),
            )
        
        dofs_count_dict = self._get_dofs_count_dict_eval()
        
        return env_cfgs, dofs_count_dict

    def _increase_task_complexity(self,):
        pass

    def _decrease_task_complexity(self,):
        pass

    def _bring_agent_closer_to_reward(self,):
        pass
    
    def _rewind_model_and_env(self,):
        pass

    def _get_curriculum(self, idx: int):
        return self.curriculum_dicts[idx]

    def get_num_embeddings_agent_min(self,) -> int:
        return self.num_embeddings_agent_min
