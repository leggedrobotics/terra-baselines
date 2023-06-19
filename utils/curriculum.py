import jax
from jax import Array
from typing import Any
from terra.config import EnvConfig
import numpy as np

class Curriculum:

    def __init__(self, rl_config) -> None:
        self.dofs = np.zeros((rl_config.num_train_envs,), dtype=np.int8)
        self.dofs_eval = np.zeros((rl_config.num_test_rollouts,), dtype=np.int8)
        self.rl_config = rl_config

        self.curriculum_dicts = [
            # {
            #     "map_width": 20,  # in meters
            #     "map_height": 20,  # in meters
            #     "max_steps_in_episode": 150,

            #     # Placeholders
            #     "n_clusters": 3,
            #     "n_tiles_per_cluster" : 3,
            #     "kernel_size_initial_sampling": 4,
            # },
            {
                "map_width": 40,  # in meters
                "map_height": 40,  # in meters
                "max_steps_in_episode": 600,
            },
            {
                "map_width": 60,  # in meters
                "map_height": 60,  # in meters
                "max_steps_in_episode": 600,
            },
        ]

        self.curriculum_len = len(self.curriculum_dicts)

        # Compute the minimum number of embedding length for the one-hot features
        self.num_embeddings_agent_min = max([max([el["map_width"], el["map_height"]]) for el in self.curriculum_dicts])
        print(f"{self.num_embeddings_agent_min=}")

    def _evaluate_progress(self, metrics_dict: dict[str, Any]) -> int:
        """
        Goes from the training metrics to a DoF (degrees of freedom) value,
        considering the current DoF.
        """
        # value_losses_individual = metrics_dict["value_losses_individual"]
        # targets_individual = metrics_dict["targets_individual"]
        # print(f"{value_losses_individual.shape=}")
        # print(f"{targets_individual.shape=}")

        # increase_dof = (value_losses_individual / targets_individual < 0.13) * (targets_individual > 0)  # TODO config

        # dofs = self.dofs + increase_dof.astype(np.int8)
        # self.dofs = [dof for dof in dofs if (dof < self.curriculum_len - 1) else 0]  # TODO else random, not 0

        self.dofs = [1 for _ in range(len(self.dofs))]
        self.dofs_eval = [1 for _ in range(len(self.dofs_eval))]
    
    def get_cfgs(self, metrics_dict: dict[str, Any]):
        self._evaluate_progress(metrics_dict)
        
        map_widths = [self.curriculum_dicts[dof]["map_width"] for dof in self.dofs]
        map_heights = [self.curriculum_dicts[dof]["map_height"] for dof in self.dofs]
        max_steps_in_episodes = [self.curriculum_dicts[dof]["max_steps_in_episode"] for dof in self.dofs]

        env_cfgs = jax.vmap(EnvConfig.parametrized)(
            np.array(map_widths),
            np.array(map_heights),
            np.array(max_steps_in_episodes)
            )
        
        dof_counts = [sum(1 for dof in self.dofs if dof == j) for j in range(self.curriculum_len)]
        dofs_count_dict = {
            f"curriculum dof count {dof}": count
            for dof, count in zip(range(self.curriculum_len), dof_counts)
        }
        return env_cfgs, dofs_count_dict

    def get_cfgs_init(self,):
        map_widths = [self.curriculum_dicts[0]["map_width"] for _ in self.dofs]
        map_heights = [self.curriculum_dicts[0]["map_height"] for _ in self.dofs]
        max_steps_in_episodes = [self.curriculum_dicts[0]["max_steps_in_episode"] for _ in self.dofs]

        env_cfgs = jax.vmap(EnvConfig.parametrized)(
            np.array(map_widths),
            np.array(map_heights),
            np.array(max_steps_in_episodes)
            )
        return env_cfgs
    
    def get_cfgs_eval(self, metrics_dict: dict[str, Any]):
        self._evaluate_progress(metrics_dict)
        
        map_widths = [self.curriculum_dicts[dof]["map_width"] for dof in self.dofs_eval]
        map_heights = [self.curriculum_dicts[dof]["map_height"] for dof in self.dofs_eval]
        max_steps_in_episodes = [self.curriculum_dicts[dof]["max_steps_in_episode"] for dof in self.dofs_eval]

        env_cfgs = jax.vmap(EnvConfig.parametrized)(
            np.array(map_widths),
            np.array(map_heights),
            np.array(max_steps_in_episodes)
            )
        
        dof_counts = [sum(1 for dof in self.dofs_eval if dof == j) for j in range(self.curriculum_len)]
        dofs_count_dict = {
            f"curriculum dof count {dof}": count
            for dof, count in zip(range(self.curriculum_len), dof_counts)
        }
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
