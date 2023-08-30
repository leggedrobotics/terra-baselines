import jax
from jax import Array
from typing import Any
from terra.config import EnvConfig
import numpy as np
from terra.config import MapType
from terra.config import RewardsType

class CurriculumTestbench:

    def __init__(self, rl_config) -> None:
        self.dofs = np.zeros((rl_config["num_train_envs"],), dtype=np.int8)
        self.rl_config = rl_config
        self.change_dof_threshold = rl_config["change_dof_threshold"]
        self.max_change_ratio = rl_config["max_change_ratio"]

        self.curriculum_dicts = [
            {
                "map_width": 60,  # in meters
                "map_height": 60,  # in meters
                "max_steps_in_episode": 30,
                "map_type": MapType.SINGLE_TILE,
                "rewards_type": RewardsType.TERMINAL_ONLY,
            },
        ]

        self.curriculum_len = len(self.curriculum_dicts)
        self.idx_sparse_rewards_levels = [i for i, level in enumerate(self.curriculum_dicts) if level["rewards_type"] == RewardsType.SPARSE]
        self.n_sparse_levels = len(self.idx_sparse_rewards_levels)

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

    def set_dofs(self, dof: int):
        self.dofs = dof * np.ones_like(self.dofs)
        self.dofs_eval = dof * np.ones_like(self.dofs_eval)

        print(f"\nStarting task {dof}:\n{self.curriculum_dicts[dof]}...\n")
    
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
    
    def get_cfgs(self, *args, **kwargs):
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
        
        dofs_count_dict = self._get_dofs_count_dict()
        return env_cfgs, dofs_count_dict

    def get_cfgs_init(self,):
        return self.get_cfgs()
    
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

    def get_curriculum_len(self,):
        return self.curriculum_len
    
    def get_dof_max_steps(self, dof):
        return self.curriculum_dicts[dof]["max_steps_in_episode"]
