from typing import Any
from terra.env import TerraEnvBatch
from terra.config import EnvConfig, MapDims

class Curriculum:

    def __init__(self) -> None:
        self.prev_curriculum = 0

        self.curriculum_dicts = [
            {
                "map_width": 20,  # in meters
                "map_height": 20,  # in meters
                "max_steps_in_episode": 150,
            },
            {
                "map_width": 40,  # in meters
                "map_height": 40,  # in meters
                "max_steps_in_episode": 300,
            },
            {
                "map_width": 60,  # in meters
                "map_height": 60,  # in meters
                "max_steps_in_episode": 600,
            },
        ]

        # Compute the minimum number of embedding length for the one-hot features
        self.num_embeddings_agent_min = max([max([el["map_width"], el["map_height"]]) for el in self.curriculum_dicts])
        print(f"{self.num_embeddings_agent_min=}")

    def evaluate_progress(self, metrics_dict: dict[str, Any]) -> int:
        """
        Goes from the training metrics to a DoF (degrees of freedom) value,
        considering the current DoF.
        """
        value_loss = metrics_dict["value_loss"]
        target = metrics_dict["target"]
        # TODO config
        if (value_loss / target < 0.13) and (target > 0) and (self.prev_curriculum < self._get_curriculum_len() - 1):
            dof = self.prev_curriculum + 1
            change_curriculum = True
        else:
            dof = self.prev_curriculum
            change_curriculum = False
        return dof, change_curriculum
    
    def start_curriculum(self):
        return self.apply_curriculum(
            curriculum_progress=0,
        )

    def apply_curriculum(self, curriculum_progress: int):
        print(f"\n~~~ Curriculum change to idx --> {curriculum_progress} ~~~\n")

        self.prev_curriculum = curriculum_progress
        curriculum = self._get_curriculum(curriculum_progress)

        map_width = curriculum["map_width"]
        map_height = curriculum["map_height"]
        print(f"{map_width=}")
        print(f"{map_height=}")
        map_dims = MapDims(
            width_m=map_width,
            height_m=map_height
        )
        env_cfg = EnvConfig.parametrized(
            map_dims=map_dims,
            max_steps_in_episode=curriculum["max_steps_in_episode"]
            )

        env = TerraEnvBatch(env_cfg=env_cfg)
        return env


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

    def _get_curriculum_len(self,) -> int:
        return len(self.curriculum_dicts)

    def get_num_embeddings_agent_min(self,) -> int:
        return self.num_embeddings_agent_min
