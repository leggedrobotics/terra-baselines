from typing import Any, Optional
from terra.env import TerraEnvBatch
from terra.config import EnvConfig, MapDims

class Curriculum:

    def __init__(self) -> None:
        self.prev_curriculum = 0

    def evaluate_progress(self, metrics_dict: dict[str, Any]) -> int:
        """
        Goes from the training metrics to a DoF (degrees of freedom) value,
        considering the current DoF.
        """
        change_curriculum = self.prev_curriculum != 1
        return 1, change_curriculum
    
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
        env_cfg = EnvConfig.from_map_dims(map_dims)

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
        """
        Defines the curriculum to follow during training.
        """
        curriculum = [
            {
                "map_width": 10,  # in meters
                "map_height": 10,  # in meters
                # "map_params": MapParamsSquareSingleTile()
            },
            {
                "map_width": 20,  # in meters
                "map_height": 20,  # in meters
                # "map_params": MapParamsSquareSingleTile()
            },
        ]

        return curriculum[idx]
