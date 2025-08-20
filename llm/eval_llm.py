import jax.numpy as jnp
import numpy as np
    
def compute_stats_llm(episode_done_once_list, episode_length_list, move_cumsum_list,
                      do_cumsum_list, areas_list, dig_tiles_per_target_map_init_list,
                      dug_tiles_per_action_map_list):
    """
    Compute statistics from the results of multiple environments.
    Args:
        episode_done_once_list (list): List of booleans indicating if the episode was done once.
        episode_length_list (list): List of integers representing the length of each episode.
        move_cumsum_list (list): List of cumulative sums of moves for each environment.
        do_cumsum_list (list): List of cumulative sums of 'do' actions for each environment.
        areas_list (list): List of areas for each environment.
        dig_tiles_per_target_map_init_list (list): List of initial dig tiles per target map.
        dug_tiles_per_action_map_list (list): List of dug tiles per action map.
    Returns:
        None: Prints the computed statistics.
    """    


    episode_done_once = jnp.array(episode_done_once_list)
    episode_length = jnp.array(episode_length_list)
    move_cumsum = jnp.array(move_cumsum_list)
    do_cumsum = jnp.array(do_cumsum_list)
    areas = jnp.array(areas_list)
    dig_tiles_per_target_map_init = jnp.array(dig_tiles_per_target_map_init_list)
    dug_tiles_per_action_map = jnp.array(dug_tiles_per_action_map_list)

    print("\nSummary of results across all environments:")
    print(f"Episode done once: {episode_done_once}")
    print(f"Episode length: {episode_length}")
    print(f"Move cumsum: {move_cumsum}")
    print(f"Do cumsum: {do_cumsum}")
    print(f"Areas: {areas}")
    print(f"Dig tiles per target map init: {dig_tiles_per_target_map_init}")
    print(f"Dug tiles per action map: {dug_tiles_per_action_map}")

    # Path efficiency -- only include finished envs
    move_cumsum *= episode_done_once
    path_efficiency = (move_cumsum / jnp.sqrt(areas))[episode_done_once]
    path_efficiency_std = path_efficiency.std()
    path_efficiency_mean = path_efficiency.mean()

    # Workspaces efficiency -- only include finished envs
    reference_workspace_area = 0.5 * np.pi * (8**2)
    n_dig_actions = do_cumsum // 2
    workspaces_efficiency = (
        reference_workspace_area
        * ((n_dig_actions * episode_done_once) / areas)[episode_done_once]
    )
    workspaces_efficiency_mean = workspaces_efficiency.mean()
    workspaces_efficiency_std = workspaces_efficiency.std()

    coverage_ratios = dug_tiles_per_action_map / dig_tiles_per_target_map_init
    coverage_scores = episode_done_once + (~episode_done_once) * coverage_ratios
    coverage_score_mean = coverage_scores.mean()
    coverage_score_std = coverage_scores.std()

    completion_rate = 100 * episode_done_once.sum() / len(episode_done_once)

    print("\nStats:\n")
    print(f"Completion: {completion_rate:.2f}%")

    print(
        f"Path efficiency: {path_efficiency_mean:.2f} ({path_efficiency_std:.2f})"
    )
    print(
        f"Workspaces efficiency: {workspaces_efficiency_mean:.2f} ({workspaces_efficiency_std:.2f})"
    )
    print(f"Coverage: {coverage_score_mean:.2f} ({coverage_score_std:.2f})")