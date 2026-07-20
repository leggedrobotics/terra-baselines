"""Compatibility entry point for the canonical policy evaluator.

Evaluation accounting lives in :mod:`eval_mcts` so PPO and PPO+MCTS always use
the same maps, horizon, first-episode success definition, and summary metrics.
"""

from eval_mcts import main, print_stats, rollout_episode

__all__ = ["rollout_episode", "print_stats", "main"]


if __name__ == "__main__":
    main()
