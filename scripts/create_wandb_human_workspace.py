#!/usr/bin/env python3
"""Create the manual, bounded Terra RL W&B workspace."""

from __future__ import annotations

import argparse
import json

ENTITY = "aless-weber-eth"
PROJECT = "mixed-agents"
WORKSPACE_NAME = "Terra RL - Human v1"


SECTIONS = (
    (
        "Task outcome",
        (
            (
                "Fixed exact success",
                "eval/update",
                (
                    "eval/promotion/exact_success_rate",
                    "eval/development/exact_success_rate",
                ),
            ),
            (
                "Fixed graded and worst-condition completion",
                "eval/update",
                (
                    "eval/promotion/macro_completion",
                    "eval/development/macro_completion",
                    "eval/promotion/worst_condition_completion",
                    "eval/development/worst_condition_completion",
                ),
            ),
            (
                "Family exact success",
                "eval/update",
                (
                    "eval/promotion/foundation_exact_success_rate",
                    "eval/promotion/trench_exact_success_rate",
                    "eval/development/foundation_exact_success_rate",
                    "eval/development/trench_exact_success_rate",
                ),
            ),
            (
                "Online training and inline-eval outcomes",
                "train/update",
                (
                    "train/episode_success_rate",
                    "train/episode_timeout_rate",
                    "online_eval/success_within_horizon_rate",
                    "online_eval/termination_within_horizon_rate",
                ),
            ),
        ),
    ),
    (
        "Curriculum population",
        (
            (
                "Active family population",
                "train/update",
                (
                    "curriculum/population/foundation",
                    "curriculum/population/trench",
                ),
            ),
            (
                "Active depth population",
                "train/update",
                (
                    "curriculum/population/Anchor",
                    "curriculum/population/One-axis",
                    "curriculum/population/Composed",
                ),
            ),
            (
                "Target effective sample size",
                "train/update",
                ("curriculum/target_ess",),
            ),
            (
                "Target entropy",
                "train/update",
                ("curriculum/target_entropy_normalized",),
            ),
        ),
    ),
    (
        "Behavior and reward",
        (
            (
                "Task progress",
                "train/update",
                (
                    "behavior/absolute_completion",
                    "behavior/dig_completion",
                    "behavior/dump_volume_completion",
                ),
            ),
            (
                "Dump quality",
                "train/update",
                (
                    "behavior/dump_purity",
                    "behavior/no_effect_action_rate",
                ),
            ),
            (
                "Action distribution",
                "train/update",
                (
                    "behavior/action_fraction/forward",
                    "behavior/action_fraction/backward",
                    "behavior/action_fraction/base_clockwise",
                    "behavior/action_fraction/base_anticlockwise",
                    "behavior/action_fraction/cabin_clockwise",
                    "behavior/action_fraction/cabin_anticlockwise",
                    "behavior/action_fraction/do",
                    "behavior/action_fraction/no_op",
                ),
            ),
            (
                "Episode reward components",
                "train/update",
                (
                    "reward/episode_return",
                    "reward/agent",
                    "reward/terminal",
                    "reward/trench",
                    "reward/existence",
                ),
            ),
        ),
    ),
    (
        "Optimization and kickstart",
        (
            (
                "PPO losses",
                "train/update",
                ("ppo/total_loss", "ppo/policy_loss", "ppo/value_loss"),
            ),
            (
                "Policy distribution",
                "train/update",
                (
                    "ppo/entropy",
                    "ppo/entropy_coef",
                    "ppo/approx_kl",
                    "ppo/clip_fraction",
                ),
            ),
            (
                "Fit and gradients",
                "train/update",
                ("ppo/explained_variance", "ppo/grad_norm"),
            ),
            (
                "Kickstart",
                "train/update",
                (
                    "kickstart/kl",
                    "kickstart/value_mse",
                    "kickstart/kl_coef",
                    "kickstart/value_coef",
                ),
            ),
        ),
    ),
)


def workspace_spec() -> dict:
    return {
        "name": WORKSPACE_NAME,
        "entity": ENTITY,
        "project": PROJECT,
        "auto_generate_panels": False,
        "visible_panel_count": sum(len(panels) for _, panels in SECTIONS),
        "sections": [
            {
                "name": name,
                "panels": [
                    {"title": title, "x": x_axis, "y": list(metrics)}
                    for title, x_axis, metrics in panels
                ],
            }
            for name, panels in SECTIONS
        ],
        "collapsed_details": {
            "line_panels": [
                {
                    "title": "Work efficiency",
                    "x": "train/update",
                    "y": [
                        "behavior/mean_episode_length",
                        "behavior/productive_workspace_cycles_per_episode",
                    ],
                },
                {
                    "title": "System",
                    "x": "train/update",
                    "y": [
                        "system/steps_per_second",
                        "system/environment_steps",
                    ],
                },
            ],
            "media": [
                "curriculum/conditions",
                "eval/promotion_conditions",
                "eval/development_conditions",
            ],
        },
    }


def create_workspace(entity: str, project: str, name: str):
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as error:
        raise RuntimeError(
            "install the preview workspace client, for example: "
            "uv run --isolated --with wandb-workspaces --with wandb "
            "python scripts/create_wandb_human_workspace.py"
        ) from error

    sections = []
    for section_name, panel_specs in SECTIONS:
        sections.append(
            ws.Section(
                name=section_name,
                panels=[
                    wr.LinePlot(title=title, x=x_axis, y=list(metrics))
                    for title, x_axis, metrics in panel_specs
                ],
                is_open=True,
                layout_settings=ws.SectionLayoutSettings(columns=4, rows=1),
            )
        )
    sections.append(
        ws.Section(
            name="Details",
            panels=[
                wr.LinePlot(
                    title="Work efficiency",
                    x="train/update",
                    y=[
                        "behavior/mean_episode_length",
                        "behavior/productive_workspace_cycles_per_episode",
                    ],
                ),
                wr.MediaBrowser(
                    title="Curriculum condition snapshot",
                    media_keys=["curriculum/conditions"],
                ),
                wr.MediaBrowser(
                    title="Fixed-evaluation condition results",
                    media_keys=[
                        "eval/promotion_conditions",
                        "eval/development_conditions",
                    ],
                ),
                wr.LinePlot(
                    title="System",
                    x="train/update",
                    y=[
                        "system/steps_per_second",
                        "system/environment_steps",
                    ],
                ),
            ],
            is_open=False,
            layout_settings=ws.SectionLayoutSettings(columns=4, rows=1),
        )
    )
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=sections,
        settings=ws.WorkspaceSettings(
            smoothing_type="none",
            ignore_outliers=False,
            sort_panels_alphabetically=False,
            max_runs=20,
        ),
        auto_generate_panels=False,
    )
    return workspace.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--name", default=WORKSPACE_NAME)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        spec = workspace_spec()
        spec.update({"entity": args.entity, "project": args.project, "name": args.name})
        print(json.dumps(spec, indent=2, sort_keys=True))
        return

    workspace = create_workspace(args.entity, args.project, args.name)
    print(workspace.url)


if __name__ == "__main__":
    main()
