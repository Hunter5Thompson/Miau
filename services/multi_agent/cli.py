"""Command line interface for running the multi-agent research workflow."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Sequence

from .team import MultiAgentSearchTeam

DEFAULT_GOAL = "Identify state-of-the-art reranker models for information retrieval"
DEFAULT_CONSTRAINTS: Sequence[str] = (
    "Prioritise peer-reviewed research papers",
    "Prefer resources published after 2022",
)


def _env_constraints(value: str | None) -> Sequence[str]:
    """Parse environment-supplied constraints (comma separated)."""

    if not value:
        return ()
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Create the ``argparse`` parser used by the CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description=(
            "Execute the multi-agent research workflow that orchestrates the "
            "Research Strategist, Data Miner, Data Analyst and Research Writer."
        )
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default=os.getenv("MULTI_AGENT_GOAL", DEFAULT_GOAL),
        help="Research goal to pursue. Defaults to the MULTI_AGENT_GOAL environment variable.",
    )
    parser.add_argument(
        "--constraint",
        "-c",
        action="append",
        dest="constraints",
        default=None,
        help="Hard constraint for the research task. Repeat the flag for multiple constraints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Initialise all agents without executing the workflow. Useful for integration tests "
            "that only need to validate wiring."
        ),
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning the resulting exit code."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    env_constraints = _env_constraints(os.getenv("MULTI_AGENT_CONSTRAINTS"))
    constraints: Iterable[str]
    if args.constraints:
        constraints = args.constraints
    elif env_constraints:
        constraints = env_constraints
    else:
        constraints = DEFAULT_CONSTRAINTS

    team = MultiAgentSearchTeam()
    if args.dry_run:
        # Trigger lazy initialisation such as LangGraph compilation without running the agents.
        if team.langgraph_app is not None:
            team.langgraph_app
        return 0

    report = team.run(args.goal, constraints=tuple(constraints))
    print("\n" + "=" * 80)
    print(report.title)
    print("=" * 80)
    print(report.content)
    if team.state is not None:
        print("\nProtokoll:")
        for entry in team.state.state_log:
            print(f"- {entry}")
    return 0


__all__ = ["run_cli", "build_parser"]


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    raise SystemExit(run_cli())
