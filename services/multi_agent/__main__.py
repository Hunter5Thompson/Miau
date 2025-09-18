"""Module entrypoint so the workflow can run via ``python -m services.multi_agent``."""

from __future__ import annotations

from .cli import run_cli


def main() -> int:
    """Execute the command line interface."""

    return run_cli()


if __name__ == "__main__":  # pragma: no cover - enables direct execution
    raise SystemExit(main())
