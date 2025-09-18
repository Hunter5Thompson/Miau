"""Entry point for running the MCP server over stdio."""

from __future__ import annotations

from .server import server


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
