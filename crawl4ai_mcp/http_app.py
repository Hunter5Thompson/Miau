"""HTTP application exposing the streamable MCP transport."""

from __future__ import annotations

from fastapi import FastAPI

from .server import server

app: FastAPI = server.streamable_http_app()

__all__ = ["app"]
