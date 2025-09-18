"""Shared exception hierarchy for the Crawl4AI MCP server."""

from __future__ import annotations


class MCPError(Exception):
    """Base error for MCP service related failures."""


class CrawlError(MCPError):
    """Raised when crawling fails."""

    def __init__(self, message: str, *, url: str | None = None) -> None:
        if url:
            message = f"{message} (url={url})"
        super().__init__(message)
        self.url = url


class RAGError(MCPError):
    """Raised when retrieval or storage fails."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


__all__ = ["MCPError", "CrawlError", "RAGError"]
