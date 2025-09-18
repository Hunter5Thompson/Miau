"""Configuration helpers for the Crawl4AI MCP server."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, HttpUrl, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the MCP server."""

    model_config = SettingsConfigDict(env_prefix="C4MCP_", env_file=".env", extra="ignore")

    # Database
    database_dsn: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/crawl4ai_mcp",
        description="Connection string for the PostgreSQL instance with pgvector.",
    )
    database_min_size: PositiveInt = Field(
        default=1,
        description="Minimum number of connections maintained by the asyncpg pool.",
    )
    database_max_size: PositiveInt = Field(
        default=10,
        description="Maximum number of connections maintained by the asyncpg pool.",
    )

    # Embeddings
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key used for embedding generation. If omitted only mocks/tests will run.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model identifier.",
    )
    embedding_dimensions: Optional[int] = Field(
        default=1536,
        description="Embedding dimensionality. None keeps provider defaults.",
    )
    embedding_batch_size: PositiveInt = Field(
        default=64,
        description="Maximum number of texts sent to the embedder in a single batch.",
    )

    # Crawling
    crawl_timeout_seconds: PositiveInt = Field(default=45, description="Per-request timeout in seconds.")
    crawl_concurrency: PositiveInt = Field(default=5, description="Max concurrent requests when crawling a domain.")
    crawl_backoff_seconds: float = Field(
        default=1.5,
        description="Base backoff in seconds for retries of HTTP failures.",
    )
    crawl_max_retries: PositiveInt = Field(default=3, description="Maximum number of retry attempts for transient failures.")
    crawl_user_agent: str = Field(
        default="Crawl4AI-MCP/0.1 (+https://github.com/modelcontextprotocol)",
        description="User agent used for crawling.",
    )

    # Chunking
    chunk_tokens: PositiveInt = Field(default=400, description="Approximate token count per chunk.")
    chunk_overlap_tokens: PositiveInt = Field(default=40, description="Overlap tokens between neighbouring chunks.")

    # RAG
    rag_default_strategy: Literal["vector", "hybrid", "rerank", "agentic"] = Field(
        default="hybrid", description="Default retrieval strategy used when none supplied."
    )
    rag_score_threshold: float = Field(
        default=0.25,
        description="Minimum relevance score for returned snippets.",
        ge=0.0,
        le=1.0,
    )
    rag_max_snippets: PositiveInt = Field(default=5, description="Default number of snippets to return.")

    # Optional HTTP transport
    http_host: str = Field(default="0.0.0.0", description="Binding host for optional HTTP transport.")
    http_port: PositiveInt = Field(default=8051, description="Binding port for optional HTTP transport.")
    http_base_url: Optional[HttpUrl] = Field(
        default=None,
        description="Public base URL advertised to clients when using HTTP transport.",
    )

    # Observability
    log_level: str = Field(default="INFO", description="Root log level.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


__all__ = ["Settings", "get_settings"]
