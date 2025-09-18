"""Pydantic models shared across the server."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Sequence

from pydantic import BaseModel, Field


@dataclass(slots=True)
class Chunk:
    """Representation of a document chunk before persistence."""

    text: str
    heading: str | None
    tokens: int
    order: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class CrawlDocument:
    """Holds the Markdown document returned from the crawler."""

    url: str
    title: str | None
    markdown: str
    metadata: dict[str, Any]
    chunks: Sequence[Chunk]


class RAGHit(BaseModel):
    """Result item returned from a retrieval query."""

    chunk_id: str = Field(..., description="UUID of the chunk")
    score: float = Field(..., ge=0.0, le=1.0)
    snippet: str = Field(..., description="Markdown snippet")
    url: str = Field(..., description="Origin document URL")
    document_id: str = Field(..., description="UUID of the source document")
    heading: str | None = Field(None, description="Heading associated with the snippet")


class IndexStatus(BaseModel):
    """Simple status summary for the index."""

    sources: int
    documents: int
    chunks: int
    last_ingest_at: datetime | None = None


class CrawlResponse(BaseModel):
    status: str
    source_id: str | None = None
    document_ids: List[str] = Field(default_factory=list)
    chunks_indexed: int = 0
    message: str | None = None


class QueryResponse(BaseModel):
    status: str
    hits: List[RAGHit] = Field(default_factory=list)
    message: str | None = None


class PurgeResponse(BaseModel):
    status: str
    deleted_sources: int
    deleted_documents: int
    deleted_chunks: int
    message: str | None = None


__all__ = [
    "Chunk",
    "CrawlDocument",
    "RAGHit",
    "IndexStatus",
    "CrawlResponse",
    "QueryResponse",
    "PurgeResponse",
]
