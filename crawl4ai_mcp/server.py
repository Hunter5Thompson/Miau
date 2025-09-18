"""Model Context Protocol server wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable

from mcp.server.fastmcp import Context, FastMCP

from .config import get_settings
from .db import Database
from .errors import CrawlError, RAGError
from .logging_config import configure_logging
from .models import CrawlResponse, PurgeResponse, QueryResponse
from .services.crawler import crawl_domain as crawl_domain_service
from .services.crawler import crawl_single_page as crawl_single_page_service
from .services.embedding import EmbeddingClient
from .services.rag_engine import IngestStats, RAGEngine

LOGGER = logging.getLogger("crawl4ai_mcp.server")

settings = get_settings()
configure_logging(settings.log_level)

database = Database(
    settings.database_dsn,
    min_size=settings.database_min_size,
    max_size=settings.database_max_size,
)
embedder = EmbeddingClient(
    api_key=settings.openai_api_key,
    model=settings.embedding_model,
    dimensions=settings.embedding_dimensions,
    batch_size=settings.embedding_batch_size,
)
rag_engine = RAGEngine(database, embedder, settings)


@asynccontextmanager
async def lifespan(app: FastMCP):
    await database.connect()
    try:
        await rag_engine.ensure_schema()
        yield
    finally:
        await database.close()


server = FastMCP(
    name="crawl4ai-mcp",
    instructions="Crawl websites with Crawl4AI and query a pgvector RAG index.",
    lifespan=lifespan,
)


async def _ingest_documents(source_url: str, documents: Iterable) -> IngestStats:
    await rag_engine.ensure_schema()
    return await rag_engine.ingest(source_url, list(documents))


@server.tool()
async def crawl_single_page(url: str, *, context: Context | None = None) -> Dict[str, Any]:
    """Crawl a single URL and index its content."""

    try:
        document = await crawl_single_page_service(url, settings)
        stats = await _ingest_documents(url, [document])
        response = CrawlResponse(
            status="ok",
            source_id=stats.source_id,
            document_ids=stats.document_ids,
            chunks_indexed=stats.chunks_indexed,
        )
        return response.model_dump()
    except CrawlError as exc:
        LOGGER.exception("crawl failed for %s", url)
        return CrawlResponse(status="error", message=str(exc)).model_dump()
    except RAGError as exc:
        LOGGER.exception("ingest failed for %s", url)
        return CrawlResponse(status="error", message=str(exc)).model_dump()


@server.tool()
async def crawl_domain(
    seed_url: str,
    max_pages: int = 20,
    sitemap: bool = True,
    *,
    context: Context | None = None,
) -> Dict[str, Any]:
    """Crawl multiple pages from a domain."""

    try:
        documents = await crawl_domain_service(
            seed_url,
            max_pages=max_pages,
            settings=settings,
            sitemap=sitemap,
        )
        stats = await _ingest_documents(seed_url, documents)
        return CrawlResponse(
            status="ok",
            source_id=stats.source_id,
            document_ids=stats.document_ids,
            chunks_indexed=stats.chunks_indexed,
        ).model_dump()
    except CrawlError as exc:
        LOGGER.exception("domain crawl failed for %s", seed_url)
        return CrawlResponse(status="error", message=str(exc)).model_dump()
    except RAGError as exc:
        LOGGER.exception("domain ingest failed for %s", seed_url)
        return CrawlResponse(status="error", message=str(exc)).model_dump()


@server.tool()
async def query_rag(
    query: str,
    k: int = 5,
    strategy: str | None = None,
    *,
    context: Context | None = None,
) -> Dict[str, Any]:
    """Query the RAG index."""

    try:
        hits = await rag_engine.query(query, k=k, strategy=strategy)
        return QueryResponse(status="ok", hits=hits).model_dump()
    except RAGError as exc:
        LOGGER.exception("rag query failed")
        return QueryResponse(status="error", message=str(exc)).model_dump()


@server.tool()
async def index_status(*, context: Context | None = None) -> Dict[str, Any]:
    """Return index statistics."""

    status = await rag_engine.index_status()
    return {"status": "ok", "index": status.model_dump()}


@server.tool()
async def purge_source(identifier: str, *, context: Context | None = None) -> Dict[str, Any]:
    """Remove a source (by UUID or URL) from the index."""

    deleted_sources, deleted_docs, deleted_chunks = await rag_engine.purge_source(identifier)
    return PurgeResponse(
        status="ok",
        deleted_sources=deleted_sources,
        deleted_documents=deleted_docs,
        deleted_chunks=deleted_chunks,
    ).model_dump()


__all__ = ["server"]
