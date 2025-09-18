"""pgvector-powered retrieval engine."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import UUID

from ..config import Settings
from ..db import Database
from ..errors import RAGError
from ..models import Chunk, CrawlDocument, IndexStatus, RAGHit
from .embedding import EmbeddingClient

LOGGER = logging.getLogger("crawl4ai_mcp.rag")

_VECTOR_SQL = """
select
    c.id as chunk_id,
    1 - (c.embedding <=> $1) as score,
    c.text,
    c.heading,
    d.url,
    d.id as document_id
from chunks c
join documents d on d.id = c.document_id
where c.embedding is not null
order by c.embedding <=> $1 asc
limit $2
"""

_HYBRID_SQL = """
with ranked as (
    select
        c.id as chunk_id,
        1 - (c.embedding <=> $1) as vector_score,
        ts_rank_cd(to_tsvector('english', c.text), plainto_tsquery($2)) as text_score,
        c.text,
        c.heading,
        d.url,
        d.id as document_id
    from chunks c
    join documents d on d.id = c.document_id
    where c.embedding is not null
)
select chunk_id,
       (0.65 * vector_score) + (0.35 * coalesce(text_score, 0)) as score,
       text,
       heading,
       url,
       document_id
from ranked
order by score desc
limit $3
"""


@dataclass(slots=True)
class IngestStats:
    source_id: str
    document_ids: list[str]
    chunks_indexed: int


class RAGEngine:
    """Handles ingestion and retrieval."""

    _DEFAULT_VECTOR_DIMENSION = 1536

    def __init__(
        self,
        database: Database,
        embedder: EmbeddingClient,
        settings: Settings,
    ) -> None:
        self._db = database
        self._embedder = embedder
        self._settings = settings
        self._schema_lock = asyncio.Lock()
        self._schema_ready = False
        self._schema_path = (
            Path(__file__).resolve().parent.parent / "migrations" / "schema.sql"
        )

    async def ensure_schema(self) -> None:
        async with self._schema_lock:
            if self._schema_ready:
                return
            sql = self._render_schema_sql()
            await self._db.run_migration(sql)
            target_dim = self._settings.embedding_dimensions or self._DEFAULT_VECTOR_DIMENSION
            await self._ensure_vector_dimension(target_dim)
            self._schema_ready = True

    def _render_schema_sql(self) -> str:
        sql = self._schema_path.read_text()
        target_dim = self._settings.embedding_dimensions
        if target_dim and target_dim != self._DEFAULT_VECTOR_DIMENSION:
            sql = sql.replace("vector(1536)", f"vector({target_dim})")
        return sql

    async def _ensure_vector_dimension(self, target_dim: int) -> None:
        if target_dim <= 0:
            raise RAGError("embedding dimension must be positive")

        async with self._db.transaction() as conn:
            row = await conn.fetchrow(
                """
                select format_type(att.atttypid, att.atttypmod) as type
                from pg_attribute att
                where att.attrelid = (
                    select oid
                    from pg_class
                    where relname = 'chunks'
                      and relnamespace = 'public'::regnamespace
                )
                  and att.attname = 'embedding'
            """
            )
            if not row:
                return

            current_dim = self._extract_vector_dimension(row.get("type"))
            if current_dim is None:
                return
            if current_dim == target_dim:
                return

            LOGGER.info(
                "updating embedding vector dimension from %s to %s", current_dim, target_dim
            )
            await conn.execute("drop index if exists idx_chunks_hnsw")
            await conn.execute(
                f"alter table chunks alter column embedding type vector({target_dim})"
            )
            await conn.execute(
                "create index if not exists idx_chunks_hnsw on chunks using hnsw (embedding vector_cosine_ops)"
            )

    @staticmethod
    def _extract_vector_dimension(type_definition: str | None) -> int | None:
        if not type_definition:
            return None
        match = re.search(r"vector\((\d+)\)", type_definition)
        if not match:
            return None
        return int(match.group(1))

    async def ingest(self, source_url: str, documents: Sequence[CrawlDocument]) -> IngestStats:
        if not documents:
            raise RAGError("no documents provided for ingestion")
        if not self._embedder.is_configured:
            raise RAGError("embedder is not configured")

        # Pre-compute embeddings to keep DB transactions short
        chunk_payloads: list[list[tuple[Chunk, list[float]]]] = []
        for document in documents:
            valid_chunks = [chunk for chunk in document.chunks if chunk.text]
            texts = [chunk.text for chunk in valid_chunks]
            embeddings = await self._embedder.embed_texts(texts) if texts else []
            if len(embeddings) != len(valid_chunks):
                raise RAGError("embedding count mismatch for document")
            chunk_payloads.append(list(zip(valid_chunks, embeddings)))

        async with self._db.transaction() as conn:
            source_row = await conn.fetchrow(
                """
                insert into sources (url, title, last_crawled_at)
                values ($1, $2, now())
                on conflict (url) do update set title = excluded.title, last_crawled_at = now()
                returning id
                """,
                source_url,
                documents[0].title,
            )
            if not source_row:
                raise RAGError("failed to upsert source")
            source_id = str(source_row["id"])

            document_ids: list[str] = []
            chunk_total = 0

            for document, chunk_embeddings in zip(documents, chunk_payloads):
                content_hash = hashlib.sha256(document.markdown.encode("utf-8")).hexdigest()
                doc_row = await conn.fetchrow(
                    """
                    insert into documents (source_id, url, title, md, meta, content_hash)
                    values ($1, $2, $3, $4, $5::jsonb, $6)
                    on conflict (url, content_hash) do nothing
                    returning id
                    """,
                    source_row["id"],
                    document.url,
                    document.title,
                    document.markdown,
                    json.dumps(document.metadata),
                    content_hash,
                )
                if doc_row is None:
                    LOGGER.info("document unchanged, skipping %s", document.url)
                    continue
                document_id = doc_row["id"]
                document_ids.append(str(document_id))
                await conn.execute("delete from chunks where document_id = $1", document_id)
                for chunk, embedding in chunk_embeddings:
                    await conn.execute(
                        """
                        insert into chunks (document_id, ord, text, heading, tokens, embedding, meta)
                        values ($1, $2, $3, $4, $5, $6, $7::jsonb)
                        """,
                        document_id,
                        chunk.order,
                        chunk.text,
                        chunk.heading,
                        chunk.tokens,
                        embedding,
                        json.dumps(chunk.metadata),
                    )
                    chunk_total += 1

            return IngestStats(source_id=source_id, document_ids=document_ids, chunks_indexed=chunk_total)

    async def query(self, query: str, *, k: int | None = None, strategy: str | None = None) -> list[RAGHit]:
        if not query:
            raise RAGError("query must not be empty")
        if not self._embedder.is_configured:
            raise RAGError("embedder is not configured")

        strategy = strategy or self._settings.rag_default_strategy
        strategy = strategy.lower()
        if strategy not in {"vector", "hybrid", "rerank", "agentic"}:
            raise RAGError(f"unknown strategy: {strategy}")
        limit = k or self._settings.rag_max_snippets
        query_vector = await self._embedder.embed_query(query)

        if strategy == "vector":
            rows = await self._db.fetch(_VECTOR_SQL, query_vector, limit)
        else:
            # Treat rerank/agentic as hybrid baseline for now
            rows = await self._db.fetch(_HYBRID_SQL, query_vector, query, limit)

        hits: list[RAGHit] = []
        for row in rows:
            score = float(row["score"])
            if score < self._settings.rag_score_threshold:
                continue
            hits.append(
                RAGHit(
                    chunk_id=str(row["chunk_id"]),
                    score=score,
                    snippet=row["text"],
                    url=row["url"],
                    document_id=str(row["document_id"]),
                    heading=row.get("heading"),
                )
            )
        return hits[:limit]

    async def purge_source(self, identifier: str) -> tuple[int, int, int]:
        async with self._db.transaction() as conn:
            if self._is_uuid(identifier):
                source_row = await conn.fetchrow("select id from sources where id = $1", UUID(identifier))
            else:
                source_row = await conn.fetchrow("select id from sources where url = $1", identifier)
            if not source_row:
                return (0, 0, 0)
            source_id = source_row["id"]
            doc_rows = await conn.fetch("select id from documents where source_id = $1", source_id)
            doc_ids = [row["id"] for row in doc_rows]
            chunk_count = 0
            if doc_ids:
                chunk_count = await conn.fetchval(
                    "select count(*) from chunks where document_id = any($1::uuid[])",
                    doc_ids,
                )
            deleted_docs = len(doc_ids)
            await conn.execute("delete from sources where id = $1", source_id)
            return (1, deleted_docs, chunk_count)

    async def index_status(self) -> IndexStatus:
        row_sources = await self._db.fetchrow("select count(*) as count from sources")
        row_documents = await self._db.fetchrow("select count(*) as count from documents")
        row_chunks = await self._db.fetchrow("select count(*) as count from chunks")
        last_ingest = await self._db.fetchrow("select max(fetched_at) as ts from documents")
        return IndexStatus(
            sources=int(row_sources["count"]) if row_sources else 0,
            documents=int(row_documents["count"]) if row_documents else 0,
            chunks=int(row_chunks["count"]) if row_chunks else 0,
            last_ingest_at=last_ingest["ts"] if last_ingest and last_ingest["ts"] else None,
        )

    @staticmethod
    def _is_uuid(value: str) -> bool:
        try:
            UUID(value)
            return True
        except Exception:
            return False


__all__ = ["RAGEngine", "IngestStats"]
