"""Database helpers built on asyncpg."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

try:  # pragma: no cover - optional dependency during tests
    import asyncpg  # type: ignore
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore

from .errors import RAGError


class Database:
    """Simple wrapper around an asyncpg pool."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 10) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: "asyncpg.Pool" | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if asyncpg is None:
            raise RAGError("asyncpg is required for database operations")
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        self._dsn, min_size=self._min_size, max_size=self._max_size
                    )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator["asyncpg.Connection"]:
        if asyncpg is None:
            raise RAGError("asyncpg is required for database operations")
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["asyncpg.Connection"]:
        async with self.connection() as conn:
            async with conn.transaction():
                yield conn

    async def execute(self, query: str, *args) -> str:
        async with self.connection() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list["asyncpg.Record"]:
        async with self.connection() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> "asyncpg.Record" | None:
        async with self.connection() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        async with self.connection() as conn:
            return await conn.fetchval(query, *args)

    async def run_migration(self, sql: str) -> None:
        try:
            async with self.connection() as conn:
                await conn.execute(sql)
        except Exception as exc:  # pragma: no cover - asyncpg already logs stack
            raise RAGError("failed to apply migration", cause=exc) from exc


__all__ = ["Database"]
