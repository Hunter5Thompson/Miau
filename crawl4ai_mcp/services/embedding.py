"""Embedding helpers using OpenAI's API."""

from __future__ import annotations

from typing import Sequence

try:  # pragma: no cover - optional import
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover - fallback during tests without dependency
    AsyncOpenAI = None  # type: ignore

from ..errors import RAGError


class EmbeddingClient:
    """Wrapper for generating embeddings."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        dimensions: int | None,
        batch_size: int = 64,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = AsyncOpenAI(api_key=api_key) if AsyncOpenAI and api_key else None

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._client is None:
            raise RAGError("embedding client is not configured (missing API key)")

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = await self._client.embeddings.create(
                input=batch,
                model=self._model,
                dimensions=self._dimensions,
            )
            # Response order preserved
            for item in response.data:
                all_embeddings.append(list(item.embedding))
        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []


__all__ = ["EmbeddingClient"]
