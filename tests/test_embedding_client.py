import pytest

from crawl4ai_mcp.errors import RAGError
from crawl4ai_mcp.services.embedding import EmbeddingClient


@pytest.mark.asyncio
async def test_embedding_client_requires_api_key():
    client = EmbeddingClient(api_key=None, model="text-embedding-3-small", dimensions=1536)
    assert client.is_configured is False
    with pytest.raises(RAGError):
        await client.embed_texts(["hello"])
