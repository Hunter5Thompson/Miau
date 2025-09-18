from crawl4ai_mcp.config import Settings


def test_settings_defaults_are_defined():
    settings = Settings()
    assert settings.database_dsn.startswith("postgresql://")
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chunk_tokens > settings.chunk_overlap_tokens
