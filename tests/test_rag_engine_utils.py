from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.services.rag_engine import RAGEngine


def test_is_uuid_helper():
    assert RAGEngine._is_uuid("12345678-1234-1234-1234-1234567890ab")
    assert not RAGEngine._is_uuid("not-a-uuid")


def test_extract_vector_dimension():
    assert RAGEngine._extract_vector_dimension("vector(2048)") == 2048
    assert RAGEngine._extract_vector_dimension("vector(1536)") == 1536
    assert RAGEngine._extract_vector_dimension("text") is None
    assert RAGEngine._extract_vector_dimension(None) is None


def test_render_schema_replaces_dimension(tmp_path):
    settings = Settings(embedding_dimensions=1024)
    engine = RAGEngine(database=object(), embedder=object(), settings=settings)
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text("create table foo (embedding vector(1536));")
    engine._schema_path = schema_file  # type: ignore[attr-defined]
    sql = engine._render_schema_sql()
    assert "vector(1024)" in sql
