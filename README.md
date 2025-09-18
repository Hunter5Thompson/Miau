# Miau

## WhisperX Service
Legacy WhisperX-related services remain under `app.py` and the `services/` package.

## Crawl4AI MCP Server
This repository now includes a Model Context Protocol (MCP) server that crawls the web using
[Crawl4AI](https://github.com/unclecode/crawl4ai) and stores Markdown chunks plus embeddings in PostgreSQL with `pgvector`.
The server exposes MCP tools for crawling individual pages, crawling entire domains, querying the RAG index, listing index status,
and purging indexed sources.

### Features
- Crawl single pages or discover entire domains via Crawl4AI's `AsyncWebCrawler` and `AsyncUrlSeeder`.
- Chunk Markdown with heading awareness and store results in PostgreSQL using `asyncpg`.
- Generate OpenAI embeddings (defaults to `text-embedding-3-small`) and persist them in `pgvector`.
- Query the index via vector or hybrid (vector + FTS) strategies through an MCP tool.
- Optional HTTP transport via FastAPI/uvicorn in addition to the stdio MCP transport.

### Local Development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-mcp.txt
export C4MCP_DATABASE_DSN="postgresql://postgres:postgres@localhost:5432/crawl4ai_mcp"
export OPENAI_API_KEY="sk-..."
python -m crawl4ai_mcp.main  # stdio transport
# or run the HTTP transport
uvicorn crawl4ai_mcp.http_app:app --port 8051
```

### Docker Compose
A dedicated compose file (`docker-compose.mcp.yml`) starts PostgreSQL (with pgvector), the MCP server (HTTP transport),
and an optional Adminer instance for inspection:

```bash
docker compose -f docker-compose.mcp.yml up --build
```

This exposes:
- MCP HTTP transport at `http://localhost:8051/mcp`
- Adminer at `http://localhost:8080`

### Running Tests
Install development dependencies (see "Local Development" above) and run:

```bash
pytest tests -q
```

### Environment Variables
The MCP server reads configuration via `pydantic-settings` with the `C4MCP_` prefix. Useful variables include:
- `C4MCP_DATABASE_DSN` – PostgreSQL DSN (default: `postgresql://postgres:postgres@localhost:5432/crawl4ai_mcp`).
- `C4MCP_EMBEDDING_MODEL` – OpenAI embedding model name (`text-embedding-3-small`).
- `C4MCP_EMBEDDING_DIMENSIONS` – Optional embedding dimensionality override.
- `OPENAI_API_KEY` – Required for embeddings when running outside of tests.

