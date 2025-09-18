from crawl4ai_mcp.chunker import chunk_markdown


def test_chunker_splits_on_headings():
    markdown = """# Title

Paragraph one has some text.

## Section
More text here spanning multiple sentences.
"""
    chunks = chunk_markdown(markdown, chunk_tokens=10, overlap_tokens=2)
    assert len(chunks) >= 2
    assert chunks[0].heading == "Title"
    assert chunks[0].metadata["heading_path"] == ["Title"]
    assert all(chunk.text for chunk in chunks)


def test_chunker_requires_larger_chunk_than_overlap():
    markdown = "# Heading\nContent"
    try:
        chunk_markdown(markdown, chunk_tokens=5, overlap_tokens=5)
    except ValueError as exc:
        assert "chunk_tokens" in str(exc)
    else:  # pragma: no cover - ensure failure visible
        assert False
