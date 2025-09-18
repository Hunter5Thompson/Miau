"""Integration helpers for external research tooling.

Each helper stays deliberately lightweight so the module is importable even when
optional dependencies (Crawl4AI, Doclin, etc.) are unavailable. When a
dependency is missing the code falls back to simple Python-only behaviour which
keeps the overall developer experience smooth while still documenting where real
integrations would plug in.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .schemas import ScrapedDocument, SearchResult

logger = logging.getLogger(__name__)


class SearchClient(Protocol):
    """Protocol used for internet search clients (DuckDuckGo, Tavily, etc.)."""

    def search(self, query: str, *, max_results: int = 5) -> List[SearchResult]:
        """Return a list of ``SearchResult`` objects for the given query."""


class DuckDuckGoSearchClient:
    """Minimal DuckDuckGo search implementation used as the default search client."""

    def __init__(
        self,
        *,
        region: str = "wt-wt",
        session: Optional[requests.Session] = None,
    ) -> None:
        self.region = region
        self.session = session or requests.Session()

    def search(self, query: str, *, max_results: int = 5) -> List[SearchResult]:
        params = {"q": query, "kl": self.region, "kp": -2}
        headers = {
            "User-Agent": "Mozilla/5.0 (MultiAgentSearchTeam; compatible)",
            "Accept-Language": "en-US,en;q=0.5",
        }
        try:
            response = self.session.get(
                "https://duckduckgo.com/html/",
                params=params,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network may be blocked in CI
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []

        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<href>[^\"]+)">(?P<title>.*?)</a>',
            re.S,
        )
        matches = pattern.finditer(response.text)
        results: List[SearchResult] = []
        for match in matches:
            href = match.group("href")
            parsed = urlparse(href)
            url = href
            if parsed.netloc == "duckduckgo.com":
                query_dict = parse_qs(parsed.query)
                uddg = query_dict.get("uddg")
                if uddg:
                    url = unquote(uddg[0])
            title = re.sub(r"<.*?>", "", match.group("title")).strip()
            snippet = None
            results.append(
                SearchResult(
                    title=title or query,
                    url=url,
                    snippet=snippet,
                    source="duckduckgo",
                    score=float(len(results) + 1),
                )
            )
            if len(results) >= max_results:
                break
        return results


class Crawl4AIScraper:
    """Lightweight wrapper that prefers Crawl4AI and falls back to ``requests``."""

    def __init__(self) -> None:
        try:  # pragma: no cover - optional dependency
            from crawl4ai import WebCrawler  # type: ignore

            self._crawler_cls = WebCrawler
        except Exception:  # pragma: no cover - executed when dependency missing
            self._crawler_cls = None
        self._session = requests.Session()

    def scrape(self, url: str) -> Optional[str]:
        if self._crawler_cls is not None:  # pragma: no cover - requires crawl4ai
            crawler = self._crawler_cls()
            try:
                page = crawler.run(url)
                if page and hasattr(page, "content"):
                    return str(page.content)
            except Exception as exc:  # pragma: no cover - depends on external site
                logger.warning("Crawl4AI scraping failed: %s", exc)

        try:
            response = self._session.get(
                url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:  # pragma: no cover - network may be blocked
            logger.warning("Fallback scraping failed: %s", exc)
            return None


class DoclinFormatter:
    """Produces structured summaries, integrating with Doclin when available."""

    def __init__(self) -> None:
        try:  # pragma: no cover - optional dependency
            from docling import Document  # type: ignore

            self._doc_class = Document
        except Exception:  # pragma: no cover - executed when dependency missing
            self._doc_class = None

    def to_structured_summary(
        self, documents: Sequence[ScrapedDocument]
    ) -> Dict[str, Any]:
        """Return a structured summary of the supplied documents."""

        summary: List[Dict[str, Any]] = []
        for document in documents:
            key_points = []
            for line in document.content.splitlines():
                line = line.strip()
                if len(line) > 12:
                    key_points.append(line)
                if len(key_points) >= 3:
                    break
            summary.append(
                {
                    "title": document.title or document.source,
                    "url": str(document.url),
                    "key_points": key_points,
                    "metadata": document.metadata,
                }
            )
        if self._doc_class is not None:  # pragma: no cover - requires docling
            try:
                structured = self._doc_class(summary=json.dumps(summary))
                return {"doclin_document": structured, "summary": summary}
            except Exception as exc:  # pragma: no cover
                logger.warning("Doclin conversion failed: %s", exc)
        return {"summary": summary}


class InMemoryVectorStore:
    """Simple vector store emulating PGVector behaviour for local execution."""

    def __init__(self) -> None:
        self._documents: Dict[str, Tuple[Dict[str, int], ScrapedDocument]] = {}

    @staticmethod
    def _tokenise(text: str) -> Dict[str, int]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, int], vec_b: Dict[str, int]) -> float:
        shared = set(vec_a) & set(vec_b)
        numerator = sum(vec_a[token] * vec_b[token] for token in shared)
        denom_a = math.sqrt(sum(value * value for value in vec_a.values()))
        denom_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if denom_a == 0 or denom_b == 0:
            return 0.0
        return numerator / (denom_a * denom_b)

    def upsert(self, document: ScrapedDocument) -> None:
        vector = self._tokenise(document.content)
        self._documents[str(document.url)] = (vector, document)

    def similarity_search(self, query: str, *, top_k: int = 5) -> List[ScrapedDocument]:
        query_vec = self._tokenise(query)
        scored: List[Tuple[float, ScrapedDocument]] = []
        for vector, document in self._documents.values():
            score = self._cosine_similarity(vector, query_vec)
            scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]


class InMemoryGraphRAG:
    """Lightweight Neo4j-inspired graph store for entity linking."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[Tuple[str, str, str]] = []

    def upsert_document(self, document: ScrapedDocument) -> None:
        node_id = str(document.url)
        self._nodes[node_id] = {
            "title": document.title,
            "source": document.source,
            "metadata": document.metadata,
        }
        # Connect by source to emphasise prioritised knowledge providers
        for existing_id, meta in list(self._nodes.items()):
            if existing_id == node_id:
                continue
            if meta.get("source") == document.source:
                self._edges.append((node_id, existing_id, "same_source"))

    def neighbours(self, node_id: str) -> List[str]:
        return [edge[1] for edge in self._edges if edge[0] == node_id]


__all__ = [
    "SearchClient",
    "DuckDuckGoSearchClient",
    "Crawl4AIScraper",
    "DoclinFormatter",
    "InMemoryVectorStore",
    "InMemoryGraphRAG",
]
