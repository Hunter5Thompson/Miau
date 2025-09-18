"""Crawl4AI powered crawling utilities."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency at runtime
    from crawl4ai import (
        AsyncUrlSeeder,
        AsyncWebCrawler,
        CacheMode,
        CrawlerRunConfig,
        SeedingConfig,
    )
except Exception:  # pragma: no cover - tests may not install crawl4ai
    AsyncUrlSeeder = None  # type: ignore
    AsyncWebCrawler = None  # type: ignore
    CacheMode = None  # type: ignore
    CrawlerRunConfig = None  # type: ignore
    SeedingConfig = None  # type: ignore

from ..chunker import chunk_markdown
from ..config import Settings
from ..errors import CrawlError
from ..models import CrawlDocument

LOGGER = logging.getLogger("crawl4ai_mcp.crawler")


T = TypeVar("T")


async def _retry(coro_factory: Callable[[], Awaitable[T]], *, retries: int, base_delay: float) -> T:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return await coro_factory()
        except Exception as exc:  # pragma: no cover - re-raised with context
            last_error = exc
            delay = base_delay * (2**attempt)
            jitter = random.uniform(0.1, 0.5)
            await asyncio.sleep(delay + jitter)
    assert last_error is not None
    raise last_error


async def _run_single_crawl(
    crawler: "AsyncWebCrawler",
    url: str,
    settings: Settings,
) -> CrawlDocument:
    async def run_once():
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS if CacheMode else None,
            user_agent=settings.crawl_user_agent,
            verbose=False,
        )
        result = await crawler.arun(url=url, config=config)
        if not getattr(result, "success", False):
            error_message = getattr(result, "error_message", "crawl failed")
            raise CrawlError(error_message or "crawl failed", url=url)
        markdown_obj = getattr(result, "markdown", None)
        if markdown_obj is None:
            raise CrawlError("crawler returned no markdown", url=url)
        if hasattr(markdown_obj, "raw_markdown"):
            markdown = markdown_obj.raw_markdown
        else:
            markdown = str(markdown_obj)
        raw_meta = getattr(result, "metadata", {}) or {}
        metadata = {
            "url": url,
            "links": getattr(result, "links", {}),
            "media": getattr(result, "media", {}),
            "metadata": raw_meta,
            "status_code": getattr(result, "status_code", None),
            "discovered_at": asyncio.get_event_loop().time(),
        }
        chunks = chunk_markdown(
            markdown,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        title = raw_meta.get("title") if isinstance(raw_meta, dict) else None
        return CrawlDocument(
            url=url,
            title=title,
            markdown=markdown,
            metadata=metadata,
            chunks=chunks,
        )

    return await _retry(
        run_once,
        retries=settings.crawl_max_retries,
        base_delay=settings.crawl_backoff_seconds,
    )


async def crawl_single_page(url: str, settings: Settings) -> CrawlDocument:
    if AsyncWebCrawler is None or CrawlerRunConfig is None:
        raise CrawlError("crawl4ai is not installed", url=url)

    async with AsyncWebCrawler() as crawler:  # type: ignore[arg-type]
        return await _run_single_crawl(crawler, url, settings)


async def _discover_domain_urls(seed_url: str, settings: Settings, max_pages: int, sitemap: bool) -> list[str]:
    if AsyncUrlSeeder is None or SeedingConfig is None:
        raise CrawlError("crawl4ai seeding utilities not available", url=seed_url)

    parsed = urlparse(seed_url)
    if not parsed.netloc:
        raise CrawlError("seed URL must include a host", url=seed_url)
    domain = parsed.netloc
    scheme = parsed.scheme or "https"

    async with AsyncUrlSeeder() as seeder:  # type: ignore[arg-type]
        config = SeedingConfig(
            source="sitemap" if sitemap else "cc",
            pattern=f"*{domain}*",
            max_urls=max_pages,
            hits_per_sec=settings.crawl_concurrency,
            live_check=False,
            extract_head=True,
        )
        results = await seeder.urls(domain, config)  # type: ignore[arg-type]
        urls = []
        for item in results:
            candidate = item.get("url")
            if not candidate:
                continue
            if candidate.startswith("//"):
                candidate = f"{scheme}:{candidate}"
            if candidate.startswith("/"):
                candidate = f"{scheme}://{domain}{candidate}"
            if urlparse(candidate).netloc.endswith(domain):
                urls.append(candidate)
        unique_urls: list[str] = []
        seen = set()
        for item in urls:
            if item not in seen:
                seen.add(item)
                unique_urls.append(item)
        return unique_urls[:max_pages]


async def crawl_domain(
    seed_url: str,
    *,
    max_pages: int,
    settings: Settings,
    sitemap: bool = True,
) -> list[CrawlDocument]:
    if AsyncWebCrawler is None:
        raise CrawlError("crawl4ai is not installed", url=seed_url)

    urls = await _discover_domain_urls(seed_url, settings, max_pages, sitemap)
    if not urls:
        raise CrawlError("no URLs discovered for domain", url=seed_url)

    async with AsyncWebCrawler() as crawler:  # type: ignore[arg-type]
        semaphore = asyncio.Semaphore(settings.crawl_concurrency)

        async def sem_task(target_url: str) -> CrawlDocument:
            async with semaphore:
                return await _run_single_crawl(crawler, target_url, settings)

        tasks = [
            sem_task(url)
            for url in urls
        ]
        documents: list[CrawlDocument] = []
        for future in asyncio.as_completed(tasks):
            try:
                documents.append(await future)
            except CrawlError as exc:
                LOGGER.warning("failed to crawl %s: %s", exc.url, exc)
        return documents


__all__ = ["crawl_single_page", "crawl_domain"]
