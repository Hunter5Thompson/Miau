"""Data schemas used by the research multi-agent workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl

logger = logging.getLogger(__name__)


class KnowledgeSource(BaseModel):
    """Represents a single knowledge source with an explicit priority weight."""

    name: str = Field(..., description="Human readable name of the source")
    kind: str = Field(
        ..., description="Type of source (paper, benchmark, blog, dataset, etc.)"
    )
    priority: int = Field(
        1,
        ge=1,
        description="Higher priority numbers indicate more trusted or relevant sources.",
    )
    weight: float = Field(
        1.0,
        ge=0.0,
        description="Continuous weight used when scoring documents coming from the source.",
    )
    url: Optional[HttpUrl] = Field(
        None, description="Canonical URL of the knowledge base, if applicable."
    )
    tags: List[str] = Field(default_factory=list, description="Additional metadata tags.")

    model_config = {"protected_namespaces": ()}


class PrioritizedKnowledgeSources(BaseModel):
    """Holds a list of knowledge sources sorted by their weight/priority."""

    sources: List[KnowledgeSource] = Field(default_factory=list)

    model_config = {"protected_namespaces": ()}

    def sorted_sources(self) -> List[KnowledgeSource]:
        """Return sources sorted by ``priority`` and ``weight`` descending."""

        return sorted(
            self.sources,
            key=lambda item: (item.priority, item.weight, item.name.lower()),
            reverse=True,
        )

    def top_sources(self, limit: Optional[int] = None) -> List[KnowledgeSource]:
        """Return the top ``limit`` sources with the highest priority."""

        ordered = self.sorted_sources()
        if limit is None:
            return ordered
        return ordered[:limit]

    def as_display(self) -> List[Dict[str, Any]]:
        """Return a serialisable representation useful for reporting."""

        return [
            {"name": src.name, "kind": src.kind, "priority": src.priority}
            for src in self.sorted_sources()
        ]


DEFAULT_PRIORITIZED_SOURCES = PrioritizedKnowledgeSources(
    sources=[
        KnowledgeSource(
            name="arXiv",
            kind="paper",
            priority=5,
            weight=1.0,
            url="https://arxiv.org",
            tags=["peer-reviewed", "ml"],
        ),
        KnowledgeSource(
            name="ACL Anthology",
            kind="paper",
            priority=5,
            weight=0.95,
            url="https://aclanthology.org",
            tags=["nlp", "peer-reviewed"],
        ),
        KnowledgeSource(
            name="OpenReview",
            kind="paper",
            priority=4,
            weight=0.9,
            url="https://openreview.net",
            tags=["peer-reviewed", "conference"],
        ),
        KnowledgeSource(
            name="Hugging Face",
            kind="model",
            priority=3,
            weight=0.85,
            url="https://huggingface.co",
            tags=["model-hub"],
        ),
        KnowledgeSource(
            name="GitHub",
            kind="code",
            priority=2,
            weight=0.6,
            url="https://github.com",
            tags=["implementation"],
        ),
        KnowledgeSource(
            name="Specialised Blogs",
            kind="blog",
            priority=1,
            weight=0.3,
            tags=["secondary"],
        ),
    ]
)


class SearchResult(BaseModel):
    """Represents a single web search result."""

    title: str
    url: HttpUrl
    snippet: Optional[str] = None
    source: Optional[str] = None
    score: float = 0.0

    model_config = {"protected_namespaces": ()}


class ScrapedDocument(BaseModel):
    """Content retrieved from the web crawler or scraping layer."""

    url: HttpUrl
    source: str
    title: Optional[str] = None
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"protected_namespaces": ()}


class ResearchPlanStep(BaseModel):
    """A single step of the research plan."""

    order: int
    agent_role: str
    description: str
    expected_output: str
    success_criteria: List[str] = Field(default_factory=list)

    model_config = {"protected_namespaces": ()}


class ResearchPlan(BaseModel):
    """Structured plan produced by the ``Research Strategist``."""

    goal: str
    rationale: str
    steps: List[ResearchPlanStep]

    model_config = {"protected_namespaces": ()}


class RerankerModelCandidate(BaseModel):
    """Candidate reranker models identified during analysis."""

    name: str
    source: str
    url: Optional[HttpUrl] = None
    score: float = 0.0
    notes: Optional[str] = None

    model_config = {"protected_namespaces": ()}


class ResearchArtifact(BaseModel):
    """Generic artifact produced by any agent in the workflow."""

    kind: str
    title: str
    content: str
    source_links: List[HttpUrl] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"protected_namespaces": ()}


class ResearchState(BaseModel):
    """Mutable state shared between the agents."""

    goal: str
    constraints: List[str] = Field(default_factory=list)
    prioritized_sources: PrioritizedKnowledgeSources
    context: Dict[str, Any] = Field(default_factory=dict)
    plan: Optional[ResearchPlan] = None
    knowledge_base: List[ScrapedDocument] = Field(default_factory=list)
    artifacts: List[ResearchArtifact] = Field(default_factory=list)
    reranker_models: List[RerankerModelCandidate] = Field(default_factory=list)
    state_log: List[str] = Field(default_factory=list)

    model_config = {"protected_namespaces": ()}

    def log(self, message: str) -> None:
        """Add a timestamped message to the state log."""

        timestamp = datetime.utcnow().isoformat()
        entry = f"{timestamp} - {message}"
        self.state_log.append(entry)
        logger.debug(entry)


@dataclass
class GatheredEvidence:
    """Container returned by the ``Data Miner`` with raw documents."""

    documents: List[ScrapedDocument]
    search_results: List[SearchResult]


__all__ = [
    "KnowledgeSource",
    "PrioritizedKnowledgeSources",
    "DEFAULT_PRIORITIZED_SOURCES",
    "SearchResult",
    "ScrapedDocument",
    "ResearchPlanStep",
    "ResearchPlan",
    "RerankerModelCandidate",
    "ResearchArtifact",
    "ResearchState",
    "GatheredEvidence",
]
