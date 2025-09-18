"""Multi-agent research orchestration components."""

from .agents import DataAnalyst, DataMiner, ResearchStrategist, ResearchWriter
from .integrations import (
    Crawl4AIScraper,
    DoclinFormatter,
    DuckDuckGoSearchClient,
    InMemoryGraphRAG,
    InMemoryVectorStore,
    SearchClient,
)
from .cli import build_parser, run_cli
from .schemas import (
    DEFAULT_PRIORITIZED_SOURCES,
    GatheredEvidence,
    KnowledgeSource,
    PrioritizedKnowledgeSources,
    ResearchArtifact,
    ResearchPlan,
    ResearchPlanStep,
    ResearchState,
    RerankerModelCandidate,
    ScrapedDocument,
    SearchResult,
)
from .team import MultiAgentSearchTeam

__all__ = [
    "DataAnalyst",
    "DataMiner",
    "ResearchStrategist",
    "ResearchWriter",
    "Crawl4AIScraper",
    "DoclinFormatter",
    "DuckDuckGoSearchClient",
    "InMemoryGraphRAG",
    "InMemoryVectorStore",
    "SearchClient",
    "DEFAULT_PRIORITIZED_SOURCES",
    "GatheredEvidence",
    "KnowledgeSource",
    "PrioritizedKnowledgeSources",
    "ResearchArtifact",
    "ResearchPlan",
    "ResearchPlanStep",
    "ResearchState",
    "RerankerModelCandidate",
    "ScrapedDocument",
    "SearchResult",
    "MultiAgentSearchTeam",
    "run_cli",
    "build_parser",
]
