"""Agent implementations for the research workflow."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .integrations import (
    Crawl4AIScraper,
    DoclinFormatter,
    DuckDuckGoSearchClient,
    InMemoryGraphRAG,
    InMemoryVectorStore,
    SearchClient,
)
from .schemas import (
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


class ResearchStrategist:
    """Agent responsible for defining the goal and building the plan."""

    def __init__(self, prioritized_sources: PrioritizedKnowledgeSources) -> None:
        self.prioritized_sources = prioritized_sources

    def define_problem(
        self, goal: str, constraints: Optional[Sequence[str]] = None
    ) -> ResearchState:
        state = ResearchState(
            goal=goal,
            constraints=list(constraints or []),
            prioritized_sources=self.prioritized_sources,
        )
        state.log("Research Strategist defined the objective and constraints.")
        return state

    def build_plan(self, state: ResearchState) -> ResearchPlan:
        steps = [
            ResearchPlanStep(
                order=1,
                agent_role="Research Strategist",
                description="Spezifiziere Zielsetzung, Erfolgsmetriken und Kontextinformationen.",
                expected_output="Projektbriefing mit klarer Problemdefinition",
                success_criteria=[
                    "Explizite Zielbeschreibung",
                    "Liste harter Constraints",
                ],
            ),
            ResearchPlanStep(
                order=2,
                agent_role="Data Miner",
                description=(
                    "Suche fokussiert nach hochwertigen Quellen (Paper vor Blogs) unter Nutzung"
                    " der priorisierten Wissensquellen."
                ),
                expected_output="Korpus relevanter Dokumente mit Metadaten",
                success_criteria=[
                    "Bevorzugung von peer-reviewten Publikationen",
                    "Abdeckung aktueller Entwicklungen",
                ],
            ),
            ResearchPlanStep(
                order=3,
                agent_role="Data Analyst",
                description=(
                    "Analysiere und strukturiere die Dokumente. Suche aktiv nach aktuellen "
                    "Reranker-Modellen und bewerte diese."
                ),
                expected_output="Strukturierte Insights, extrahierte Modelle und Bewertungsmatrix",
                success_criteria=[
                    "Identifizierte Reranker-Modelle",
                    "Bewertete Evidenzqualität",
                ],
            ),
            ResearchPlanStep(
                order=4,
                agent_role="Research Writer",
                description="Verdichte Ergebnisse zu einem Research-Report mit Handlungsempfehlungen.",
                expected_output="Veröffentlichungsreifer Bericht",
                success_criteria=[
                    "Klare Zusammenfassung",
                    "Quellennachweise",
                ],
            ),
        ]
        rationale = (
            "Die Planung stellt sicher, dass hochwertigere Wissensquellen höher gewichtet werden. "
            "Durch den Fokus auf peer-reviewte Publikationen und modellbezogene Recherchen entsteht "
            "eine belastbare Entscheidungsgrundlage."
        )
        plan = ResearchPlan(goal=state.goal, rationale=rationale, steps=steps)
        state.plan = plan
        state.log("Research Strategist entwickelte einen vierstufigen Plan.")
        return plan


class DataMiner:
    """Agent gathering evidence using the configured search and scraping stack."""

    def __init__(
        self,
        *,
        prioritized_sources: PrioritizedKnowledgeSources,
        search_client: Optional[SearchClient] = None,
        scraper: Optional[Crawl4AIScraper] = None,
        vector_store: Optional[InMemoryVectorStore] = None,
        graph_store: Optional[InMemoryGraphRAG] = None,
        max_results_per_source: int = 3,
    ) -> None:
        self.prioritized_sources = prioritized_sources
        self.search_client = search_client or DuckDuckGoSearchClient()
        self.scraper = scraper or Crawl4AIScraper()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.graph_store = graph_store or InMemoryGraphRAG()
        self.max_results_per_source = max_results_per_source

    def gather(self, state: ResearchState) -> GatheredEvidence:
        search_results: List[SearchResult] = []
        documents: List[ScrapedDocument] = []

        for source in self.prioritized_sources.sorted_sources():
            for query in self._queries_for_source(state, source):
                for result in self.search_client.search(
                    query, max_results=self.max_results_per_source
                ):
                    weighted_result = self._apply_source_weight(result, source)
                    search_results.append(weighted_result)
                    document = self._scrape_result(weighted_result, source)
                    if document is None:
                        continue
                    documents.append(document)
                    self.vector_store.upsert(document)
                    self.graph_store.upsert_document(document)
            state.log(
                f"Data Miner suchte nach '{state.goal}' mit Schwerpunkt auf {source.name}."
            )

        state.knowledge_base.extend(documents)
        state.log(
            "Data Miner sammelte %d Dokumente und speicherte sie im Vector- und Graph-Index."
            % len(documents)
        )
        return GatheredEvidence(documents=documents, search_results=search_results)

    def _queries_for_source(
        self, state: ResearchState, source: KnowledgeSource
    ) -> Iterable[str]:
        keywords = list(state.context.get("keywords", []))
        base_queries = [state.goal] + keywords
        return [f"{query} {source.name}" for query in base_queries]

    def _apply_source_weight(
        self, result: SearchResult, source: KnowledgeSource
    ) -> SearchResult:
        weighted_score = (result.score or 1.0) * source.priority * source.weight
        return result.model_copy(update={"score": weighted_score, "source": source.name})

    def _scrape_result(
        self, result: SearchResult, source: KnowledgeSource
    ) -> Optional[ScrapedDocument]:
        page_text = self.scraper.scrape(str(result.url))
        if not page_text:
            return None
        return ScrapedDocument(
            url=result.url,
            source=source.name,
            title=result.title,
            content=page_text,
            metadata={
                "source_kind": source.kind,
                "priority": source.priority,
                "weight": source.weight,
            },
        )


class DataAnalyst:
    """Agent that synthesises gathered data and searches for reranker models."""

    def __init__(
        self,
        *,
        prioritized_sources: PrioritizedKnowledgeSources,
        doc_formatter: Optional[DoclinFormatter] = None,
        vector_store: Optional[InMemoryVectorStore] = None,
        graph_store: Optional[InMemoryGraphRAG] = None,
        search_client: Optional[SearchClient] = None,
        reranker_limit: int = 5,
    ) -> None:
        self.prioritized_sources = prioritized_sources
        self.doc_formatter = doc_formatter or DoclinFormatter()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.graph_store = graph_store or InMemoryGraphRAG()
        self.search_client = search_client or DuckDuckGoSearchClient()
        self.reranker_limit = reranker_limit

    def refine(
        self, state: ResearchState, evidence: GatheredEvidence
    ) -> ResearchArtifact:
        relevant_documents = self._select_relevant_documents(state, evidence)
        structured = self.doc_formatter.to_structured_summary(relevant_documents)
        rerankers = self._search_reranker_models(state)
        state.reranker_models = rerankers

        insights = self._extract_key_points(structured)
        content_lines = ["Synthese der wichtigsten Erkenntnisse:", ""]
        content_lines.extend(f"- {insight}" for insight in insights)
        if rerankers:
            content_lines.append("")
            content_lines.append("Identifizierte Reranker-Modelle:")
            for candidate in rerankers:
                detail = f"- {candidate.name} (Quelle: {candidate.source})"
                if candidate.url:
                    detail += f" – {candidate.url}"
                content_lines.append(detail)

        metadata: Dict[str, Any] = {
            "structured_summary": structured,
            "reranker_candidates": [candidate.model_dump() for candidate in rerankers],
        }
        graph_context = self._graph_context(relevant_documents)
        if graph_context:
            metadata["graph_neighbours"] = graph_context

        artifact = ResearchArtifact(
            kind="analysis",
            title="Strukturierte Analyse der Recherche",
            content="\n".join(content_lines),
            source_links=[document.url for document in relevant_documents],
            metadata=metadata,
        )
        state.artifacts.append(artifact)
        state.log(
            "Data Analyst verdichtete die Evidenz und identifizierte Reranker-Modelle."
        )
        return artifact

    def _select_relevant_documents(
        self, state: ResearchState, evidence: GatheredEvidence
    ) -> List[ScrapedDocument]:
        vector_hits = self.vector_store.similarity_search(state.goal, top_k=6)
        if not vector_hits:
            return evidence.documents
        seen = {str(doc.url) for doc in vector_hits}
        for document in evidence.documents:
            if str(document.url) not in seen:
                vector_hits.append(document)
                seen.add(str(document.url))
        return vector_hits

    def _extract_key_points(self, structured_summary: Dict[str, Any]) -> List[str]:
        insights: List[str] = []
        for item in structured_summary.get("summary", []):
            points = item.get("key_points") or []
            if not points:
                continue
            insights.append(f"{item['title']}: {points[0]}")
        return insights

    def _graph_context(self, documents: Sequence[ScrapedDocument]) -> Dict[str, List[str]]:
        context: Dict[str, List[str]] = {}
        for doc in documents:
            neighbours = self.graph_store.neighbours(str(doc.url))
            if neighbours:
                context[str(doc.url)] = neighbours
        return context

    def _search_reranker_models(self, state: ResearchState) -> List[RerankerModelCandidate]:
        queries = [
            f"{state.goal} reranker model",
            "state-of-the-art reranker model 2024",
            "information retrieval reranker benchmark",
        ]
        candidates: Dict[str, RerankerModelCandidate] = {}
        for source in self.prioritized_sources.sorted_sources():
            if source.kind not in {"paper", "model", "benchmark"}:
                continue
            for query in queries:
                weighted_query = f"{query} {source.name}"
                results = self.search_client.search(weighted_query, max_results=2)
                for result in results:
                    score = (result.score or 1.0) * source.priority * source.weight
                    candidate = RerankerModelCandidate(
                        name=result.title,
                        source=source.name,
                        url=result.url,
                        score=score,
                        notes="Identifiziert über priorisierte Reranker-Recherche",
                    )
                    key = str(result.url)
                    if key not in candidates or candidates[key].score < score:
                        candidates[key] = candidate
        ordered = sorted(
            candidates.values(), key=lambda candidate: candidate.score, reverse=True
        )
        return ordered[: self.reranker_limit]


class ResearchWriter:
    """Agent assembling the final research report."""

    def __init__(self, template_name: str = "default") -> None:
        self.template_name = template_name

    def generate(
        self, state: ResearchState, analysis: ResearchArtifact
    ) -> ResearchArtifact:
        lines = [
            f"Forschungsbericht: {state.goal}",
            "= " + "=" * (len(state.goal) + 17),
            "",
            "Zusammenfassung:",
            analysis.content,
            "",
            "Constraints:",
        ]
        lines.extend(f"- {constraint}" for constraint in state.constraints or ["Keine"])
        if state.plan:
            lines.append("")
            lines.append("Forschungsplan:")
            for step in state.plan.steps:
                lines.append(
                    f"- Schritt {step.order}: {step.agent_role} – {step.description}"
                )
        if state.reranker_models:
            lines.append("")
            lines.append("Empfohlene Reranker-Modelle:")
            for candidate in state.reranker_models:
                lines.append(
                    f"- {candidate.name} (Quelle: {candidate.source}, Score: {candidate.score:.2f})"
                )
        lines.append("")
        lines.append("Priorisierte Wissensquellen:")
        for source in state.prioritized_sources.sorted_sources():
            lines.append(
                f"- {source.name} ({source.kind}, Priorität {source.priority})"
            )

        artifact = ResearchArtifact(
            kind="report",
            title=f"Research Report – {state.goal}",
            content="\n".join(lines),
            source_links=[link for link in analysis.source_links],
            metadata={
                "state_log": list(state.state_log),
                "reranker_models": [candidate.model_dump() for candidate in state.reranker_models],
                "prioritized_sources": state.prioritized_sources.as_display(),
                "template": self.template_name,
            },
        )
        state.artifacts.append(artifact)
        state.log("Research Writer generierte den finalen Bericht.")
        return artifact


__all__ = [
    "ResearchStrategist",
    "DataMiner",
    "DataAnalyst",
    "ResearchWriter",
]
