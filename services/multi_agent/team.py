"""High level orchestration for the research multi-agent search workflow."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, TypedDict

try:  # pragma: no cover - optional runtime dependency
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - executed when LangGraph is unavailable
    END = "__end__"
    StateGraph = None

from .agents import DataAnalyst, DataMiner, ResearchStrategist, ResearchWriter
from .integrations import (
    Crawl4AIScraper,
    DoclinFormatter,
    DuckDuckGoSearchClient,
    InMemoryGraphRAG,
    InMemoryVectorStore,
    SearchClient,
)
from .schemas import (
    DEFAULT_PRIORITIZED_SOURCES,
    GatheredEvidence,
    PrioritizedKnowledgeSources,
    ResearchArtifact,
    ResearchState,
)


class MultiAgentSearchTeam:
    """Coordinates the five research agents end-to-end."""

    def __init__(
        self,
        *,
        prioritized_sources: Optional[PrioritizedKnowledgeSources] = None,
        search_client: Optional[SearchClient] = None,
        scraper: Optional[Crawl4AIScraper] = None,
        doc_formatter: Optional[DoclinFormatter] = None,
        vector_store: Optional[InMemoryVectorStore] = None,
        graph_store: Optional[InMemoryGraphRAG] = None,
    ) -> None:
        self.prioritized_sources = (
            prioritized_sources
            or DEFAULT_PRIORITIZED_SOURCES.model_copy(deep=True)
        )
        vector_store = vector_store or InMemoryVectorStore()
        graph_store = graph_store or InMemoryGraphRAG()
        search_client = search_client or DuckDuckGoSearchClient()
        scraper = scraper or Crawl4AIScraper()
        doc_formatter = doc_formatter or DoclinFormatter()

        self.strategist = ResearchStrategist(self.prioritized_sources)
        self.miner = DataMiner(
            prioritized_sources=self.prioritized_sources,
            search_client=search_client,
            scraper=scraper,
            vector_store=vector_store,
            graph_store=graph_store,
        )
        self.analyst = DataAnalyst(
            prioritized_sources=self.prioritized_sources,
            doc_formatter=doc_formatter,
            vector_store=vector_store,
            graph_store=graph_store,
            search_client=search_client,
        )
        self.writer = ResearchWriter()
        self._state: Optional[ResearchState] = None
        self._langgraph_app = self._build_langgraph_app()

    @property
    def state(self) -> Optional[ResearchState]:
        """Return the most recent ``ResearchState`` after ``run`` was executed."""

        return self._state

    @property
    def langgraph_app(self) -> Any:
        """Return the compiled LangGraph app if the dependency is installed."""

        return self._langgraph_app

    def run(
        self, goal: str, *, constraints: Optional[Sequence[str]] = None
    ) -> ResearchArtifact:
        """Execute the full multi-agent research workflow."""

        if self._langgraph_app is not None:
            initial_state: Dict[str, Any] = {
                "goal": goal,
                "constraints": list(constraints or []),
            }
            result: Dict[str, Any] = self._langgraph_app.invoke(initial_state)
            report = result.get("report")
            state = result.get("state")
            if isinstance(state, ResearchState):
                self._state = state
            if isinstance(report, ResearchArtifact):
                return report
            raise RuntimeError(
                "LangGraph execution completed without producing a research report."
            )

        state = self.strategist.define_problem(goal, constraints)
        self.strategist.build_plan(state)
        evidence = self.miner.gather(state)
        analysis = self.analyst.refine(state, evidence)
        report = self.writer.generate(state, analysis)
        self._state = state
        return report

    # ------------------------------------------------------------------
    # LangGraph integration helpers
    # ------------------------------------------------------------------
    def _build_langgraph_app(self) -> Any:
        """Construct a LangGraph ``StateGraph`` when the dependency is available."""

        if StateGraph is None:  # pragma: no cover - optional dependency path
            return None

        class GraphState(TypedDict, total=False):
            goal: str
            constraints: Sequence[str]
            state: ResearchState
            evidence: GatheredEvidence
            analysis: ResearchArtifact
            report: ResearchArtifact

        graph = StateGraph(GraphState)
        graph.add_node("define_problem", self._lg_define_problem)
        graph.add_node("gather", self._lg_gather)
        graph.add_node("analyse", self._lg_analyse)
        graph.add_node("write", self._lg_write)
        graph.set_entry_point("define_problem")
        graph.add_edge("define_problem", "gather")
        graph.add_edge("gather", "analyse")
        graph.add_edge("analyse", "write")
        graph.add_edge("write", END)
        return graph.compile()

    def _lg_define_problem(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        goal = payload.get("goal", "")
        constraints = payload.get("constraints")
        state = self.strategist.define_problem(goal, constraints)
        self.strategist.build_plan(state)
        return {"state": state}

    def _lg_gather(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload["state"]
        evidence = self.miner.gather(state)
        return {"state": state, "evidence": evidence}

    def _lg_analyse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload["state"]
        evidence = payload["evidence"]
        analysis = self.analyst.refine(state, evidence)
        return {"state": state, "analysis": analysis}

    def _lg_write(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload["state"]
        analysis = payload["analysis"]
        report = self.writer.generate(state, analysis)
        self._state = state
        return {"state": state, "report": report}


__all__ = ["MultiAgentSearchTeam"]
