# Multi-Agent Research Team

Dieses Verzeichnis enthält den modularen Multi-Agenten-Stack, der in der Projekt-README beschrieben wird. Kernfeatures:

- **Priorisierte Wissensquellen** mit expliziten Gewichten (Paper > Blogs).
- **Integrationen** für DuckDuckGo/Tavily, Crawl4AI, Docling, PGVector & Neo4j.
- **LangGraph**-kompatible Orchestrierung via `MultiAgentSearchTeam.langgraph_app`.
- **CLI** (`python -m services.multi_agent`), die Ziele und Constraints aus Argumenten oder Umgebungsvariablen liest.

## Entwicklung

```bash
python -m compileall services/multi_agent
python -m services.multi_agent --dry-run
```

## Container

```bash
cd services/multi_agent
docker compose up --build
```

Die Container-Variante nutzt [`Requirments_Agents.txt`](../../Requirments_Agents.txt) für die Abhängigkeiten.
