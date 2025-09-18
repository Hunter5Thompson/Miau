# Miau Research Orchestration

Dieses Repository erweitert den ursprünglichen **Miau WhisperX Service** um ein klar strukturiertes Multi-Agenten-Rechercheteam. Der Fokus liegt auf qualitativ hochwertigen Wissensquellen (Paper > Blogposts), modernen RAG-Bausteinen und einem reproduzierbaren Container-Setup, das die komplette Pipeline ausführbar macht.

## Projektüberblick

Der Multi-Agenten-Workflow besteht aus fünf Rollen, die einander die Staffel übergeben:

1. **Research Strategist** – schärft Zielsetzung, Constraints und den Arbeitsplan.
2. **Data Miner** – führt priorisierte Websuchen durch, nutzt Crawl4AI für Scraping und speichert Ergebnisse in Vektor- und Graph-Indizes.
3. **Data Analyst** – formatiert Dokumente strukturiert mit Docling, analysiert Evidenz und sucht gezielt nach aktuellen Reranker-Modellen.
4. **Research Writer** – fasst Erkenntnisse zu einem Research-Report zusammen.
5. **LangGraph Orchestrator** – optionaler StateGraph, der die Pipeline zustandsbasiert ausführt.

Die Agenten teilen sich einen `ResearchState`, der über Pydantic validiert wird und damit eine typsichere Grundlage für LangGraph, PGVector und Neo4j schafft.

## Technologie-Stack

| Komponente | Zweck |
|------------|-------|
| **Pydantic** | Definiert Schemas für Wissensquellen, Artefakte und den geteilten Zustand |
| **LangGraph** | Optionaler StateGraph, der das Agenten-Team orkestriert |
| **DuckDuckGo / Tavily** | Internet-Suche (DuckDuckGo ist als Default integriert, Tavily lässt sich austauschen) |
| **Crawl4AI** | Robustes Webscraping mit Fallback auf `requests` |
| **Docling** | Structured Output / Summaries |
| **PGVector** | Hybrid-RAG Speicher für Dokumente |
| **Neo4j** | GraphRAG zur Verbindung thematischer Nachbarn |

> Eine auf das Multi-Agenten-Team zugeschnittene Abhängigkeitsliste findet sich in [`Requirments_Agents.txt`](./Requirments_Agents.txt).

## Verzeichnisstruktur

```
services/
└── multi_agent/
    ├── agents.py            # Implementierung der einzelnen Rollen
    ├── integrations.py      # Suche, Scraping, Vektor- und Graphspeicher
    ├── schemas.py           # Pydantic-Modelle & Priorisierungen
    ├── team.py              # Orchestrator + optionale LangGraph-Pipeline
    ├── cli.py               # Kommandozeilen-Einstiegspunkt
    ├── __main__.py          # ermöglicht `python -m services.multi_agent`
    └── README.md (optional, siehe unten)
```

## Priorisierte Wissensquellen

Die Standardkonfiguration (`DEFAULT_PRIORITIZED_SOURCES`) gewichtet hochwertige Quellen höher:

- arXiv & ACL Anthology (Priority 5)
- OpenReview (Priority 4)
- Hugging Face Model Hub (Priority 3)
- GitHub (Priority 2)
- Spezialisierte Blogs (Priority 1)

Diese Gewichtung fließt sowohl in die Suchanfragen als auch in die Score-Normalisierung ein.

## Containerisierung

Für die Multi-Agenten-Pipeline wurde ein eigenständiger Container-Stack ergänzt:

- [`services/multi_agent/Dockerfile`](services/multi_agent/Dockerfile) baut ein leichtgewichtiges Image auf Basis von `python:3.11-slim` und installiert `Requirments_Agents.txt`.
- [`services/multi_agent/docker-compose.yml`](services/multi_agent/docker-compose.yml) startet den Container und erlaubt die Steuerung über Umgebungsvariablen.

### Schnelleinstieg

```bash
# Repository klonen
cd services/multi_agent

# Container bauen und starten
docker compose up --build
```

Wichtige Umgebungsvariablen (via `.env` oder `docker compose`):

| Variable | Beschreibung |
|----------|--------------|
| `MULTI_AGENT_GOAL` | Ziel der Recherche (Standard: „Identify state-of-the-art reranker models…“) |
| `MULTI_AGENT_CONSTRAINTS` | Kommagetrennte Liste von Constraints |

Die Standardausgabe des Containers ist ein strukturierter Research-Report inklusive Protokoll.

## Lokale Nutzung

Die CLI kann auch lokal ohne Docker verwendet werden:

```bash
python -m services.multi_agent "Find the best reranker models for hybrid RAG"

# Mehrere Constraints
python -m services.multi_agent "Trend-Analyse zu Rerankern" \
  --constraint "Fokus auf ACL 2023-2024" \
  --constraint "Evaluierung anhand von BEIR Benchmarks"
```

Für Integrations-Tests ohne Netzwerkanfragen:

```bash
python -m services.multi_agent --dry-run
```

## Weiterführende Hinweise

- Die LangGraph-Integration ist optional und wird nur aktiviert, wenn das Paket installiert ist.
- PGVector- und Neo4j-Verbindungen können durch Ersetzen der In-Memory-Implementierungen (`InMemoryVectorStore`, `InMemoryGraphRAG`) hergestellt werden.
- Der Data Analyst sucht standardmäßig nach Reranker-Modellen mit Fokus auf peer-reviewte Veröffentlichungen.

Viel Erfolg beim Erkunden des Multi-Agenten-Stacks!
