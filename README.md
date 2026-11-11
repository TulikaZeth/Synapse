<div align="center">

# Synapse — GraphRAG Document Intelligence Platform

**Hybrid Knowledge Graph + Dense Vector Retrieval-Augmented Generation with Multi-Hop Reasoning & Verifiable Citations**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-FF6F00)](https://www.trychroma.com/)
[![Neo4j](https://img.shields.io/badge/Graph%20DB-Neo4j%20%7C%20NetworkX-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-46%20Passing-2EA44F)](#testing--verification)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Why GraphRAG?](#why-graphrag)
- [System Architecture](#system-architecture)
- [Core Workflows](#core-workflows)
  - [1. Asynchronous Ingestion & Knowledge Extraction](#1-asynchronous-ingestion--knowledge-extraction)
  - [2. Hybrid Retrieval & Grounded Generation](#2-hybrid-retrieval--grounded-generation)
- [Data Models & Polyglot Storage](#data-models--polyglot-storage)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Option A: Local Development (Zero External Dependencies)](#option-a-local-development-zero-external-dependencies)
  - [Option B: Docker Compose (Backend + Neo4j)](#option-b-docker-compose-backend--neo4j)
- [Configuration](#configuration)
- [REST API Reference](#rest-api-reference)
- [Testing & Verification](#testing--verification)
- [Implementation Roadmap](#implementation-roadmap)

---

## Overview

**Synapse** is an enterprise-ready **GraphRAG** (Knowledge Graph-Enhanced Retrieval-Augmented Generation) platform that bridges unstructured semantic vector search with structured knowledge graph traversal.

When documents (`.pdf`, `.docx`, `.txt`) are uploaded, Synapse asynchronously parses them, splits text along natural sentence boundaries, indexes dense embeddings in **ChromaDB**, extracts typed entities and directed semantic relationships into **Neo4j** (or an embedded **NetworkX** graph), and fuses vector and graph evidence using **Reciprocal Rank Fusion (RRF)** to synthesize answers backed by verified `[SOURCE X]` citations.

---

## Why GraphRAG?

Traditional vector-only RAG pipelines struggle on queries requiring:

1. **Multi-hop reasoning** across separate chunks or documents (e.g., $A \xrightarrow{\text{USES}} B \xrightarrow{\text{DEPENDS\_ON}} C$).
2. **Indirect entity relationship discovery** where keywords alone do not connect two concepts.
3. **Traceable provenance** linking every generated claim back to exact document pages, chunks, and supporting evidence excerpts.

| Capability | Standard Vector RAG | Synapse GraphRAG |
|---|---|---|
| Semantic chunk similarity | Yes | Yes (ChromaDB cosine search) |
| Multi-hop entity path discovery | No | Yes ($N$-hop BFS & path traversal) |
| Rank fusion across modalities | No | Yes (Reciprocal Rank Fusion, $k_0 = 60$) |
| Offline / zero-API-key fallback | Rarely | Yes (Embedded NetworkX + NLP rules + deterministic embeddings) |
| Strict citation verification | Limited | Yes (`[SOURCE X]` verified against retrieved chunks) |

---

## System Architecture

Synapse follows a layered, polyglot persistence architecture designed for graceful degradation—running out-of-the-box locally with zero external services, and scaling up to PostgreSQL, Neo4j, and Gemini API via environment variables.

```mermaid
flowchart TD
    Client["HTTP Client / Swagger UI / Frontend (Planned)"]

    subgraph Gateway["FastAPI Application Gateway (/api/v1)"]
        Health["GET /health"]
        R_Docs["/documents<br/>Upload, Status, Chunks, Delete"]
        R_Graph["/graph<br/>Entities, Relations, Explore"]
        R_Query["/query<br/>Vector, Graph, Hybrid QA"]
        R_Eval["/evaluation<br/>Benchmark Runner & Results"]
    end

    subgraph Ingestion["Async Ingestion Pipeline (BackgroundTasks)"]
        Parser["Multi-Format Parser<br/>pypdf / pdfplumber / python-docx / TXT"]
        Chunker["Semantic Sliding-Window Chunker<br/>500 chars | 100 overlap"]
        Extractor["NER & Relation Extractor<br/>Gemini API / Rule-Based NLP Fallback"]
    end

    subgraph Retrieval["Hybrid Retrieval & Grounded Generation"]
        Linker["Query Entity Linker"]
        VecSearch["Dense Vector Search (Top-K)"]
        GraphWalk["N-Hop Subgraph & Multi-Hop Paths"]
        RRF["Reciprocal Rank Fusion (RRF)"]
        Generator["Grounded Generator & Citation Verifier<br/>Gemini API / Extractive Fallback"]
    end

    subgraph Storage["Polyglot Persistence Tier"]
        SQL[("SQLite / PostgreSQL<br/>documents, chunks, query_logs, evaluation_runs")]
        Chroma[("ChromaDB<br/>graphrag_chunks Collection")]
        Graph[("Neo4j / Embedded NetworkX<br/>Entities & Directed Relations")]
    end

    Client <--> Gateway
    R_Docs -->|Triggers Async| Parser
    Parser --> Chunker
    Chunker --> Extractor
    Chunker --> SQL
    Chunker --> Chroma
    Extractor --> Graph

    R_Query --> Linker
    Linker --> VecSearch
    Linker --> GraphWalk
    VecSearch <--> Chroma
    GraphWalk <--> Graph
    VecSearch --> RRF
    GraphWalk --> RRF
    RRF <--> SQL
    RRF --> Generator
    Generator --> R_Query
    R_Query --> SQL

    R_Graph <--> Graph
    R_Eval <--> SQL
    Health --> SQL & Chroma & Graph
```

---

## Core Workflows

### 1. Asynchronous Ingestion & Knowledge Extraction

Document uploads return immediately (`status: "PENDING"`) while heavy parsing, chunking, embedding, and graph extraction execute asynchronously via FastAPI `BackgroundTasks`.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant API as POST /documents/upload
    participant DB as SQLite / PostgreSQL
    participant BG as Background Pipeline
    participant Parser as Parser & Chunker
    participant Chroma as ChromaDB
    participant Extractor as Gemini / NLP Extractor
    participant Graph as Neo4j / NetworkX

    User->>API: Upload file (.pdf / .docx / .txt)
    API->>API: Validate extension & save to ./data/uploads
    API->>DB: Insert Document (status="PENDING")
    API->>BG: Schedule run_ingestion_pipeline(doc_id)
    API-->>User: 200 OK { id, status: "PENDING" }

    BG->>DB: Update status = "PROCESSING"
    BG->>Parser: Extract text & clean control chars/whitespace
    Parser-->>BG: Overlapping 500-char sentence-aligned chunks
    BG->>DB: Bulk insert ChunkModel rows
    BG->>Chroma: Upsert chunk embeddings + page metadata
    BG->>Extractor: Extract entities & (S, P, O, Evidence) triples
    BG->>Graph: Upsert Entity nodes & RELATION edges
    BG->>DB: Update counts & status = "INDEXED"

    User->>API: Poll GET /documents/{id}/status
    API-->>User: { status: "INDEXED", chunk_count, entity_count, relationship_count }
```

### 2. Hybrid Retrieval & Grounded Generation

Queries in `hybrid` mode combine semantic similarity from ChromaDB with multi-hop relational paths from the Knowledge Graph using **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{vector}, \text{graph}\}} \frac{w_m}{k_0 + r_m(d)}$$

where $k_0 = 60$, default weights $w_{\text{vector}} = 0.5$ and $w_{\text{graph}} = 0.5$ (or `1.0 / 0.0` in `vector`-only mode, `0.0 / 1.0` in `graph`-only mode), and $r_m(d)$ is the 1-based rank of chunk $d$.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant API as POST /api/v1/query
    participant Linker as Query Entity Linker
    participant Chroma as ChromaDB
    participant Graph as Neo4j / NetworkX
    participant RRF as RRF Ranker
    participant Gen as Grounded Generator

    User->>API: { question, retrieval_mode: "hybrid", top_k: 5, graph_depth: 2 }
    API->>Linker: Match query tokens against Graph entities/aliases
    par Parallel Retrieval
        API->>Chroma: Vector similarity search (Top-K * 2)
        API->>Graph: Explore N-hop subgraphs & multi-hop paths
    end
    Chroma-->>RRF: Ranked vector chunks + cosine scores
    Graph-->>RRF: Supporting chunk IDs + Fact Chains + Subgraph
    RRF->>RRF: Compute RRF scores & hydrate missing graph chunks
    RRF-->>Gen: Top-K evidence chunks + [GRAPH KNOWLEDGE] chains
    Gen->>Gen: Synthesize answer & verify [SOURCE X] citations
    Gen-->>API: Answer + VerifiedCitation[] + grounding status
    API->>API: Log query metrics to query_logs
    API-->>User: 200 OK { answer, sources, entities, graph_subgraph, latency_ms }
```

---

## Data Models & Polyglot Storage

### Relational Schema (`SQLAlchemy` — SQLite / PostgreSQL)

```mermaid
erDiagram
    DOCUMENTS ||--o{ CHUNKS : "has (CASCADE DELETE)"
    DOCUMENTS {
        string id PK "UUID v4"
        string filename
        string file_type "pdf | docx | txt"
        int file_size
        string file_path
        string status "PENDING | PROCESSING | INDEXED | FAILED"
        text error_message
        int chunk_count
        int entity_count
        int relationship_count
        datetime created_at
        datetime updated_at
    }
    CHUNKS {
        string id PK "UUID v4 (Chroma ID)"
        string document_id FK
        int chunk_index
        int page_number
        text content
        int token_count
        string embedding_id
        datetime created_at
    }
    QUERY_LOGS {
        string id PK "qry_*"
        text query_text
        string retrieval_mode "vector | graph | hybrid"
        int top_k
        int graph_depth
        text generated_answer
        json retrieved_chunks
        json retrieved_entities
        json citations
        float latency_ms
        datetime created_at
    }
    EVALUATION_RUNS {
        string id PK
        string name
        string retrieval_mode
        float recall_at_k
        float precision_at_k
        float mrr
        float faithfulness
        float answer_relevance
        float avg_latency_ms
        int total_tokens
        datetime created_at
    }
```

### Property Graph Schema (`Neo4j` / Embedded `NetworkX`)

- **`Entity` Nodes**: Canonical normalized `id` (e.g., `TRANSFORMER`), display `name`, `type` (`PERSON`, `ORGANIZATION`, `LOCATION`, `TECHNOLOGY`, `CONCEPT`, `EVENT`), `description`, `aliases[]`, and `document_ids[]`.
- **`RELATION` Edges**: `id`, `source`, `target`, canonical uppercase `type` (`USES`, `DEVELOPED_BY`, `BASED_ON`, `DEPENDS_ON`, `PART_OF`, `TYPE_OF`, `INTRODUCED`, `RELATED_TO`), verbatim `evidence`, `confidence` (`0.0`–`1.0`), `document_id`, and `chunk_id`.

### Vector Store (`ChromaDB`)

- Collection `graphrag_chunks` configured with cosine distance (`hnsw:space: cosine`).
- Uses a built-in zero-dependency 384-dimensional L2-normalized embedding function (`FastDeterministicEmbeddingFunction`) for instant offline execution, storing `chunk_id`, `document_id`, `filename`, `page_number`, `chunk_index`, and `token_count` metadata.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI, Uvicorn, Pydantic v2, `pydantic-settings` |
| **Relational ORM** | SQLAlchemy 2.0 (SQLite default, PostgreSQL-ready) |
| **Vector Database** | ChromaDB (persistent local store) |
| **Knowledge Graph** | Neo4j (`neo4j` Bolt driver) with Embedded NetworkX + JSON fallback |
| **Document Parsing** | `pypdf`, `pdfplumber` (table-aware fallback), `python-docx` |
| **LLM & Extraction** | Google Gemini (`google-genai`) with Rule-Based NLP & Extractive Synthesis fallbacks |
| **Testing** | Pytest, FastAPI `TestClient` (`httpx`) |
| **Containerization** | Docker, Docker Compose (Neo4j 5.18 APOC + FastAPI) |

---

## Project Structure

```text
Synapse/
├── implementation.md               # Master technical specification & roadmap
├── README.md                       # Project documentation
├── docker-compose.yml              # Neo4j + FastAPI multi-container stack
├── .env.example                    # Environment configuration template
├── .gitignore                      # Ignores venv/, data/, *.db, caches
└── backend/
    ├── Dockerfile                  # Backend container image definition
    ├── pyproject.toml              # Build & pytest configuration
    ├── requirements.txt            # Python dependencies
    ├── config.py                   # Centralized Pydantic settings
    ├── database.py                 # SQLAlchemy models & session management
    ├── main.py                     # FastAPI entrypoint, lifespan, /health
    ├── api/
    │   ├── documents.py            # /api/v1/documents endpoints
    │   ├── graph.py                # /api/v1/graph exploration endpoints
    │   ├── query.py                # /api/v1/query hybrid QA endpoint
    │   └── evaluation.py           # /api/v1/evaluation endpoints
    ├── services/
    │   ├── parser.py               # PDF, DOCX, TXT multi-format parser & cleaner
    │   ├── chunker.py              # Semantic sliding-window chunker
    │   ├── extractor.py            # Gemini + Rule-based NER & relation extractor
    │   ├── pipeline.py             # Async ingestion orchestrator
    │   ├── vector_store.py         # ChromaDB client & embedding wrapper
    │   ├── graph_store.py          # Neo4j + Embedded NetworkX dual backend
    │   ├── retriever.py            # Entity linker, graph traversal & RRF fusion
    │   └── generator.py            # Grounded synthesis & [SOURCE X] verifier
    └── tests/
        ├── test_milestone1.py      # Storage, DB cascades & API tests (8 tests)
        ├── test_milestone2.py      # Parsing, chunking, extraction & pipeline tests (29 tests)
        └── test_milestone3.py      # RRF, multi-hop, citations & query API tests (9 tests)
```

---

## Quick Start

### Option A: Local Development (Zero External Dependencies)

Works out-of-the-box with SQLite, local ChromaDB, embedded NetworkX graph, and offline fallbacks (no API key or external DB required).

#### 1. Create a virtual environment and install dependencies

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\pip install -r backend\requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

#### 2. (Optional) Configure environment variables

Copy `.env.example` to `.env` if you want to enable Gemini LLM extraction/synthesis, PostgreSQL, or Neo4j:
```bash
cp .env.example .env
```

#### 3. Start the FastAPI server

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**macOS / Linux:**
```bash
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 4. Open interactive API docs

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

---

### Option B: Docker Compose (Backend + Neo4j)

Launches Neo4j 5.18 (with APOC plugin) and the FastAPI backend together:

```bash
docker compose up --build
```

- **FastAPI Backend:** http://localhost:8000
- **Neo4j Browser UI:** http://localhost:7474 (default credentials: `neo4j` / `password`)

---

## Configuration

All settings are managed via [`backend/config.py`](backend/config.py) and loaded from `.env`:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Runtime environment (`development` / `production`) |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Uvicorn server bind host and port |
| `DATABASE_URL` | `sqlite:///./data/graphrag.db` | SQLAlchemy URI (supports PostgreSQL URLs) |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Local persistence path for ChromaDB |
| `CHROMA_COLLECTION_NAME` | `graphrag_chunks` | ChromaDB collection name |
| `GRAPH_PERSIST_FILE` | `./data/embedded_graph.json` | JSON persistence path for NetworkX fallback |
| `NEO4J_ENABLED` | `false` | Set `true` to connect to Neo4j (falls back if unreachable) |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `password` | Neo4j authentication credentials |
| `GEMINI_API_KEY` | `None` | Google Gemini API key (optional; uses offline fallbacks if unset) |
| `LLM_MODEL` | `gemini-1.5-flash` | Gemini model ID for extraction and grounded QA |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Sliding-window chunk target and overlap in characters |
| `UPLOAD_DIR` | `./data/uploads` | Local directory for uploaded source files |

---

## REST API Reference

All feature endpoints are mounted under `/api/v1`.

### System Health
- `GET /health` or `GET /api/v1/health` — Checks database, ChromaDB, and active graph store status.

### Document Management (`/api/v1/documents`)
- `POST /api/v1/documents/upload` — Upload `.pdf`, `.docx`, or `.txt` file (`multipart/form-data`) and trigger background ingestion.
- `GET /api/v1/documents` — List uploaded documents and processing status.
- `GET /api/v1/documents/{document_id}/status` — Poll status (`PENDING`, `PROCESSING`, `INDEXED`, `FAILED`) and chunk/entity/edge counts.
- `GET /api/v1/documents/{document_id}/chunks` — Inspect extracted text chunks and page numbers.
- `DELETE /api/v1/documents/{document_id}` — Cascading delete across SQLite/PostgreSQL, ChromaDB vectors, Knowledge Graph edges/orphan nodes, and disk storage.

### Knowledge Graph (`/api/v1/graph`)
- `GET /api/v1/graph/entities?entity_type=TECHNOLOGY&search=transformer&limit=50` — List/filter entities sorted by degree.
- `GET /api/v1/graph/relationships?source_entity=TRANSFORMER&limit=100` — List directed edges with supporting evidence.
- `POST /api/v1/graph/explore` — Explore the $N$-hop neighborhood subgraph around a focal entity.
- `GET /api/v1/graph/stats` — Active graph backend metrics.

### Hybrid Query & Reasoning (`/api/v1/query`)

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does self-attention work in the Transformer architecture?",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "graph_depth": 2,
    "include_citations": true
  }'
```

**Response:**
```json
{
  "query_id": "qry_a1b2c3d4",
  "question": "How does self-attention work in the Transformer architecture?",
  "answer": "The Transformer relies entirely on self-attention mechanisms... [SOURCE 1]",
  "retrieval_mode": "hybrid",
  "sources": [
    {
      "citation_id": 1,
      "document_id": "f5a7a942-...",
      "filename": "attention.pdf",
      "page": 1,
      "chunk_id": "chk_001",
      "excerpt": "The Transformer architecture relies entirely on an attention mechanism..."
    }
  ],
  "entities": ["Transformer", "Self-Attention"],
  "graph_subgraph": {
    "nodes": [
      {"id": "TRANSFORMER", "label": "Transformer", "type": "TECHNOLOGY", "degree": 2}
    ],
    "edges": [
      {"id": "rel_123", "source": "TRANSFORMER", "target": "SELF_ATTENTION", "type": "USES"}
    ]
  },
  "latency_ms": 42.8,
  "model_used": "extractive_synthesizer",
  "is_fully_grounded": true
}
```

### Evaluation (`/api/v1/evaluation`)
- `POST /api/v1/evaluation/run` — Trigger retrieval mode comparison runs (`vector`, `graph`, `hybrid`).
- `GET /api/v1/evaluation/results` — Retrieve historical evaluation run records.

---

## Testing & Verification

Run the full automated test suite (46 tests covering storage engines, ingestion pipeline, RRF retrieval, citation verification, and FastAPI endpoints):

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\pytest -v backend\tests\
```

**macOS / Linux:**
```bash
PYTHONPATH=. pytest -v backend/tests/
```

---


