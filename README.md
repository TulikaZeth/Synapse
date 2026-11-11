# Synapse - GraphRAG Document Intelligence Platform

**GraphRAG** bridges unstructured dense semantic vector retrieval (ChromaDB) with structured property graph traversal (Neo4j / Embedded NetworkX graph engine) and relational metadata persistence (SQLite / PostgreSQL) to power multi-hop reasoning with verifiable source citations.

---

## Milestone 1 Implementation Overview

Milestone 1 establishes the foundational storage layer, relational schemas, dual-graph engine, vector store, and FastAPI backend services.

### Implemented Modules

1. **Relational Database (`backend/database.py`)**
   - Tables:
     - `documents`: Document records, file size, status (`PENDING`, `PROCESSING`, `INDEXED`, `FAILED`), chunk/entity counters.
     - `chunks`: Positional chunk metadata, page provenance, token counts, and vector IDs.
     - `query_logs`: Complete log of natural language queries, retrieval modes, latency, retrieved chunks, and citation records.
     - `evaluation_runs`: Benchmark runs tracking Recall@K, Precision@K, MRR, Faithfulness, and Answer Relevance.
   - Cascading delete: Removing a document automatically deletes its chunks.

2. **Dense Vector Store (`backend/services/vector_store.py`)**
   - ChromaDB persistent storage integration.
   - Built-in zero-dependency deterministic 384-dimensional embedding function with L2-normalized cosine distance.
   - Operations: `add_chunks`, `query`, `get_chunk`, `delete_by_document_id`, `get_stats`.

3. **Unified Knowledge Graph Store (`backend/services/graph_store.py`)**
   - Automatic fallback: Seamlessly connects to Neo4j if available and configured; otherwise operates on embedded NetworkX engine with JSON disk persistence.
   - Canonical entity normalization, alias tracking, and document ID provenance.
   - Directed relationships with evidence excerpts, confidence scores, and chunk citations.
   - Graph operations: `upsert_entity`, `upsert_relationship`, `explore_subgraph` (N-hop traversal), `find_multi_hop_paths`, `delete_by_document_id`.

4. **FastAPI Application Server (`backend/main.py`)**
   - Comprehensive `/health` check returning real-time status of the database, vector store, and graph store.
   - Routers mounted under `/api/v1`:
     - `/api/v1/documents`: Upload with file-type validation (`.pdf`, `.docx`, `.txt`), status, chunk inspection, and cascade deletion.
     - `/api/v1/graph`: Entity exploration, relationship retrieval, and subgraph visualization endpoint.
     - `/api/v1/query`: Multi-mode query endpoint (`vector`, `graph`, `hybrid`).
     - `/api/v1/evaluation`: Benchmark runner and historical results.

5. **Containerization & Orchestration (`docker-compose.yml`, `backend/Dockerfile`)**
   - Multi-container stack orchestrating Neo4j (APOC enabled) and FastAPI backend.

---

## Running the Backend

### Prerequisites
- Python 3.11+
- Virtual environment (`venv`)

### 1. Setup Environment
```bash
python -m venv venv
.\venv\Scripts\pip install -r backend\requirements.txt
```

### 2. Run Test Suite
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\pytest -v backend\tests\test_milestone1.py
```

### 3. Start Development Server
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

