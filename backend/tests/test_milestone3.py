"""
Milestone 3 Test Suite: Hybrid Retrieval & Grounded Generation

Tests for:
1. Query Entity Linker
2. Reciprocal Rank Fusion (RRF) scoring & list fusion
3. Multi-hop Graph Traversal & Fact Chain Discovery
4. Context Assembler & Block Formatting
5. Citation Verification Engine ([SOURCE X])
6. Grounded Generator (Extractive Synthesizer & Gemini API fallback)
7. End-to-End POST /api/v1/query endpoint across vector, graph, and hybrid modes
"""

import os
import shutil
import tempfile
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.config import settings
from backend.database import Base, DocumentModel, ChunkModel, QueryLogModel, get_db
from backend.services.vector_store import VectorStoreService
from backend.services.graph_store import EmbeddedGraphStore
from backend.services.retriever import (
    link_query_entities,
    compute_rrf_score,
    fuse_ranked_lists,
    discover_graph_evidence,
    execute_hybrid_retrieval,
    RetrievedEvidence,
    FactChain,
    HybridRetrievalResult,
    RRF_K0
)
from backend.services.generator import (
    build_grounded_context,
    verify_citations,
    generate_grounded_answer,
    VerifiedCitation,
    GenerationResult
)
from backend.main import app


@pytest.fixture(scope="session")
def m3_temp_dir():
    temp_dir = tempfile.mkdtemp(prefix="graphrag_m3_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def m3_db_session():
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def m3_client(m3_db_session):
    def override_get_db():
        try:
            yield m3_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def populated_stores(m3_temp_dir):
    """Populate vector store and graph store with connected sample knowledge."""
    vs = VectorStoreService(
        persist_dir=os.path.join(m3_temp_dir, "chroma_m3"),
        collection_name="test_m3_chunks"
    )
    vs.initialize()

    gs = EmbeddedGraphStore(
        persist_file=os.path.join(m3_temp_dir, "graph_m3.json")
    )

    doc_id = "doc_attention_paper"
    chunks = [
        {
            "id": "chk_m3_01",
            "content": "The Transformer architecture relies entirely on an attention mechanism to draw global dependencies.",
            "document_id": doc_id,
            "filename": "attention.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "token_count": 15
        },
        {
            "id": "chk_m3_02",
            "content": "Self-attention layers connect all positions to compute representations without sequential recurrence.",
            "document_id": doc_id,
            "filename": "attention.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "token_count": 14
        },
        {
            "id": "chk_m3_03",
            "content": "Multi-head attention uses scaled dot-product attention computed across parallel representation subspaces.",
            "document_id": doc_id,
            "filename": "attention.pdf",
            "page_number": 3,
            "chunk_index": 2,
            "token_count": 16
        }
    ]
    vs.add_chunks(chunks)

    # Entities
    gs.upsert_entity({
        "id": "TRANSFORMER",
        "name": "Transformer",
        "type": "TECHNOLOGY",
        "description": "Neural network architecture based on self-attention",
        "aliases": ["Transformer Model"],
        "document_id": doc_id
    })
    gs.upsert_entity({
        "id": "SELF_ATTENTION",
        "name": "Self-Attention",
        "type": "CONCEPT",
        "description": "Mechanism relating different positions of a sequence",
        "document_id": doc_id
    })
    gs.upsert_entity({
        "id": "SCALED_DOT_PRODUCT",
        "name": "Scaled Dot-Product",
        "type": "CONCEPT",
        "description": "Core attention calculation",
        "document_id": doc_id
    })

    # Relationships
    gs.upsert_relationship({
        "source": "TRANSFORMER",
        "target": "SELF_ATTENTION",
        "type": "USES",
        "evidence": "The Transformer architecture relies entirely on an attention mechanism.",
        "confidence": 0.95,
        "document_id": doc_id,
        "chunk_id": "chk_m3_01"
    })
    gs.upsert_relationship({
        "source": "SELF_ATTENTION",
        "target": "SCALED_DOT_PRODUCT",
        "type": "DEPENDS_ON",
        "evidence": "Multi-head attention uses scaled dot-product attention.",
        "confidence": 0.92,
        "document_id": doc_id,
        "chunk_id": "chk_m3_03"
    })

    return vs, gs


# =============================================================================
# 1. RRF Scoring & Fusion Tests
# =============================================================================

def test_compute_rrf_score_formula():
    # RRF = w_vector / (60 + r_vector) + w_graph / (60 + r_graph)
    # Item ranked #1 in both: 0.5 / 61 + 0.5 / 61 = 1 / 61 ~= 0.016393
    score_both = compute_rrf_score(vector_rank=1, graph_rank=1, w_vector=0.5, w_graph=0.5, k_0=60)
    expected = round((0.5 / 61) + (0.5 / 61), 6)
    assert score_both == expected

    # Item ranked only in vector (#1)
    score_vec_only = compute_rrf_score(vector_rank=1, graph_rank=None, w_vector=0.5, w_graph=0.5, k_0=60)
    assert score_vec_only == round(0.5 / 61, 6)

    # Item appearing in both must outrank item appearing in only one
    assert score_both > score_vec_only


def test_fuse_ranked_lists():
    vector_chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "filename": "f1.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "content": "Content 1",
            "token_count": 10,
            "score": 0.9
        },
        {
            "chunk_id": "c2",
            "document_id": "d1",
            "filename": "f1.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "content": "Content 2",
            "token_count": 10,
            "score": 0.8
        }
    ]
    # Graph discovered c2 as rank #1, c3 as rank #2
    graph_chunk_ids = ["c2", "c3"]

    fused = fuse_ranked_lists(
        vector_chunks=vector_chunks,
        graph_chunk_ids=graph_chunk_ids,
        retrieval_mode="hybrid",
        top_k=5
    )

    # c2 appears in BOTH lists (vector #2, graph #1), so it should rank first!
    assert len(fused) >= 2
    assert fused[0].chunk_id == "c2"
    assert fused[0].graph_rank == 1
    assert fused[0].vector_rank == 2
    assert fused[0].rrf_score > fused[1].rrf_score


# =============================================================================
# 2. Query Entity Linking & Graph Traversal Tests
# =============================================================================

def test_query_entity_linker(populated_stores, monkeypatch):
    vs, gs = populated_stores
    import backend.services.retriever as ret_mod
    monkeypatch.setattr(ret_mod, "graph_store", gs)

    query = "How does the Transformer model use Self-Attention?"
    matched = link_query_entities(query)
    matched_ids = [m["id"] for m in matched]

    assert "TRANSFORMER" in matched_ids
    assert "SELF_ATTENTION" in matched_ids


def test_discover_graph_evidence_and_fact_chains(populated_stores, monkeypatch):
    vs, gs = populated_stores
    import backend.services.retriever as ret_mod
    monkeypatch.setattr(ret_mod, "graph_store", gs)

    linked = [
        {"id": "TRANSFORMER", "name": "Transformer"},
        {"id": "SCALED_DOT_PRODUCT", "name": "Scaled Dot-Product"}
    ]
    chunk_ids, fact_chains, subgraph = discover_graph_evidence(linked, graph_depth=2)

    assert "chk_m3_01" in chunk_ids or "chk_m3_03" in chunk_ids
    assert len(fact_chains) >= 1
    assert fact_chains[0].source_entity == "TRANSFORMER"
    assert fact_chains[0].target_entity == "SCALED_DOT_PRODUCT"
    assert "Self-Attention" in fact_chains[0].formatted_chain() or "SELF_ATTENTION" in fact_chains[0].formatted_chain()
    assert len(subgraph["nodes"]) >= 2


# =============================================================================
# 3. Context Construction & Citation Verification Tests
# =============================================================================

def test_build_grounded_context():
    evidence = [
        RetrievedEvidence(
            chunk_id="chk_001",
            document_id="doc_1",
            filename="paper.pdf",
            page_number=3,
            chunk_index=0,
            content="Self-attention correlates tokens across the sequence.",
            token_count=10,
            vector_score=0.95,
            rrf_score=0.016
        )
    ]
    chain = FactChain(
        source_entity="TRANSFORMER",
        target_entity="ATTENTION",
        hops=["(Transformer) -[USES]-> (Attention)"],
        evidence_text="Uses attention mechanisms."
    )
    result = HybridRetrievalResult(
        query="Explain self-attention",
        retrieval_mode="hybrid",
        top_chunks=evidence,
        linked_entities=["Transformer"],
        fact_chains=[chain],
        subgraph={"nodes": [], "edges": []}
    )

    context_str, source_map = build_grounded_context(result)
    assert "[SOURCE 1 | Document: paper.pdf | Page: 3 | Chunk: chk_001...]" in context_str
    assert "Self-attention correlates tokens across the sequence." in context_str
    assert "[GRAPH KNOWLEDGE]:" in context_str
    assert "[FACT CHAIN]: (Transformer) -[USES]-> (Attention)" in context_str
    assert 1 in source_map


def test_verify_citations():
    source_map = {
        1: RetrievedEvidence(
            chunk_id="c1", document_id="d1", filename="doc1.pdf",
            page_number=1, chunk_index=0, content="Excerpt 1", token_count=5
        ),
        2: RetrievedEvidence(
            chunk_id="c2", document_id="d1", filename="doc1.pdf",
            page_number=2, chunk_index=1, content="Excerpt 2", token_count=5
        )
    }

    # Generated text citing valid [SOURCE 1] and invalid [SOURCE 99]
    generated = "Transformers rely on attention [SOURCE 1], but not recurrent units [SOURCE 99]."
    verified, unverified = verify_citations(generated, source_map)

    assert len(verified) == 1
    assert verified[0].citation_id == 1
    assert verified[0].chunk_id == "c1"
    assert "[SOURCE 99]" in unverified


# =============================================================================
# 4. Grounded Generator Tests (Extractive Synthesizer Fallback)
# =============================================================================

def test_generate_grounded_answer_extractive():
    evidence = [
        RetrievedEvidence(
            chunk_id="chk_01",
            document_id="doc_1",
            filename="paper.pdf",
            page_number=1,
            chunk_index=0,
            content="The Transformer relies entirely on self-attention mechanisms to compute representations.",
            token_count=12,
            vector_score=0.92,
            rrf_score=0.016
        )
    ]
    ret_res = HybridRetrievalResult(
        query="What does the Transformer rely on?",
        retrieval_mode="hybrid",
        top_chunks=evidence,
        linked_entities=["Transformer"],
        fact_chains=[],
        subgraph={"nodes": [], "edges": []}
    )

    gen_res = generate_grounded_answer(ret_res, use_llm=False)
    assert isinstance(gen_res, GenerationResult)
    assert "[SOURCE 1]" in gen_res.answer
    assert len(gen_res.verified_citations) == 1
    assert gen_res.verified_citations[0].citation_id == 1
    assert gen_res.is_fully_grounded is True


def test_generate_grounded_answer_empty():
    ret_res = HybridRetrievalResult(
        query="Quantum gravity in black holes?",
        retrieval_mode="hybrid",
        top_chunks=[],
        linked_entities=[],
        fact_chains=[],
        subgraph={"nodes": [], "edges": []}
    )
    gen_res = generate_grounded_answer(ret_res, use_llm=False)
    assert "no relevant evidence" in gen_res.answer.lower()
    assert len(gen_res.verified_citations) == 0


# =============================================================================
# 5. End-to-End POST /api/v1/query Endpoint Integration Tests
# =============================================================================

def test_api_v1_query_modes(m3_client, populated_stores, monkeypatch):
    vs, gs = populated_stores
    import backend.services.retriever as ret_mod
    monkeypatch.setattr(ret_mod, "vector_store", vs)
    monkeypatch.setattr(ret_mod, "graph_store", gs)

    # 1. Hybrid query
    res_hybrid = m3_client.post("/api/v1/query", json={
        "question": "How does self-attention work in the Transformer architecture?",
        "retrieval_mode": "hybrid",
        "top_k": 3,
        "graph_depth": 2,
        "include_citations": True
    })
    assert res_hybrid.status_code == 200
    data_hybrid = res_hybrid.json()
    assert "query_id" in data_hybrid
    assert data_hybrid["retrieval_mode"] == "hybrid"
    assert len(data_hybrid["answer"]) > 20
    assert len(data_hybrid["sources"]) > 0
    assert data_hybrid["sources"][0]["citation_id"] >= 1
    assert "Transformer" in data_hybrid["entities"] or "Self-Attention" in data_hybrid["entities"]
    assert "nodes" in data_hybrid["graph_subgraph"]
    assert data_hybrid["latency_ms"] > 0

    # 2. Vector-only query
    res_vector = m3_client.post("/api/v1/query", json={
        "question": "scaled dot-product attention",
        "retrieval_mode": "vector",
        "top_k": 2
    })
    assert res_vector.status_code == 200
    data_vector = res_vector.json()
    assert data_vector["retrieval_mode"] == "vector"
    assert len(data_vector["sources"]) > 0

    # 3. Invalid mode validation rejection
    res_invalid = m3_client.post("/api/v1/query", json={
        "question": "test query",
        "retrieval_mode": "unsupported_mode"
    })
    assert res_invalid.status_code == 400
    assert "Invalid retrieval mode" in res_invalid.json()["detail"]
