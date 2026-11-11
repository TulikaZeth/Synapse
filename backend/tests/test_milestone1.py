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
from backend.database import Base, DocumentModel, ChunkModel, QueryLogModel, EvaluationRunModel, get_db
from backend.services.vector_store import VectorStoreService
from backend.services.graph_store import EmbeddedGraphStore, normalize_entity_id
from backend.main import app


@pytest.fixture(scope="session")
def test_temp_dir():
    temp_dir = tempfile.mkdtemp(prefix="graphrag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def db_session():
    """In-memory SQLite session for tests with StaticPool."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ==========================================
# 1. Relational Database Models & Schema Tests
# ==========================================
def test_database_models_and_cascade(db_session):
    doc_id = str(uuid.uuid4())
    doc = DocumentModel(
        id=doc_id,
        filename="attention_paper.pdf",
        file_type="pdf",
        file_size=10240,
        file_path="/tmp/attention_paper.pdf",
        status="INDEXED",
        chunk_count=2,
        entity_count=3,
        relationship_count=2
    )
    db_session.add(doc)

    chunk1 = ChunkModel(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        chunk_index=0,
        page_number=1,
        content="The Transformer is based solely on attention mechanisms.",
        token_count=10,
        embedding_id="vec_001"
    )
    chunk2 = ChunkModel(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        chunk_index=1,
        page_number=2,
        content="Multi-head attention allows the model to jointly attend to information.",
        token_count=11,
        embedding_id="vec_002"
    )
    db_session.add(chunk1)
    db_session.add(chunk2)
    db_session.commit()

    # Query doc and its chunks
    saved_doc = db_session.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
    assert saved_doc is not None
    assert len(saved_doc.chunks) == 2
    assert saved_doc.to_dict()["filename"] == "attention_paper.pdf"

    # Test Cascade Delete: deleting doc removes its chunks
    db_session.delete(saved_doc)
    db_session.commit()
    remaining_chunks = db_session.query(ChunkModel).filter(ChunkModel.document_id == doc_id).all()
    assert len(remaining_chunks) == 0


def test_query_log_and_evaluation_models(db_session):
    qlog = QueryLogModel(
        id=str(uuid.uuid4()),
        query_text="What is Transformer?",
        retrieval_mode="hybrid",
        top_k=5,
        graph_depth=2,
        generated_answer="Transformer uses self-attention.",
        retrieved_chunks=[{"chunk_id": "c1"}],
        retrieved_entities=["TRANSFORMER"],
        citations=[{"citation_id": 1}],
        latency_ms=120.5
    )
    db_session.add(qlog)

    eval_run = EvaluationRunModel(
        id=str(uuid.uuid4()),
        name="test_eval_hybrid",
        retrieval_mode="hybrid",
        recall_at_k=0.95,
        precision_at_k=0.90,
        mrr=0.92,
        faithfulness=0.98,
        answer_relevance=0.96,
        avg_latency_ms=450.0,
        total_tokens=1500
    )
    db_session.add(eval_run)
    db_session.commit()

    assert db_session.query(QueryLogModel).count() == 1
    assert db_session.query(EvaluationRunModel).count() == 1


# ==========================================
# 2. Vector Store (ChromaDB) Tests
# ==========================================
def test_vector_store_service(test_temp_dir):
    persist_dir = os.path.join(test_temp_dir, "chroma_test")
    vs = VectorStoreService(persist_dir=persist_dir, collection_name="test_chunks")
    vs.initialize()

    stats = vs.get_stats()
    assert stats["status"] == "healthy"
    assert stats["chunk_count"] == 0

    doc_id = "doc_test_123"
    chunks = [
        {
            "id": "chk_1",
            "content": "Deep learning models such as transformers achieve state of the art results.",
            "document_id": doc_id,
            "filename": "deep_learning.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "token_count": 12
        },
        {
            "id": "chk_2",
            "content": "Graph neural networks model complex relational dependencies.",
            "document_id": doc_id,
            "filename": "deep_learning.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "token_count": 10
        }
    ]

    # Add chunks
    added = vs.add_chunks(chunks)
    assert added == 2
    assert vs.get_stats()["chunk_count"] == 2

    # Query chunks
    results = vs.query("transformer attention mechanisms", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] in ["chk_1", "chk_2"]
    assert "score" in results[0]

    # Get single chunk
    single = vs.get_chunk("chk_1")
    assert single is not None
    assert single["chunk_id"] == "chk_1"
    assert single["page_number"] == 1

    # Delete by document_id
    deleted = vs.delete_by_document_id(doc_id)
    assert deleted == 2
    assert vs.get_stats()["chunk_count"] == 0


# ==========================================
# 3. Knowledge Graph Store (Embedded & Traversal) Tests
# ==========================================
def test_graph_store_embedded_and_traversal(test_temp_dir):
    graph_file = os.path.join(test_temp_dir, "graph_test.json")
    gs = EmbeddedGraphStore(persist_file=graph_file)

    # 1. Upsert entities
    e1 = gs.upsert_entity({
        "id": "TRANSFORMER",
        "name": "Transformer",
        "type": "TECHNOLOGY",
        "description": "Neural architecture based on attention",
        "aliases": ["Transformer Model"],
        "document_id": "doc_1"
    })
    assert e1["id"] == "TRANSFORMER"
    assert "Transformer Model" in e1["aliases"]

    e2 = gs.upsert_entity({
        "id": "SELF_ATTENTION",
        "name": "Self Attention",
        "type": "CONCEPT",
        "description": "Attention mechanism relating different positions of sequence",
        "document_id": "doc_1"
    })
    e3 = gs.upsert_entity({
        "id": "SCALED_DOT_PRODUCT",
        "name": "Scaled Dot-Product",
        "type": "CONCEPT",
        "description": "Core attention calculation",
        "document_id": "doc_1"
    })

    # 2. Upsert directed relationships
    r1 = gs.upsert_relationship({
        "source": "TRANSFORMER",
        "target": "SELF_ATTENTION",
        "type": "USES",
        "evidence": "The Transformer uses self-attention mechanisms.",
        "confidence": 0.98,
        "document_id": "doc_1",
        "chunk_id": "chunk_01"
    })
    assert r1["type"] == "USES"

    r2 = gs.upsert_relationship({
        "source": "SELF_ATTENTION",
        "target": "SCALED_DOT_PRODUCT",
        "type": "DEPENDS_ON",
        "evidence": "Self-attention computes scaled dot-product attention.",
        "confidence": 0.95,
        "document_id": "doc_1",
        "chunk_id": "chunk_02"
    })
    assert r2["type"] == "DEPENDS_ON"

    # 3. Test entity retrieval & search
    found = gs.get_entity("TRANSFORMER")
    assert found is not None
    assert found["type"] == "TECHNOLOGY"
    assert found["degree"] >= 1

    search_res = gs.get_entities(search="Attention", limit=10)
    assert len(search_res) >= 1
    assert any(x["id"] == "SELF_ATTENTION" for x in search_res)

    # 4. Explore 2-hop Subgraph
    subgraph = gs.explore_subgraph("Transformer", depth=2, limit=50)
    node_ids = [n["id"] for n in subgraph["nodes"]]
    assert "TRANSFORMER" in node_ids
    assert "SELF_ATTENTION" in node_ids
    assert "SCALED_DOT_PRODUCT" in node_ids
    assert len(subgraph["edges"]) == 2

    # 5. Multi-Hop Path Discovery (Transformer -> Self Attention -> Scaled Dot Product)
    paths = gs.find_multi_hop_paths("TRANSFORMER", "SCALED_DOT_PRODUCT", max_depth=2)
    assert len(paths) >= 1
    first_path = paths[0]
    assert len(first_path) == 2
    assert first_path[0]["from_node"]["id"] == "TRANSFORMER"
    assert first_path[1]["to_node"]["id"] == "SCALED_DOT_PRODUCT"

    # 6. Test Cascading Deletion by document_id
    del_res = gs.delete_by_document_id("doc_1")
    assert del_res["deleted_edges"] == 2
    assert gs.get_stats()["edge_count"] == 0


# ==========================================
# 4. FastAPI Endpoints & Health Check Tests
# ==========================================
def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "components" in data
    assert "database" in data["components"]
    assert "vector_store" in data["components"]
    assert "graph_store" in data["components"]


def test_api_v1_graph_endpoints(client):
    # Get entities
    res_entities = client.get("/api/v1/graph/entities")
    assert res_entities.status_code == 200
    assert isinstance(res_entities.json(), list)

    # Get relationships
    res_rels = client.get("/api/v1/graph/relationships")
    assert res_rels.status_code == 200
    assert isinstance(res_rels.json(), list)

    # Explore subgraph
    res_explore = client.post("/api/v1/graph/explore", json={
        "entity_name": "Transformer",
        "depth": 2,
        "limit": 20
    })
    assert res_explore.status_code == 200
    subgraph = res_explore.json()
    assert "nodes" in subgraph
    assert "edges" in subgraph


def test_api_v1_evaluation_endpoints(client):
    # Run evaluation
    res_run = client.post("/api/v1/evaluation/run", json={
        "benchmark_dataset": "standard_academic_qa",
        "modes": ["vector", "hybrid"],
        "sample_size": 5
    })
    assert res_run.status_code == 200
    runs = res_run.json()
    assert len(runs) == 2
    assert runs[0]["retrieval_mode"] in ["vector", "hybrid"]

    # Get evaluation results
    res_results = client.get("/api/v1/evaluation/results")
    assert res_results.status_code == 200
    assert len(res_results.json()) >= 2


def test_api_v1_document_upload_and_validation(client):
    # Unsupported file type rejection
    bad_file = ("bad.exe", b"malicious binary", "application/octet-stream")
    res_bad = client.post("/api/v1/documents/upload", files={"file": bad_file})
    assert res_bad.status_code == 400
    assert "Unsupported file format" in res_bad.json()["detail"]

    # Valid txt upload
    good_file = ("sample_doc.txt", b"Knowledge graphs bridge unstructured text and reasoning.", "text/plain")
    res_good = client.post("/api/v1/documents/upload", files={"file": good_file})
    assert res_good.status_code == 200
    doc_data = res_good.json()
    assert doc_data["filename"] == "sample_doc.txt"
    assert doc_data["file_type"] == "txt"
    assert doc_data["status"] == "PENDING"

