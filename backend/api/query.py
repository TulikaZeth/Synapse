import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db, QueryLogModel
from backend.services.retriever import execute_hybrid_retrieval
from backend.services.generator import generate_grounded_answer

router = APIRouter(prefix="/query", tags=["Query & Reasoning"])


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to answer")
    retrieval_mode: str = Field("hybrid", description="Retrieval mode: 'vector', 'graph', or 'hybrid'")
    top_k: int = Field(5, ge=1, le=20, description="Top-K vector chunks to retrieve")
    graph_depth: int = Field(2, ge=1, le=3, description="Graph traversal depth")
    include_citations: bool = Field(True, description="Enforce and verify source citations")


class SourceCitation(BaseModel):
    citation_id: int
    document_id: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[int] = None
    chunk_id: str
    excerpt: str


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    retrieval_mode: str
    sources: List[SourceCitation]
    entities: List[str]
    graph_subgraph: Dict[str, Any]
    latency_ms: float
    model_used: Optional[str] = None
    is_fully_grounded: Optional[bool] = True


@router.post("", response_model=QueryResponse)
def execute_query(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Execute GraphRAG hybrid retrieval and grounded generation:
    1. Query Entity Linking
    2. Dense Vector Retrieval + Multi-Hop Graph Traversal
    3. Reciprocal Rank Fusion (RRF)
    4. Grounded LLM Generation with Citation Verification ([SOURCE X])
    5. Query Audit Logging
    """
    start_time = time.perf_counter()
    query_id = f"qry_{uuid.uuid4().hex[:8]}"

    valid_modes = {"vector", "graph", "hybrid"}
    mode = request.retrieval_mode.lower()
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid retrieval mode '{request.retrieval_mode}'. Allowed: {', '.join(valid_modes)}"
        )

    # 1. Execute Hybrid Retrieval & RRF Fusion
    retrieval_result = execute_hybrid_retrieval(
        question=request.question,
        retrieval_mode=mode,
        top_k=request.top_k,
        graph_depth=request.graph_depth
    )

    # 2. Grounded Generation & Citation Verification
    generation_result = generate_grounded_answer(
        retrieval_result=retrieval_result,
        use_llm=True
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Convert verified citations to response schema
    sources = [
        SourceCitation(
            citation_id=c.citation_id,
            document_id=c.document_id,
            filename=c.filename,
            page=c.page,
            chunk_id=c.chunk_id,
            excerpt=c.excerpt
        )
        for c in generation_result.verified_citations
    ]

    # Format graph subgraph for UI canvas rendering
    visual_subgraph = {
        "nodes": [
            {
                "id": n.get("id"),
                "label": n.get("name") or n.get("id"),
                "type": n.get("type", "CONCEPT"),
                "degree": n.get("degree", 0),
                "description": n.get("description", "")
            }
            for n in retrieval_result.subgraph.get("nodes", [])
        ],
        "edges": [
            {
                "id": e.get("id") or f"e_{idx}",
                "source": e.get("source"),
                "target": e.get("target"),
                "type": e.get("type", "RELATED_TO"),
                "evidence": e.get("evidence", ""),
                "confidence": e.get("confidence", 1.0)
            }
            for idx, e in enumerate(retrieval_result.subgraph.get("edges", []))
        ]
    }

    # Persist query log for analytics and evaluation
    try:
        log_entry = QueryLogModel(
            id=query_id,
            query_text=request.question,
            retrieval_mode=mode,
            top_k=request.top_k,
            graph_depth=request.graph_depth,
            generated_answer=generation_result.answer,
            retrieved_chunks=[c.to_dict() for c in retrieval_result.top_chunks],
            retrieved_entities=retrieval_result.linked_entities,
            citations=[s.model_dump() for s in sources],
            latency_ms=latency_ms
        )
        db.add(log_entry)
        db.commit()
    except Exception:
        db.rollback()

    return QueryResponse(
        query_id=query_id,
        question=request.question,
        answer=generation_result.answer,
        retrieval_mode=mode,
        sources=sources,
        entities=retrieval_result.linked_entities,
        graph_subgraph=visual_subgraph,
        latency_ms=latency_ms,
        model_used=generation_result.model_used,
        is_fully_grounded=generation_result.is_fully_grounded
    )
