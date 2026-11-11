"""
Hybrid Retrieval & Multi-Hop Reasoning Engine

Implements:
1. Query Entity Linker: Detects entities mentioned in the natural language query.
2. Parallel Vector Retrieval: Dense cosine similarity search via ChromaDB.
3. Graph Multi-Hop Traversal: Discovers N-hop subgraphs and intermediate fact chains.
4. Reciprocal Rank Fusion (RRF):
     RRF_Score(d) = sum_{m in {vector, graph}} (w_m / (k_0 + r_m(d)))
   where k_0 = 60, default weights w_vector=0.5, w_graph=0.5.
5. Evidence Collector: Merges unique chunks with page-level provenance.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

from backend.config import settings
from backend.services.vector_store import vector_store
from backend.services.graph_store import graph_store, normalize_entity_id
from backend.database import SessionLocal, ChunkModel, DocumentModel

logger = logging.getLogger(__name__)

RRF_K0 = 60  # Standard RRF smoothing constant


@dataclass
class RetrievedEvidence:
    """A retrieved chunk enriched with fusion scores and provenance."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int]
    chunk_index: int
    content: str
    token_count: int
    vector_rank: Optional[int] = None
    graph_rank: Optional[int] = None
    vector_score: float = 0.0
    rrf_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "token_count": self.token_count,
            "vector_rank": self.vector_rank,
            "graph_rank": self.graph_rank,
            "vector_score": self.vector_score,
            "rrf_score": self.rrf_score,
        }


@dataclass
class FactChain:
    """An intermediate relational path discovery chain."""
    source_entity: str
    target_entity: str
    hops: List[str]
    evidence_text: str = ""

    def formatted_chain(self) -> str:
        return " -> ".join(self.hops)


@dataclass
class HybridRetrievalResult:
    """Full payload resulting from hybrid retrieval."""
    query: str
    retrieval_mode: str
    top_chunks: List[RetrievedEvidence] = field(default_factory=list)
    linked_entities: List[str] = field(default_factory=list)
    fact_chains: List[FactChain] = field(default_factory=list)
    subgraph: Dict[str, Any] = field(default_factory=lambda: {"nodes": [], "edges": []})


# =============================================================================
# 1. Query Entity Linker
# =============================================================================

def link_query_entities(query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Match entities from the knowledge graph that appear in the user query.
    Matches against entity canonical IDs, display names, and aliases.
    """
    all_entities = graph_store.get_entities(limit=200)
    if not all_entities:
        return []

    query_lower = query_text.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))

    matched = []
    for ent in all_entities:
        name_lower = ent.get("name", "").lower()
        id_norm = ent.get("id", "").lower().replace("_", " ")
        aliases = [a.lower() for a in ent.get("aliases", [])]

        # Check full string containment or multi-token overlap
        is_match = False
        if name_lower and name_lower in query_lower:
            is_match = True
        elif id_norm and id_norm in query_lower:
            is_match = True
        elif any(alias in query_lower for alias in aliases):
            is_match = True
        else:
            # Check single distinctive token (min length 4 to avoid stop words)
            name_words = {w for w in re.findall(r'\b\w+\b', name_lower) if len(w) >= 4}
            if name_words and name_words.issubset(query_tokens):
                is_match = True

        if is_match:
            matched.append(ent)
            if len(matched) >= limit:
                break

    return matched


# =============================================================================
# 2. Reciprocal Rank Fusion (RRF)
# =============================================================================

def compute_rrf_score(
    vector_rank: Optional[int],
    graph_rank: Optional[int],
    w_vector: float = 0.5,
    w_graph: float = 0.5,
    k_0: int = RRF_K0
) -> float:
    """
    Compute RRF score for an item appearing at given 1-based ranks:
      RRF = (w_vector / (k_0 + vector_rank)) + (w_graph / (k_0 + graph_rank))
    """
    score = 0.0
    if vector_rank is not None and vector_rank > 0:
        score += w_vector / (k_0 + vector_rank)
    if graph_rank is not None and graph_rank > 0:
        score += w_graph / (k_0 + graph_rank)
    return round(score, 6)


def fuse_ranked_lists(
    vector_chunks: List[Dict[str, Any]],
    graph_chunk_ids: List[str],
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    k_0: int = RRF_K0
) -> List[RetrievedEvidence]:
    """
    Combine vector search results and graph-derived chunks via Reciprocal Rank Fusion.
    """
    if retrieval_mode == "vector":
        w_vector, w_graph = 1.0, 0.0
    elif retrieval_mode == "graph":
        w_vector, w_graph = 0.0, 1.0
    else:  # hybrid
        w_vector, w_graph = 0.5, 0.5

    # Map of chunk_id -> dict with metadata & ranks
    chunk_map: Dict[str, Dict[str, Any]] = {}

    # 1. Process vector results
    for rank_idx, v_chunk in enumerate(vector_chunks, start=1):
        cid = v_chunk["chunk_id"]
        chunk_map[cid] = {
            "chunk_id": cid,
            "document_id": v_chunk.get("document_id", ""),
            "filename": v_chunk.get("filename", "unknown"),
            "page_number": v_chunk.get("page_number"),
            "chunk_index": v_chunk.get("chunk_index", 0),
            "content": v_chunk.get("content", ""),
            "token_count": v_chunk.get("token_count", 0),
            "vector_score": v_chunk.get("score", 0.0),
            "vector_rank": rank_idx,
            "graph_rank": None,
        }

    # 2. Process graph chunk IDs (ranked by graph traversal order / degree)
    missing_cids = []
    for rank_idx, cid in enumerate(graph_chunk_ids, start=1):
        if cid in chunk_map:
            chunk_map[cid]["graph_rank"] = rank_idx
        else:
            missing_cids.append((cid, rank_idx))

    # Hydrate any graph chunk IDs not already in the vector results
    if missing_cids:
        db = SessionLocal()
        try:
            ids_to_fetch = [c[0] for c in missing_cids]
            chunk_records = db.query(ChunkModel).filter(ChunkModel.id.in_(ids_to_fetch)).all()
            doc_ids = {c.document_id for c in chunk_records}
            docs = {d.id: d.filename for d in db.query(DocumentModel).filter(DocumentModel.id.in_(doc_ids)).all()}

            rec_map = {c.id: c for c in chunk_records}
            for cid, rank_idx in missing_cids:
                rec = rec_map.get(cid)
                if rec:
                    fname = docs.get(rec.document_id, "unknown")
                    chunk_map[cid] = {
                        "chunk_id": cid,
                        "document_id": rec.document_id,
                        "filename": fname,
                        "page_number": rec.page_number,
                        "chunk_index": rec.chunk_index,
                        "content": rec.content,
                        "token_count": rec.token_count,
                        "vector_score": 0.0,
                        "vector_rank": None,
                        "graph_rank": rank_idx,
                    }
        finally:
            db.close()

    # 3. Calculate RRF scores for all candidate chunks
    evidence_list: List[RetrievedEvidence] = []
    for cid, data in chunk_map.items():
        rrf = compute_rrf_score(
            vector_rank=data["vector_rank"],
            graph_rank=data["graph_rank"],
            w_vector=w_vector,
            w_graph=w_graph,
            k_0=k_0
        )
        evidence = RetrievedEvidence(
            chunk_id=cid,
            document_id=data["document_id"],
            filename=data["filename"],
            page_number=data["page_number"],
            chunk_index=data["chunk_index"],
            content=data["content"],
            token_count=data["token_count"],
            vector_rank=data["vector_rank"],
            graph_rank=data["graph_rank"],
            vector_score=data["vector_score"],
            rrf_score=rrf,
        )
        evidence_list.append(evidence)

    # 4. Sort descending by RRF score (tie-break on vector_score)
    evidence_list.sort(key=lambda e: (e.rrf_score, e.vector_score), reverse=True)

    return evidence_list[:top_k]


# =============================================================================
# 3. Multi-Hop Graph Traversal & Fact-Chain Discovery
# =============================================================================

def discover_graph_evidence(
    linked_entities: List[Dict[str, Any]],
    graph_depth: int = 2,
    max_entities: int = 5
) -> Tuple[List[str], List[FactChain], Dict[str, Any]]:
    """
    Traverse graph neighborhoods and multi-hop paths for linked entities.

    Returns:
        - List of supporting chunk IDs found on graph edges.
        - List of FactChain objects representing relational hops.
        - Merged subgraph for visualization (nodes + edges).
    """
    if not linked_entities:
        return [], [], {"nodes": [], "edges": []}

    supporting_chunk_ids: List[str] = []
    fact_chains: List[FactChain] = []
    seen_nodes: Dict[str, Dict[str, Any]] = {}
    seen_edges: List[Dict[str, Any]] = []

    focus_entities = linked_entities[:max_entities]

    # 1. Collect N-hop subgraphs around focal entities
    for ent in focus_entities:
        ent_id = ent.get("id") or ent.get("name") or str(ent)
        sub = graph_store.explore_subgraph(
            entity_name=ent_id,
            depth=graph_depth,
            limit=25
        )
        for n in sub.get("nodes", []):
            nid = n.get("id") or n.get("name")
            if nid:
                seen_nodes[nid] = n
        for e in sub.get("edges", []):
            seen_edges.append(e)
            cid = e.get("chunk_id")
            if cid and cid not in supporting_chunk_ids:
                supporting_chunk_ids.append(cid)

    # 2. Discover multi-hop paths between pairs of linked entities
    if len(focus_entities) >= 2:
        for i in range(len(focus_entities)):
            for j in range(i + 1, len(focus_entities)):
                src = focus_entities[i].get("id") or focus_entities[i].get("name")
                tgt = focus_entities[j].get("id") or focus_entities[j].get("name")
                if not (src and tgt):
                    continue
                paths = graph_store.find_multi_hop_paths(source_name=src, target_name=tgt, max_depth=graph_depth)
                for path in paths:
                    hops = []
                    evidence_texts = []
                    for seg in path:
                        from_name = seg["from_node"].get("name", seg["from_node"].get("id"))
                        to_name = seg["to_node"].get("name", seg["to_node"].get("id"))
                        edge_type = seg["edge"].get("type", "CONNECTED_TO")
                        hops.append(f"({from_name}) -[{edge_type}]-> ({to_name})")
                        ev = seg["edge"].get("evidence")
                        if ev:
                            evidence_texts.append(ev)
                        cid = seg["edge"].get("chunk_id")
                        if cid and cid not in supporting_chunk_ids:
                            supporting_chunk_ids.append(cid)

                    fact_chains.append(FactChain(
                        source_entity=src,
                        target_entity=tgt,
                        hops=hops,
                        evidence_text=" | ".join(evidence_texts)
                    ))

    merged_subgraph = {
        "nodes": list(seen_nodes.values()),
        "edges": seen_edges
    }

    return supporting_chunk_ids, fact_chains, merged_subgraph


# =============================================================================
# 4. Master Hybrid Retrieval Coordinator
# =============================================================================

def execute_hybrid_retrieval(
    question: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    graph_depth: int = 2
) -> HybridRetrievalResult:
    """
    Execute end-to-end hybrid retrieval:
      - Query Entity Linking
      - Dense Vector Search
      - Multi-hop Graph Traversal
      - Reciprocal Rank Fusion (RRF)
    """
    logger.info("Executing retrieval for query: '%s' (mode: %s, top_k: %d, depth: %d)",
                question, retrieval_mode, top_k, graph_depth)

    # 1. Link entities in query text
    linked_entities = link_query_entities(question)
    entity_names = [e.get("name") or e.get("id") for e in linked_entities]

    # 2. Vector retrieval
    vector_chunks: List[Dict[str, Any]] = []
    if retrieval_mode in ("vector", "hybrid"):
        # Fetch slightly more candidates for fusion
        fetch_k = top_k * 2 if retrieval_mode == "hybrid" else top_k
        vector_chunks = vector_store.query(query_text=question, top_k=fetch_k)

    # 3. Graph traversal & multi-hop discovery
    graph_chunk_ids: List[str] = []
    fact_chains: List[FactChain] = []
    subgraph: Dict[str, Any] = {"nodes": [], "edges": []}

    if retrieval_mode in ("graph", "hybrid") and linked_entities:
        graph_chunk_ids, fact_chains, subgraph = discover_graph_evidence(
            linked_entities=linked_entities,
            graph_depth=graph_depth
        )

    # 4. Reciprocal Rank Fusion
    top_chunks = fuse_ranked_lists(
        vector_chunks=vector_chunks,
        graph_chunk_ids=graph_chunk_ids,
        retrieval_mode=retrieval_mode,
        top_k=top_k
    )

    return HybridRetrievalResult(
        query=question,
        retrieval_mode=retrieval_mode,
        top_chunks=top_chunks,
        linked_entities=entity_names,
        fact_chains=fact_chains,
        subgraph=subgraph
    )
