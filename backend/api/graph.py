from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from backend.services.graph_store import graph_store

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


class SubgraphExploreRequest(BaseModel):
    entity_name: str = Field(..., description="Name or ID of starting entity")
    depth: int = Field(2, ge=1, le=5, description="Traversal depth")
    limit: int = Field(50, ge=1, le=200, description="Maximum nodes to return")


class EntityResponse(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = ""
    aliases: Optional[List[str]] = []
    document_ids: Optional[List[str]] = []
    degree: Optional[int] = 0


@router.get("/entities", response_model=List[Dict[str, Any]])
def get_entities(
    entity_type: Optional[str] = Query(None, description="Filter by type (PERSON, ORGANIZATION, etc.)"),
    limit: int = Query(50, ge=1, le=200, description="Max entities to return"),
    search: Optional[str] = Query(None, description="Search by name, ID, or description")
):
    """Retrieve entities from the knowledge graph with optional type and text filtering."""
    return graph_store.get_entities(entity_type=entity_type, limit=limit, search=search)


@router.get("/relationships", response_model=List[Dict[str, Any]])
def get_relationships(
    source_entity: Optional[str] = Query(None, description="Filter by source entity"),
    limit: int = Query(100, ge=1, le=500, description="Max relationships to return")
):
    """Retrieve relationships (edges) from the knowledge graph."""
    return graph_store.get_relationships(source_entity=source_entity, limit=limit)


@router.post("/explore", response_model=Dict[str, Any])
def explore_subgraph(request: SubgraphExploreRequest):
    """
    Explore the N-hop subgraph neighborhood around a focal entity node.
    Returns nodes and edges formatted for visual canvas rendering.
    """
    subgraph = graph_store.explore_subgraph(
        entity_name=request.entity_name,
        depth=request.depth,
        limit=request.limit
    )
    return subgraph


@router.get("/stats")
def get_graph_stats():
    """Retrieve status and metrics of the active graph store backend."""
    return graph_store.get_stats()

