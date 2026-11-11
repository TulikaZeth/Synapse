import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx

from backend.config import settings

logger = logging.getLogger(__name__)


def normalize_entity_id(name: str) -> str:
    """Normalize entity name to canonical uppercase ID."""
    if not name:
        return ""
    return name.strip().upper().replace(" ", "_")


class EmbeddedGraphStore:
    """
    In-memory NetworkX MultiDiGraph with JSON persistence fallback.
    Provides fast, standalone graph storage without requiring an external Neo4j daemon.
    """

    def __init__(self, persist_file: Optional[str] = None):
        self.persist_file = persist_file or settings.GRAPH_PERSIST_FILE
        self.graph = nx.MultiDiGraph()
        self._load()

    def _load(self):
        """Load graph state from persistent JSON file if present."""
        if os.path.exists(self.persist_file):
            try:
                with open(self.persist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data, edges="links", directed=True, multigraph=True)
                logger.info("Loaded embedded graph from %s (%d nodes, %d edges)",
                            self.persist_file, self.graph.number_of_nodes(), self.graph.number_of_edges())
            except Exception as e:
                logger.warning("Could not load embedded graph file %s: %s. Starting fresh.", self.persist_file, e)
                self.graph = nx.MultiDiGraph()
        else:
            self.graph = nx.MultiDiGraph()

    def persist(self):
        """Persist graph state to disk."""
        try:
            Path(self.persist_file).parent.mkdir(parents=True, exist_ok=True)
            data = nx.node_link_data(self.graph, edges="links")
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to persist embedded graph: %s", e)

    def upsert_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert an entity node."""
        raw_name = entity.get("name") or entity.get("id") or ""
        entity_id = entity.get("id") or normalize_entity_id(raw_name)
        display_name = entity.get("name") or raw_name
        entity_type = (entity.get("type") or "CONCEPT").upper()
        description = entity.get("description") or ""
        aliases = list(set(entity.get("aliases") or []))
        doc_ids = set(entity.get("document_ids") or [])
        if "document_id" in entity and entity["document_id"]:
            doc_ids.add(entity["document_id"])

        if self.graph.has_node(entity_id):
            node_data = self.graph.nodes[entity_id]
            # Merge document IDs
            existing_docs = set(node_data.get("document_ids", []))
            existing_docs.update(doc_ids)
            node_data["document_ids"] = list(existing_docs)

            # Merge aliases
            existing_aliases = set(node_data.get("aliases", []))
            existing_aliases.update(aliases)
            node_data["aliases"] = list(existing_aliases)

            # Update description if new one is longer
            if len(description) > len(node_data.get("description", "")):
                node_data["description"] = description
            if entity_type and entity_type != "CONCEPT":
                node_data["type"] = entity_type
            result = dict(node_data)
        else:
            node_data = {
                "id": entity_id,
                "name": display_name,
                "type": entity_type,
                "description": description,
                "aliases": list(aliases),
                "document_ids": list(doc_ids),
            }
            self.graph.add_node(entity_id, **node_data)
            result = dict(node_data)

        self.persist()
        return result

    def upsert_relationship(self, rel: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert a directed relationship between two entities."""
        source = rel.get("source") or ""
        target = rel.get("target") or ""
        source_id = normalize_entity_id(source)
        target_id = normalize_entity_id(target)
        rel_type = (rel.get("type") or "RELATED_TO").upper().replace(" ", "_")

        # Ensure source and target nodes exist
        if not self.graph.has_node(source_id):
            self.upsert_entity({"id": source_id, "name": source, "type": "CONCEPT"})
        if not self.graph.has_node(target_id):
            self.upsert_entity({"id": target_id, "name": target, "type": "CONCEPT"})

        rel_id = rel.get("id") or f"rel_{uuid.uuid4().hex[:12]}"
        edge_data = {
            "id": rel_id,
            "source": source_id,
            "target": target_id,
            "type": rel_type,
            "evidence": rel.get("evidence", ""),
            "confidence": float(rel.get("confidence", 1.0)),
            "document_id": rel.get("document_id", ""),
            "chunk_id": rel.get("chunk_id", ""),
        }

        # Check if identical edge already exists
        exists = False
        if self.graph.has_edge(source_id, target_id):
            for edge_key, data in self.graph.get_edge_data(source_id, target_id).items():
                if data.get("type") == rel_type and data.get("chunk_id") == edge_data["chunk_id"]:
                    data.update(edge_data)
                    exists = True
                    break

        if not exists:
            self.graph.add_edge(source_id, target_id, key=rel_id, **edge_data)

        self.persist()
        return edge_data

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        norm_id = normalize_entity_id(entity_id)
        if self.graph.has_node(norm_id):
            data = dict(self.graph.nodes[norm_id])
            data["degree"] = self.graph.degree(norm_id)
            return data
        return None

    def get_entities(
        self,
        entity_type: Optional[str] = None,
        limit: int = 50,
        search: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        results = []
        search_lower = search.lower() if search else None
        type_upper = entity_type.upper() if entity_type else None

        for node_id, data in self.graph.nodes(data=True):
            if type_upper and data.get("type") != type_upper:
                continue
            if search_lower:
                name_match = search_lower in data.get("name", "").lower()
                id_match = search_lower in node_id.lower()
                desc_match = search_lower in data.get("description", "").lower()
                if not (name_match or id_match or desc_match):
                    continue

            item = dict(data)
            item["id"] = data.get("id") or node_id
            item["name"] = data.get("name") or node_id
            item["degree"] = self.graph.degree(node_id)
            results.append(item)
            if len(results) >= limit:
                break

        # Sort by degree descending
        results.sort(key=lambda x: x.get("degree", 0), reverse=True)
        return results

    def get_relationships(
        self,
        source_entity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        results = []
        source_norm = normalize_entity_id(source_entity) if source_entity else None

        for u, v, data in self.graph.edges(data=True):
            if source_norm and u != source_norm:
                continue
            results.append(dict(data))
            if len(results) >= limit:
                break

        return results

    def explore_subgraph(
        self,
        entity_name: str,
        depth: int = 2,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Traverse N-hop neighborhood starting from entity_name.
        Returns nodes and edges ready for visualization.
        """
        start_node = normalize_entity_id(entity_name)
        if not self.graph.has_node(start_node):
            # Check if there is an alias or fuzzy name match
            matched = None
            for node_id, data in self.graph.nodes(data=True):
                if entity_name.lower() in data.get("name", "").lower() or entity_name.lower() in [a.lower() for a in data.get("aliases", [])]:
                    matched = node_id
                    break
            if matched:
                start_node = matched
            else:
                return {"nodes": [], "edges": []}

        visited_nodes: Set[str] = {start_node}
        current_layer: Set[str] = {start_node}

        for _ in range(depth):
            next_layer: Set[str] = set()
            for node in current_layer:
                # Successors and predecessors
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                for neighbor in neighbors:
                    if neighbor not in visited_nodes:
                        next_layer.add(neighbor)
                        visited_nodes.add(neighbor)
                        if len(visited_nodes) >= limit:
                            break
                if len(visited_nodes) >= limit:
                    break
            current_layer = next_layer
            if len(visited_nodes) >= limit or not current_layer:
                break

        # Collect node payloads
        nodes = []
        for nid in visited_nodes:
            ndata = dict(self.graph.nodes[nid])
            ndata["id"] = ndata.get("id") or nid
            ndata["name"] = ndata.get("name") or nid
            ndata["degree"] = self.graph.degree(nid)
            nodes.append(ndata)

        # Collect induced edges
        edges = []
        subgraph = self.graph.subgraph(visited_nodes)
        for u, v, k, edata in subgraph.edges(data=True, keys=True):
            edges.append(dict(edata))

        return {"nodes": nodes, "edges": edges}

    def find_multi_hop_paths(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = 2
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between source and target entities up to max_depth.
        Returns list of path segments with intermediate edges and evidence.
        """
        s_id = normalize_entity_id(source_name)
        t_id = normalize_entity_id(target_name)

        if not (self.graph.has_node(s_id) and self.graph.has_node(t_id)):
            return []

        try:
            # Simple paths up to max_depth
            undirected = self.graph.to_undirected()
            raw_paths = list(nx.all_simple_paths(undirected, source=s_id, target=t_id, cutoff=max_depth))
        except Exception:
            return []

        structured_paths = []
        for p in raw_paths[:10]:
            path_segments = []
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                edge_info = None
                if self.graph.has_edge(u, v):
                    edge_data_dict = self.graph.get_edge_data(u, v)
                    edge_info = list(edge_data_dict.values())[0]
                elif self.graph.has_edge(v, u):
                    edge_data_dict = self.graph.get_edge_data(v, u)
                    edge_info = list(edge_data_dict.values())[0]

                from_data = dict(self.graph.nodes[u])
                from_data["id"] = from_data.get("id") or u
                from_data["name"] = from_data.get("name") or u

                to_data = dict(self.graph.nodes[v])
                to_data["id"] = to_data.get("id") or v
                to_data["name"] = to_data.get("name") or v

                path_segments.append({
                    "from_node": from_data,
                    "to_node": to_data,
                    "edge": edge_info or {"type": "CONNECTED_TO", "evidence": ""}
                })
            structured_paths.append(path_segments)

        return structured_paths

    def delete_by_document_id(self, document_id: str) -> Dict[str, int]:
        """Delete relationships and update/clean nodes associated with document_id."""
        edges_to_remove = []
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if data.get("document_id") == document_id:
                edges_to_remove.append((u, v, k))

        for u, v, k in edges_to_remove:
            self.graph.remove_edge(u, v, key=k)

        # Update nodes document_ids list and clean up orphans
        nodes_to_remove = []
        for nid, data in self.graph.nodes(data=True):
            doc_ids = data.get("document_ids", [])
            if document_id in doc_ids:
                doc_ids.remove(document_id)
                data["document_ids"] = doc_ids
            # If node has no docs left and zero edges, remove it
            if len(doc_ids) == 0 and self.graph.degree(nid) == 0:
                nodes_to_remove.append(nid)

        for nid in nodes_to_remove:
            self.graph.remove_node(nid)

        self.persist()
        return {
            "deleted_edges": len(edges_to_remove),
            "deleted_nodes": len(nodes_to_remove)
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "backend": "embedded_networkx",
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "persist_file": self.persist_file
        }


class Neo4jGraphStore:
    """Neo4j driver implementation of GraphStore."""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def verify_connectivity(self):
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def upsert_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        raw_name = entity.get("name") or entity.get("id") or ""
        entity_id = entity.get("id") or normalize_entity_id(raw_name)
        display_name = entity.get("name") or raw_name
        entity_type = (entity.get("type") or "CONCEPT").upper()
        description = entity.get("description") or ""
        aliases = list(set(entity.get("aliases") or []))
        doc_ids = list(set(entity.get("document_ids") or []))
        if "document_id" in entity and entity["document_id"]:
            doc_ids.append(entity["document_id"])

        query = """
        MERGE (e:Entity {id: $id})
        ON CREATE SET e.name = $name, e.type = $type, e.description = $description,
                      e.aliases = $aliases, e.document_ids = $doc_ids
        ON MATCH SET e.name = $name,
                     e.document_ids = apoc.coll.toSet(e.document_ids + $doc_ids),
                     e.aliases = apoc.coll.toSet(e.aliases + $aliases)
        RETURN e.id as id, e.name as name, e.type as type, e.description as description,
               e.aliases as aliases, e.document_ids as document_ids
        """
        with self.driver.session() as session:
            result = session.run(query, id=entity_id, name=display_name, type=entity_type,
                                 description=description, aliases=aliases, doc_ids=doc_ids)
            record = result.single()
            return dict(record) if record else entity

    def upsert_relationship(self, rel: Dict[str, Any]) -> Dict[str, Any]:
        source = normalize_entity_id(rel.get("source", ""))
        target = normalize_entity_id(rel.get("target", ""))
        rel_type = (rel.get("type") or "RELATED_TO").upper().replace(" ", "_")
        rel_id = rel.get("id") or f"rel_{uuid.uuid4().hex[:12]}"

        query = f"""
        MERGE (s:Entity {{id: $source}})
        MERGE (t:Entity {{id: $target}})
        MERGE (s)-[r:{rel_type} {{id: $rel_id}}]->(t)
        SET r.evidence = $evidence, r.confidence = $confidence,
            r.document_id = $doc_id, r.chunk_id = $chunk_id
        RETURN r.id as id, s.id as source, t.id as target, type(r) as type,
               r.evidence as evidence, r.confidence as confidence
        """
        with self.driver.session() as session:
            result = session.run(query, source=source, target=target, rel_id=rel_id,
                                 evidence=rel.get("evidence", ""),
                                 confidence=float(rel.get("confidence", 1.0)),
                                 doc_id=rel.get("document_id", ""),
                                 chunk_id=rel.get("chunk_id", ""))
            record = result.single()
            return dict(record) if record else rel

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        norm_id = normalize_entity_id(entity_id)
        query = """
        MATCH (e:Entity {id: $id})
        OPTIONAL MATCH (e)-[r]-()
        RETURN e.id as id, e.name as name, e.type as type, e.description as description,
               e.aliases as aliases, e.document_ids as document_ids, count(r) as degree
        """
        with self.driver.session() as session:
            record = session.run(query, id=norm_id).single()
            return dict(record) if record and record["id"] else None

    def get_entities(self, entity_type: Optional[str] = None, limit: int = 50, search: Optional[str] = None) -> List[Dict[str, Any]]:
        where_clauses = []
        params: Dict[str, Any] = {"limit": limit}
        if entity_type:
            where_clauses.append("e.type = $type")
            params["type"] = entity_type.upper()
        if search:
            where_clauses.append("(toLower(e.name) CONTAINS toLower($search) OR toLower(e.id) CONTAINS toLower($search))")
            params["search"] = search

        where_stmt = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
        MATCH (e:Entity)
        {where_stmt}
        OPTIONAL MATCH (e)-[r]-()
        RETURN e.id as id, e.name as name, e.type as type, e.description as description,
               e.aliases as aliases, e.document_ids as document_ids, count(r) as degree
        ORDER BY degree DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            return [dict(rec) for rec in session.run(query, **params)]

    def get_relationships(self, source_entity: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        where_stmt = "WHERE s.id = $source" if source_entity else ""
        if source_entity:
            params["source"] = normalize_entity_id(source_entity)

        query = f"""
        MATCH (s:Entity)-[r]->(t:Entity)
        {where_stmt}
        RETURN r.id as id, s.id as source, t.id as target, type(r) as type,
               r.evidence as evidence, r.confidence as confidence,
               r.document_id as document_id, r.chunk_id as chunk_id
        LIMIT $limit
        """
        with self.driver.session() as session:
            return [dict(rec) for rec in session.run(query, **params)]

    def explore_subgraph(self, entity_name: str, depth: int = 2, limit: int = 50) -> Dict[str, Any]:
        norm_id = normalize_entity_id(entity_name)
        query = f"""
        MATCH path = (s:Entity {{id: $id}})-[*1..{depth}]-(t:Entity)
        WITH nodes(path) as ns, relationships(path) as rs
        UNWIND ns as n
        UNWIND rs as r
        WITH collect(DISTINCT n) as all_nodes, collect(DISTINCT r) as all_rels
        RETURN [node in all_nodes[..$limit] | {{id: node.id, name: node.name, type: node.type, description: node.description}}] as nodes,
               [rel in all_rels[..$limit] | {{id: rel.id, source: startNode(rel).id, target: endNode(rel).id, type: type(rel), evidence: rel.evidence}}] as edges
        """
        with self.driver.session() as session:
            record = session.run(query, id=norm_id, limit=limit).single()
            if record:
                return {"nodes": record["nodes"], "edges": record["edges"]}
        return {"nodes": [], "edges": []}

    def find_multi_hop_paths(self, source_name: str, target_name: str, max_depth: int = 2) -> List[List[Dict[str, Any]]]:
        s_id = normalize_entity_id(source_name)
        t_id = normalize_entity_id(target_name)
        query = f"""
        MATCH p = (s:Entity {{id: $source}})-[*1..{max_depth}]-(t:Entity {{id: $target}})
        RETURN [n in nodes(p) | {{id: n.id, name: n.name, type: n.type}}] as nodes,
               [r in relationships(p) | {{id: r.id, source: startNode(r).id, target: endNode(r).id, type: type(r), evidence: r.evidence}}] as edges
        LIMIT 10
        """
        results = []
        with self.driver.session() as session:
            for record in session.run(query, source=s_id, target=t_id):
                nodes = record["nodes"]
                edges = record["edges"]
                segments = []
                for i, edge in enumerate(edges):
                    segments.append({
                        "from_node": nodes[i],
                        "to_node": nodes[i + 1],
                        "edge": edge
                    })
                results.append(segments)
        return results

    def delete_by_document_id(self, document_id: str) -> Dict[str, int]:
        query = """
        MATCH ()-[r {document_id: $doc_id}]->()
        DELETE r
        """
        with self.driver.session() as session:
            summary = session.run(query, doc_id=document_id).consume()
            return {"deleted_edges": summary.counters.relationships_deleted, "deleted_nodes": 0}

    def get_stats(self) -> Dict[str, Any]:
        with self.driver.session() as session:
            node_count = session.run("MATCH (n:Entity) RETURN count(n) as c").single()["c"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
            return {
                "status": "healthy",
                "backend": "neo4j",
                "node_count": node_count,
                "edge_count": edge_count,
                "uri": self.uri
            }


class GraphStoreService:
    """
    Unified Graph Store Service.
    Automatically connects to Neo4j if configured and reachable;
    otherwise falls back gracefully to EmbeddedGraphStore.
    """

    def __init__(self):
        self._backend: Any = None
        self.backend_type: str = "embedded_networkx"
        self._initialize()

    def _initialize(self):
        if settings.NEO4J_ENABLED and settings.NEO4J_URI:
            try:
                neo_backend = Neo4jGraphStore(
                    uri=settings.NEO4J_URI,
                    user=settings.NEO4J_USER,
                    password=settings.NEO4J_PASSWORD
                )
                neo_backend.verify_connectivity()
                self._backend = neo_backend
                self.backend_type = "neo4j"
                logger.info("Connected successfully to Neo4j graph store at %s", settings.NEO4J_URI)
                return
            except Exception as e:
                logger.warning("Neo4j connection failed (%s). Falling back to EmbeddedGraphStore.", e)

        # Fallback to Embedded NetworkX Store
        self._backend = EmbeddedGraphStore(persist_file=settings.GRAPH_PERSIST_FILE)
        self.backend_type = "embedded_networkx"
        logger.info("Initialized EmbeddedGraphStore (NetworkX)")

    def upsert_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        return self._backend.upsert_entity(entity)

    def upsert_relationship(self, rel: Dict[str, Any]) -> Dict[str, Any]:
        return self._backend.upsert_relationship(rel)

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self._backend.get_entity(entity_id)

    def get_entities(self, entity_type: Optional[str] = None, limit: int = 50, search: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._backend.get_entities(entity_type=entity_type, limit=limit, search=search)

    def get_relationships(self, source_entity: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        return self._backend.get_relationships(source_entity=source_entity, limit=limit)

    def explore_subgraph(self, entity_name: str, depth: int = 2, limit: int = 50) -> Dict[str, Any]:
        return self._backend.explore_subgraph(entity_name=entity_name, depth=depth, limit=limit)

    def find_multi_hop_paths(self, source_name: str, target_name: str, max_depth: int = 2) -> List[List[Dict[str, Any]]]:
        return self._backend.find_multi_hop_paths(source_name=source_name, target_name=target_name, max_depth=max_depth)

    def delete_by_document_id(self, document_id: str) -> Dict[str, int]:
        return self._backend.delete_by_document_id(document_id)

    def get_stats(self) -> Dict[str, Any]:
        try:
            return self._backend.get_stats()
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": self.backend_type,
                "error": str(e),
                "node_count": 0,
                "edge_count": 0
            }


# Singleton instance
graph_store = GraphStoreService()

