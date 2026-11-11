import hashlib
import logging
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from backend.config import settings

logger = logging.getLogger(__name__)


class FastDeterministicEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Self-contained 384-dimensional normalized embedding function.
    Uses n-gram hashing and term weighting with L2 normalization.
    Guarantees instant zero-dependency execution without network downloads,
    while providing true cosine similarity semantics.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "fast_deterministic_embedding"

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "FastDeterministicEmbeddingFunction":
        return FastDeterministicEmbeddingFunction(dim=config.get("dim", 384))

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for text in input:
            vec = [0.0] * self.dim
            words = text.lower().replace(",", " ").replace(".", " ").replace(";", " ").split()
            if not words:
                embeddings.append(vec)
                continue

            for i, word in enumerate(words):
                # Unigram term feature
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[h] += 1.0
                # Bigram phrase feature
                if i < len(words) - 1:
                    bigram = f"{word}_{words[i+1]}"
                    bh = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16) % self.dim
                    vec[bh] += 1.5

            # L2 normalization for cosine similarity
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [round(x / norm, 6) for x in vec]
            embeddings.append(vec)
        return embeddings


class VectorStoreService:
    """ChromaDB Vector Store wrapper for GraphRAG document chunks."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_fn: Optional[EmbeddingFunction] = None
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_fn = embedding_fn or FastDeterministicEmbeddingFunction(dim=384)
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self._initialized = False

    def initialize(self):
        """Initialize ChromaDB PersistentClient and collection."""
        try:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            self._initialized = True
            logger.info("ChromaDB initialized at %s with collection '%s'", self.persist_dir, self.collection_name)
        except Exception as e:
            logger.error("Failed to initialize ChromaDB: %s", e)
            self._initialized = False
            raise e

    def _ensure_initialized(self):
        if not self._initialized or self.collection is None:
            self.initialize()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add document chunks to ChromaDB.
        Each chunk dict must contain:
          - id (str): Chunk UUID
          - content (str): Text content
          - document_id (str): Parent document UUID
          - filename (str): Source filename
          - page_number (int, optional): Page number
          - chunk_index (int): Positional index
          - token_count (int): Token count
        """
        self._ensure_initialized()
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = str(chunk.get("id") or chunk.get("chunk_id"))
            content = chunk.get("content", "")
            doc_id = str(chunk.get("document_id", ""))
            filename = str(chunk.get("filename", "unknown"))
            page_num = chunk.get("page_number")
            page_num_val = int(page_num) if page_num is not None else -1
            chunk_index = int(chunk.get("chunk_index", 0))
            token_count = int(chunk.get("token_count", 0))

            ids.append(chunk_id)
            documents.append(content)
            metadatas.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "filename": filename,
                "page_number": page_num_val,
                "chunk_index": chunk_index,
                "token_count": token_count,
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(ids)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute dense vector search against chunks.
        Returns list of matching chunks with similarity score (0.0 - 1.0).
        """
        self._ensure_initialized()
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        where_filter = {"document_id": document_id} if document_id else None

        results = self.collection.query(
            query_texts=[query_text],
            n_results=actual_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

            for i, chunk_id in enumerate(ids):
                # Cosine distance to similarity: similarity = 1 - distance
                dist = distances[i] if i < len(distances) else 0.0
                similarity = max(0.0, min(1.0, 1.0 - dist))
                meta = metas[i] if i < len(metas) else {}
                page_val = meta.get("page_number")
                if page_val == -1:
                    page_val = None

                formatted_results.append({
                    "chunk_id": chunk_id,
                    "document_id": meta.get("document_id"),
                    "filename": meta.get("filename"),
                    "page_number": page_val,
                    "chunk_index": meta.get("chunk_index"),
                    "token_count": meta.get("token_count"),
                    "content": docs[i],
                    "score": round(similarity, 4),
                    "distance": round(dist, 4)
                })

        return formatted_results

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to a document."""
        self._ensure_initialized()
        try:
            existing = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception as e:
            logger.error("Error deleting chunks for document %s: %s", document_id, e)
            return 0

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by its ID."""
        self._ensure_initialized()
        res = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if res and res.get("ids") and len(res["ids"]) > 0:
            meta = res["metadatas"][0]
            page_val = meta.get("page_number")
            if page_val == -1:
                page_val = None
            return {
                "chunk_id": chunk_id,
                "content": res["documents"][0],
                "document_id": meta.get("document_id"),
                "filename": meta.get("filename"),
                "page_number": page_val,
                "chunk_index": meta.get("chunk_index"),
                "token_count": meta.get("token_count")
            }
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        try:
            self._ensure_initialized()
            count = self.collection.count()
            return {
                "status": "healthy",
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir,
                "chunk_count": count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "chunk_count": 0
            }


# Singleton instance
vector_store = VectorStoreService()
