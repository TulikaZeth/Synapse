"""
Document Ingestion Pipeline Orchestrator

Coordinates the complete flow:
  1. Parse document (PDF / DOCX / TXT) → structured pages
  2. Chunk text with semantic sliding-window → overlapping chunks
  3. Extract entities & relationships (Gemini / rule-based) → knowledge triples
  4. Persist chunks to relational DB + ChromaDB vectors
  5. Persist entities & relationships to knowledge graph store
  6. Update document status and counters
"""

import logging
import traceback
from typing import Optional

from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import SessionLocal, DocumentModel, ChunkModel
from backend.services.parser import parse_document
from backend.services.chunker import chunk_document
from backend.services.extractor import extract_from_chunks
from backend.services.vector_store import vector_store
from backend.services.graph_store import graph_store

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(document_id: str, use_llm: bool = True) -> None:
    """
    Execute the full ingestion pipeline for a document.

    This function is designed to be called as a background task from the
    upload endpoint. It manages its own database session.

    Args:
        document_id: UUID of the document record already created in the DB.
        use_llm: Whether to attempt LLM-based extraction (falls back to rules).
    """
    db: Session = SessionLocal()
    try:
        doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not doc:
            logger.error("Ingestion pipeline: Document %s not found.", document_id)
            return

        # Mark as PROCESSING
        doc.status = "PROCESSING"
        db.commit()
        logger.info("Starting ingestion for '%s' (%s)", doc.filename, document_id)

        # ── Step 1: Parse ──────────────────────────────────────────────
        parsed = parse_document(doc.file_path)
        if parsed.error:
            doc.status = "FAILED"
            doc.error_message = f"Parsing failed: {parsed.error}"
            db.commit()
            logger.error("Parsing failed for '%s': %s", doc.filename, parsed.error)
            return

        if not parsed.full_text or len(parsed.full_text.strip()) < 10:
            doc.status = "FAILED"
            doc.error_message = "Document is empty or contains no extractable text."
            db.commit()
            logger.error("Empty document: '%s'", doc.filename)
            return

        logger.info("Parsed '%s': %d pages, %d chars.",
                     doc.filename, parsed.total_pages, len(parsed.full_text))

        # ── Step 2: Chunk ──────────────────────────────────────────────
        chunks = chunk_document(
            parsed_doc=parsed,
            document_id=document_id,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        if not chunks:
            doc.status = "FAILED"
            doc.error_message = "Chunking produced zero chunks."
            db.commit()
            logger.error("Zero chunks for '%s'", doc.filename)
            return

        logger.info("Chunked '%s': %d chunks.", doc.filename, len(chunks))

        # ── Step 3: Persist chunks to relational DB ────────────────────
        chunk_models = []
        for c in chunks:
            cm = ChunkModel(
                id=c.id,
                document_id=document_id,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                content=c.content,
                token_count=c.token_count,
                embedding_id=c.id,  # Will match ChromaDB ID
            )
            chunk_models.append(cm)
            db.add(cm)

        db.flush()  # Flush so chunks are queryable

        # ── Step 4: Index chunks in ChromaDB ───────────────────────────
        chroma_payloads = [
            {
                "id": c.id,
                "content": c.content,
                "document_id": document_id,
                "filename": doc.filename,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
            }
            for c in chunks
        ]
        indexed_count = vector_store.add_chunks(chroma_payloads)
        logger.info("Indexed %d chunks in ChromaDB for '%s'.", indexed_count, doc.filename)

        # ── Step 5: Extract entities & relationships ───────────────────
        chunk_dicts = [c.to_dict() for c in chunks]
        entities, relationships = extract_from_chunks(
            chunks=chunk_dicts,
            document_id=document_id,
            use_llm=use_llm,
        )
        logger.info(
            "Extracted %d entities, %d relationships from '%s'.",
            len(entities), len(relationships), doc.filename
        )

        # ── Step 6: Persist to knowledge graph ─────────────────────────
        for entity_dict in entities:
            graph_store.upsert_entity(entity_dict)

        for rel_dict in relationships:
            graph_store.upsert_relationship(rel_dict)

        # ── Step 7: Update document counters and mark INDEXED ──────────
        doc.chunk_count = len(chunks)
        doc.entity_count = len(entities)
        doc.relationship_count = len(relationships)
        doc.status = "INDEXED"
        doc.error_message = None
        db.commit()

        logger.info(
            "Ingestion complete for '%s': %d chunks, %d entities, %d relationships.",
            doc.filename, doc.chunk_count, doc.entity_count, doc.relationship_count
        )

    except Exception as e:
        logger.error("Ingestion pipeline error for %s: %s\n%s",
                      document_id, e, traceback.format_exc())
        try:
            doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
            if doc:
                doc.status = "FAILED"
                doc.error_message = f"Pipeline error: {str(e)[:500]}"
                db.commit()
        except Exception:
            pass
    finally:
        db.close()

