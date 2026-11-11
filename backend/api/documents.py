import os
import shutil
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.config import settings
from backend.database import get_db, DocumentModel, ChunkModel
from backend.services.vector_store import vector_store
from backend.services.graph_store import graph_store
from backend.services.pipeline import run_ingestion_pipeline

router = APIRouter(prefix="/documents", tags=["Document Management"])


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    error_message: Optional[str] = None
    chunk_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    chunk_count: int
    entity_count: int
    relationship_count: int
    error_message: Optional[str] = None


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Upload a document (.pdf, .docx, .txt) for ingestion into the GraphRAG pipeline.
    Initializes document record in DB and stores the file in UPLOAD_DIR.
    """
    settings.ensure_directories()
    filename = file.filename or "unknown_file"
    ext = os.path.splitext(filename)[1].lower().lstrip(".")

    allowed_extensions = {"pdf", "docx", "txt"}
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(allowed_extensions)}"
        )

    doc_id = str(uuid.uuid4())
    save_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{filename}")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    doc = DocumentModel(
        id=doc_id,
        filename=filename,
        file_type=ext,
        file_size=file_size,
        file_path=save_path,
        status="PENDING",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Launch background ingestion pipeline
    background_tasks.add_task(run_ingestion_pipeline, doc_id)

    return doc.to_dict()


@router.get("", response_model=List[DocumentResponse])
def list_documents(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """List all ingested documents with metadata and processing status."""
    docs = db.query(DocumentModel).order_by(DocumentModel.created_at.desc()).limit(limit).all()
    return [d.to_dict() for d in docs]


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    """Retrieve real-time processing status and extracted statistics for a document."""
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": doc.id,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
        "entity_count": doc.entity_count,
        "relationship_count": doc.relationship_count,
        "error_message": doc.error_message,
    }


@router.get("/{document_id}/chunks")
def get_document_chunks(document_id: str, db: Session = Depends(get_db)):
    """List all chunks associated with a document."""
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = db.query(ChunkModel).filter(ChunkModel.document_id == document_id).order_by(ChunkModel.chunk_index.asc()).all()
    return [c.to_dict() for c in chunks]


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """
    Cascading deletion of a document across:
    1. Relational database (document & chunks)
    2. ChromaDB dense vector collection
    3. Neo4j / Embedded Knowledge Graph nodes & edges
    4. Local disk storage
    """
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 1. Delete vectors from ChromaDB
    deleted_vectors = vector_store.delete_by_document_id(document_id)

    # 2. Delete relationships & orphan entities from Graph Store
    graph_cleanup = graph_store.delete_by_document_id(document_id)

    # 3. Delete file from local filesystem
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception:
            pass

    # 4. Delete from Relational DB (cascades to chunks)
    db.delete(doc)
    db.commit()

    return {
        "message": f"Document '{doc.filename}' ({document_id}) deleted successfully",
        "deleted_vectors": deleted_vectors,
        "deleted_graph_elements": graph_cleanup
    }

