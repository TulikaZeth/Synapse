import uuid
from datetime import datetime, timezone
from typing import Generator
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from backend.config import settings

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(512), nullable=False)
    status = Column(String(30), nullable=False, default="PENDING")
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    entity_count = Column(Integer, default=0)
    relationship_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "document_id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "file_path": self.file_path,
            "status": self.status,
            "error_message": self.error_message,
            "chunk_count": self.chunk_count,
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ChunkModel(Base):
    __tablename__ = "chunks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    embedding_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utcnow)

    # Relationships
    document = relationship("DocumentModel", back_populates="chunks")

    def to_dict(self):
        return {
            "id": self.id,
            "chunk_id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "content": self.content,
            "token_count": self.token_count,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class QueryLogModel(Base):
    __tablename__ = "query_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    retrieval_mode = Column(String(20), nullable=False)
    top_k = Column(Integer, nullable=False)
    graph_depth = Column(Integer, nullable=False)
    generated_answer = Column(Text, nullable=False)
    retrieved_chunks = Column(JSON, nullable=False, default=list)
    retrieved_entities = Column(JSON, nullable=False, default=list)
    citations = Column(JSON, nullable=False, default=list)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "query_id": self.id,
            "query_text": self.query_text,
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "graph_depth": self.graph_depth,
            "generated_answer": self.generated_answer,
            "retrieved_chunks": self.retrieved_chunks,
            "retrieved_entities": self.retrieved_entities,
            "citations": self.citations,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EvaluationRunModel(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    retrieval_mode = Column(String(20), nullable=False)
    recall_at_k = Column(Float, nullable=False)
    precision_at_k = Column(Float, nullable=False)
    mrr = Column(Float, nullable=False)
    faithfulness = Column(Float, nullable=False)
    answer_relevance = Column(Float, nullable=False)
    avg_latency_ms = Column(Float, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "retrieval_mode": self.retrieval_mode,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Database Engine and Session Factory
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables in the database if they do not exist."""
    settings.ensure_directories()
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for yielding database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

