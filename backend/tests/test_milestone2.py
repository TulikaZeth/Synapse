"""
Milestone 2 Test Suite
Tests for: Document Parser, Semantic Chunker, Entity/Relationship Extractor,
           and end-to-end Ingestion Pipeline.
"""

import os
import shutil
import tempfile
import uuid
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, DocumentModel, ChunkModel
from backend.services.parser import (
    parse_document, parse_txt, parse_docx, parse_pdf,
    clean_text, ParsedDocument, ParsedPage,
)
from backend.services.chunker import (
    chunk_text, chunk_document, split_into_sentences,
    estimate_tokens, Chunk,
)
from backend.services.extractor import (
    extract_from_text, extract_from_chunks,
    _extract_with_rules, _normalize_entity_type, _normalize_rel_type,
    ExtractionResult,
)
from backend.services.pipeline import run_ingestion_pipeline
from backend.services.vector_store import VectorStoreService
from backend.services.graph_store import EmbeddedGraphStore


@pytest.fixture(scope="session")
def test_temp_dir():
    temp_dir = tempfile.mkdtemp(prefix="graphrag_m2_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_txt_file(test_temp_dir):
    """Create a realistic multi-paragraph TXT document for testing."""
    content = """The Transformer Architecture and Modern NLP

The Transformer model was introduced by Google researchers in the paper
"Attention Is All You Need" published in 2017. The architecture relies
entirely on self-attention mechanisms, dispensing with recurrence and
convolutions entirely.

Self-attention, sometimes called intra-attention, is an attention mechanism
relating different positions of a single sequence in order to compute a
representation of the sequence. The Transformer uses multi-head attention,
which allows the model to jointly attend to information from different
representation subspaces at different positions.

The key innovation of the Transformer is the Scaled Dot-Product Attention
mechanism. Given queries Q, keys K, and values V, the attention function
computes a weighted sum of values based on the compatibility of queries
with keys. This computation is highly parallelizable across the sequence,
making it much faster than recurrent approaches.

OpenAI built GPT (Generative Pre-trained Transformer) on top of the
Transformer decoder architecture. Google developed BERT (Bidirectional
Encoder Representations from Transformers) using the encoder side.
Both models achieved state-of-the-art results on numerous NLP benchmarks.

Multi-head attention is a key component that enables the Transformer
to capture different types of relationships between tokens. Each head
learns to attend to different aspects of the input, and the outputs
are concatenated and linearly transformed.

Deep learning frameworks such as PyTorch and TensorFlow provide
implementations of Transformer layers. These frameworks are developed
by Meta and Google respectively. Stanford University and MIT have
contributed significantly to Transformer research.
"""
    path = os.path.join(test_temp_dir, "transformer_paper.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


@pytest.fixture
def sample_docx_file(test_temp_dir):
    """Create a simple DOCX file for testing."""
    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_paragraph("Knowledge Graphs and Retrieval-Augmented Generation")
    doc.add_paragraph(
        "Knowledge graphs represent structured information as entities and "
        "relationships. Neo4j is a popular graph database used to store and "
        "query these knowledge graphs efficiently."
    )
    doc.add_paragraph(
        "Retrieval-Augmented Generation (RAG) combines information retrieval "
        "with language model generation. ChromaDB and Pinecone are vector "
        "databases commonly used in RAG systems."
    )
    doc.add_paragraph(
        "GraphRAG extends traditional RAG by incorporating knowledge graph "
        "traversal into the retrieval pipeline, enabling multi-hop reasoning "
        "across document boundaries."
    )

    path = os.path.join(test_temp_dir, "knowledge_graphs.docx")
    doc.save(path)
    return path


# =============================================================================
# 1. Text Cleaning Tests
# =============================================================================

class TestTextCleaning:
    def test_strips_control_characters(self):
        dirty = "Hello\x00World\x07 test\x0b data"
        cleaned = clean_text(dirty)
        assert "\x00" not in cleaned
        assert "\x07" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned

    def test_normalizes_whitespace(self):
        text = "Too   many    spaces   here"
        cleaned = clean_text(text)
        assert "  " not in cleaned
        assert "Too many spaces here" == cleaned

    def test_preserves_paragraph_boundaries(self):
        text = "Paragraph one.\n\nParagraph two.\n\n\n\nParagraph three."
        cleaned = clean_text(text)
        assert "Paragraph one.\n\nParagraph two.\n\nParagraph three." == cleaned

    def test_strips_leading_trailing_whitespace(self):
        text = "   \n\n  Hello world.  \n\n  "
        cleaned = clean_text(text)
        assert cleaned == "Hello world."


# =============================================================================
# 2. Document Parser Tests
# =============================================================================

class TestDocumentParser:
    def test_parse_txt_file(self, sample_txt_file):
        result = parse_document(sample_txt_file)
        assert result.error is None
        assert result.file_type == "txt"
        assert result.total_pages >= 1
        assert len(result.full_text) > 100
        assert "Transformer" in result.full_text
        assert "self-attention" in result.full_text

    def test_parse_docx_file(self, sample_docx_file):
        result = parse_document(sample_docx_file)
        assert result.error is None
        assert result.file_type == "docx"
        assert len(result.full_text) > 50
        assert "Knowledge" in result.full_text
        assert "Neo4j" in result.full_text

    def test_parse_nonexistent_file(self):
        result = parse_document("/nonexistent/path/file.txt")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_parse_unsupported_extension(self, test_temp_dir):
        bad_path = os.path.join(test_temp_dir, "test.xyz")
        with open(bad_path, "w") as f:
            f.write("test content")
        result = parse_document(bad_path)
        assert result.error is not None
        assert "Unsupported" in result.error

    def test_parse_txt_encoding_fallback(self, test_temp_dir):
        """Test that latin-1 encoded files are handled correctly."""
        path = os.path.join(test_temp_dir, "latin1_doc.txt")
        with open(path, "wb") as f:
            f.write("Café résumé naïve".encode("latin-1"))
        result = parse_txt(path)
        assert result.error is None
        assert "Caf" in result.full_text


# =============================================================================
# 3. Semantic Chunker Tests
# =============================================================================

class TestSemanticChunker:
    def test_sentence_splitting(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_into_sentences(text)
        assert len(sentences) >= 3

    def test_token_estimation(self):
        text = "A" * 400  # 400 chars → ~100 tokens
        tokens = estimate_tokens(text)
        assert 95 <= tokens <= 105

    def test_basic_chunking(self):
        doc_id = str(uuid.uuid4())
        text = "Sentence one. " * 30 + "Sentence two. " * 30
        chunks = chunk_text(text, document_id=doc_id, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1

        # Verify metadata on each chunk
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == doc_id
            assert chunk.chunk_index == i
            assert len(chunk.content) > 0
            assert chunk.token_count > 0

    def test_chunk_size_respected(self):
        doc_id = str(uuid.uuid4())
        text = "This is a test sentence for chunking. " * 50
        chunks = chunk_text(text, document_id=doc_id, chunk_size=500, chunk_overlap=100)
        for chunk in chunks:
            # Allow some tolerance for sentence boundary preservation
            assert len(chunk.content) <= 600, \
                f"Chunk too long: {len(chunk.content)} chars"

    def test_overlap_preserves_context(self):
        doc_id = str(uuid.uuid4())
        text = ("Alpha bravo. " * 15 + "Charlie delta. " * 15 +
                "Echo foxtrot. " * 15)
        chunks = chunk_text(text, document_id=doc_id, chunk_size=200, chunk_overlap=80)
        if len(chunks) >= 2:
            # Some text from the tail of chunk 0 should appear at the
            # start of chunk 1 (overlap)
            tail_words = set(chunks[0].content.split()[-5:])
            head_words = set(chunks[1].content.split()[:10])
            overlap_words = tail_words & head_words
            assert len(overlap_words) > 0, "Expected overlap between consecutive chunks"

    def test_chunk_document_preserves_page_numbers(self, sample_txt_file):
        parsed = parse_document(sample_txt_file)
        doc_id = str(uuid.uuid4())
        chunks = chunk_document(parsed, document_id=doc_id, chunk_size=300, chunk_overlap=80)
        assert len(chunks) > 1
        # TXT file has single page
        for chunk in chunks:
            assert chunk.page_number == 1

    def test_empty_text_produces_no_chunks(self):
        doc_id = str(uuid.uuid4())
        chunks = chunk_text("", document_id=doc_id)
        assert chunks == []

    def test_long_sentence_force_split(self):
        doc_id = str(uuid.uuid4())
        long_sentence = "word " * 300  # ~1500 chars, one sentence
        chunks = chunk_text(long_sentence, document_id=doc_id, chunk_size=500, chunk_overlap=100)
        assert len(chunks) >= 2


# =============================================================================
# 4. Entity & Relationship Extractor Tests (Rule-based)
# =============================================================================

class TestExtractor:
    def test_normalize_entity_type(self):
        assert _normalize_entity_type("tech") == "TECHNOLOGY"
        assert _normalize_entity_type("ORG") == "ORGANIZATION"
        assert _normalize_entity_type("place") == "LOCATION"
        assert _normalize_entity_type("PERSON") == "PERSON"
        assert _normalize_entity_type("unknown") == "CONCEPT"

    def test_normalize_rel_type(self):
        assert _normalize_rel_type("uses") == "USES"
        assert _normalize_rel_type("developed by") == "DEVELOPED_BY"
        assert _normalize_rel_type("PART_OF") == "PART_OF"

    def test_rule_based_entity_extraction(self):
        text = (
            "The Transformer model was developed by Google. "
            "It uses self-attention mechanisms and multi-head attention. "
            "PyTorch and TensorFlow are popular deep learning frameworks. "
            "OpenAI created GPT based on the Transformer architecture."
        )
        result = _extract_with_rules(text)
        assert len(result.entities) > 0

        entity_names = {e.name.lower() for e in result.entities}
        # Should find at least some of these
        assert "transformer" in entity_names or "Transformer" in {e.name for e in result.entities}
        assert any("google" in n for n in entity_names)
        assert any("pytorch" in n or "tensorflow" in n for n in entity_names)

    def test_rule_based_entity_types(self):
        text = (
            "Google and OpenAI are organizations working on deep learning "
            "and transformer technology in California."
        )
        result = _extract_with_rules(text)
        type_map = {e.name.lower(): e.type for e in result.entities}
        # Google/OpenAI should be ORGANIZATION
        if "google" in type_map:
            assert type_map["google"] == "ORGANIZATION"
        if "openai" in type_map:
            assert type_map["openai"] == "ORGANIZATION"
        # California should be LOCATION
        if "california" in type_map:
            assert type_map["california"] == "LOCATION"

    def test_rule_based_relationship_extraction(self):
        text = "The Transformer uses self-attention for sequence modeling."
        result = _extract_with_rules(text)
        # Should find entities: Transformer, Self-Attention
        entity_names_lower = {e.name.lower() for e in result.entities}
        assert "transformer" in entity_names_lower or "self-attention" in entity_names_lower

    def test_extract_from_text_api(self):
        text = (
            "Google developed the Transformer architecture. "
            "BERT uses the encoder component of the Transformer. "
            "Stanford University published research on attention mechanisms."
        )
        # Force rule-based (no LLM)
        result = extract_from_text(text, use_llm=False)
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) > 0

    def test_extract_from_chunks_batch(self):
        chunks = [
            {
                "id": "chunk_001",
                "content": "Google created TensorFlow for deep learning research. "
                           "PyTorch was developed by Meta for neural network training."
            },
            {
                "id": "chunk_002",
                "content": "The Transformer architecture uses self-attention. "
                           "BERT is based on the Transformer encoder."
            },
        ]
        entities, relationships = extract_from_chunks(
            chunks=chunks, document_id="doc_test", use_llm=False
        )
        assert len(entities) > 0
        # Should have entities from both chunks, deduplicated
        entity_names = {e["name"].lower() for e in entities}
        assert any("google" in n or "tensorflow" in n or "transformer" in n for n in entity_names)

    def test_empty_text_extraction(self):
        result = extract_from_text("", use_llm=False)
        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    def test_short_text_extraction(self):
        result = extract_from_text("Hi", use_llm=False)
        assert len(result.entities) == 0


# =============================================================================
# 5. End-to-End Ingestion Pipeline Tests
# =============================================================================

class TestIngestionPipeline:
    @pytest.fixture
    def pipeline_db_session(self, test_temp_dir):
        """Create a database session for pipeline testing."""
        db_path = os.path.join(test_temp_dir, "pipeline_test.db")
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return engine, TestSession

    def test_full_pipeline_txt(self, sample_txt_file, test_temp_dir, pipeline_db_session, monkeypatch):
        """Test the complete ingestion pipeline with a TXT file."""
        engine, TestSession = pipeline_db_session

        # Create vector store and graph store for this test
        vs = VectorStoreService(
            persist_dir=os.path.join(test_temp_dir, "chroma_pipeline"),
            collection_name="pipeline_test"
        )
        vs.initialize()

        gs = EmbeddedGraphStore(
            persist_file=os.path.join(test_temp_dir, "pipeline_graph.json")
        )

        # Patch the pipeline module to use our test instances
        import backend.services.pipeline as pipeline_mod
        monkeypatch.setattr(pipeline_mod, "SessionLocal", TestSession)
        monkeypatch.setattr(pipeline_mod, "vector_store", vs)
        monkeypatch.setattr(pipeline_mod, "graph_store", gs)

        # Create document record
        db = TestSession()
        doc_id = str(uuid.uuid4())
        doc = DocumentModel(
            id=doc_id,
            filename="transformer_paper.txt",
            file_type="txt",
            file_size=os.path.getsize(sample_txt_file),
            file_path=sample_txt_file,
            status="PENDING",
        )
        db.add(doc)
        db.commit()
        db.close()

        # Run pipeline (synchronously for testing, rule-based extraction)
        run_ingestion_pipeline(doc_id, use_llm=False)

        # Verify results
        db = TestSession()
        doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        assert doc is not None
        assert doc.status == "INDEXED", f"Expected INDEXED, got {doc.status}: {doc.error_message}"
        assert doc.chunk_count > 0
        assert doc.entity_count > 0
        assert doc.relationship_count >= 0

        # Verify chunks were persisted
        chunks = db.query(ChunkModel).filter(ChunkModel.document_id == doc_id).all()
        assert len(chunks) == doc.chunk_count
        assert all(c.content for c in chunks)
        assert all(c.token_count > 0 for c in chunks)
        db.close()

        # Verify vectors were indexed in ChromaDB
        vs_stats = vs.get_stats()
        assert vs_stats["chunk_count"] == doc.chunk_count

        # Verify vector search works
        results = vs.query("self-attention Transformer", top_k=3)
        assert len(results) > 0
        assert results[0]["score"] > 0

        # Verify graph store has entities
        graph_stats = gs.get_stats()
        assert graph_stats["node_count"] > 0

    def test_pipeline_with_docx(self, sample_docx_file, test_temp_dir, pipeline_db_session, monkeypatch):
        """Test ingestion pipeline with a DOCX file."""
        engine, TestSession = pipeline_db_session

        vs = VectorStoreService(
            persist_dir=os.path.join(test_temp_dir, "chroma_docx"),
            collection_name="docx_test"
        )
        vs.initialize()

        gs = EmbeddedGraphStore(
            persist_file=os.path.join(test_temp_dir, "docx_graph.json")
        )

        import backend.services.pipeline as pipeline_mod
        monkeypatch.setattr(pipeline_mod, "SessionLocal", TestSession)
        monkeypatch.setattr(pipeline_mod, "vector_store", vs)
        monkeypatch.setattr(pipeline_mod, "graph_store", gs)

        db = TestSession()
        doc_id = str(uuid.uuid4())
        doc = DocumentModel(
            id=doc_id,
            filename="knowledge_graphs.docx",
            file_type="docx",
            file_size=os.path.getsize(sample_docx_file),
            file_path=sample_docx_file,
            status="PENDING",
        )
        db.add(doc)
        db.commit()
        db.close()

        run_ingestion_pipeline(doc_id, use_llm=False)

        db = TestSession()
        doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        assert doc.status == "INDEXED", f"Expected INDEXED, got {doc.status}: {doc.error_message}"
        assert doc.chunk_count > 0
        assert doc.entity_count > 0
        db.close()

    def test_pipeline_missing_file_fails_gracefully(self, test_temp_dir, pipeline_db_session, monkeypatch):
        """Pipeline should mark document as FAILED when file doesn't exist."""
        engine, TestSession = pipeline_db_session

        import backend.services.pipeline as pipeline_mod
        monkeypatch.setattr(pipeline_mod, "SessionLocal", TestSession)

        db = TestSession()
        doc_id = str(uuid.uuid4())
        doc = DocumentModel(
            id=doc_id,
            filename="ghost.txt",
            file_type="txt",
            file_size=100,
            file_path="/nonexistent/ghost.txt",
            status="PENDING",
        )
        db.add(doc)
        db.commit()
        db.close()

        run_ingestion_pipeline(doc_id, use_llm=False)

        db = TestSession()
        doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        assert doc.status == "FAILED"
        assert doc.error_message is not None
        assert "not found" in doc.error_message.lower() or "parsing failed" in doc.error_message.lower()
        db.close()

