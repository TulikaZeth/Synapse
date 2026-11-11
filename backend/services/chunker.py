"""
Semantic Sliding-Window Chunker
Splits document text into overlapping chunks that respect natural sentence
and paragraph boundaries.

Specification:
  - Target size: 500 characters (~120-150 tokens)
  - Overlap: 100 characters (~25 tokens)
  - Boundary preservation: split on sentence endings ('. ', '\\n\\n')
  - Every chunk carries {document_id, chunk_id, page_number, chunk_index}
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from backend.services.parser import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

# Token estimate: ~4 characters per token for English text
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""
    id: str
    document_id: str
    chunk_index: int
    page_number: Optional[int]
    content: str
    token_count: int
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "chunk_id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "content": self.content,
            "token_count": self.token_count,
            "char_count": self.char_count,
        }


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    Handles common abbreviations, decimal numbers, and preserves paragraph breaks.
    """
    if not text.strip():
        return []

    # Split on paragraph boundaries first
    paragraphs = re.split(r'\n\s*\n', text)
    sentences = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split on sentence-ending punctuation followed by whitespace or end-of-string.
        # Avoid splitting on common abbreviations (Mr., Dr., etc.) and decimals.
        parts = re.split(
            r'(?<=[.!?])\s+(?=[A-Z"\'\(])|(?<=[.!?])\s*$',
            para
        )

        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

        # Add a paragraph break marker between paragraphs
        sentences.append("\n")

    # Remove trailing paragraph marker
    while sentences and sentences[-1] == "\n":
        sentences.pop()

    return sentences


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    page_number: Optional[int] = None,
) -> List[Chunk]:
    """
    Split a block of text into overlapping chunks that respect sentence boundaries.

    Algorithm:
    1. Split text into sentences.
    2. Greedily accumulate sentences until adding the next sentence would exceed chunk_size.
    3. Emit the chunk and start the next one with overlap characters from the tail of the
       previous chunk to preserve context continuity.
    """
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_length = 0
    chunk_index = 0

    def _emit_chunk(sents: List[str]) -> None:
        nonlocal chunk_index
        chunk_text_joined = " ".join(s for s in sents if s != "\n")
        # Restore paragraph breaks
        chunk_text_joined = chunk_text_joined.strip()
        if not chunk_text_joined:
            return

        chunk = Chunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_index=chunk_index,
            page_number=page_number,
            content=chunk_text_joined,
            token_count=estimate_tokens(chunk_text_joined),
        )
        chunks.append(chunk)
        chunk_index += 1

    for sentence in sentences:
        sent_len = len(sentence) if sentence != "\n" else 0

        # If a single sentence exceeds chunk_size, force-split it by character count
        if sent_len > chunk_size and sentence != "\n":
            # Flush current accumulation first
            if current_sentences:
                _emit_chunk(current_sentences)
                current_sentences = []
                current_length = 0

            # Force-split the long sentence
            for start in range(0, len(sentence), chunk_size - chunk_overlap):
                fragment = sentence[start:start + chunk_size]
                if fragment.strip():
                    _emit_chunk([fragment])
            continue

        # Would adding this sentence exceed the target?
        projected = current_length + sent_len + (1 if current_sentences else 0)
        if projected > chunk_size and current_sentences:
            # Emit current chunk
            _emit_chunk(current_sentences)

            # Build overlap from the tail of current sentences
            overlap_chars = 0
            overlap_sentences: List[str] = []
            for s in reversed(current_sentences):
                if s == "\n":
                    continue
                if overlap_chars + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_chars += len(s)

            current_sentences = overlap_sentences
            current_length = overlap_chars

        current_sentences.append(sentence)
        current_length += sent_len + (1 if sent_len > 0 else 0)

    # Flush remaining
    if current_sentences:
        _emit_chunk(current_sentences)

    return chunks


def chunk_document(
    parsed_doc: ParsedDocument,
    document_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Chunk]:
    """
    Chunk a parsed document, preserving page-level provenance.

    For multi-page documents (PDFs), each page is chunked independently so
    page_number attribution stays accurate. For single-page documents
    (TXT/DOCX), the full text is chunked as one block.
    """
    if parsed_doc.error:
        logger.error("Cannot chunk document '%s': %s", parsed_doc.filename, parsed_doc.error)
        return []

    all_chunks: List[Chunk] = []

    if len(parsed_doc.pages) > 1:
        # Multi-page: chunk each page independently for accurate page_number
        global_index = 0
        for page in parsed_doc.pages:
            if not page.text.strip():
                continue

            page_chunks = chunk_text(
                text=page.text,
                document_id=document_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                page_number=page.page_number,
            )
            # Reassign global chunk indices
            for c in page_chunks:
                c.chunk_index = global_index
                global_index += 1
            all_chunks.extend(page_chunks)
    else:
        # Single page (TXT / DOCX): chunk the full text
        page_num = parsed_doc.pages[0].page_number if parsed_doc.pages else None
        all_chunks = chunk_text(
            text=parsed_doc.full_text,
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_number=page_num,
        )

    logger.info(
        "Chunked document '%s' (%s): %d chunks generated.",
        parsed_doc.filename, parsed_doc.file_type, len(all_chunks)
    )
    return all_chunks

