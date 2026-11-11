"""
Multi-Format Document Parser
Supports PDF (page-by-page), DOCX (paragraph + table), and TXT extraction.
Applies text cleaning: strips non-printable control characters,
normalizes redundant whitespace, and preserves paragraph boundaries.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """Represents a single logical page of extracted text."""
    page_number: int  # 1-based
    text: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class ParsedDocument:
    """Result of parsing a document file into structured pages."""
    filename: str
    file_type: str
    total_pages: int = 0
    pages: List[ParsedPage] = field(default_factory=list)
    full_text: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        if self.pages and not self.full_text:
            self.full_text = "\n\n".join(p.text for p in self.pages if p.text.strip())
            self.total_pages = len(self.pages)


def clean_text(text: str) -> str:
    """
    Apply standard text cleaning:
    1. Strip non-printable control characters (keep newlines and tabs).
    2. Normalize redundant whitespace within lines.
    3. Collapse 3+ consecutive newlines into double newlines (paragraph boundaries).
    4. Strip leading/trailing whitespace.
    """
    # Remove non-printable chars except \n, \r, \t
    text = re.sub(r'[^\S \n\r\t]', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize multiple spaces within lines to single space
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Collapse 3+ newlines into paragraph boundaries
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def parse_pdf(file_path: str) -> ParsedDocument:
    """Extract text from PDF page-by-page using pypdf, with pdfplumber fallback."""
    filename = os.path.basename(file_path)
    pages: List[ParsedPage] = []

    try:
        import pypdf

        reader = pypdf.PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = clean_text(raw_text)
            pages.append(ParsedPage(page_number=i + 1, text=cleaned))

        # If pypdf yielded mostly empty pages, try pdfplumber
        total_chars = sum(p.char_count for p in pages)
        if total_chars < 50 and len(pages) > 0:
            logger.info("pypdf extracted minimal text from %s, trying pdfplumber.", filename)
            pages = _parse_pdf_pdfplumber(file_path)

    except ImportError:
        logger.warning("pypdf not installed, trying pdfplumber for %s", filename)
        pages = _parse_pdf_pdfplumber(file_path)
    except Exception as e:
        logger.error("pypdf failed for %s: %s, trying pdfplumber.", filename, e)
        try:
            pages = _parse_pdf_pdfplumber(file_path)
        except Exception as e2:
            return ParsedDocument(filename=filename, file_type="pdf", error=str(e2))

    doc = ParsedDocument(filename=filename, file_type="pdf", pages=pages)
    doc.full_text = "\n\n".join(p.text for p in pages if p.text.strip())
    doc.total_pages = len(pages)
    logger.info("Parsed PDF '%s': %d pages, %d characters.", filename, doc.total_pages, len(doc.full_text))
    return doc


def _parse_pdf_pdfplumber(file_path: str) -> List[ParsedPage]:
    """Fallback PDF parser using pdfplumber for scanned/complex layouts."""
    import pdfplumber

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            # Also extract tables as text
            tables = page.extract_tables() or []
            table_text_parts = []
            for table in tables:
                for row in table:
                    if row:
                        table_text_parts.append(" | ".join(str(cell or "") for cell in row))
            if table_text_parts:
                raw_text = raw_text + "\n\n" + "\n".join(table_text_parts)

            cleaned = clean_text(raw_text)
            pages.append(ParsedPage(page_number=i + 1, text=cleaned))
    return pages


def parse_docx(file_path: str) -> ParsedDocument:
    """Extract text from DOCX file via python-docx, including paragraphs and tables."""
    filename = os.path.basename(file_path)

    try:
        from docx import Document as DocxDocument

        doc = DocxDocument(file_path)
        content_parts: List[str] = []

        # Extract body paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                content_parts.append(text)

        # Extract table content
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    content_parts.append(" | ".join(cells))

        full_text = clean_text("\n\n".join(content_parts))

        # DOCX doesn't have inherent page numbers; treat as single logical page
        pages = [ParsedPage(page_number=1, text=full_text)] if full_text else []

        result = ParsedDocument(
            filename=filename, file_type="docx", pages=pages,
            full_text=full_text, total_pages=len(pages)
        )
        logger.info("Parsed DOCX '%s': %d characters.", filename, len(full_text))
        return result

    except Exception as e:
        logger.error("Failed to parse DOCX '%s': %s", filename, e)
        return ParsedDocument(filename=filename, file_type="docx", error=str(e))


def parse_txt(file_path: str) -> ParsedDocument:
    """Parse plain text file with UTF-8 and fallback encoding normalization."""
    filename = os.path.basename(file_path)

    raw_bytes = Path(file_path).read_bytes()
    text = None

    # Try UTF-8 first, then fallbacks
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if text is None:
        text = raw_bytes.decode("utf-8", errors="replace")

    full_text = clean_text(text)
    pages = [ParsedPage(page_number=1, text=full_text)] if full_text else []

    result = ParsedDocument(
        filename=filename, file_type="txt", pages=pages,
        full_text=full_text, total_pages=len(pages)
    )
    logger.info("Parsed TXT '%s': %d characters.", filename, len(full_text))
    return result


def parse_document(file_path: str) -> ParsedDocument:
    """
    Universal document parser dispatcher.
    Detects file type by extension and routes to the appropriate parser.
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    filename = os.path.basename(file_path)

    if not os.path.exists(file_path):
        return ParsedDocument(filename=filename, file_type=ext, error=f"File not found: {file_path}")

    parsers = {
        "pdf": parse_pdf,
        "docx": parse_docx,
        "txt": parse_txt,
    }

    parser_fn = parsers.get(ext)
    if parser_fn is None:
        return ParsedDocument(
            filename=filename, file_type=ext,
            error=f"Unsupported file type: .{ext}. Supported: pdf, docx, txt"
        )

    return parser_fn(file_path)

