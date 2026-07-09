"""
research_gap_agent.Ingestion.parser

Enhanced PDF parser for scientific research papers using pymupdf4llm.

This module provides :func:`parse_pdf`, which extracts high-quality text from a
PDF using pymupdf4llm (a library specifically designed for LLM consumption).
pymupdf4llm provides better text extraction, table handling, and document structure
preservation compared to basic PyMuPDF.

The implementation focuses on:
- High-quality text extraction optimized for LLMs
- Table extraction and preservation
- Document structure preservation (headings, sections)
- Robust error handling and structured logging
- Fallback to basic PyMuPDF if pymupdf4llm fails

Requirements:
- PyMuPDF (installed as ``PyMuPDF``)
- pymupdf4llm (new dependency)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    fitz = None  # type: ignore[assignment]
    _FITZ_IMPORT_ERROR = exc
else:
    _FITZ_IMPORT_ERROR = None

try:
    import pymupdf4llm
except ImportError as exc:  # pragma: no cover
    pymupdf4llm = None  # type: ignore[assignment]
    _PYMUPDF4LLM_IMPORT_ERROR = exc
else:
    _PYMUPDF4LLM_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


_WHITESPACE_RE = re.compile(r"[\t\r\f\v ]+")
_NEWLINE_RE = re.compile(r"\n{3,}")


def _normalize_text(text: str) -> str:
    """Normalize extracted text.

    pymupdf4llm generally provides clean text, but we still normalize:
    - Converts Windows newlines to ``\n``.
    - Collapses repeated horizontal whitespace into a single space.
    - Collapses runs of 3+ newlines into at most two.
    - Strips leading/trailing whitespace.
    - Removes hyphenation across line breaks (e.g., "com-\nplementary" -> "complementary")

    Args:
        text: Raw extracted page text.

    Returns:
        Normalized text.
    """

    # Normalize newline style
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove hyphenation across line breaks (common in PDFs)
    # Pattern: word at end of line followed by hyphen, newline, then continuation
    # Example: "com-\nplementary" -> "complementary"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Collapse spaces/tabs but keep newlines
    text = _WHITESPACE_RE.sub(" ", text)

    # Reduce excessive blank lines
    text = _NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def _parse_with_pymupdf4llm(pdf_path: str) -> Dict[str, Any]:
    """Parse PDF using pymupdf4llm for enhanced LLM-optimized text extraction.

    pymupdf4llm provides:
    - Better text extraction quality
    - Table extraction and preservation
    - Document structure preservation
    - Markdown-formatted output

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A dictionary with the shape::

            {
                "full_text": str,
                "metadata": {
                    "title": str,
                    "author": str,
                    "page_count": int
                }
            }

    Raises:
        RuntimeError: If pymupdf4llm is not available.
        Exception: If parsing fails.
    """
    if pymupdf4llm is None:  # pragma: no cover
        raise RuntimeError(
            "pymupdf4llm is required for enhanced PDF parsing. "
            "Install with `pip install pymupdf4llm`."
        ) from _PYMUPDF4LLM_IMPORT_ERROR

    logger.info("Parsing PDF with pymupdf4llm: %s", pdf_path)

    try:
        # Use pymupdf4llm to extract text in markdown format
        # This provides better structure preservation for LLMs
        markdown_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Also extract basic metadata using PyMuPDF
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for metadata extraction.") from _FITZ_IMPORT_ERROR
            
        document = fitz.open(pdf_path)
        page_count = len(document)
        
        info = document.metadata or {}
        title = str(info.get("title") or "").strip()
        author = str(info.get("author") or "").strip()
        
        if not title:
            title = str(info.get("Title") or "").strip()
        if not author:
            author = str(info.get("Author") or "").strip()
            
        document.close()
        
        # Normalize the markdown text
        full_text = _normalize_text(markdown_text)
        
        metadata: Dict[str, Any] = {
            "title": title,
            "author": author,
            "page_count": int(page_count),
            "parser": "pymupdf4llm",
        }

        logger.info(
            "Parsed PDF with pymupdf4llm: %s (pages=%s, extracted_chars=%s)",
            pdf_path,
            page_count,
            len(full_text),
        )

        return {"full_text": full_text, "metadata": metadata}

    except Exception as exc:
        logger.exception("pymupdf4llm parsing failed for %s", pdf_path)
        raise


def _parse_with_pymupdf(pdf_path: str) -> Dict[str, Any]:
    """Fallback PDF parser using basic PyMuPDF.

    This is used as a fallback when pymupdf4llm fails or is not available.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A dictionary with the shape::

            {
                "full_text": str,
                "metadata": {
                    "title": str,
                    "author": str,
                    "page_count": int
                }
            }
    """
    if fitz is None:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF is required to parse PDFs. Install with `pip install PyMuPDF`."
        ) from _FITZ_IMPORT_ERROR

    logger.info("Parsing PDF with PyMuPDF fallback: %s", pdf_path)

    try:
        document: Optional["fitz.Document"] = fitz.open(pdf_path)
    except FileNotFoundError:
        logger.exception("PDF not found: %s", pdf_path)
        raise
    except Exception as exc:
        logger.exception("Failed to open PDF: %s", pdf_path)
        raise ValueError(f"Failed to open PDF: {pdf_path}") from exc

    with document:
        page_count = len(document)

        info: Dict[str, Any] = {}
        try:
            info = document.metadata or {}
        except Exception:
            logger.debug("Unable to read PDF metadata for %s", pdf_path, exc_info=True)
            info = {}

        title = str(info.get("title") or "").strip()
        author = str(info.get("author") or "").strip()

        if not title:
            title = str(info.get("Title") or "").strip()
        if not author:
            author = str(info.get("Author") or "").strip()

        pages_text: list[str] = []

        for page_index in range(page_count):
            try:
                page = document.load_page(page_index)
            except Exception:
                logger.exception(
                    "Failed loading page %s of %s; skipping page.",
                    page_index,
                    pdf_path,
                )
                continue

            try:
                # Try multiple extraction methods for better compatibility
                raw_text = page.get_text("text") or ""
                
                # If regular text extraction fails, try other methods
                if not raw_text or len(raw_text.strip()) < 10:
                    logger.debug("Regular text extraction minimal for page %s, trying 'blocks'", page_index)
                    try:
                        blocks = page.get_text("blocks")
                        if blocks:
                            raw_text = "\n".join([b[4] for b in blocks if b[4] and len(b[4].strip()) > 10])
                    except Exception as block_exc:
                        logger.debug("Blocks extraction failed: %s", block_exc)
                    
                if not raw_text or len(raw_text.strip()) < 10:
                    logger.debug("Blocks extraction minimal for page %s, trying 'words'", page_index)
                    try:
                        words = page.get_text("words")
                        if words:
                            raw_text = " ".join([w[4] for w in words if w[4]])
                    except Exception as word_exc:
                        logger.debug("Words extraction failed: %s", word_exc)
                        
                if not raw_text or len(raw_text.strip()) < 10:
                    logger.debug("Words extraction minimal for page %s, trying 'html'", page_index)
                    try:
                        html_text = page.get_text("html")
                        if html_text:
                            # Simple HTML to text conversion
                            import re
                            raw_text = re.sub(r'<[^>]+>', ' ', html_text)
                            raw_text = re.sub(r'\s+', ' ', raw_text).strip()
                    except Exception as html_exc:
                        logger.debug("HTML extraction failed: %s", html_exc)
                        
            except Exception:
                logger.exception(
                    "Failed extracting text from page %s of %s; skipping page.",
                    page_index,
                    pdf_path,
                )
                continue

            normalized = _normalize_text(raw_text)

            if not normalized:
                logger.debug("Skipping empty page %s for %s (all extraction methods failed)", page_index, pdf_path)
                continue

            pages_text.append(normalized)

        full_text = "\n\n".join(pages_text)

        metadata: Dict[str, Any] = {
            "title": title,
            "author": author,
            "page_count": int(page_count),
            "parser": "pymupdf_fallback",
        }

        logger.info(
            "Parsed PDF with PyMuPDF fallback: %s (pages=%s, extracted_chars=%s)",
            pdf_path,
            page_count,
            len(full_text),
        )

        # Warn if no text was extracted (likely image-based PDF)
        if not full_text or len(full_text.strip()) < 50:
            logger.warning(
                "PDF appears to be image-based or corrupted. No text extracted from %s. "
                "This PDF requires OCR processing or is not a text-based PDF.",
                pdf_path
            )
            # Add a note to metadata for UI display
            metadata["extraction_warning"] = "No text could be extracted from this PDF. It may be image-based and require OCR processing."

        return {"full_text": full_text, "metadata": metadata}


def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    """Parse a PDF file and extract its text and metadata.

    This function first tries to use pymupdf4llm for enhanced LLM-optimized parsing,
    with fallback to basic PyMuPDF if it fails.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A dictionary with the shape::

            {
                "full_text": str,
                "metadata": {
                    "title": str,
                    "author": str,
                    "page_count": int
                }
            }

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        RuntimeError: If both pymupdf4llm and PyMuPDF are not available.
        ValueError: If the PDF cannot be opened/parsed.
    """
    # Try pymupdf4llm first for enhanced parsing
    try:
        return _parse_with_pymupdf4llm(pdf_path)
    except Exception as exc:
        logger.warning(
            "pymupdf4llm parsing failed for %s, falling back to PyMuPDF: %s",
            pdf_path,
            exc
        )
        # Fallback to basic PyMuPDF
        return _parse_with_pymupdf(pdf_path)


if __name__ == "__main__":
    result = parse_pdf("data/uploads/sample_paper.pdf")

    print("\nMetadata")
    print(result["metadata"])

    print("\nText Length")
    print(len(result["full_text"]))

    print("\nFirst 1000 Characters")
    print(result["full_text"][:1000])
