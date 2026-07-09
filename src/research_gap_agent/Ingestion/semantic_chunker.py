"""
research_gap_agent.Ingestion.semantic_chunker

Production-ready semantic chunking service for parsed PDF documents.

This module provides :class:`SemanticChunkerService`, which takes a parsed
document dictionary (as returned by parser.py) and produces a list of
clean Python dictionaries representing semantic chunks.

Chunk boundary detection is performed using LangChain Experimental's
``SemanticChunker``. Embeddings are created only to support semantic boundary
detection; embedding vectors are NOT returned.

Output chunk dictionaries have the shape::

    {
        "chunk_id": "uuid",
        "text": "chunk content",
        "metadata": {
            "source": "paper.pdf",
            "filename": "paper.pdf",
            "chunk_index": 0
        }
    }

Notes:
- This file is self-contained and does not depend on LangChain Document
  objects at the output boundary.
- The implementation is defensive for edge cases: empty/None text,
  very small documents, and chunking failures.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

try:
    from langchain_experimental.text_splitter import SemanticChunker
except ModuleNotFoundError:
    SemanticChunker = None  # type: ignore[assignment]

# HuggingFaceEmbeddings import is intentionally lazy/optional.
# Import-time side effects in transformers/langchain_huggingface can crash
# Streamlit startup in some environments.
HuggingFaceEmbeddings = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


ParsedDocument = Dict[str, Any]
ChunkDict = Dict[str, Any]


@lru_cache(maxsize=4)
def _get_cached_chunk_embeddings(model_name: str, embeddings_cls):
    return embeddings_cls(model_name=model_name)


class SemanticChunkerService:
    """
    Semantic chunker service using LangChain Experimental ``SemanticChunker``.

    Responsibilities:
    - Validate and extract the input text + metadata from a parsed document.
    - Use SemanticChunker for semantic boundary detection.
    - Create a HuggingFace embeddings model for chunk boundary detection only.
    - Return output as plain Python dictionaries (no LangChain objects).
    """

    def __init__(
        self,
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize the service.

        Args:
            embeddings_model_name: HuggingFace model name used by
                ``HuggingFaceEmbeddings``.
        """
        self._embeddings_model_name = embeddings_model_name

    def _build_embeddings(self):
        """
        Build embeddings used only for semantic chunk boundary detection.

        Returns:
            Initialized HuggingFaceEmbeddings instance.

        Raises:
            RuntimeError: if required dependencies are missing.
        """
        try:
            # Import lazily to avoid import-time crashes.
            from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "HuggingFaceEmbeddings dependency missing or failed to import. "
                "Semantic chunking is unavailable in this environment."
            ) from exc

        # Reuse the same embeddings model across Streamlit reruns/documents.
        return _get_cached_chunk_embeddings(
            self._embeddings_model_name,
            _HuggingFaceEmbeddings,
        )

    def _safe_get_text(self, parsed_document: ParsedDocument) -> str:
        """
        Extract and normalize the input text from a parsed document.

        Args:
            parsed_document: Parsed document dictionary from parser.py.

        Returns:
            Extracted text as a string (possibly empty).
        """
        text = parsed_document.get("text")
        if text is None:
            return ""
        if not isinstance(text, str):
            # Defensive conversion; if it’s not a string, convert to str.
            return str(text)
        return text

    def _safe_get_metadata(self, parsed_document: ParsedDocument) -> Dict[str, Any]:
        """
        Extract and normalize metadata from a parsed document.

        Args:
            parsed_document: Parsed document dictionary from parser.py.

        Returns:
            Metadata dictionary (always returned as a dict).
        """
        metadata = parsed_document.get("metadata")
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            # Defensive conversion.
            return {"metadata": metadata}
        return metadata

    def _create_chunk_dict(
        self,
        *,
        chunk_text: str,
        base_metadata: Dict[str, Any],
        chunk_index: int,
    ) -> ChunkDict:
        """
        Create a clean chunk dictionary for output.

        Args:
            chunk_text: Chunk content.
            base_metadata: Original metadata from parsed document.
            chunk_index: Index of chunk in document order.

        Returns:
            Chunk dictionary.
        """
        # Preserve original document metadata and append required fields.
        # Ensure we don't mutate the caller's dict.
        chunk_metadata = dict(base_metadata)
        chunk_metadata["chunk_index"] = chunk_index
        # chunk_id is kept at top-level, as required.
        return {
            "chunk_id": str(uuid.uuid4()),
            "text": chunk_text,
            "metadata": chunk_metadata,
        }

    def chunk_document(self, parsed_document: dict) -> list[dict]:
        """
        Chunk a parsed document using semantic chunk boundaries.

        Args:
            parsed_document: Parsed document dictionary from parser.py, expected
                to include:
                - "text": str | None
                - "metadata": dict (including at least source/filename when available)

        Returns:
            List of chunk dictionaries, each containing:
            - chunk_id (uuid string)
            - text (chunk content)
            - metadata (original metadata + chunk_index)

        Edge cases:
        - Empty/None text => returns [].
        - Very small documents => returns a single chunk (best-effort).
        - Chunking failures => returns fallback chunking (best-effort),
          and logs the exception.
        """
        if parsed_document is None:
            logger.warning("chunk_document received None parsed_document.")
            return []

        text = self._safe_get_text(parsed_document).strip()
        base_metadata = self._safe_get_metadata(parsed_document)

        if not text:
            logger.info("Parsed document has empty/None text; returning no chunks.")
            return []

        # If the document is very small, semantic boundary detection may be noisy.
        # We still return a single chunk for robustness.
        # Threshold chosen to be conservative; SemanticChunker can handle small docs
        # but this avoids edge-case instability.
        # For small documents, semantic boundary detection may not behave well.
        # However, returning 1 chunk too aggressively makes the UI look like
        # chunking "didn't work". We only apply a 1-chunk short-circuit for
        # extremely small inputs.
        if len(text) < 120:
            logger.info("Parsed document is extremely small (<120 chars); using 1-chunk output.")
            return [
                self._create_chunk_dict(
                    chunk_text=text,
                    base_metadata=base_metadata,
                    chunk_index=0,
                )
            ]

        try:
            # Some environments may have intermittent import issues; re-try import at runtime.
            semantic_chunker_cls = SemanticChunker
            if semantic_chunker_cls is None:  # pragma: no cover
                try:
                    from langchain_experimental.text_splitter import (  # type: ignore
                        SemanticChunker as _SemanticChunker,
                    )

                    semantic_chunker_cls = _SemanticChunker
                except Exception as exc:
                    raise RuntimeError(
                        "SemanticChunker dependency missing or failed to import. "
                        "Install `langchain-experimental` to enable semantic chunking."
                    ) from exc

            if semantic_chunker_cls is None:  # pragma: no cover
                raise RuntimeError(
                    "SemanticChunker dependency missing. "
                    "Install `langchain-experimental` to enable semantic chunking."
                )

            embeddings = self._build_embeddings()

            # LangChain Experimental SemanticChunker:
            # - breakpoint_threshold_type="percentile" as required
            # - it will compute boundaries using embeddings
            chunker = semantic_chunker_cls(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
            )

            # SemanticChunker expects input text as a string and returns a list
            # of LangChain "Document"-like objects (depending on version).
            chunk_docs = chunker.create_documents([text])
            logger.info("SemanticChunker returned %s chunk_docs", len(chunk_docs))

            chunks: List[ChunkDict] = []
            for idx, cd in enumerate(chunk_docs):
                # Defensive extraction from Document-like structures.
                chunk_text: Optional[str] = None
                if hasattr(cd, "page_content"):
                    chunk_text = getattr(cd, "page_content")
                elif hasattr(cd, "text"):
                    chunk_text = getattr(cd, "text")
                elif isinstance(cd, dict):
                    chunk_text = cd.get("page_content") or cd.get("text")
                else:
                    chunk_text = str(cd)

                if chunk_text is None:
                    continue

                chunk_text_str = str(chunk_text).strip()
                if not chunk_text_str:
                    continue

                chunks.append(
                    self._create_chunk_dict(
                        chunk_text=chunk_text_str,
                        base_metadata=base_metadata,
                        chunk_index=idx,
                    )
                )

            # If semantic chunker didn't split, do a deterministic fallback so
            # the UI/pipeline can actually see chunking happen.
            #
            # This avoids the common "everything is one chunk" outcome.
            # If semantic chunker didn't split but the input is large enough to
            # benefit from multiple chunks, apply deterministic fallback splitting.
            if len(chunks) <= 1 and len(text) >= 300:
                logger.warning(
                    "SemanticChunker did not split (chunks=%s) for a document; applying fallback splitting.",
                    len(chunks),
                )

                def _fallback_split(document_text: str) -> List[str]:
                    # 1) Prefer paragraph splits
                    parts = [p.strip() for p in document_text.split("\n\n") if p.strip()]
                    if len(parts) > 1:
                        # Re-pack to target size blocks
                        target = 1200
                        out: List[str] = []
                        buf: List[str] = []
                        buf_len = 0
                        for p in parts:
                            p_len = len(p)
                            if buf and (buf_len + p_len) > target:
                                out.append("\n\n".join(buf).strip())
                                buf = [p]
                                buf_len = p_len
                            else:
                                buf.append(p)
                                buf_len += p_len
                        if buf:
                            out.append("\n\n".join(buf).strip())
                        return [c for c in out if c]

                    # 2) Otherwise sentence split
                    sentences = [s.strip() for s in document_text.split(". ") if s.strip()]
                    if not sentences:
                        return [document_text]

                    target = 1200
                    out2: List[str] = []
                    buf2: List[str] = []
                    buf2_len = 0
                    for s in sentences:
                        s2 = s if s.endswith(".") else f"{s}."
                        s_len = len(s2)
                        if buf2 and (buf2_len + s_len) > target:
                            out2.append(" ".join(buf2).strip())
                            buf2 = [s2]
                            buf2_len = s_len
                        else:
                            buf2.append(s2)
                            buf2_len += s_len
                    if buf2:
                        out2.append(" ".join(buf2).strip())
                    return [c for c in out2 if c]

                fallback_chunks_text = _fallback_split(text)

                # Guarantee multi-chunk output for sufficiently large documents.
                # Some extracted PDFs produce text with few paragraph/sentence delimiters,
                # which can cause the heuristic splitter to return 1 chunk.
                if len(fallback_chunks_text) <= 1 and len(text) >= 300:
                    logger.warning(
                        "Fallback splitter produced <=1 chunk; applying char-based split to force chunking."
                    )

                    target = 1200
                    overlap = 200
                    forced: List[str] = []
                    start = 0
                    text_len = len(text)
                    while start < text_len:
                        end = min(start + target, text_len)
                        part = text[start:end].strip()
                        if part:
                            forced.append(part)
                        if end >= text_len:
                            break
                        start = max(end - overlap, 0)

                    fallback_chunks_text = forced

                logger.info(
                    "Fallback splitter produced %s chunks",
                    len(fallback_chunks_text),
                )

                chunks = [
                    self._create_chunk_dict(
                        chunk_text=ct,
                        base_metadata=base_metadata,
                        chunk_index=i,
                    )
                    for i, ct in enumerate(fallback_chunks_text)
                    if ct and ct.strip()
                ]

            if not chunks:
                logger.warning(
                    "SemanticChunker/fallback produced no usable chunks; falling back to 1-chunk output."
                )
                return [
                    self._create_chunk_dict(
                        chunk_text=text,
                        base_metadata=base_metadata,
                        chunk_index=0,
                    )
                ]

            logger.info(
                "Chunking complete. chunks=%s chars_input=%s chunk_sizes_sample=%s",
                len(chunks),
                len(text),
                [len(c.get("text", "")) for c in chunks[:3]],
            )
            return chunks

        except Exception as exc:
            logger.exception(
                "Chunking failed for parsed_document; using deterministic fallback. Error: %s",
                exc,
            )

            # Deterministic char-based split so UI never collapses to 1 chunk
            # for sufficiently large documents.
            target = 1200
            overlap = 200
            forced: List[str] = []
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + target, text_len)
                part = text[start:end].strip()
                if part:
                    forced.append(part)

                if end >= text_len:
                    break

                start = max(end - overlap, 0)

            fallback_chunks_text = forced or [text]

            return [
                self._create_chunk_dict(
                    chunk_text=ct,
                    base_metadata=base_metadata,
                    chunk_index=i,
                )
                for i, ct in enumerate(fallback_chunks_text)
                if ct and ct.strip()
            ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Small usage example
    example_parsed_document: dict = {
        "text": "This is a small example. " * 50,
        "metadata": {
            "source": "paper.pdf",
            "filename": "paper.pdf",
            "title": "Example Title",
            "authors": ["Author A"],
        },
    }

    service = SemanticChunkerService()
    output_chunks = service.chunk_document(example_parsed_document)

    logger.info("Generated %s chunks. First chunk:", len(output_chunks))
    if output_chunks:
        logger.info(output_chunks[0])
