"""
research_gap_agent.Ingestion.embedder

Automated embedding service for the ingestion pipeline.

Pipeline:
PDF Upload
→ parser.py
→ semantic_chunker.py
→ embedder.py
→ pinecone_upsert.py

Contract (no extra transformation layer):
Input  to embedder.embed_chunks:
[
    {"chunk_id": str, "text": str, "metadata": dict},
    ...
]

Output from embedder.embed_chunks:
[
    {"chunk_id": str, "text": str, "embedding": list[float], "metadata": dict},
    ...
]

This module provides :class:`EmbeddingService`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

import logging

logger = logging.getLogger(__name__)


ChunkIn = Dict[str, Any]
ChunkOut = Dict[str, Any]


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # all-MiniLM-L6-v2 produces 384-d embeddings
    embedding_dim: int = 384


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name)


class EmbeddingService:
    """
    Sentence-transformers based embedding service.

    Public API:
        embed_chunks(chunks: list[dict]) -> list[dict]

    The output is ready to be consumed by pinecone_upsert.upsert_chunks(embedded_chunks).
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()
        self._model = None  # lazy init to avoid import-time/launch issues

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            self._model = _load_sentence_transformer(self._config.model_name)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: sentence-transformers. "
                "Install it to enable embeddings."
            ) from exc

        return self._model

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _safe_metadata(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return {"metadata": value}

    @staticmethod
    def _normalize_embeddings(embeddings: Any) -> List[List[float]]:
        """Convert model output into a stable list-of-lists shape."""
        if embeddings is None:
            return []

        try:
            embeddings_list = embeddings.tolist()  # type: ignore[assignment]
        except Exception:
            embeddings_list = embeddings

        if not isinstance(embeddings_list, list):
            return [[float(embeddings_list)]]

        if not embeddings_list:
            return []

        first_item = embeddings_list[0]
        if isinstance(first_item, (int, float)):
            return [[float(value) for value in embeddings_list]]

        normalized: List[List[float]] = []
        for row in embeddings_list:
            if row is None:
                normalized.append([])
                continue
            if isinstance(row, list):
                normalized.append([float(value) for value in row])
                continue
            try:
                normalized.append([float(value) for value in list(row)])
            except TypeError:
                normalized.append([float(row)])

        return normalized

    def embed_semantic_chunks(self, parsed_document: dict) -> list[dict]:
        """Chunk a parsed document and embed the resulting semantic chunks."""
        try:
            from research_gap_agent.Ingestion.semantic_chunker import (  # type: ignore
                SemanticChunkerService,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: semantic_chunker service could not be imported."
            ) from exc

        chunker = SemanticChunkerService()
        chunks = chunker.chunk_document(parsed_document)
        return self.embed_chunks(chunks)

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Embed a list of semantic chunks in batch and return them fully enriched.

        Steps:
        1) Extract all chunk texts
        2) Generate embeddings in batch
        3) Attach embeddings back to corresponding chunks
        4) Preserve chunk_id, metadata, and original text
        5) Return a fully enriched chunk list ready for Pinecone upsert
        """
        if not chunks:
            return []

        # Extract texts + preserve ordering
        texts: List[str] = []
        cleaned_inputs: List[ChunkIn] = []

        for ch in chunks:
            if not isinstance(ch, dict):
                # defensive: skip invalid items
                continue

            chunk_id = ch.get("chunk_id")
            text = self._safe_text(ch.get("text"))
            metadata = self._safe_metadata(ch.get("metadata"))

            # Preserve original dict shape minimally while ensuring required keys
            cleaned_inputs.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    # keep reference to original if needed later
                }
            )
            texts.append(text)

        if not cleaned_inputs:
            return []

        model = self._get_model()

        # sentence-transformers returns an array-like object for encode().
        # Use normalize_embeddings=False (default) to keep raw vectors;
        # Pinecone can handle similarity based on its configured metric.
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        embeddings_list = self._normalize_embeddings(embeddings)

        # Defensive: handle mismatch lengths
        n = len(cleaned_inputs)
        if len(embeddings_list) != n:
            logger.warning(
                "Embedding count mismatch: inputs=%s embeddings=%s. "
                "Will embed up to min length.",
                n,
                len(embeddings_list),
            )
        out_len = min(n, len(embeddings_list))

        embedded_chunks: List[ChunkOut] = []
        for i in range(out_len):
            inp = cleaned_inputs[i]
            embedded_chunks.append(
                {
                    "chunk_id": inp.get("chunk_id"),
                    "text": inp.get("text"),
                    "embedding": embeddings_list[i],
                    "metadata": inp.get("metadata", {}),
                }
            )

        # If there are extra inputs (shouldn't happen), append them with empty embeddings
        for i in range(out_len, n):
            inp = cleaned_inputs[i]
            embedded_chunks.append(
                {
                    "chunk_id": inp.get("chunk_id"),
                    "text": inp.get("text"),
                    "embedding": [],
                    "metadata": inp.get("metadata", {}),
                }
            )

        return embedded_chunks
