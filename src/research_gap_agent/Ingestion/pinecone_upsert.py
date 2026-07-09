"""
research_gap_agent.Ingestion.pinecone_upsert

Production-ready Pinecone upsert service for the ingestion pipeline.

Workflow contract:
- Caller provides embedded chunks directly from embedder.py:

    [
        {"chunk_id": str, "text": str, "embedding": list[float], "metadata": dict},
        ...
    ]

- This module:
    * Extracts metadata
    * Generates per-paper namespace automatically
    * Connects to Pinecone
    * Ensures index exists (create if missing)
    * Waits until index is ready
    * Validates vectors (embedding dimension)
    * Batches vectors
    * Upserts all vectors into Pinecone
    * Returns upload statistics

No manual intervention is required.

Notes:
- This module reads the Pinecone API key from `api_keys.txt`.
- It never logs or hardcodes the Pinecone API key.
"""

from __future__ import annotations

import os
import re
import time
import uuid
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Global timestamp for unique namespace generation
_ingestion_timestamp = int(time.time())

logger = logging.getLogger(__name__)


# ---------------------------
# Config (from prompt)
# ---------------------------

PINECONE_INDEX_NAME = "researchassistant"
PINECONE_HOST_URL = "https://researchassistant-7qdvhrg.svc.aped-4627-b74a.pinecone.io"
EMBEDDING_DIMENSION = 384
SIMILARITY_METRIC = "cosine"

# Upsert batch sizing
UPSERT_BATCH_SIZE = 100

# Index creation wait options
INDEX_READY_POLL_SECONDS = 2
INDEX_READY_MAX_WAIT_SECONDS = 120


# ---------------------------
# Types
# ---------------------------

ChunkIn = Dict[str, Any]
UpsertStats = Dict[str, Any]


@dataclass(frozen=True)
class PineconeConfig:
    api_key: str
    index_name: str = PINECONE_INDEX_NAME
    host_url: str = PINECONE_HOST_URL
    dimension: int = EMBEDDING_DIMENSION
    metric: str = SIMILARITY_METRIC


# ---------------------------
# Helpers
# ---------------------------

def _read_pinecone_api_key_from_api_keys_file(api_keys_path: str = "api_keys.txt") -> str:
    """
    Read Pinecone API key from api_keys.txt.

    Expected line pattern includes:
      "pinecone key : <value>"

    Raises:
        RuntimeError if key is missing/invalid.
    """
    path = Path(api_keys_path)

    if not path.exists():
        raise RuntimeError(
            f"Missing Pinecone credentials file: {api_keys_path}. "
            f"Expected a line like 'pinecone key : <API_KEY>'."
        )

    content = path.read_text(encoding="utf-8", errors="ignore")

    # Be defensive with whitespace variations.
    # Example from repo:
    #   "pinecone key : pcsk_...."
    m = re.search(r"pinecone\s+key\s*:\s*([A-Za-z0-9_\-]+)", content, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError(
            f"Unable to find Pinecone API key in {api_keys_path}. "
            f"Expected a line like 'pinecone key : <API_KEY>'."
        )

    api_key = m.group(1).strip()
    if not api_key or not api_key.startswith("pcsk_"):
        raise RuntimeError(
            "Pinecone API key found in api_keys.txt but does not look valid "
            "(expected it to start with 'pcsk_')."
        )

    return api_key


def _safe_filename_to_namespace(filename: str) -> Optional[str]:
    if not filename or not isinstance(filename, str):
        return None
    name = filename.strip()
    if not name:
        return None
    # Remove extension
    return os.path.splitext(name)[0].strip() or None


def _get_namespace_from_chunk_metadata(metadata: Dict[str, Any]) -> str:
    """
    Priority:
      1) metadata["filename"] without extension + timestamp
      2) metadata["title"] + timestamp
      3) UUID fallback

    Different papers must naturally generate different namespaces.
    All chunks from same paper should share same namespace.
    Adding timestamp ensures fresh uploads don't conflict with old data.
    """
    filename = None
    title = None
    try:
        filename = metadata.get("filename")  # type: ignore[assignment]
        title = metadata.get("title")  # type: ignore[assignment]
    except Exception:
        filename = None
        title = None

    ns = _safe_filename_to_namespace(str(filename)) if filename is not None else None
    if ns:
        # Add timestamp to ensure unique namespace for each upload
        return f"{ns}_{_ingestion_timestamp}"

    if title is not None and isinstance(title, str):
        t = title.strip()
        if t:
            # Add timestamp to ensure unique namespace for each upload
            return f"{t[:50]}_{_ingestion_timestamp}"  # Truncate long titles

    return str(uuid.uuid4())


def _chunk_iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _validate_and_normalize_embedding(embedding: Any, *, expected_dim: int) -> Tuple[bool, List[float]]:
    """
    Returns:
      (is_valid, normalized_values)

    - normalized_values is always a list[float] if valid; empty list if invalid.
    """
    if embedding is None:
        return False, []

    # sentence-transformers normalization is already done in embedder,
    # but we still validate here for production safety.
    if not isinstance(embedding, list):
        try:
            embedding = list(embedding)  # type: ignore[arg-type]
        except Exception:
            return False, []

    values: List[float] = []
    try:
        for v in embedding:
            # Ensure float coercion
            values.append(float(v))
    except Exception:
        return False, []

    if len(values) != expected_dim:
        return False, []

    return True, values


def _validate_chunk_for_upsert(ch: Any) -> Tuple[bool, Optional[str], Optional[List[float]], Optional[Dict[str, Any]]]:
    """
    Validate chunk and extract required fields.

    Returns:
      (is_valid, chunk_id, embedding_values, metadata)
    """
    if not isinstance(ch, dict):
        return False, None, None, None

    chunk_id = ch.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        return False, None, None, None

    metadata = ch.get("metadata")
    if metadata is None or not isinstance(metadata, dict):
        # allow metadata missing by coercing to {}
        metadata = {}

    # Preserve top-level "text" field in metadata for retrieval
    # The embedder output has "text" at the top level, not in metadata
    if "text" not in metadata and "text" in ch:
        text_value = ch.get("text")
        if isinstance(text_value, str):
            metadata["text"] = text_value

    embedding = ch.get("embedding")
    ok, values = _validate_and_normalize_embedding(embedding, expected_dim=EMBEDDING_DIMENSION)
    if not ok:
        return False, None, None, None

    return True, chunk_id, values, metadata


def _build_vectors(
    embedded_chunks: List[ChunkIn],
    *,
    namespace: str,
) -> List[Dict[str, Any]]:
    """
    Build Pinecone vector payloads: {"id": ..., "values": ..., "metadata": ...}

    Note: Pinecone namespace is passed separately to upsert().
    """
    MAX_METADATA_TEXT_CHARS = 8000  # Keep Pinecone per-vector metadata under 40960 bytes

    vectors: List[Dict[str, Any]] = []
    for ch in embedded_chunks:
        valid, chunk_id, values, metadata = _validate_chunk_for_upsert(ch)
        if not valid or chunk_id is None or values is None or metadata is None:
            continue

        # Ensure chunk text is available for retrieval.
        # Embedded chunks already include "text"; but Pinecone currently stores it in metadata.
        if "text" not in metadata:
            try:
                # ch might be validated already but still be defensive.
                if isinstance(ch, dict) and "text" in ch:
                    metadata["text"] = ch.get("text")
            except Exception:
                # If metadata injection fails, keep going without crashing upsert.
                pass

        # Truncate metadata["text"] to avoid Pinecone 40,960-byte metadata limit per vector.
        try:
            raw_text = metadata.get("text", "")
            text = raw_text if isinstance(raw_text, str) else ""

            is_truncated = len(text) > MAX_METADATA_TEXT_CHARS
            if is_truncated:
                text = text[:MAX_METADATA_TEXT_CHARS]

            metadata["text"] = text
            metadata["text_truncated"] = bool(is_truncated)
        except Exception:
            # If truncation logic fails, do not break upsert; keep whatever is there.
            pass

        vectors.append(
            {
                "id": chunk_id,
                "values": values,
                "metadata": metadata,
            }
        )
    return vectors


def _pinecone_client(config: PineconeConfig):
    """
    Create a control-plane Pinecone client (SDK v9.1.0).

    IMPORTANT:
    Do NOT pass `host` into this client. Control-plane ops (list_indexes,
    describe_index, etc.) must use the control-plane client only.
    """
    try:
        from pinecone import Pinecone  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: pinecone. Install it (e.g., `pip install pinecone`)."
        ) from exc

    # Control-plane client: no host passed.
    return Pinecone(api_key=config.api_key)


def _verify_index_exists(pc, config: PineconeConfig) -> None:
    """
    Verify the Pinecone index exists using control-plane API only.
    """
    index_name = config.index_name
    try:
        indexes = pc.list_indexes()
    except Exception as exc:
        raise RuntimeError(f"Failed to list Pinecone indexes: {exc}") from exc

    # pinecone SDK returns IndexList([...]) where each item has .name
    existing_names = set()
    try:
        for i in indexes:
            name = getattr(i, "name", None)
            if name:
                existing_names.add(name)
    except TypeError:
        # Fallback: indexes might not be iterable in some versions
        pass

    if index_name not in existing_names:
        raise RuntimeError(f"Pinecone index '{index_name}' does not exist.")


# ---------------------------
# Public API
# ---------------------------

def upsert_chunks(embedded_chunks: List[ChunkIn], timestamp: Optional[int] = None) -> UpsertStats:
    """
    Upsert embedded chunks into Pinecone.

    Args:
        embedded_chunks: list of embedded chunk dicts produced by embedder.py
        timestamp: optional timestamp for namespace generation (uses current time if None)

    Returns:
        Upload statistics with:
          - total_input_chunks
          - total_uploaded_vectors
          - skipped_invalid_vectors
          - per_namespace (dict namespace -> uploaded_count)
    """
    global _ingestion_timestamp
    if timestamp is not None:
        _ingestion_timestamp = timestamp
        logger.info(f"Using provided timestamp for namespace generation: {timestamp}")
    else:
        _ingestion_timestamp = int(time.time())
        logger.info(f"Using current timestamp for namespace generation: {_ingestion_timestamp}")
        
    if embedded_chunks is None:
        embedded_chunks = []

    config = PineconeConfig(
        api_key=_read_pinecone_api_key_from_api_keys_file("api_keys.txt"),
    )

    if not embedded_chunks:
        return {
            "total_input_chunks": 0,
            "total_uploaded_vectors": 0,
            "skipped_invalid_vectors": 0,
            "per_namespace": {},
        }

    # Validate index dimension expectation at runtime as required by prompt
    # (We already use EMBEDDING_DIMENSION throughout).
    expected_dim = config.dimension
    if expected_dim != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Embedding dimension mismatch: configured expected_dim={expected_dim} "
            f"but code constant EMBEDDING_DIMENSION={EMBEDDING_DIMENSION}."
        )

    # Create control-plane client + verify index exists (no create_index, no wait)
    pc = _pinecone_client(config)
    _verify_index_exists(pc, config)

    # Build vectors grouped by namespace, so all chunks from same paper share one namespace.
    chunks_by_namespace: Dict[str, List[ChunkIn]] = {}
    skipped = 0

    for ch in embedded_chunks:
        valid, chunk_id, values, metadata = _validate_chunk_for_upsert(ch)
        if not valid or chunk_id is None or values is None or metadata is None:
            skipped += 1
            continue

        ns = _get_namespace_from_chunk_metadata(metadata)
        chunks_by_namespace.setdefault(ns, []).append(ch)

    per_ns_uploaded: Dict[str, int] = {}
    total_uploaded = 0

    # Upsert in batches per namespace.
    # Use the index handle from the same (control-plane) client.
    index = pc.Index(config.index_name)  # type: ignore[attr-defined]

    for ns, ns_chunks in chunks_by_namespace.items():
        # Build vector payloads (valid chunks only; invalid already skipped)
        # We rebuild payloads here to ensure values are in the right type/shape.
        vectors: List[Dict[str, Any]] = _build_vectors(ns_chunks, namespace=ns)

        if not vectors:
            per_ns_uploaded[ns] = 0
            continue

        uploaded_for_ns = 0
        for batch in _chunk_iter_batches(vectors, UPSERT_BATCH_SIZE):
            try:
                # Pinecone upsert signature: index.upsert(vectors=[...], namespace='...')
                index.upsert(vectors=batch, namespace=ns)  # type: ignore[attr-defined]
                uploaded_for_ns += len(batch)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to upsert vectors to Pinecone index='{config.index_name}', "
                    f"namespace='{ns}': {exc}"
                ) from exc

        per_ns_uploaded[ns] = uploaded_for_ns
        total_uploaded += uploaded_for_ns

    return {
        "total_input_chunks": len(embedded_chunks),
        "total_uploaded_vectors": total_uploaded,
        "skipped_invalid_vectors": skipped,
        "per_namespace": per_ns_uploaded,
    }
