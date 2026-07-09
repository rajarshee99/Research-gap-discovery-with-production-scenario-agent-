"""src.research_gap_agent.nodes.retrieval

LangGraph retrieval node for the Research Gap Discovery & Industry Alignment Agent.

Responsibilities (retrieval-only):
- Read the user's query from the LangGraph state.
- Generate a query embedding using the *same* embedding model and logic as the ingestion pipeline
  (see :mod:`research_gap_agent.Ingestion.embedder`).
- Connect to the existing Pinecone index populated during ingestion.
- Perform a similarity search and retrieve top-k matching chunks.
- Store results in ``state["retrieved_docs"]``.

This node does not implement any LLM generation, reranking, routing, or analysis.

Notes on Pinecone metadata:
- During upsert, ingestion stores chunk text under ``metadata["text"]``.
- This node preserves Pinecone-returned metadata and similarity scores.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast, Union

from research_gap_agent.Ingestion.embedder import EmbeddingService
from research_gap_agent.state.state import ResearchState


logger = logging.getLogger(__name__)


# Keep these values aligned with ingestion pipeline constants in:
#   src/research_gap_agent/Ingestion/pinecone_upsert.py
PINECONE_INDEX_NAME = "researchassistant"
EMBEDDING_DIMENSION = 384

DEFAULT_TOP_K = 10  # Increased from 5 to provide more context and reduce hallucination


class RetrievedDoc(TypedDict):
    """A single retrieved chunk."""

    id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str]


def _read_pinecone_api_key_from_api_keys_file(api_keys_path: str = "api_keys.txt") -> str:
    """Read Pinecone API key from ``api_keys.txt``.

    Expected line pattern includes::

        pinecone key : <value>

    Raises:
        RuntimeError: if key is missing or invalid.
    """

    path = Path(api_keys_path)
    if not path.exists():
        raise RuntimeError(
            f"Missing Pinecone credentials file: {api_keys_path}. "
            "Expected a line like 'pinecone key : <API_KEY>'."
        )

    content = path.read_text(encoding="utf-8", errors="ignore")

    m = re.search(r"pinecone\s+key\s*:\s*([A-Za-z0-9_\-]+)", content, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError(
            f"Unable to find Pinecone API key in {api_keys_path}. "
            "Expected a line like 'pinecone key : <API_KEY>'."
        )

    api_key = m.group(1).strip()
    if not api_key or not api_key.startswith("pcsk_"):
        raise RuntimeError(
            "Pinecone API key found in api_keys.txt but does not look valid "
            "(expected it to start with 'pcsk_')."
        )

    return api_key


def _extract_query_embedding(embedding_chunks: List[Dict[str, Any]]) -> List[float]:
    """Extract the single query embedding vector."""

    if not embedding_chunks:
        raise RuntimeError("Embedding generation returned no vectors for the query.")

    first = embedding_chunks[0]
    embedding = first.get("embedding")
    if embedding is None:
        raise RuntimeError("Embedding generation returned an item without an 'embedding' field.")
    if not isinstance(embedding, list):
        raise RuntimeError("Embedding vector has an unexpected type; expected a list of floats.")

    values: List[float] = []
    for v in embedding:
        values.append(float(v))

    if len(values) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Query embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(values)}"
        )

    return values


def retrieval_node(state: ResearchState) -> ResearchState:
    """LangGraph retrieval node.

    Args:
        state: The current LangGraph ``ResearchState``.

    Returns:
        The updated state including ``state["retrieved_docs"]``.

    Raises:
        ValueError: if the query is empty.
        RuntimeError: for Pinecone/embedding failures.
    """

    start_time = time.perf_counter()

    # Read query from state
    query = (state.get("query") or "").strip()
    logger.info("Retrieval node - incoming query: %r", query)

    if not query:
        err = "Query must not be empty for retrieval."
        logger.error(err)
        state.setdefault("errors", []).append(err)
        raise ValueError(err)

    # Enhance query with conversation history for better follow-up retrieval
    messages = state.get("messages", [])
    if messages and len(messages) > 1:  # More than just the current message
        # Get previous messages for context (excluding current message)
        previous_messages = messages[:-1]  # All messages except current
        recent_messages = previous_messages[-2:]  # Last 2 previous messages for context
        context_parts = []
        for msg in recent_messages:
            # Handle both dict and LangChain message objects
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                # LangChain message object
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
            
            if content and isinstance(content, str) and role in ["user", "assistant", "human", "ai"]:
                # Truncate long messages to avoid embedding issues
                if len(content) > 200:
                    content = content[:200]
                context_parts.append(content)
        
        if context_parts:
            # Combine context with current query for better retrieval
            enhanced_query = " ".join(context_parts + [query])
            logger.info("Retrieval node - enhanced query with conversation history: %r", enhanced_query[:100] + "...")
            query = enhanced_query

    # Config: top-k (configurable via environment variable or state override)
    top_k: int = DEFAULT_TOP_K
    # optional: allow overriding via state for experiments without changing schema
    if isinstance(state.get("retrieval_top_k"), int):
        top_k = cast(int, state["retrieval_top_k"])
    if top_k <= 0:
        top_k = DEFAULT_TOP_K

    # 1) Generate embedding (must match ingestion model/logic)
    logger.info("Retrieval node - generating query embedding")
    try:
        embedder = EmbeddingService()

        # embedder.embed_chunks expects list[dict] contract:
        #   {"chunk_id": str, "text": str, "metadata": dict}
        embedded_query_chunks = embedder.embed_chunks(
            [
                {
                    "chunk_id": "query",
                    "text": query,
                    "metadata": {},
                }
            ]
        )

        query_vector = _extract_query_embedding(embedded_query_chunks)
    except Exception as exc:
        logger.exception("Retrieval node - embedding generation failed")
        err = f"Query embedding failed: {exc}"
        state.setdefault("errors", []).append(err)
        raise RuntimeError(err) from exc

    # 2) Connect to Pinecone
    logger.info("Retrieval node - connecting to Pinecone index=%s", PINECONE_INDEX_NAME)
    matches = []  # Initialize outside try block for scope

    try:
        from pinecone import Pinecone  # type: ignore

        api_key = _read_pinecone_api_key_from_api_keys_file("api_keys.txt")
        pc = Pinecone(api_key=api_key)

        # Verify index exists (control-plane only)
        indexes = pc.list_indexes()
        existing_names = {getattr(i, "name", None) for i in indexes}
        if PINECONE_INDEX_NAME not in existing_names:
            raise RuntimeError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist.")

        index = pc.Index(PINECONE_INDEX_NAME)  # type: ignore[attr-defined]

        logger.info("Retrieval node - pinecone search top_k=%s", top_k)

        # Check if a specific namespace is provided (for targeting recent uploads)
        target_namespace = state.get("target_namespace")
        
        logger.info("Retrieval node - target_namespace: %s", target_namespace)
        
        if target_namespace:
            logger.info("Retrieval node - searching specific namespace: %s", target_namespace)
            try:
                search_response = index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=target_namespace
                )
                matches = getattr(search_response, "matches", None) or []
                logger.info("Retrieval node - found %s matches in target namespace", len(matches))
            except Exception as exc:
                logger.warning("Retrieval node - failed to query target namespace '%s': %s", target_namespace, exc)
                matches = []
        else:
            # First try search without namespace (default namespace)
            search_response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )

            matches = getattr(search_response, "matches", None) or []

            # If no matches in default namespace, try searching all namespaces
            # This is needed because ingestion pipeline uses per-paper namespaces
            if not matches:
                logger.info("Retrieval node - no matches in default namespace, searching all namespaces")
                index_stats = index.describe_index_stats()  # type: ignore[attr-defined]
                namespaces = index_stats.namespaces or {}

                all_matches = []
                try:
                    for ns_name in namespaces.keys():
                        ns_response = index.query(
                            vector=query_vector,
                            top_k=top_k,
                            include_metadata=True,
                            namespace=ns_name
                        )
                        ns_matches = getattr(ns_response, "matches", None) or []
                        if ns_matches:
                            logger.debug("Retrieval node - found %s matches in namespace '%s'", len(ns_matches), ns_name)
                            all_matches.extend(ns_matches)
                except Exception as ns_exc:
                    logger.warning("Retrieval node - failed to query namespaces: %s", ns_exc)

                # If we found matches in namespaces, use those instead
                if all_matches:
                    matches = all_matches
                    logger.info("Retrieval node - total matches across all namespaces: %s", len(matches))
                    # Sort by score across all namespaces
                    matches.sort(key=lambda m: float(getattr(m, "score", 0)), reverse=True)
                    matches = matches[:top_k]  # Keep only top_k overall

    except TimeoutError as exc:
        logger.exception("Retrieval node - Pinecone timeout")
        err = f"Pinecone query timed out: {exc}"
        state.setdefault("errors", []).append(err)
        raise RuntimeError(err) from exc
    except Exception as exc:
        logger.exception("Retrieval node - Pinecone search failed")
        err = f"Pinecone retrieval failed: {exc}"
        state.setdefault("errors", []).append(err)
        raise RuntimeError(err) from exc

    # 3) Parse results
    retrieved_docs: List[RetrievedDoc] = []
    try:
        # Use the matches we collected (either from default namespace or all namespaces)
        for m in matches:
            # Pinecone match shape:
            #   {"id": str, "score": float, "metadata": dict, ...}
            doc_id = cast(str, getattr(m, "id", None) or m.get("id"))
            score = float(getattr(m, "score", None) or m.get("score"))
            metadata_any = getattr(m, "metadata", None)
            metadata: Dict[str, Any] = (
                dict(metadata_any) if isinstance(metadata_any, dict) else m.get("metadata", {})
            )
            text = metadata.get("text") if isinstance(metadata, dict) else None

            retrieved_docs.append(
                {
                    "id": doc_id,
                    "score": score,
                    "metadata": metadata,
                    "text": cast(Optional[str], text if isinstance(text, str) else None),
                }
            )

        # Ensure ordering by similarity (Pinecone generally returns in descending score, but we enforce it)
        retrieved_docs.sort(key=lambda d: d["score"], reverse=True)

    except Exception as exc:
        logger.exception("Retrieval node - failed to parse Pinecone response")
        err = f"Failed to parse Pinecone retrieval results: {exc}"
        state.setdefault("errors", []).append(err)
        raise RuntimeError(err) from exc

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "Retrieval node - retrieved_docs=%s elapsed_ms=%.2f",
        len(retrieved_docs),
        elapsed_ms,
    )

    # 4) Return updated state
    # Create a new dict to ensure LangGraph properly propagates state
    state_out: Dict[str, Any] = dict(state)
    state_out["retrieved_docs"] = retrieved_docs
    return cast(ResearchState, state_out)

