import logging
from typing import Any, Dict, Optional

from research_gap_agent.Ingestion.parser import parse_pdf
from research_gap_agent.Ingestion.semantic_chunker import SemanticChunkerService
from research_gap_agent.Ingestion.embedder import EmbeddingService

logger = logging.getLogger(__name__)


def ingest_pdf(pdf_path: str, timestamp: Optional[int] = None) -> Dict[str, Any]:
    """Ingest a PDF for Phase 1 testing.

    This function parses the PDF into text + basic metadata, then performs
    semantic chunking and embedding generation. It implements the full ingestion
    pipeline: parsing → chunking → embedding.

    Args:
        pdf_path: Path to the PDF file on disk.
        timestamp: Optional timestamp for namespace generation (uses current time if None).

    Returns:
        The parsed document with chunks and embeddings:

        {
            "full_text": str,
            "metadata": {
                "title": str,
                "author": str,
                "page_count": int
            },
            "chunks": list[dict],  # Semantic chunks before embedding
            "chunk_count": int,
            "embedded_chunks": list[dict],  # Chunks with embedding vectors
            "embedded_chunk_count": int
        }

    Raises:
        Any exception raised by parsing, chunking, or embedding
        is logged and re-raised.
    """

    logger.info("Starting ingestion for PDF: %s", pdf_path)

    # Inject deterministic upload metadata so downstream stages can preserve
    # it for namespace generation + vector metadata.
    # Example expected namespace: filename without extension.
    pdf_filename = pdf_path.split("/")[-1].split("\\")[-1]
    pdf_source = "upload"
    
    # Use provided timestamp or current time for consistency
    import time
    ingestion_timestamp = timestamp if timestamp is not None else int(time.time())

    try:
        result: Dict[str, Any] = parse_pdf(pdf_path)
        
        # Ensure the result metadata includes filename for namespace extraction
        if isinstance(result.get("metadata"), dict):
            result["metadata"].setdefault("filename", pdf_filename)
            result["metadata"].setdefault("source", pdf_source)
            result["metadata"]["ingestion_timestamp"] = ingestion_timestamp
        else:
            result["metadata"] = {"filename": pdf_filename, "source": pdf_source, "ingestion_timestamp": ingestion_timestamp}
            
    except Exception:
        logger.exception("Failed to parse PDF during ingestion: %s", pdf_path)
        raise

    # Phase automation: once parsing completes, run semantic chunking immediately.
    # semantic_chunker.py expects input dict keys: {"text": str, "metadata": dict}
    chunker = SemanticChunkerService()
    try:
        base_metadata = (
            result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        )

        # Ensure required metadata keys exist for downstream namespace + vector metadata.
        # Keep any existing keys from parser (title/author/page_count).
        base_metadata = dict(base_metadata)
        base_metadata.setdefault("filename", pdf_filename)
        base_metadata.setdefault("source", pdf_source)

        # Preprocess text to remove references section for better chunking
        full_text = result.get("full_text", "")
        text_for_chunking = full_text
        
        # Remove references section if present (common in academic papers)
        # Only remove if it's at the end to avoid cutting off main content
        if "References" in full_text:
            ref_index = full_text.find("References")
            # Only remove if References appears in the last 25% of the document
            if ref_index > len(full_text) * 0.75:
                text_for_chunking = full_text[:ref_index].strip()
                logger.info(
                    "Removed references section from %s chars to %s chars for chunking",
                    len(full_text),
                    len(text_for_chunking)
                )
            else:
                logger.info(
                    "References section found at position %s (too early, not removing) in document of %s chars",
                    ref_index,
                    len(full_text)
                )
        elif "Bibliography" in full_text:
            bib_index = full_text.find("Bibliography")
            # Only remove if Bibliography appears in the last 25% of the document
            if bib_index > len(full_text) * 0.75:
                text_for_chunking = full_text[:bib_index].strip()
                logger.info(
                    "Removed bibliography section from %s chars to %s chars for chunking",
                    len(full_text),
                    len(text_for_chunking)
                )
            else:
                logger.info(
                    "Bibliography section found at position %s (too early, not removing) in document of %s chars",
                    bib_index,
                    len(full_text)
                )
        
        # Remove administrative/footer content sections
        # These appear at the end and have low relevance for content queries
        # Only remove if they appear in the last 15% of the document
        footer_indicators = [
            "All authors have read and agreed",
            "Funding: This research received",
            "Institutional Review Board Statement",
            "Informed Consent Statement",
            "Data Availability Statement",
            "Conflicts of Interest",
            "Disclaimer/Publisher's Note"
        ]
        
        for indicator in footer_indicators:
            if indicator in text_for_chunking:
                footer_index = text_for_chunking.find(indicator)
                # Only remove if footer content appears in the last 15% of the document
                if footer_index > len(text_for_chunking) * 0.85:
                    text_before_footer = text_for_chunking[:footer_index].strip()
                    logger.info(
                        "Removed footer content from %s chars to %s chars for chunking",
                        len(text_for_chunking),
                        len(text_before_footer)
                    )
                    text_for_chunking = text_before_footer
                    break
                else:
                    logger.info(
                        "Footer indicator '%s' found at position %s (too early, not removing) in document of %s chars",
                        indicator[:30],
                        footer_index,
                        len(text_for_chunking)
                    )
        
        # Extract and prioritize Abstract and Introduction for better generic query handling
        # This ensures "what is this paper about" type questions work better
        abstract_text = ""
        main_content_text = text_for_chunking
        
        if "Abstract" in text_for_chunking:
            abstract_start = text_for_chunking.find("Abstract")
            abstract_end = abstract_start + len("Abstract")
            # Find the end of abstract (usually before Introduction or Keywords)
            intro_start = text_for_chunking.find("Introduction")
            keywords_start = text_for_chunking.find("Keywords")
            
            # Only extract abstract if Introduction comes after it
            if intro_start != -1 and intro_start > abstract_end:
                # Extract text between Abstract and Introduction
                abstract_text = text_for_chunking[abstract_end:intro_start].strip()
                # Keep the full text for chunking, just mark where abstract is
                main_content_text = text_for_chunking
                logger.info(
                    "Extracted abstract (%s chars) before Introduction at position %s",
                    len(abstract_text),
                    intro_start
                )
            elif keywords_start != -1 and keywords_start > abstract_end:
                # Extract text between Abstract and Keywords
                abstract_text = text_for_chunking[abstract_end:keywords_start].strip()
                main_content_text = text_for_chunking
                logger.info(
                    "Extracted abstract (%s chars) before Keywords at position %s",
                    len(abstract_text),
                    keywords_start
                )
            else:
                # Fallback: take first 1500 chars after Abstract for summary
                abstract_end_limit = min(abstract_end + 1500, len(text_for_chunking))
                abstract_text = text_for_chunking[abstract_end:abstract_end_limit].strip()
                main_content_text = text_for_chunking
                logger.info(
                    "Extracted partial abstract (%s chars) using fallback method",
                    len(abstract_text)
                )
        
        # Create a dedicated summary chunk for generic queries
        # This will be processed separately to ensure better matching
        summary_chunk = ""
        if abstract_text:
            # Add title if available
            title = result.get("metadata", {}).get("title", "")
            if title:
                summary_chunk = f"Title: {title}\n\n"
            summary_chunk += f"Abstract: {abstract_text}"
            # Add first sentence of introduction for context
            if "Introduction" in main_content_text:
                intro_start = main_content_text.find("Introduction")
                intro_first_sentence = main_content_text[intro_start:intro_start+300].split('.')[0] + "."
                summary_chunk += f"\n\nIntroduction: {intro_first_sentence}"
        
        # Use the full main content for chunking, optionally prepend summary
        # The key fix: don't replace main content with just abstract/intro
        text_for_chunking = main_content_text
        if summary_chunk:
            # Prepend summary for better retrieval, but keep full content
            text_for_chunking = summary_chunk + "\n\n" + main_content_text
            logger.info(
                "Prepended summary chunk (%s chars) to full content (%s chars) for better generic query handling",
                len(summary_chunk),
                len(main_content_text)
            )

        parsed_for_chunking: Dict[str, Any] = {
            "text": text_for_chunking,
            "metadata": base_metadata,
        }

        chunk_docs = chunker.chunk_document(parsed_for_chunking)
        logger.info(
            "Semantic chunking completed for PDF: %s (chunks=%s)",
            pdf_path,
            len(chunk_docs),
        )

        # Return semantic chunks so the Streamlit UI can display them.
        result["chunks"] = chunk_docs
        result["chunk_count"] = len(chunk_docs)
    except Exception:
        logger.exception("Semantic chunking failed for PDF: %s", pdf_path)
        raise

    # Phase automation: once chunking completes, run embedding immediately.
    # embedder.py expects input: [{"chunk_id": str, "text": str, "metadata": dict}, ...]
    embedder = EmbeddingService()
    try:
        embedded_chunks = embedder.embed_chunks(chunk_docs)
        logger.info(
            "Embedding completed for PDF: %s (embedded_chunks=%s)",
            pdf_path,
            len(embedded_chunks),
        )

        # Return embedded chunks for UI/debug visibility.
        result["embedded_chunks"] = embedded_chunks
        result["embedded_chunk_count"] = len(embedded_chunks)

        # Phase automation: automatically upsert into Pinecone.
        # Caller should not perform any preprocessing or manual steps.
        try:
            from research_gap_agent.Ingestion.pinecone_upsert import upsert_chunks  # type: ignore

            upsert_stats = upsert_chunks(embedded_chunks, timestamp=ingestion_timestamp)
            logger.info(f"Pinecone upsert completed with stats: {upsert_stats}")
            result["pinecone_upsert_stats"] = upsert_stats
            result["pinecone_uploaded_vectors"] = upsert_stats.get("total_uploaded_vectors")
            result["pinecone_skipped_invalid_vectors"] = upsert_stats.get(
                "skipped_invalid_vectors"
            )
        except Exception:
            # In production you may choose to fail-fast; for robustness we fail-fast here
            # to keep the pipeline consistent and observable.
            logger.exception("Pinecone upsert failed for PDF: %s", pdf_path)
            raise
    except Exception:
        logger.exception("Embedding failed for PDF: %s", pdf_path)
        raise

    # Debug prints (useful during Phase 1 testing)
    print("\n===== PDF PARSED =====")
    metadata = result.get("metadata", {})
    print(metadata)

    full_text = result.get("full_text", "")
    if isinstance(full_text, str):
        print(f"Text Length: {len(full_text)}")
    else:
        print("Text Length: unknown (non-string full_text)")

    chunk_count = result.get("chunk_count")
    if chunk_count is not None:
        print(f"Chunk Count: {chunk_count}")

    embedded_count = result.get("embedded_chunk_count")
    if embedded_count is not None:
        print(f"Embedded Chunk Count: {embedded_count}")

        # Show first embedding info
        embedded_chunks = result.get("embedded_chunks")
        if embedded_chunks and len(embedded_chunks) > 0:
            first_embedded = embedded_chunks[0]
            embedding = first_embedded.get("embedding", [])
            if embedding:
                print(f"First Embedding Dimensions: {len(embedding)}")

    logger.info(
        "Ingestion completed for PDF: %s (pages=%s, text_chars=%s, chunks=%s, embedded=%s)",
        pdf_path,
        metadata.get("page_count"),
        len(full_text) if isinstance(full_text, str) else None,
        result.get("chunk_count"),
        result.get("embedded_chunk_count"),
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-test PDF ingestion pipeline.")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default="data/uploads/Artificial_Endocrine_System (1).pdf",
        help="Path to the PDF to ingest (defaults to the uploaded test PDF).",
    )
    args = parser.parse_args()

    parsed = ingest_pdf(args.pdf_path)
    metadata = parsed.get("metadata", {})
    full_text = parsed.get("full_text", "")
    print("\n===== SELF TEST COMPLETE =====")
    print("pdf_path:", args.pdf_path)
    print("metadata:", metadata)
    print("text_chars:", len(full_text) if isinstance(full_text, str) else "unknown")

