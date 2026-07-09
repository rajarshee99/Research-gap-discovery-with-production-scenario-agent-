"""
LangGraph web search node for the Research Gap Discovery & Industry Alignment Agent.

This node is responsible for performing web search when the PDF context is insufficient,
specifically for the oss120B model. It uses Tavily for web search and Jina AI for content
extraction, then chunks and ranks the content before sending to LLM.

Responsibilities:
- Determine if PDF context is sufficient for answering the query
- If insufficient, perform web search using Tavily
- Extract content from URLs using Jina AI
- Chunk and rank extracted content by relevance
- Store top-k chunks in the LangGraph state for LLM
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, cast

from research_gap_agent.services.tavily_service import TavilyService
from research_gap_agent.services.jina_service import JinaAIService
from research_gap_agent.llm.factory import LLMFactory
from research_gap_agent.state.state import ResearchState

logger = logging.getLogger(__name__)

# Thresholds for determining context sufficiency
MIN_RETRIEVED_DOCS = 1  # Reduced from 2 - even 1 good doc should be sufficient
MIN_AVG_SCORE = 0.3  # Reduced from 0.5 - lower threshold to prioritize PDF content
MIN_TOTAL_TEXT_LENGTH = 200  # Reduced from 500 - even small amount of text should be tried first

# Chunking and ranking parameters
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Character overlap between chunks
TOP_K_CHUNKS = 5  # Number of top chunks to send to LLM


def _is_pdf_context_sufficient(
    retrieved_docs: Optional[List[Dict[str, Any]]],
    query: str
) -> bool:
    """
    Determine if the retrieved PDF context is sufficient to answer the query.
    
    Args:
        retrieved_docs: List of retrieved document chunks from Pinecone.
        query: The user's query.
    
    Returns:
        True if PDF context is likely sufficient, False otherwise.
    """
    if not retrieved_docs or len(retrieved_docs) == 0:
        logger.info("PDF context insufficient: No documents retrieved")
        return False
    
    # Check if we have enough documents
    if len(retrieved_docs) < MIN_RETRIEVED_DOCS:
        logger.info(
            "PDF context may be insufficient: Only %d documents retrieved (threshold: %d)",
            len(retrieved_docs),
            MIN_RETRIEVED_DOCS
        )
        return False
    
    # Check average similarity score
    scores = [doc.get("score", 0) for doc in retrieved_docs if doc.get("score") is not None]
    if scores:
        avg_score = sum(scores) / len(scores)
        if avg_score < MIN_AVG_SCORE:
            logger.info(
                "PDF context may be insufficient: Average score %.2f (threshold: %.2f)",
                avg_score,
                MIN_AVG_SCORE
            )
            return False
    
    # Check total text length
    total_text_length = sum(
        len(doc.get("text", "")) for doc in retrieved_docs if doc.get("text")
    )
    if total_text_length < MIN_TOTAL_TEXT_LENGTH:
        logger.info(
            "PDF context may be insufficient: Total text length %d (threshold: %d)",
            total_text_length,
            MIN_TOTAL_TEXT_LENGTH
        )
        return False
    
    logger.info(
        "PDF context appears sufficient: %d docs, avg_score=%.2f, total_text=%d",
        len(retrieved_docs),
        sum(scores) / len(scores) if scores else 0,
        total_text_length
    )
    return True


def _llm_decides_web_search_needed(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    selected_llm: str
) -> bool:
    """
    Use LLM reasoning to determine if web search is needed for answering the query.
    
    This function is specifically designed for oss120b to leverage its reasoning capabilities
    to make an intelligent decision about whether web search would be beneficial.
    
    Args:
        query: The user's query.
        retrieved_docs: List of retrieved document chunks from PDF.
        selected_llm: The selected LLM model.
    
    Returns:
        True if LLM determines web search is needed, False otherwise.
    """
    # Only use LLM decision for oss120b (the model with reasoning capabilities)
    if selected_llm != "oss120b":
        logger.info("LLM decision - not using oss120b, skipping LLM-based decision")
        return False
    
    try:
        # Prepare context for the LLM decision with more comprehensive information
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Use top 5 docs for better context
            doc_text = doc.get("text", "")
            score = doc.get("score", 0.0)
            if doc_text:
                # Provide more context with similarity scores
                if len(doc_text) > 800:
                    doc_text = doc_text[:800] + "... [truncated]"
                context_parts.append(f"Document {i} (relevance: {score:.2f}): {doc_text}")
        
        context_summary = "\n\n".join(context_parts) if context_parts else "No documents retrieved."
        
        # Build decision prompt for the LLM with clearer instructions
        decision_prompt = f"""You are an intelligent research assistant that determines whether web search is needed to answer a user's question accurately.

User Question: {query}

Available PDF Context:
{context_summary}

Your task: Determine if web search is necessary to provide a complete and accurate answer.

Consider these factors:
1. Does the PDF context contain sufficient and relevant information to answer the question comprehensively?
2. Is the question asking for current/recent information that might not be in the PDF (e.g., recent developments, current status)?
3. Is the question asking for information outside the scope of the PDF (e.g., comparisons with other research, external context)?
4. Would web search provide valuable supplementary information that would significantly improve the answer quality?
5. Are the retrieved documents highly relevant (high relevance scores) to the question?

IMPORTANT: Only recommend web search if the PDF context is genuinely insufficient or the question explicitly requires current/external information. If the PDF contains relevant information, prefer using it to avoid potential hallucination from web sources.

Answer with EXACTLY one word: "YES" if web search is needed, "NO" if it is not needed.

Answer:"""

        # Get LLM instance (using oss120b for reasoning)
        factory = LLMFactory()
        llm = factory.get_llm("oss120b")
        
        # Get LLM decision
        decision = llm.generate(decision_prompt).strip().upper()
        
        logger.info("LLM decision result: %s", decision)
        
        # Parse the decision
        if decision == "YES":
            logger.info("LLM decision - web search is needed")
            return True
        elif decision == "NO":
            logger.info("LLM decision - web search is not needed")
            return False
        else:
            # If LLM doesn't give a clear answer, be conservative and don't search
            logger.warning("LLM decision - unclear response '%s', defaulting to NO web search", decision)
            return False
            
    except Exception as exc:
        logger.exception("LLM decision - failed, defaulting to NO web search")
        # On error, be conservative and don't perform web search
        return False


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for better ranking and processing.
    
    Args:
        text: Input text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between chunks.
    
    Returns:
        List of text chunks.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        
        # Try to break at sentence boundaries for better readability
        if end < text_len:
            # Look for sentence endings in the last 20% of the chunk
            break_point = chunk.rfind('.')
            if break_point > chunk_size * 0.8:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap if end < text_len else end
    
    # Remove empty chunks
    chunks = [c for c in chunks if c and len(c) > 50]
    
    return chunks


def _rank_chunks_by_relevance(
    chunks: List[str],
    query: str,
    urls: List[str]
) -> List[Dict[str, Any]]:
    """
    Rank chunks by relevance to the query using simple keyword matching.
    
    Args:
        chunks: List of text chunks.
        query: User's query for relevance scoring.
        urls: Corresponding URLs for each chunk.
    
    Returns:
        List of ranked chunks with metadata.
    """
    if not chunks:
        return []
    
    # Extract keywords from query (simple approach)
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
                 'below', 'between', 'under', 'again', 'further', 'then', 'once',
                 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
                 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
                 'if', 'or', 'because', 'until', 'while', 'although', 'though'}
    
    query_keywords = query_words - stop_words
    
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        
        # Calculate relevance score
        score = 0.0
        
        # Exact phrase matches (higher weight)
        for phrase in query_lower.split():
            if phrase in chunk_lower:
                score += 2.0
        
        # Keyword matches
        chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
        matching_keywords = chunk_words & query_keywords
        score += len(matching_keywords) * 1.5
        
        # Density of keywords (keywords per 100 chars)
        if len(chunk) > 0:
            keyword_density = len(matching_keywords) / (len(chunk) / 100)
            score += keyword_density * 0.5
        
        # Position in original text (earlier chunks might be more relevant)
        score += (len(chunks) - i) * 0.1
        
        scored_chunks.append({
            "chunk": chunk,
            "score": score,
            "url": urls[i] if i < len(urls) else "unknown",
            "chunk_index": i
        })
    
    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info(
        "Ranked %d chunks, top score: %.2f, bottom score: %.2f",
        len(scored_chunks),
        scored_chunks[0]["score"] if scored_chunks else 0,
        scored_chunks[-1]["score"] if scored_chunks else 0
    )
    
    return scored_chunks


def web_search_node(state: ResearchState) -> ResearchState:
    """
    LangGraph web search node for conditional web search and content extraction.
    
    This node:
    1. Checks if PDF context is sufficient for answering the query
    2. If insufficient and using oss120B model, performs web search via Tavily
    3. Extracts key information from search results using Jina AI
    4. Stores results in state for use by the generator node
    
    Args:
        state: The current LangGraph ResearchState.
    
    Returns:
        The updated state including web search results (if performed).
    
    Raises:
        RuntimeError: For Tavily or Jina AI service failures.
    """
    start_time = time.perf_counter()
    
    # Read query and model selection from state
    query = (state.get("query") or "").strip()
    selected_llm = state.get("selected_llm", "")
    retrieved_docs = state.get("retrieved_docs", [])
    
    logger.info(
        "Web search node - query: %r, selected_llm: %s, retrieved_docs: %d",
        query,
        selected_llm,
        len(retrieved_docs) if retrieved_docs else 0
    )
    
    # Initialize web search state
    state_out: Dict[str, Any] = dict(state)
    state_out["web_search_performed"] = False
    state_out["web_search_results"] = None
    state_out["web_search_query"] = None
    state_out["pdf_context_sufficient"] = False
    
    # Only perform web search for oss120B model
    if selected_llm != "oss120b":
        logger.info("Web search node - skipping (not using oss120b model)")
        state_out["pdf_context_sufficient"] = True  # Assume sufficient for other models
        return cast(ResearchState, state_out)
    
    # For oss120b, use LLM reasoning to decide if web search is needed
    # For other models, use simple PDF context sufficiency check
    if selected_llm == "oss120b":
        logger.info("Web search node - using LLM reasoning to decide if web search is needed")
        should_search = _llm_decides_web_search_needed(query, retrieved_docs, selected_llm)
        state_out["pdf_context_sufficient"] = not should_search  # Mark as sufficient if not searching
    else:
        # For other models, use simple heuristic
        is_sufficient = _is_pdf_context_sufficient(retrieved_docs, query)
        state_out["pdf_context_sufficient"] = is_sufficient
        should_search = not is_sufficient
    
    if not should_search:
        logger.info("Web search node - skipping web search (determined not needed)")
        state_out["web_search_performed"] = False
        state_out["web_search_results"] = None
        return cast(ResearchState, state_out)
    
    logger.info("Web search node - web search needed, performing search to supplement")
    
    try:
        # Step 1: Perform web search using Tavily
        tavily_service = TavilyService()
        
        # Use the original query without enhancement for consistent behavior
        search_results = tavily_service.search(
            query=query,
            max_results=5,  # Keep at 5 for quality over quantity
            search_depth="advanced"  # Use advanced search for better results
        )
        
        logger.info(
            "Web search node - Tavily search completed: %d results",
            search_results.get("total_results", 0)
        )
        
        # Step 2: Extract content using Jina AI (always use Jina as per new pipeline)
        if search_results.get("results"):
            logger.info("Step 2: Using Jina AI to read content from Tavily-found URLs")
            jina_service = JinaAIService()
            jina_results = jina_service.extract_from_search_results(
                search_results=search_results["results"],
                query=query,
                max_results=5,
                max_length=8000  # Increased to get more content for chunking
            )
            extracted_content = jina_results.get("extracted_content", [])
            
            logger.info(
                "Web search node - Content extraction completed: %d URLs processed",
                len(extracted_content)
            )
            
            # Step 3: Chunk the extracted content from all URLs
            all_chunks = []
            all_urls = []
            
            for content_item in extracted_content:
                url = content_item.get("url", "")
                key_info = content_item.get("key_information", "")
                status = content_item.get("status", "")
                
                if status == "success" and key_info:
                    # Chunk the content
                    chunks = _chunk_text(key_info, CHUNK_SIZE, CHUNK_OVERLAP)
                    logger.info("Chunked content from %s: %d chunks", url, len(chunks))
                    
                    # Track which URL each chunk came from
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_urls.append(url)
            
            logger.info("Total chunks from all URLs: %d", len(all_chunks))
            
            # Step 4: Rank chunks by relevance to query
            if all_chunks:
                ranked_chunks = _rank_chunks_by_relevance(all_chunks, query, all_urls)
                
                # Step 5: Select top-k chunks
                top_k_chunks = ranked_chunks[:TOP_K_CHUNKS]
                logger.info("Selected top %d chunks for LLM", len(top_k_chunks))
                
                # Format top-k chunks for LLM
                formatted_chunks = []
                for i, chunk_data in enumerate(top_k_chunks, 1):
                    formatted_chunks.append({
                        "chunk_number": i,
                        "text": chunk_data["chunk"],
                        "url": chunk_data["url"],
                        "relevance_score": chunk_data["score"]
                    })
            else:
                formatted_chunks = []
                logger.warning("No chunks available for ranking")
            
            # Store combined results with top-k chunks
            web_search_results = {
                "tavily_search": search_results,
                "jina_extraction": {
                    "extracted_content": extracted_content,
                    "total_extracted": len(extracted_content),
                    "failed_urls": []
                },
                "chunking": {
                    "total_chunks": len(all_chunks),
                    "top_k_chunks": TOP_K_CHUNKS,
                    "selected_chunks": len(formatted_chunks)
                },
                "top_k_chunks": formatted_chunks,  # This is what the LLM will use
                "performed_at": time.time(),
                "web_search_performed": True,
            }
            
            # If extraction failed but Tavily has an answer, use that as fallback
            if len(extracted_content) == 0 and search_results.get("answer"):
                logger.info("Content extraction failed, using Tavily's AI answer as fallback")
                web_search_results["fallback_answer"] = search_results["answer"]
            
            state_out["web_search_performed"] = True
            state_out["web_search_results"] = web_search_results
            state_out["web_search_query"] = query
            
            logger.info("Web search node - completed successfully")
        else:
            logger.warning("Web search node - No search results from Tavily")
            state_out["web_search_performed"] = False
            state_out["web_search_results"] = None
    
    except Exception as exc:
        logger.exception("Web search node - failed")
        error_msg = f"Web search failed: {exc}"
        state_out.setdefault("errors", []).append(error_msg)
        # Don't raise - allow the pipeline to continue with PDF context only
        state_out["web_search_performed"] = False
        state_out["web_search_results"] = None
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "Web search node - completed in %.2f ms, web_search_performed=%s",
        elapsed_ms,
        state_out["web_search_performed"]
    )
    
    return cast(ResearchState, state_out)


if __name__ == "__main__":
    # Self-test
    logging.basicConfig(level=logging.INFO)
    
    # Test with insufficient context
    test_state: ResearchState = {
        "messages": [],
        "query": "What are the latest developments in quantum computing?",
        "selected_llm": "oss120b",
        "target_namespace": None,
        "uploaded_files": [],
        "file_payloads": [],
        "parsed_documents": [],
        "vector_store_path": None,
        "metadata": {},
        "analysis_results": {},
        "cross_paper_gaps": [],
        "industry_gap_alignment": {},
        "current_node": "",
        "errors": [],
        "retrieved_docs": [],  # Empty to trigger web search
        "answer": None,
        "web_search_performed": False,
        "web_search_results": None,
        "web_search_query": None,
        "pdf_context_sufficient": False,
    }
    
    print("Testing web search node with insufficient PDF context...")
    try:
        result_state = web_search_node(test_state)
        print(f"\nWeb search performed: {result_state['web_search_performed']}")
        print(f"PDF context sufficient: {result_state['pdf_context_sufficient']}")
        if result_state['web_search_results']:
            tavily_results = result_state['web_search_results']['tavily_search']
            print(f"Tavily results: {tavily_results['total_results']}")
            jina_results = result_state['web_search_results']['jina_extraction']
            print(f"Jina extracted: {jina_results['total_extracted']}")
    except Exception as e:
        print(f"Error: {e}")
