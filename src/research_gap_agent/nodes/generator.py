"""LangGraph generator node for the Research Gap Discovery & Industry Alignment Agent.

This node is responsible for generating the final answer using Retrieval-Augmented Generation (RAG).
It reads the user query and retrieved documents from the LangGraph state, constructs a RAG prompt,
and uses the selected LLM to generate the answer.

Responsibilities (generation-only):
- Read the user's query from the LangGraph state
- Read retrieved documents from the LangGraph state
- Build a clean RAG prompt with context and instructions
- Obtain the LLM using LLMFactory
- Generate the answer using the selected model
- Store the answer in the LangGraph state

This node does not implement:
- Retrieval logic
- Embedding generation
- Pinecone connections
- Routing logic
- Web search
- Research gap analysis
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, List, Optional, cast
from urllib.parse import urlparse

from research_gap_agent.llm.factory import LLMFactory
from research_gap_agent.state.state import ResearchState

logger = logging.getLogger(__name__)

# Default LLM model to use if none is specified
DEFAULT_LLM_MODEL = "oss20b"

# Token limits for different models (increased to provide more context and reduce hallucination)
MODEL_TOKEN_LIMITS = {
    "oss20b": 3000,  # Increased from 2000 to provide more PDF context
    "oss120b": 4000,  # Increased from 3500 to provide more context for advanced reasoning
}

# Approximate token to character ratio (rough estimate for text)
TOKEN_TO_CHAR_RATIO = 4


def _truncate_context_by_tokens(
    context_parts: List[str],
    max_chars: int,
    priority_order: Optional[List[int]] = None
) -> List[str]:
    """Truncate context parts to fit within character limit (proxy for token limit).
    
    Args:
        context_parts: List of context strings (documents or web sources).
        max_chars: Maximum total characters allowed.
        priority_order: Optional list of indices indicating priority order for keeping content.
    
    Returns:
        Truncated list of context parts that fit within the limit.
    """
    if not context_parts:
        return []
    
    # Calculate total characters
    total_chars = sum(len(part) for part in context_parts)
    
    if total_chars <= max_chars:
        return context_parts
    
    logger.info(
        "Context truncation needed: total_chars=%d, max_chars=%d, parts=%d",
        total_chars, max_chars, len(context_parts)
    )
    
    # If priority order is provided, sort by priority
    if priority_order:
        # Create (index, part) pairs and sort by priority
        indexed_parts = list(enumerate(context_parts))
        indexed_parts.sort(key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else len(priority_order))
        sorted_parts = [part for _, part in indexed_parts]
    else:
        sorted_parts = context_parts
    
    # Truncate from the end (least priority first)
    truncated_parts = []
    current_chars = 0
    
    for part in reversed(sorted_parts):
        part_chars = len(part)
        if current_chars + part_chars <= max_chars:
            truncated_parts.insert(0, part)
            current_chars += part_chars
        else:
            # Try to add a truncated version
            remaining_chars = max_chars - current_chars
            if remaining_chars > 100:  # Only add if we can keep meaningful content
                truncated_part = part[:remaining_chars] + "... [truncated]"
                truncated_parts.insert(0, truncated_part)
                current_chars += len(truncated_part)
            break
    
    logger.info(
        "Context truncated from %d to %d parts, %d to %d chars",
        len(context_parts), len(truncated_parts), total_chars, current_chars
    )
    
    return truncated_parts


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text using character ratio.
    
    Args:
        text: Input text.
    
    Returns:
        Estimated token count.
    """
    return len(text) // TOKEN_TO_CHAR_RATIO


def _is_research_gap_query(query: str) -> bool:
    """Check if the query is asking about research gaps."""
    gap_keywords = [
        "research gap", "gaps in research", "research gaps", "knowledge gap",
        "future research", "future work", "limitations", "study limitations",
        "what is missing", "what remains", "open questions", "unexplored",
        "research opportunities", "research needs", "further research",
        "areas for future", "potential research", "research directions",
        "gap analysis", "identify gaps", "find gaps", "discover gaps"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in gap_keywords)


def _build_rag_prompt(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    messages: Optional[List[Any]] = None,
    web_search_results: Optional[Dict[str, Any]] = None,
    model_name: str = DEFAULT_LLM_MODEL
) -> str:
    """Build a production-quality RAG prompt with conversation history and web search results.

    Args:
        query: The user's original question.
        retrieved_docs: List of retrieved document chunks.
        messages: Optional conversation history for follow-up questions.
        web_search_results: Optional web search results from Tavily + Jina AI.
        model_name: The model being used to determine token limits.

    Returns:
        A formatted RAG prompt string.
    """
    # Get token limit for the model
    token_limit = MODEL_TOKEN_LIMITS.get(model_name, MODEL_TOKEN_LIMITS[DEFAULT_LLM_MODEL])
    max_context_chars = token_limit * TOKEN_TO_CHAR_RATIO
    
    # Reserve space for query, instructions, conversation history, and overhead
    # The prompt template itself adds ~1000-1500 tokens with instructions
    max_context_chars -= 4000  # Reduced reservation to allow more context
    
    logger.info(
        "Building RAG prompt for model %s with token_limit=%d, max_context_chars=%d",
        model_name, token_limit, max_context_chars
    )
    
    # Limit the number of documents to include (prioritize PDF content with more context)
    max_pdf_docs = 8  # Increased from 5 to provide more context and reduce hallucination
    max_web_sources = 3  # Increased from 2 to provide more web context when available
    
    # Format retrieved documents as context (with full content)
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:max_pdf_docs], 1):
        doc_text = doc.get("text", "")
        if doc_text:
            # Increase PDF document length limit to provide more context
            max_doc_chars = 3000  # Increased from 2000 to provide more context
            if len(doc_text) > max_doc_chars:
                doc_text = doc_text[:max_doc_chars] + "... [truncated]"
            context_parts.append(f"Document {i}:\n{doc_text}")
    
    logger.info("Included %d PDF documents (max %d)", len(context_parts), max_pdf_docs)

    # Add web search context if available (limited)
    web_context_parts = []
    if web_search_results and web_search_results.get("web_search_performed"):
        # Use the new top-k chunks format
        top_k_chunks = web_search_results.get("top_k_chunks", [])
        
        if top_k_chunks:
            logger.info("Using top-k chunks format: %d chunks available", len(top_k_chunks))
            for chunk_data in top_k_chunks[:max_web_sources]:
                chunk_number = chunk_data.get("chunk_number", 0)
                chunk_text = chunk_data.get("text", "")
                url = chunk_data.get("url", "Unknown URL")
                relevance_score = chunk_data.get("relevance_score", 0.0)
                
                if chunk_text:
                    # Extract domain name for display
                    try:
                        domain = urlparse(url).netloc
                        if domain.startswith('www.'):
                            domain = domain[4:]  # Remove www. prefix
                        source_name = domain
                    except:
                        source_name = "Unknown Source"
                    
                    # Limit individual chunk length to provide more context
                    max_web_chars = 2000  # Increased from 1500 to provide more web context
                    if len(chunk_text) > max_web_chars:
                        chunk_text = chunk_text[:max_web_chars] + "... [truncated]"
                    
                    # Make website name more prominent with clear formatting
                    web_context_parts.append(
                        f"=== WEB SOURCE {chunk_number}: {source_name.upper()} ===\n"
                        f"URL: {url}\n"
                        f"Relevance Score: {relevance_score:.2f}\n"
                        f"Content:\n{chunk_text}"
                    )
        else:
            # Fallback to old format if top_k_chunks not available
            logger.info("top_k_chunks not available, using fallback extraction format")
            jina_extraction = web_search_results.get("jina_extraction", {})
            extracted_content = jina_extraction.get("extracted_content", [])
            
            # If Jina extraction failed, try Tavily's fallback answer
            if not extracted_content and web_search_results.get("fallback_answer"):
                logger.info("Using Tavily's AI answer as fallback for web context")
                fallback_answer = web_search_results["fallback_answer"]
                max_fallback_chars = 2000
                if len(fallback_answer) > max_fallback_chars:
                    fallback_answer = fallback_answer[:max_fallback_chars] + "... [truncated]"
                web_context_parts.append(f"Web Search Summary:\n{fallback_answer}")
            elif extracted_content:
                for i, content in enumerate(extracted_content[:max_web_sources], 1):
                    url = content.get("url", "Unknown URL")
                    key_info = content.get("key_information", "")
                    status = content.get("status", "")
                    
                    if status == "success" and key_info:
                        try:
                            domain = urlparse(url).netloc
                            if domain.startswith('www.'):
                                domain = domain[4:]
                            source_name = domain
                        except:
                            source_name = "Unknown Source"
                        
                        max_web_chars = 2000  # Increased from 1500 to provide more web context
                        if len(key_info) > max_web_chars:
                            key_info = key_info[:max_web_chars] + "... [truncated]"
                        # Use same prominent format for fallback
                        web_context_parts.append(
                            f"=== WEB SOURCE {i}: {source_name.upper()} ===\n"
                            f"URL: {url}\n"
                            f"Content:\n{key_info}"
                        )
    
    logger.info("Included %d web sources (max %d)", len(web_context_parts), max_web_sources)

    # Combine all context parts (PDF docs first, then web sources)
    all_context_parts = context_parts + web_context_parts
    
    # Calculate total context size and log it
    total_context_chars = sum(len(part) for part in all_context_parts)
    logger.info("Total context size: %d chars (%d parts)", total_context_chars, len(all_context_parts))
    
    # If context is still too large, apply additional truncation
    if total_context_chars > max_context_chars and max_context_chars > 0:
        logger.warning(
            "Context %d chars exceeds limit %d, applying additional truncation",
            total_context_chars, max_context_chars
        )
        all_context_parts = _truncate_context_by_tokens(all_context_parts, max_context_chars)
        total_context_chars = sum(len(part) for part in all_context_parts)
        logger.info("After truncation: %d chars (%d parts)", total_context_chars, len(all_context_parts))
    
    # Build final context string
    if all_context_parts:
        full_context = "\n\n".join(all_context_parts)
        
        # Add web search delimiter if web sources were included
        if web_context_parts:
            # Add delimiter at the right position
            if context_parts:
                pdf_context = "\n\n".join(context_parts)
                web_context = "\n\n".join(web_context_parts)
                full_context = f"{pdf_context}\n\n=== WEB SEARCH RESULTS ===\n{web_context}\n=== END WEB SEARCH RESULTS ==="
    else:
        full_context = "No relevant documents were retrieved from the PDF."

    # Build conversation history for follow-up questions (limited to prevent token overflow)
    conversation_history = ""
    if messages and len(messages) > 1:  # More than just the current message
        conversation_parts = []
        # Only include the last 1 previous message to save tokens
        recent_messages = messages[-2:-1]  # Get only the immediate previous message
        for msg in recent_messages:
            # Handle both dict and LangChain message objects
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                # LangChain message object
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
            
            if content and isinstance(content, str):
                # Aggressively truncate long messages
                if len(content) > 200:
                    content = content[:200] + "... [truncated]"
                conversation_parts.append(f"{role.capitalize()}: {content}")
        
        if conversation_parts:
            conversation_history = f"\n\nConversation History:\n" + "\n".join(conversation_parts)

    # Build the RAG prompt with clear instructions
    # Determine if web search was performed based on context content
    web_search_performed = "=== WEB SOURCE" in full_context or "=== WEB SEARCH RESULTS ===" in full_context
    
    # Check if this is a research gap query
    is_gap_query = _is_research_gap_query(query)
    
    # Log context size before building prompt
    logger.info("Context size: %d chars, web_search: %s, is_gap_query: %s", len(full_context), web_search_performed, is_gap_query)
    
    if web_search_performed:
        if is_gap_query:
            # Specialized prompt for research gap discovery queries
            prompt = f"""You are a research gap discovery assistant that analyzes PDF documents and web search results to identify research gaps and future research directions.

User Question:
{query}{conversation_history}

Context from Retrieved Documents:
{full_context}

Instructions:
1. ANALYZE BOTH SOURCES: Combine information from PDF documents and web search to identify research gaps.
2. PDF documents provide the foundation - analyze what has been studied, methodologies used, and findings reported.
3. Web search results provide broader context - use them to identify what's missing, what others are researching, and emerging trends.
4. IDENTIFY SPECIFIC GAPS: Look for:
   - Methodological limitations mentioned in the PDF
   - Populations or conditions not studied
   - Technologies or approaches not explored
   - Contradictory findings that need resolution
   - Emerging areas mentioned in web search but not in PDF
5. SYNTHESIZE GAPS: Combine PDF limitations with web search insights to propose specific, actionable research gaps.
6. PRIORITIZE GAPS: Focus on the most significant and researchable gaps based on both sources.
7. **IMPORTANT - ALWAYS CITE WEBSITES**: When using information from web sources, you MUST prominently mention the website name. Use formats like:
   - "According to research from NATURE.COM..."
   - "Studies from SCIENCEDAILY.COM indicate..."
   - "As reported on NATIONALACADEMIES.ORG..."
   - Always include the full website name in uppercase when citing web sources.
8. When using information from PDF documents, cite the document number (e.g., "According to Document 1...").
9. **CRITICAL - DO NOT HALLUCINATE GAPS**: Do not invent research gaps or limitations that are not explicitly mentioned or clearly implied by the provided context. Only identify gaps supported by evidence from the PDF or web search results.
10. Be specific and evidence-based in your gap identification. Avoid speculative or vague gap suggestions.
11. Provide specific, actionable research directions that could realistically be pursued based on the actual limitations and findings in the context.

Answer:"""
        else:
            # Standard prompt when web search was performed for non-gap queries
            prompt = f"""You are a helpful research assistant that answers questions based on the provided context from both PDF documents and web search.

User Question:
{query}{conversation_history}

Context from Retrieved Documents:
{full_context}

Instructions:
1. PRIORITIZE PDF DOCUMENTS: First and foremost, answer using information from the PDF documents (Document 1, Document 2, etc.).
2. Use web search results ONLY to supplement or update information that is not available in the PDF documents.
3. Web search results are provided as ranked chunks with relevance scores - higher scores indicate better relevance to your question.
4. **IMPORTANT - ALWAYS CITE WEBSITES**: When using information from web sources, you MUST prominently mention the website name. Use formats like:
   - "According to GITHUB.COM..."
   - "As reported on STACKOVERFLOW.COM..."
   - "Information from WIKIPEDIA.ORG suggests..."
   - Always include the full website name in uppercase when citing web sources.
5. When using information from PDF documents, cite the document number (e.g., "According to Document 1...").
6. If there are differences between PDF and web information, prioritize the PDF content as it's the uploaded document, but acknowledge web context.
7. If the answer cannot be found in either the PDF or web search context, clearly state that the information is not available in the provided context.
8. **CRITICAL - DO NOT HALLUCINATE**: Do not invent, fabricate, or add information that is not present in the provided context. If you don't know the answer from the context, say so explicitly.
9. Be specific and precise in your answers. Avoid vague or speculative statements unless supported by the context.
10. For follow-up questions, consider the conversation history to understand the context.
11. Provide a comprehensive answer that prioritizes PDF content and uses web search only as supplementary information.

Answer:"""
    else:
        # Standard prompt when only PDF context is available
        prompt = f"""You are a helpful research assistant that answers questions based on the provided context.

User Question:
{query}{conversation_history}

Context from Retrieved Documents:
{full_context}

Instructions:
1. Answer the user's question using ONLY the information provided in the context above.
2. If the answer cannot be found in the provided context, clearly state that the information is not available in the retrieved documents.
3. **CRITICAL - DO NOT HALLUCINATE**: Do not invent, fabricate, or add information that is not present in the context. If you don't know the answer from the context, say so explicitly.
4. Be specific and precise in your answers. Avoid vague or speculative statements unless supported by the context.
5. If the context contains relevant information, provide a clear and concise answer.
6. If the context contains conflicting information, acknowledge the conflict and present both perspectives.
7. Cite the specific document numbers when referencing information (e.g., "According to Document 1...").
8. For follow-up questions, consider the conversation history to understand the context, but still base your answer primarily on the retrieved documents.

Answer:"""

    # Log final prompt size for monitoring
    estimated_tokens = _estimate_tokens(prompt)
    logger.info(
        "Final RAG prompt: estimated_tokens=%d, characters=%d, model=%s, limit=%d",
        estimated_tokens, len(prompt), model_name, token_limit
    )
    
    if estimated_tokens > token_limit:
        logger.error(
            "Prompt exceeds token limit: estimated_tokens=%d, limit=%d. "
            "This WILL cause API errors. Consider reducing context further.",
            estimated_tokens, token_limit
        )
        # As a last resort, truncate the entire prompt if it's way too large
        if estimated_tokens > token_limit * 1.5:
            logger.warning("Prompt is severely oversized, applying emergency truncation")
            # Truncate context section only
            if "Context from Retrieved Documents:" in prompt:
                parts = prompt.split("Context from Retrieved Documents:")
                if len(parts) == 2:
                    header = parts[0]
                    context_section = parts[1]
                    # Keep only first 30% of context
                    context_limit = len(context_section) // 3
                    truncated_context = context_section[:context_limit] + "\n\n... [emergency truncation applied]"
                    prompt = header + "Context from Retrieved Documents:" + truncated_context
                    estimated_tokens = _estimate_tokens(prompt)
                    logger.warning(
                        "Emergency truncation applied: new estimated_tokens=%d",
                        estimated_tokens
                    )

    return prompt


def generator_node(state: ResearchState) -> ResearchState:
    """LangGraph generator node for RAG-based answer generation.

    Args:
        state: The current LangGraph ResearchState.

    Returns:
        The updated state including state["answer"].

    Raises:
        ValueError: If the query is empty or retrieved_docs is missing/empty.
        RuntimeError: For LLM initialization or generation failures.
    """
    start_time = time.perf_counter()

    # Read query from state
    query = (state.get("query") or "").strip()
    logger.info("Generator node - incoming query: %r", query)

    if not query:
        err = "Query must not be empty for generation."
        logger.error(err)
        state.setdefault("errors", []).append(err)
        raise ValueError(err)

    # Read retrieved documents from state
    retrieved_docs = state.get("retrieved_docs")
    if retrieved_docs is None:
        err = "Retrieved documents not found in state. Retrieval node must run before generator."
        logger.warning(err)
        state.setdefault("errors", []).append(err)
        # Initialize empty list to allow generator to proceed
        retrieved_docs = []

    if not isinstance(retrieved_docs, list):
        err = f"Retrieved documents must be a list, got {type(retrieved_docs).__name__}."
        logger.error(err)
        state.setdefault("errors", []).append(err)
        raise ValueError(err)

    logger.info("Generator node - number of retrieved documents: %d", len(retrieved_docs))

    if not retrieved_docs:
        err = "No retrieved documents available for generation."
        logger.warning(err)
        state.setdefault("errors", []).append(err)
        # We'll still proceed with empty context rather than raising an error
        # This allows the LLM to respond that no information is available

    # Select LLM model
    selected_llm = state.get("selected_llm")
    if selected_llm:
        model_name = selected_llm
        logger.info("Generator node - using selected LLM from state: %s", model_name)
    else:
        model_name = DEFAULT_LLM_MODEL
        logger.info("Generator node - no selected LLM in state, using default: %s", model_name)

    # Build RAG prompt with conversation history and web search results
    logger.info("Generator node - building RAG prompt")
    try:
        messages = state.get("messages", [])
        web_search_results = state.get("web_search_results")
        rag_prompt = _build_rag_prompt(query, retrieved_docs, messages, web_search_results, model_name)
        logger.debug("Generator node - RAG prompt length: %d characters", len(rag_prompt))
    except Exception as exc:
        logger.exception("Generator node - failed to build RAG prompt")
        err = f"Failed to build RAG prompt: {exc}"
        state.setdefault("errors", []).append(err)
        raise RuntimeError(err) from exc

    # Obtain LLM instance
    logger.info("Generator node - obtaining LLM instance via factory")
    try:
        factory = LLMFactory()
        llm = factory.get_llm(model_name)
        logger.info("Generator node - successfully obtained LLM: %s", llm.model_name)
    except Exception as exc:
        logger.exception("Generator node - failed to obtain LLM instance")
        err = f"Failed to obtain LLM '{model_name}': {exc}"
        state.setdefault("errors", []).append(err)
        raise RuntimeError(err) from exc

    # Generate answer
    logger.info("Generator node - generating answer with LLM")
    try:
        answer = llm.generate(rag_prompt)
        logger.info(
            "Generator node - answer generated, length: %d characters", len(answer)
        )
    except TimeoutError as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error("Generator node - LLM generation timeout after %.2f ms", elapsed_ms)
        err = f"LLM generation timed out: {exc}"
        state.setdefault("errors", []).append(err)
        raise TimeoutError(err) from exc
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.exception("Generator node - LLM generation failed after %.2f ms", elapsed_ms)
        err = f"LLM generation failed: {exc}"
        state.setdefault("errors", []).append(err)
        raise RuntimeError(err) from exc

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "Generator node - completed successfully, elapsed_ms: %.2f", elapsed_ms
    )

    # Store answer in state
    state_out: Dict[str, Any] = dict(state)
    state_out["answer"] = answer

    return cast(ResearchState, state_out)


def generator_node_stream(state: ResearchState) -> Iterator[str]:
    """LangGraph generator node for RAG-based answer generation with streaming.

    This is a streaming version that yields answer chunks as they are generated.

    Args:
        state: The current LangGraph ResearchState.

    Yields:
        Answer chunks as they are generated by the LLM.

    Raises:
        ValueError: If the query is empty or retrieved_docs is missing/empty.
        RuntimeError: For LLM initialization or generation failures.
    """
    start_time = time.perf_counter()

    # Read query from state
    query = (state.get("query") or "").strip()
    logger.info("Generator node stream - incoming query: %r", query)

    if not query:
        err = "Query must not be empty for generation."
        logger.error(err)
        raise ValueError(err)

    # Read retrieved documents from state
    retrieved_docs = state.get("retrieved_docs")
    if retrieved_docs is None:
        err = "Retrieved documents not found in state. Retrieval node must run before generator."
        logger.warning(err)
        retrieved_docs = []

    if not isinstance(retrieved_docs, list):
        err = f"Retrieved documents must be a list, got {type(retrieved_docs).__name__}."
        logger.error(err)
        raise ValueError(err)

    logger.info("Generator node stream - number of retrieved documents: %d", len(retrieved_docs))

    # Select LLM model
    selected_llm = state.get("selected_llm")
    if selected_llm:
        model_name = selected_llm
        logger.info("Generator node stream - using selected LLM from state: %s", model_name)
    else:
        model_name = DEFAULT_LLM_MODEL
        logger.info("Generator node stream - no selected LLM in state, using default: %s", model_name)

    # Build RAG prompt with conversation history and web search results
    logger.info("Generator node stream - building RAG prompt")
    try:
        messages = state.get("messages", [])
        web_search_results = state.get("web_search_results")
        rag_prompt = _build_rag_prompt(query, retrieved_docs, messages, web_search_results, model_name)
        logger.debug("Generator node stream - RAG prompt length: %d characters", len(rag_prompt))
    except Exception as exc:
        logger.exception("Generator node stream - failed to build RAG prompt")
        err = f"Failed to build RAG prompt: {exc}"
        raise RuntimeError(err) from exc

    # Obtain LLM instance
    logger.info("Generator node stream - obtaining LLM instance via factory")
    try:
        factory = LLMFactory()
        llm = factory.get_llm(model_name)
        logger.info("Generator node stream - successfully obtained LLM: %s", llm.model_name)
    except Exception as exc:
        logger.exception("Generator node stream - failed to obtain LLM instance")
        err = f"Failed to obtain LLM '{model_name}': {exc}"
        raise RuntimeError(err) from exc

    # Generate answer with streaming
    logger.info("Generator node stream - generating answer with LLM streaming")
    try:
        # Use the streaming method
        for chunk in llm.generate_stream(rag_prompt):
            yield chunk
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Generator node stream - streaming completed, elapsed_ms: %.2f",
            elapsed_ms,
        )
    except TimeoutError as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error("Generator node stream - LLM generation timeout after %.2f ms", elapsed_ms)
        err = f"LLM generation timed out: {exc}"
        raise TimeoutError(err) from exc
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.exception(
            "Generator node stream - LLM generation failed after %.2f ms",
            elapsed_ms,
        )
        err = f"LLM generation failed: {exc}"
        raise RuntimeError(err) from exc
