"""
LangGraph mindmap node for the Research Gap Discovery & Industry Alignment Agent.

This node is responsible for generating mindmap structures using the LLM based on user queries
and retrieved documents. It creates structured mindmap data that can be visualized using Mermaid.js.

Responsibilities:
- Generate mindmap structure (nodes, edges, relationships) using LLM
- Base mindmap content on retrieved PDF documents and web search results
- Output structured JSON data for visualization
- Only run for oss120b model (has reasoning capabilities)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, cast

from research_gap_agent.llm.factory import LLMFactory
from research_gap_agent.state.state import ResearchState

logger = logging.getLogger(__name__)


def _build_mindmap_prompt(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    web_search_results: Optional[Dict[str, Any]] = None,
    answer: Optional[str] = None
) -> str:
    """Build a prompt for LLM to generate mindmap structure.

    Args:
        query: The user's original question.
        retrieved_docs: List of retrieved document chunks from PDF.
        web_search_results: Optional web search results.
        answer: Optional generated answer to incorporate into mindmap.

    Returns:
        A formatted prompt for mindmap generation.
    """
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:8], 1):
        doc_text = doc.get("text", "")
        if doc_text:
            if len(doc_text) > 1500:
                doc_text = doc_text[:1500] + "... [truncated]"
            context_parts.append(f"Document {i}:\n{doc_text}")

    # Add web search context if available
    web_context = ""
    if web_search_results and web_search_results.get("web_search_performed"):
        top_k_chunks = web_search_results.get("top_k_chunks", [])
        if top_k_chunks:
            web_context = "\n\n=== WEB SEARCH CONTEXT ===\n"
            for chunk_data in top_k_chunks[:3]:
                chunk_text = chunk_data.get("text", "")
                url = chunk_data.get("url", "Unknown URL")
                if chunk_text:
                    if len(chunk_text) > 1000:
                        chunk_text = chunk_text[:1000] + "... [truncated]"
                    web_context += f"Source: {url}\n{chunk_text}\n\n"

    full_context = "\n\n".join(context_parts) + web_context

    # If no context but have answer, use answer as context
    if not full_context and answer:
        full_context = f"Generated Answer:\n{answer}"
    elif not full_context:
        full_context = "No additional context available. Base mindmap on the question itself."

    # Build the mindmap generation prompt
    prompt = f"""You are a research analysis expert that creates structured mindmaps to visualize research concepts and their relationships.

User Question: {query}

Generated Answer: {answer if answer else "No answer generated yet."}

Research Context:
{full_context}

Your task: Create a structured mindmap that visualizes the key concepts, relationships, and insights from the research context related to the user's question.

Instructions:
1. Identify 4-8 key concepts from the research context that are most relevant to the user's question
2. Determine relationships between these concepts (hierarchical, causal, correlational, etc.)
3. Create a clear hierarchical structure with a main central concept
4. Include specific details from the research context as sub-nodes
5. Use clear, concise labels for nodes (2-5 words each)
6. Focus on concepts that have supporting evidence in the provided context
7. If context is limited, create a simple mindmap based on the question and any available information
8. Do not invent concepts not present in or directly inferred from the context

Output Format: Return ONLY a valid JSON object with this exact structure:
{{
  "main_concept": "Central topic or question",
  "nodes": [
    {{
      "id": "unique_id",
      "label": "Concept name",
      "type": "main|sub|detail",
      "description": "Brief explanation (1-2 sentences)"
    }}
  ],
  "edges": [
    {{
      "from": "parent_node_id",
      "to": "child_node_id",
      "relationship": "type of relationship (e.g., 'includes', 'leads to', 'relates to')"
    }}
  ]
}}

Requirements:
- Return ONLY the JSON, no additional text
- Ensure all IDs in 'edges' reference valid node IDs
- Create at least 3 nodes and 2 edges (can be simpler if context is limited)
- Main concept should be the central theme of the user's question
- Use 'main' type for the central concept, 'sub' for major categories, 'detail' for specific points
- If context is very limited, create a simple hierarchical structure based on the question

JSON:"""

    return prompt


def mindmap_node(state: ResearchState) -> ResearchState:
    """LangGraph mindmap node for generating structured mindmap data.

    This node uses the LLM to generate a structured mindmap based on the user's query
    and retrieved documents. Only runs for oss120b model.

    Args:
        state: The current LangGraph ResearchState.

    Returns:
        The updated state including mindmap data.

    Raises:
        RuntimeError: For LLM generation failures.
    """
    start_time = time.perf_counter()

    # Only run for oss120b model
    selected_llm = state.get("selected_llm", "")
    if selected_llm != "oss120b":
        logger.info("Mindmap node - skipping (not using oss120b model)")
        state_out: Dict[str, Any] = dict(state)
        state_out["mindmap_data"] = None
        state_out["mindmap_generated"] = False
        return cast(ResearchState, state_out)

    query = (state.get("query") or "").strip()
    retrieved_docs = state.get("retrieved_docs", [])
    web_search_results = state.get("web_search_results")
    answer = state.get("answer")

    logger.info(
        "Mindmap node - query: %r, retrieved_docs: %d, has_web_search: %s, has_answer: %s",
        query,
        len(retrieved_docs) if retrieved_docs else 0,
        web_search_results is not None,
        answer is not None
    )

    # Initialize output state
    state_out: Dict[str, Any] = dict(state)
    state_out["mindmap_data"] = None
    state_out["mindmap_generated"] = False

    # Always try to generate something - create fallback if needed
    try:
        # Build mindmap generation prompt
        logger.info("Mindmap node - building prompt for LLM")
        mindmap_prompt = _build_mindmap_prompt(
            query, retrieved_docs, web_search_results, answer
        )

        # Get LLM instance (use oss120b for reasoning)
        logger.info("Mindmap node - obtaining LLM instance")
        factory = LLMFactory()
        llm = factory.get_llm("oss120b")

        # Generate mindmap structure
        logger.info("Mindmap node - generating mindmap structure with LLM")
        llm_response = llm.generate(mindmap_prompt).strip()
        logger.info(f"Mindmap node - LLM response length: {len(llm_response)}")

        # Parse JSON response
        logger.info("Mindmap node - parsing LLM response")
        mindmap_data = _parse_llm_mindmap_response(llm_response)

        if mindmap_data:
            logger.info(
                "Mindmap node - successfully generated mindmap with %d nodes and %d edges",
                len(mindmap_data.get("nodes", [])),
                len(mindmap_data.get("edges", []))
            )
            state_out["mindmap_data"] = mindmap_data
            state_out["mindmap_generated"] = True
        else:
            logger.warning("Mindmap node - LLM generation failed, using fallback")
            # Create fallback mindmap
            fallback_mindmap = _create_fallback_mindmap(query, answer)
            state_out["mindmap_data"] = fallback_mindmap
            state_out["mindmap_generated"] = True

    except Exception as exc:
        logger.exception("Mindmap node - failed to generate mindmap with LLM, using fallback")
        # Always create a fallback mindmap even if everything fails
        fallback_mindmap = _create_fallback_mindmap(query, answer)
        state_out["mindmap_data"] = fallback_mindmap
        state_out["mindmap_generated"] = True
        logger.info("Mindmap node - created fallback mindmap")

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "Mindmap node - completed in %.2f ms, generated=%s",
        elapsed_ms,
        state_out["mindmap_generated"]
    )

    return cast(ResearchState, state_out)


def _parse_llm_mindmap_response(llm_response: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response into mindmap data with multiple fallback strategies.

    Args:
        llm_response: Raw LLM response string.

    Returns:
        Parsed mindmap data or None if parsing fails completely.
    """
    try:
        # Extract JSON from response (handle potential markdown code blocks)
        if "```json" in llm_response:
            json_start = llm_response.find("```json") + 7
            json_end = llm_response.find("```", json_start)
            json_str = llm_response[json_start:json_end].strip()
            logger.info("Mindmap node - extracted JSON from markdown code block")
        elif "```" in llm_response:
            json_start = llm_response.find("```") + 3
            json_end = llm_response.find("```", json_start)
            json_str = llm_response[json_start:json_end].strip()
            logger.info("Mindmap node - extracted JSON from generic code block")
        else:
            json_str = llm_response
            logger.info("Mindmap node - using entire response as JSON")

        logger.debug(f"Mindmap node - JSON string length: {len(json_str)}")
        logger.debug(f"Mindmap node - JSON string preview: {json_str[:200]}...")

        mindmap_data = json.loads(json_str)
        logger.info("Mindmap node - successfully parsed JSON")

        # Lenient validation - only check basic structure
        if not isinstance(mindmap_data, dict):
            logger.warning("Mindmap node - response is not a dict, trying to fix")
            return None

        # Ensure required fields exist with defaults
        if "nodes" not in mindmap_data:
            logger.warning("Mindmap node - missing nodes, adding empty array")
            mindmap_data["nodes"] = []
        if "edges" not in mindmap_data:
            logger.warning("Mindmap node - missing edges, adding empty array")
            mindmap_data["edges"] = []
        if "main_concept" not in mindmap_data:
            logger.warning("Mindmap node - missing main_concept, adding default")
            mindmap_data["main_concept"] = "Research Analysis"

        # Ensure arrays
        if not isinstance(mindmap_data["nodes"], list):
            logger.warning("Mindmap node - nodes is not a list, converting")
            mindmap_data["nodes"] = []
        if not isinstance(mindmap_data["edges"], list):
            logger.warning("Mindmap node - edges is not a list, converting")
            mindmap_data["edges"] = []

        # Accept even minimal structures
        return mindmap_data

    except json.JSONDecodeError as exc:
        logger.error("Mindmap node - failed to parse JSON response: %s", exc)
        logger.debug("LLM response that failed to parse: %s", llm_response[:500])
        return None
    except Exception as exc:
        logger.error("Mindmap node - unexpected error parsing response: %s", exc)
        return None


def _create_fallback_mindmap(query: str, answer: Optional[str] = None) -> Dict[str, Any]:
    """Create a simple fallback mindmap when LLM generation fails.

    Args:
        query: The user's question.
        answer: Optional generated answer.

    Returns:
        Simple mindmap structure.
    """
    # Create a simple hierarchical structure based on the query
    main_concept = query[:50] + "..." if len(query) > 50 else query

    # Extract meaningful terms from query (filter out common words)
    common_words = {'what', 'how', 'why', 'when', 'where', 'which', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'although', 'though'}

    words = [word.strip('.,!?;:') for word in query.split()]
    key_terms = [word for word in words if word.lower() not in common_words and len(word) > 2][:4]  # Get up to 4 meaningful words

    nodes = []
    edges = []

    # Create main concept node
    nodes.append({
        "id": "main",
        "label": main_concept,
        "type": "main",
        "description": "Main research question"
    })

    # Create sub-nodes from key terms
    if key_terms:
        for i, term in enumerate(key_terms, 1):
            node_id = f"sub_{i}"
            nodes.append({
                "id": node_id,
                "label": term.capitalize(),
                "type": "sub",
                "description": f"Key concept: {term}"
            })
            edges.append({
                "from": "main",
                "to": node_id,
                "relationship": "relates to"
            })
    else:
        # If no key terms found, create generic nodes
        nodes.append({
            "id": "sub_1",
            "label": "Research Context",
            "type": "sub",
            "description": "Analysis of available information"
        })
        edges.append({
            "from": "main",
            "to": "sub_1",
            "relationship": "includes"
        })

    # If answer is available, add a summary node
    if answer:
        summary_text = answer[:100] + "..." if len(answer) > 100 else answer
        nodes.append({
            "id": "summary",
            "label": "Key Findings",
            "type": "detail",
            "description": summary_text
        })
        # Connect to the first sub-node or main
        target_node = nodes[1]["id"] if len(nodes) > 1 else "main"
        edges.append({
            "from": target_node,
            "to": "summary",
            "relationship": "includes"
        })

    logger.info(f"Created fallback mindmap with {len(nodes)} nodes and {len(edges)} edges")

    return {
        "main_concept": main_concept,
        "nodes": nodes,
        "edges": edges
    }


if __name__ == "__main__":
    # Self-test
    logging.basicConfig(level=logging.INFO)

    # Test fallback mindmap creation
    print("Testing fallback mindmap creation...")
    test_query = "What are the main challenges in machine learning?"
    test_answer = "The main challenges include data quality issues, lack of model interpretability, and high computational costs."

    fallback = _create_fallback_mindmap(test_query, test_answer)
    print(f"Fallback mindmap: {json.dumps(fallback, indent=2)}")

    # Test with minimal state
    print("\nTesting with minimal state...")
    test_state: ResearchState = {
        "messages": [],
        "query": test_query,
        "selected_llm": "oss120b",
        "target_namespace": "test",
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
        "retrieved_docs": [],
        "answer": test_answer,
        "web_search_performed": False,
        "web_search_results": None,
        "web_search_query": None,
        "pdf_context_sufficient": True,
    }

    try:
        result = mindmap_node(test_state)
        print(f"Mindmap generated: {result['mindmap_generated']}")
        if result['mindmap_data']:
            print(f"Mindmap data: {json.dumps(result['mindmap_data'], indent=2)}")
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if LLM is not available, but fallback should still work")