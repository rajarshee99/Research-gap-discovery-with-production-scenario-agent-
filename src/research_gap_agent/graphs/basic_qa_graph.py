"""Basic LangGraph for research gap discovery and industry alignment.

This implements a complete Retrieval-Augmented Generation (RAG) workflow:
START -> Retrieval -> Web Search (conditional for oss120b) -> Generator -> Mindmap (conditional for oss120b) -> END

The graph:
1. Retrieves relevant documents using the retrieval node
2. Optionally performs web search if using oss120b and PDF context is insufficient
3. Generates an answer using the retrieved documents and optionally web search results
4. Optionally generates a mindmap if using oss120b
5. Returns the final answer and mindmap in the state

Future extensions may add nodes such as:
- Router
- Reranker
- Domain detection
- Industry gap analysis
"""

import logging

from langgraph.graph import StateGraph, START, END

from research_gap_agent.state.state import ResearchState
from research_gap_agent.nodes.retrieval import retrieval_node
from research_gap_agent.nodes.web_search import web_search_node
from research_gap_agent.nodes.generator import generator_node
from research_gap_agent.nodes.mindmap_node import mindmap_node

logger = logging.getLogger(__name__)


def create_basic_qa_graph():
    """Create the basic QA LangGraph with complete RAG workflow.

    Returns:
        A compiled StateGraph with the following flow:
        START -> retrieval -> web_search -> generator -> mindmap -> END
    """
    logger.info("Creating basic QA LangGraph with RAG workflow")

    # Initialize the graph with ResearchState
    graph = StateGraph(ResearchState)
    logger.debug("Initialized StateGraph with ResearchState")

    # Add the retrieval node
    graph.add_node("retrieval", retrieval_node)
    logger.debug("Registered retrieval node")

    # Add the web search node (conditional for oss120b)
    graph.add_node("web_search", web_search_node)
    logger.debug("Registered web_search node")

    # Add the generator node
    graph.add_node("generator", generator_node)
    logger.debug("Registered generator node")

    # Add the mindmap node (conditional for oss120b)
    graph.add_node("mindmap", mindmap_node)
    logger.debug("Registered mindmap node")

    # Define the workflow: START -> retrieval -> web_search -> generator -> mindmap -> END
    graph.add_edge(START, "retrieval")
    graph.add_edge("retrieval", "web_search")
    graph.add_edge("web_search", "generator")
    graph.add_edge("generator", "mindmap")
    graph.add_edge("mindmap", END)
    logger.debug("Defined workflow edges: START -> retrieval -> web_search -> generator -> mindmap -> END")

    # Compile the graph
    compiled_graph = graph.compile()
    logger.info("Successfully compiled basic QA LangGraph")

    return compiled_graph


# Export the compiled graph for easy import
basic_qa_graph = create_basic_qa_graph()
logger.info("Exported basic_qa_graph instance")
