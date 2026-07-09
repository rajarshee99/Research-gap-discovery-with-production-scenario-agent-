from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PaperMetadata(TypedDict):
    title: str
    authors: List[str]
    publication_year: Optional[str]
    abstract: str
    journal: Optional[str]


class AnalysisResults(TypedDict):
    executive_summary: Optional[str]
    methodology: Optional[str]
    key_contributions: List[str]
    strengths_weaknesses: Dict[str, List[str]]  # {"strengths": [...], "weaknesses": [...]}
    research_gaps: List[str]
    industry_alignment: Dict[str, Any]  # {"use_cases": [...], "market_relevance": ...}


class AgentState(TypedDict, total=False):
    """
    The parent state that travels through the Research Gap Discovery
    and Industry Alignment LangGraph.
    """
    # --- Conversational & Input State ---
    messages: Annotated[List[BaseMessage], add_messages]
    query: str  # User's query/question
    selected_llm: Optional[str]  # Selected LLM model (e.g., "oss20b", "oss120b") populated by Router

    # --- Document & Processing State ---
    uploaded_files: List[str]          # List of uploaded PDF names
    file_payloads: List[Dict[str, Any]]  # Serialized raw PDF bytes and metadata from the UI
    parsed_documents: List[Dict[str, Any]]  # Extracted text, page counts, and structural content
    vector_store_path: Optional[str]   # Path/ID to the FAISS index for this session

    # --- Analysis & Extraction State ---
    metadata: Dict[str, PaperMetadata]  # Extracted metadata keyed by filename
    analysis_results: Dict[str, AnalysisResults]  # Analysis keyed by filename

    # --- Synthesis & Global Discovery State ---
    cross_paper_gaps: List[str]        # Discovered gaps from synthesizing all papers
    industry_gap_alignment: Dict[str, Any]  # Final alignment report with industry needs

    # --- Flow Control & Errors ---
    current_node: str                  # Currently executing node name
    errors: List[str]                  # List of errors encountered during execution

    # --- Retrieval & Generation State ---
    retrieved_docs: Optional[List[Dict[str, Any]]]  # Retrieved documents from Pinecone
    answer: Optional[str]              # Generated answer from LLM
    target_namespace: Optional[str]   # Specific Pinecone namespace to search (for recent uploads)

    # --- Web Search State (for oss120B model) ---
    web_search_performed: bool        # Whether web search was performed
    web_search_results: Optional[Dict[str, Any]]  # Results from Tavily + Jina AI extraction
    web_search_query: Optional[str]   # Query used for web search
    pdf_context_sufficient: bool       # Whether PDF context was deemed sufficient

    # --- Mindmap State (for oss120B model) ---
    mindmap_generated: bool           # Whether mindmap was successfully generated
    mindmap_data: Optional[Dict[str, Any]]  # Structured mindmap data (nodes, edges)


# Type alias for compatibility with existing imports
ResearchState = AgentState
