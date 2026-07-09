from io import BytesIO
from pathlib import Path

import logging
import sys
import time
from typing import Any, Dict

import streamlit as st
try:
    # Try importing with different possible import names
    try:
        from streamlit_mermaid import st_mermaid
        mermaid_available = True
    except ImportError:
        try:
            import streamlit_mermaid
            mermaid_available = True
            # Create a wrapper function if st_mermaid is not directly available
            def st_mermaid(syntax, height=None):
                return streamlit_mermaid.st_mermaid(syntax, height=height)
        except ImportError:
            mermaid_available = False
            logging.warning("streamlit-mermaid not available, mindmap visualization disabled")
except Exception as e:
    mermaid_available = False
    logging.warning(f"Error importing streamlit-mermaid: {e}")

logger = logging.getLogger(__name__)

# Ensure `src/` is on sys.path so `import research_gap_agent...` works when running `streamlit run app.py`.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_gap_agent.Ingestion.pdf_ingestion import ingest_pdf
from research_gap_agent.graphs.basic_qa_graph import basic_qa_graph
from research_gap_agent.state.state import ResearchState

try:
    import fitz
except ImportError:
    fitz = None


# Professional theme configuration
st.set_page_config(
    page_title="Research Gap Discovery Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional research tool styling
st.markdown("""
<style>
    /* Global styles - Professional Dark Theme */
    .stApp {
        background-color: #0a0e27;
    }
    
    /* Typography - Professional research styling */
    h1, h2, h3, h4, h5, h6 {
        color: #e8eaf6;
        font-weight: 500;
        letter-spacing: -0.5px;
    }
    
    p, span, div, label {
        color: #b0b8c8;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        background-color: #0a0e27;
    }
    
    /* Sidebar styling - Professional */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    [data-testid="stSidebar"] > div {
        padding: 1.5rem;
    }
    
    /* Card styling - Professional cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4);
    }
    
    /* Button styling - Professional */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.3);
    }
    
    .stButton > button:disabled {
        background-color: #374151;
        color: #6b7280;
        cursor: not-allowed;
    }
    
    /* File uploader styling - Professional */
    [data-testid="stFileUploader"] {
        border: 2px dashed #374151;
        border-radius: 6px;
        padding: 2rem;
        background-color: #111827;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
    }
    
    /* Chat message styling - Professional */
    .stChatMessage {
        background-color: #111827;
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3);
        border: 1px solid #1f2937;
    }
    
    /* Status indicators - Professional */
    .status-ready {
        color: #10b981;
        font-weight: 500;
    }
    
    .status-processing {
        color: #f59e0b;
        font-weight: 500;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: 500;
    }
    
    /* Info boxes - Professional */
    .info-box {
        background-color: #1e3a5f;
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    .success-box {
        background-color: #1e4031;
        border-left: 3px solid #10b981;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    /* Expander styling - Professional */
    [data-testid="stExpander"] {
        border: 1px solid #1f2937;
        border-radius: 6px;
        background-color: #111827;
    }
    
    /* Progress bar - Professional */
    [data-testid="stProgress"] > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Input fields - Professional */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #111827;
        color: #e8eaf6;
        border: 1px solid #374151;
        border-radius: 4px;
    }
    
    /* Metrics - Professional */
    [data-testid="stMetricValue"] {
        color: #e8eaf6;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af;
        font-size: 0.85rem;
    }
    
    /* Dividers - Professional */
    hr {
        border-color: #1f2937;
    }
    
    /* Streamlit native components override */
    .st-ae {
        background-color: #111827;
        border-color: #1f2937;
    }
    
    /* Text area and input focus */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
    }
    
    /* Select dropdown styling */
    .stSelectbox > div > div > select {
        background-color: #111827;
        color: #e8eaf6;
    }
    
    /* Chat input styling */
    [data-testid="stChatInput"] > div > div > textarea {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 6px;
    }
    
    [data-testid="stChatInput"] > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
    }
</style>
""", unsafe_allow_html=True)


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to the **Research Gap Discovery Agent**. \n\n"
                    "Upload your research PDFs to analyze research gaps, "
                    "identify industry alignment opportunities, and receive intelligent analysis."
                ),
            }
        ]

    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

    if "processed_file_count" not in st.session_state:
        st.session_state.processed_file_count = 0

    if "selected_llm" not in st.session_state:
        st.session_state.selected_llm = "oss20b"

    if "ai_status" not in st.session_state:
        st.session_state.ai_status = "Ready"

    if "current_namespace" not in st.session_state:
        st.session_state.current_namespace = None
    
    # Processing state to prevent duplicate processing on reruns
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    
    # Short-term memory for graph state persistence
    if "graph_memory" not in st.session_state:
        st.session_state.graph_memory = {
            "last_retrieved_docs": None,
            "last_web_search_results": None,
            "last_web_search_query": None,
            "last_mindmap_data": None,
            "last_mindmap_generated": False,
            "conversation_context": [],
            "memory_size": 5
        }

    # Mindmap display toggle
    if "show_mindmap" not in st.session_state:
        st.session_state.show_mindmap = True


def mindmap_to_mermaid(mindmap_data: Dict[str, Any]) -> str:
    """Convert mindmap JSON data to Mermaid mindmap syntax.

    Args:
        mindmap_data: Dictionary with nodes, edges, and main_concept.

    Returns:
        Mermaid mindmap syntax string.
    """
    if not mindmap_data:
        return ""

    main_concept = mindmap_data.get("main_concept", "Central Concept")
    nodes = mindmap_data.get("nodes", [])
    edges = mindmap_data.get("edges", [])

    # Create node lookup
    node_lookup = {node["id"]: node for node in nodes}

    # Build Mermaid mindmap syntax
    mermaid_lines = ["mindmap", f"  root(({main_concept}))"]

    # Add nodes hierarchically
    # First, identify root-level nodes (nodes that are children of root or have no incoming edges)
    node_children = {node["id"]: [] for node in nodes}
    for edge in edges:
        parent = edge["from"]
        child = edge["to"]
        if parent in node_children:
            node_children[parent].append(child)

    # Find root-level nodes (nodes that are not children of any other node, or are children of main concept)
    all_child_ids = {edge["to"] for edge in edges}
    root_level_nodes = [node for node in nodes if node["id"] not in all_child_ids]

    # If no clear hierarchy, use the first node as root
    if not root_level_nodes and nodes:
        root_level_nodes = [nodes[0]]

    # If still no nodes, create a simple structure from main concept
    if not nodes:
        # Create a simple fallback structure
        mermaid_lines = ["mindmap", f"  root(({main_concept}))"]
        return "\n".join(mermaid_lines)

    # Add nodes to Mermaid syntax
    for node in root_level_nodes:
        # Sanitize node ID for Mermaid (remove special characters)
        safe_id = node["id"].replace("-", "_").replace(" ", "_").replace("(", "_").replace(")", "_")
        mermaid_lines.append(f"    {safe_id}({node['label']})")
        # Add children recursively
        _add_children_to_mermaid(mermaid_lines, node["id"], node_children, node_lookup, level=2)

    return "\n".join(mermaid_lines)


def _add_children_to_mermaid(
    mermaid_lines: list,
    parent_id: str,
    node_children: Dict[str, list],
    node_lookup: Dict[str, Dict],
    level: int
):
    """Recursively add children to Mermaid syntax."""
    if parent_id not in node_children:
        return

    children = node_children[parent_id]
    for child_id in children:
        if child_id in node_lookup:
            child_node = node_lookup[child_id]
            indent = "  " * level
            # Sanitize child ID for Mermaid
            safe_id = child_id.replace("-", "_").replace(" ", "_").replace("(", "_").replace(")", "_")
            mermaid_lines.append(f"{indent}{safe_id}({child_node['label']})")
            # Recursively add children
            _add_children_to_mermaid(mermaid_lines, child_id, node_children, node_lookup, level + 1)


def get_pdf_details(uploaded_file) -> dict:
    details = {
        "name": uploaded_file.name,
        "size_mb": uploaded_file.size / (1024 * 1024),
        "pages": None,
    }

    if fitz is not None:
        pdf_bytes = uploaded_file.getvalue()
        with fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf") as document:
            details["pages"] = len(document)

    return details


def save_uploaded_pdfs(uploaded_files: list) -> list[str]:
    """Save uploaded PDF files to data/uploads/ preserving original filenames."""

    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for uploaded_file in uploaded_files:
        destination = uploads_dir / uploaded_file.name
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_paths.append(str(destination))

    return saved_paths


def render_header():
    """Render professional header with model and status indicators."""
    # Get current model display
    model_display = "OSS 20B" if st.session_state.selected_llm == "oss20b" else "OSS 120B"
    model_capability = "Fast Inference" if st.session_state.selected_llm == "oss20b" else "Advanced Reasoning"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1f2937;">
        <div>
            <h1 style="margin: 0; font-size: 1.75rem; color: #e8eaf6; font-weight: 500;">Research Gap Discovery Agent</h1>
            <p style="margin: 0.4rem 0 0 0; color: #9ca3af; font-size: 0.85rem;">
                AI-powered research analysis and gap identification
            </p>
        </div>
        <div style="text-align: right; display: flex; gap: 0.75rem; align-items: center;">
            <div style="background: #111827; border: 1px solid #374151; color: #b0b8c8; padding: 0.5rem 1rem; border-radius: 4px; font-weight: 500; font-size: 0.8rem;">
                <span style="color: #6b7280; font-size: 0.75rem; margin-right: 0.4rem;">MODEL</span>
                {model_display}
            </div>
            <div style="background: #111827; border: 1px solid #374151; color: #b0b8c8; padding: 0.5rem 1rem; border-radius: 4px; font-weight: 500; font-size: 0.8rem;">
                <span style="color: #6b7280; font-size: 0.75rem; margin-right: 0.4rem;">CAPABILITY</span>
                {model_capability}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_indicator():
    """Render AI status indicator."""
    status = st.session_state.ai_status
    
    status_colors = {
        "Ready": "#10b981",
        "Processing": "#f59e0b", 
        "Error": "#ef4444"
    }
    
    status_descriptions = {
        "Ready": "System ready for queries",
        "Processing": "Processing request...",
        "Error": "System error occurred"
    }
    
    color = status_colors.get(status, "#6b7280")
    description = status_descriptions.get(status, status)
    
    # Add processing warning
    processing_warning = ""
    if st.session_state.is_processing:
        processing_warning = " <span style='color: #f59e0b; font-size: 0.75rem; margin-left: 0.5rem;'>(active)</span>"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; padding: 0.5rem 0.75rem; background: #111827; border: 1px solid #1f2937; border-radius: 4px;">
        <div style="width: 8px; height: 8px; background: {color}; border-radius: 50%; box-shadow: 0 0 8px {color};"></div>
        <span style="color: #e8eaf6; font-weight: 500; font-size: 0.85rem;">
            {status}{processing_warning}
        </span>
        <span style="color: #6b7280; font-size: 0.75rem; margin-left: auto;">
            {description}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render professional sidebar with session information."""
    with st.sidebar:
        st.markdown("### Session Information")

        st.markdown("#### Document Status")
        if st.session_state.documents_processed:
            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 1rem; margin: 0.5rem 0;">
                <div style="color: #6b7280; font-size: 0.75rem; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Documents</div>
                <div style="color: #e8eaf6; font-size: 1.25rem; font-weight: 600;">{st.session_state.processed_file_count}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 1rem; margin: 0.5rem 0;">
                <div style="color: #6b7280; font-size: 0.75rem; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Queries</div>
                <div style="color: #e8eaf6; font-size: 1.25rem; font-weight: 600;">{len(st.session_state.messages) - 1}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.current_namespace:
                st.markdown(f"""
                <div style="color: #6b7280; font-size: 0.75rem; margin-top: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">Namespace</div>
                <div style="color: #9ca3af; font-size: 0.8rem; margin-top: 0.25rem; font-family: monospace; background: #0a0e27; padding: 0.4rem 0.6rem; border-radius: 3px; border: 1px solid #1f2937;">
                    {st.session_state.current_namespace[:40]}{'...' if len(st.session_state.current_namespace) > 40 else ''}
                </div>
                """, unsafe_allow_html=True)

            # Mindmap toggle (only show if using oss120b)
            if st.session_state.selected_llm == "oss120b" and mermaid_available:
                st.markdown("---")
                st.markdown("#### Visualization Options")
                show_mindmap = st.checkbox(
                    "Show Mindmaps",
                    value=st.session_state.show_mindmap,
                    help="Display interactive mindmaps for research concepts (oss120b only)"
                )
                st.session_state.show_mindmap = show_mindmap
        else:
            st.markdown('<div style="color: #6b7280; font-size: 0.85rem; padding: 1rem 0;">No documents loaded</div>', unsafe_allow_html=True)


def render_upload_section():
    """Render professional file upload section."""
    # Model selection - professional styling
    st.markdown("### Model Configuration")
    st.markdown("Select the AI model for analysis")
    
    model_options = {
        "oss20b": "OSS 20B - Fast Inference",
        "oss120b": "OSS 120B - Advanced Reasoning"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "AI Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.selected_llm),
            key="main_model_selector",
            label_visibility="visible",
            disabled=st.session_state.is_processing
        )
        
        if selected_model != st.session_state.selected_llm and not st.session_state.is_processing:
            st.session_state.selected_llm = selected_model
            st.rerun()
    
    with col2:
        # Display current model details
        if st.session_state.selected_llm == "oss20b":
            st.markdown("""
            <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 0.75rem; text-align: center;">
                <div style="color: #6b7280; font-size: 0.7rem; margin-bottom: 0.25rem;">SPEED</div>
                <div style="color: #10b981; font-weight: 600; font-size: 0.85rem;">Fast</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 0.75rem; text-align: center;">
                <div style="color: #6b7280; font-size: 0.7rem; margin-bottom: 0.25rem;">CAPABILITY</div>
                <div style="color: #3b82f6; font-weight: 600; font-size: 0.85rem;">Advanced</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Document Ingestion")
    st.markdown("Upload research papers for analysis")
    
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research papers in PDF format",
        key="pdf_uploader"
    )
    
    if uploaded_files:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Selected Files")
            for i, file in enumerate(uploaded_files, 1):
                details = get_pdf_details(file)
                st.markdown(f"""
                <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="background: #1f2937; border-radius: 3px; padding: 0.4rem;">
                            <span style="font-size: 0.9rem;">PDF</span>
                        </div>
                        <div style="flex: 1;">
                            <div style="font-weight: 500; color: #e8eaf6; font-size: 0.9rem;">{details['name']}</div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;">
                                {details['size_mb']:.2f} MB {f'• {details['pages']} pages' if details['pages'] else ''}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Actions")
            if st.button("Process Documents", type="primary", use_container_width=True, disabled=st.session_state.is_processing):
                process_documents(uploaded_files)


def process_documents(uploaded_files):
    """Process uploaded PDF documents."""
    # Prevent duplicate processing on Streamlit reruns
    if st.session_state.is_processing:
        logger.info("Already processing documents, skipping duplicate request")
        return
    
    try:
        st.session_state.is_processing = True
        saved_file_paths = save_uploaded_pdfs(uploaded_files)

        # Use consistent timestamp for all documents in this batch
        import time
        batch_timestamp = int(time.time())
        logger.info(f"Processing documents with batch timestamp: {batch_timestamp}")

        with st.spinner("🔄 Processing documents..."):
            results: list[dict] = []
            for saved_file_path in saved_file_paths:
                # Pass the batch timestamp to ensure consistent namespace generation
                result = ingest_pdf(saved_file_path, timestamp=batch_timestamp)
                results.append(result)

        st.session_state.documents_processed = True
        st.session_state.processed_file_count = len(results)
        
        # Get namespace from Pinecone upsert stats to ensure it matches what was actually stored
        if results and isinstance(results[0], dict):
            upsert_stats = results[0].get("pinecone_upsert_stats", {})
            per_namespace = upsert_stats.get("per_namespace", {})
            
            if per_namespace:
                # Get the actual namespace used for storage (includes timestamp)
                stored_namespace = list(per_namespace.keys())[0]
                st.session_state.current_namespace = stored_namespace
                logger.info(f"Set namespace from Pinecone upsert stats: {stored_namespace}")
            else:
                # Fallback to filename-based namespace (shouldn't happen normally)
                metadata = results[0].get("metadata", {})
                if isinstance(metadata, dict) and "filename" in metadata:
                    filename = metadata["filename"]
                    base_namespace = filename.rsplit('.', 1)[0] if '.' in filename else filename
                    st.session_state.current_namespace = base_namespace
                    logger.warning(f"Using fallback namespace (no upsert stats): {base_namespace}")

        # Display processing results
        render_processing_results(results)
        
        st.success(f"✅ Successfully processed {len(results)} document(s)")
        st.session_state.is_processing = False
        st.rerun()

    except Exception as exc:
        st.error(f"❌ Error processing documents: {str(exc)}")
        logger.exception("Document processing failed")
        st.session_state.is_processing = False


def render_processing_results(results):
    """Render processing results in a professional format."""
    st.markdown("### Processing Results")
    
    for idx, result in enumerate(results, 1):
        with st.expander(f"Document {idx} - Analysis Details", expanded=False):
            metadata = result.get("metadata", {})
            full_text = result.get("full_text", "")
            chunk_count = result.get("chunk_count", 0)
            embedded_chunk_count = result.get("embedded_chunk_count", 0)
            
            # Metadata section
            st.markdown("#### Document Metadata")
            if metadata:
                for key, value in metadata.items():
                    st.markdown(f"""
                    <div style="background: #0a0e27; border: 1px solid #1f2937; border-radius: 3px; padding: 0.5rem 1rem; margin: 0.25rem 0;">
                        <span style="color: #6b7280; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">{key}</span>
                        <span style="color: #9ca3af; margin-left: 0.75rem; font-size: 0.85rem;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Statistics
            st.markdown("#### Processing Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 1rem; text-align: center;">
                    <div style="color: #6b7280; font-size: 0.7rem; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Chunks</div>
                    <div style="color: #e8eaf6; font-size: 1.25rem; font-weight: 600;">{chunk_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 1rem; text-align: center;">
                    <div style="color: #6b7280; font-size: 0.7rem; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Vectors</div>
                    <div style="color: #e8eaf6; font-size: 1.25rem; font-weight: 600;">{embedded_chunk_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 1rem; text-align: center;">
                    <div style="color: #6b7280; font-size: 0.7rem; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Characters</div>
                    <div style="color: #e8eaf6; font-size: 1.25rem; font-weight: 600;">{len(full_text):,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Preview
            st.markdown("#### Content Preview")
            preview_text = full_text[:500] + "..." if len(full_text) > 500 else full_text
            st.code(preview_text, language=None)


def render_chat_interface():
    """Render professional chat interface."""
    st.markdown("### Research Analysis")

    if not st.session_state.documents_processed:
        st.markdown("""
        <div style="background: #1e3a5f; border-left: 3px solid #3b82f6; padding: 1rem; border-radius: 4px; margin: 1rem 0; color: #e2e8f0;">
            <strong>Document Required:</strong> Upload and process documents to enable analysis features.
        </div>
        """, unsafe_allow_html=True)
        return

    # Display chat messages
    logger.info(f"Rendering chat interface. Total messages: {len(st.session_state.messages)}")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display mindmap if available and enabled
    if st.session_state.show_mindmap and mermaid_available:
        last_mindmap_generated = st.session_state.graph_memory.get("last_mindmap_generated", False)
        last_mindmap_data = st.session_state.graph_memory.get("last_mindmap_data")

        logger.info(f"DEBUG UI: show_mindmap = {st.session_state.show_mindmap}, mermaid_available = {mermaid_available}")
        logger.info(f"DEBUG UI: last_mindmap_generated = {last_mindmap_generated}, last_mindmap_data present = {last_mindmap_data is not None}")

        # For testing - always show a simple mindmap if oss120b is selected
        if st.session_state.selected_llm == "oss120b":
            with st.expander("🧠 Research Mindmap (Test Mode)", expanded=True):
                st.markdown("### Concept Visualization")
                st.markdown("Interactive mindmap showing key concepts and relationships from the research.")

                # Try to use actual mindmap data if available
                if last_mindmap_generated and last_mindmap_data:
                    try:
                        mermaid_syntax = mindmap_to_mermaid(last_mindmap_data)
                        logger.info(f"DEBUG UI: Using actual mindmap data, syntax length = {len(mermaid_syntax)}")
                        if mermaid_syntax:
                            st_mermaid(mermaid_syntax, height="500px")
                        else:
                            st.warning("Mindmap conversion failed")
                    except Exception as exc:
                        logger.exception("Failed to render actual mindmap")
                        st.error(f"Failed to render mindmap: {str(exc)}")
                else:
                    # Use a simple test mindmap
                    logger.info("DEBUG UI: Using test mindmap")
                    test_mermaid = """mindmap
  root((Research Analysis))
    main_concept((Main Question))
      key_terms((Key Concepts))
        findings((Key Findings))"""
                    st_mermaid(test_mermaid, height="400px")
                    st.info("Test mindmap displayed (actual mindmap data not available)")

        elif last_mindmap_generated and last_mindmap_data:
            # Original logic for other models
            with st.expander("🧠 Research Mindmap", expanded=True):
                st.markdown("### Concept Visualization")
                st.markdown("Interactive mindmap showing key concepts and relationships from the research.")

                try:
                    mermaid_syntax = mindmap_to_mermaid(last_mindmap_data)
                    logger.info(f"DEBUG UI: mermaid_syntax length = {len(mermaid_syntax)}")

                    if mermaid_syntax:
                        st_mermaid(mermaid_syntax, height="500px")
                        logger.info("DEBUG UI: Successfully displayed mindmap")
                    else:
                        st.info("Mindmap data could not be converted to visualization format.")
                        logger.warning("DEBUG UI: mermaid_syntax was empty")
                except Exception as exc:
                    logger.exception("Failed to render mindmap")
                    st.error(f"Failed to render mindmap: {str(exc)}")
        else:
            logger.warning(f"DEBUG UI: Mindmap not displayed - generated: {last_mindmap_generated}, data: {last_mindmap_data is not None}")

    # Chat input
    if prompt := st.chat_input(
        "Ask about research methodology, findings, gaps, or any aspect of your documents...",
        disabled=not st.session_state.documents_processed or st.session_state.is_processing
    ):
        handle_user_query(prompt)


def handle_user_query(query: str) -> None:
    """Handle user query with streaming response."""
    logger.info(f"handle_user_query called with query: {query[:50]}...")
    
    # Prevent duplicate processing on Streamlit reruns
    if st.session_state.is_processing:
        logger.info("Already processing a query, skipping duplicate request")
        return
    
    try:
        st.session_state.is_processing = True
        st.session_state.ai_status = "Processing"
        st.session_state.messages.append({"role": "user", "content": query})
        logger.info(f"User message added. Total messages: {len(st.session_state.messages)}")

        # Use the current session's namespace to search only the uploaded document
        # This prevents retrieving documents from previous sessions
        target_namespace = st.session_state.current_namespace
        logger.info(f"Using namespace for retrieval: {target_namespace}")
        
        state: ResearchState = {
            "messages": st.session_state.messages,
            "query": query,
            "selected_llm": st.session_state.selected_llm,
            "target_namespace": target_namespace,  # Search only current document
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
            "retrieved_docs": None,
            "answer": None,
            "web_search_performed": False,
            "web_search_results": None,
            "web_search_query": None,
            "pdf_context_sufficient": False,
        }

        # Use the LangGraph directly for simplicity
        spinner_text = f"Searching and generating answer using {st.session_state.selected_llm}..."
        if st.session_state.selected_llm == "oss120b":
            spinner_text += " (with web search capability)"
        
        with st.spinner(spinner_text):
            result_state = basic_qa_graph.invoke(state)

        # Extract answer and web search info
        answer = result_state.get("answer", "No answer generated.")
        errors = result_state.get("errors", [])
        web_search_performed = result_state.get("web_search_performed", False)
        pdf_context_sufficient = result_state.get("pdf_context_sufficient", False)
        retrieved_docs = result_state.get("retrieved_docs")
        web_search_results = result_state.get("web_search_results")
        mindmap_generated = result_state.get("mindmap_generated", False)
        mindmap_data = result_state.get("mindmap_data")

        logger.info(f"Query processing completed. Answer length: {len(answer)}, Errors: {len(errors)}")
        logger.info(f"DEBUG: mindmap_generated = {mindmap_generated}, mindmap_data present = {mindmap_data is not None}")
        if mindmap_data:
            logger.info(f"DEBUG: mindmap_data keys = {mindmap_data.keys()}, nodes = {len(mindmap_data.get('nodes', []))}")

        # Update short-term memory with current query results
        st.session_state.graph_memory["last_retrieved_docs"] = retrieved_docs
        st.session_state.graph_memory["last_web_search_results"] = web_search_results
        st.session_state.graph_memory["last_web_search_query"] = result_state.get("web_search_query")
        st.session_state.graph_memory["last_mindmap_data"] = mindmap_data
        st.session_state.graph_memory["last_mindmap_generated"] = mindmap_generated

        logger.info(f"DEBUG: Updated graph_memory - last_mindmap_generated = {st.session_state.graph_memory['last_mindmap_generated']}")
        logger.info(f"DEBUG: Updated graph_memory - last_mindmap_data present = {st.session_state.graph_memory['last_mindmap_data'] is not None}")
        
        # Add conversation context to memory (keep last N exchanges)
        conversation_entry = {
            "query": query,
            "answer": answer,
            "has_web_search": web_search_performed,
            "has_mindmap": mindmap_generated,
            "timestamp": time.time()
        }
        st.session_state.graph_memory["conversation_context"].append(conversation_entry)
        
        # Keep only the most recent entries based on memory_size
        memory_size = st.session_state.graph_memory.get("memory_size", 5)
        if len(st.session_state.graph_memory["conversation_context"]) > memory_size:
            st.session_state.graph_memory["conversation_context"] = st.session_state.graph_memory["conversation_context"][-memory_size:]

        # Add web search status to answer if using oss120b
        if st.session_state.selected_llm == "oss120b":
            status_info = "\n\n---\n**Analysis Context:** "
            if web_search_performed:
                status_info += "Web search was performed to supplement PDF context with current information."
            elif pdf_context_sufficient:
                status_info += "PDF context was sufficient; web search was not needed."
            else:
                status_info += "Using PDF context only."

            # Add mindmap status - now always generates with fallback
            if mindmap_generated:
                status_info += "\n\n**🧠 Mindmap:** A research mindmap has been generated based on this analysis. See the visualization below."

            answer = answer + status_info

        # Add assistant response to chat
        if errors:
            error_msg = f"Errors occurred: {'; '.join(errors)}\n\n{answer}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            logger.info(f"Added error message to chat. Total messages: {len(st.session_state.messages)}")
        else:
            st.session_state.messages.append({"role": "assistant", "content": answer})
            logger.info(f"Added assistant response to chat. Total messages: {len(st.session_state.messages)}")

        # Reset AI status and processing flag BEFORE rerun
        st.session_state.ai_status = "Ready"
        st.session_state.is_processing = False
        
        logger.info("About to trigger rerun to update UI")
        # Force rerun to update the UI and show the new message
        st.rerun()

    except Exception as exc:
        logger.exception("Failed to process user query")
        error_msg = f"Sorry, I encountered an error: {str(exc)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.session_state.ai_status = "Error"
        st.session_state.is_processing = False
        # Force rerun to show the error message
        st.rerun()


def main():
    """Main application entry point."""
    initialize_state()
    
    # Render UI components
    render_header()
    render_status_indicator()
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_upload_section()
        
        if st.session_state.documents_processed:
            st.markdown("---")
            st.markdown("### Session Statistics")
            
            # Memory info
            memory = st.session_state.graph_memory
            recent_queries = len(memory.get("conversation_context", []))
            
            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 1rem; margin: 0.5rem 0;">
                <div style="color: #6b7280; font-size: 0.7rem; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Queries Processed</div>
                <div style="color: #e8eaf6; font-size: 1.25rem; font-weight: 600;">{recent_queries}</div>
            </div>
            """, unsafe_allow_html=True)
            
            web_search_used = memory.get("last_web_search_results") is not None
            search_text = "Web Enhanced" if web_search_used else "PDF Only"
            search_color = "#3b82f6" if web_search_used else "#6b7280"
            
            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #374151; border-radius: 4px; padding: 1rem; margin: 0.5rem 0;">
                <div style="color: #6b7280; font-size: 0.7rem; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Analysis Mode</div>
                <div style="color: {search_color}; font-size: 1rem; font-weight: 600;">{search_text}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        render_chat_interface()


if __name__ == "__main__":
    main()
