# Data Flow Diagram - Research Gap Discovery and Industry Alignment Agent

## Overview
This document shows the data flow from file to file in the Research Gap Discovery and Industry Alignment Agent project.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                      (app.py - Streamlit)                        │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PDF Upload Flow                             │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌──────────────────────┐      ┌──────────────────────┐
        │   uploader.py        │      │   pdf_ingestion.py   │
        │   (normalize files)  │─────▶│   (main ingestion)   │
        └──────────────────────┘      └──────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
        ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
        │   parser.py          │  │ semantic_chunker.py  │  │   embedder.py        │
        │ (PDF text extraction)│ │ (chunk documents)    │  │ (generate embeddings)│
        └──────────────────────┘  └──────────────────────┘  └──────────────────────┘
                    │                       │                       │
                    └───────────────────────┼───────────────────────┘
                                            │
                                            ▼
                                ┌──────────────────────┐
                                │ pinecone_upsert.py   │
                                │ (upload to vector DB)│
                                └──────────────────────┘
                                            │
                                            ▼
                                ┌──────────────────────┐
                                │   Pinecone Vector DB │
                                │   (External Service) │
                                └──────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Query Processing Flow                        │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      chat.py                                     │
│              (build chat payload)                                │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   basic_qa_graph.py                              │
│              (LangGraph orchestration)                           │
└─────────────────────────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐  ┌──────────────┐  ┌──────────────┐
│retrieval│  │ web_search   │  │  generator   │
│  .py    │  │    .py       │  │    .py      │
└─────────┘  └──────────────┘  └──────────────┘
    │               │               │
    │               │               │
    └───────────────┼───────────────┘
                    │
                    ▼
            ┌──────────────┐
            │ mindmap_node │
            │    .py       │
            └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │   state.py   │
            │ (AgentState) │
            └──────────────┘
```

## Detailed File-to-File Data Flow

### 1. PDF Ingestion Pipeline

**Entry Point:** `app.py`
- User uploads PDF files through Streamlit file uploader
- Files are saved to `data/uploads/` directory
- Calls `save_uploaded_pdfs()` function

**Flow:**
```
app.py
  ├─▶ uploader.py (normalize_uploaded_files, serialize_uploaded_files)
  ├─▶ pdf_ingestion.py (ingest_pdf)
  │     ├─▶ parser.py (parse_pdf)
  │     │     ├─▶ pymupdf4llm (primary parsing)
  │     │     └─▶ PyMuPDF (fallback parsing)
  │     ├─▶ semantic_chunker.py (SemanticChunkerService.chunk_document)
  │     ├─▶ embedder.py (EmbeddingService.embed_chunks)
  │     └─▶ pinecone_upsert.py (upsert_chunks)
  │           └─▶ Pinecone Vector DB (external service)
  └─▶ Update session state with processing results
```

**Data Transformations:**
1. **uploader.py**: Converts file objects to serialized format with metadata
2. **parser.py**: Extracts text and metadata from PDF, returns structured dict
3. **semantic_chunker.py**: Splits text into semantic chunks with context
4. **embedder.py**: Converts text chunks to embedding vectors
5. **pinecone_upsert.py**: Uploads vectors to Pinecone with metadata

### 2. Query Processing Pipeline

**Entry Point:** `app.py`
- User submits query through chat interface
- Query is processed through LangGraph workflow

**Flow:**
```
app.py
  ├─▶ chat.py (build_chat_payload)
  ├─▶ basic_qa_graph.py (basic_qa_graph)
  │     ├─▶ state.py (ResearchState - state management)
  │     ├─▶ retrieval.py (retrieval_node)
  │     │     ├─▶ embedder.py (EmbeddingService - query embedding)
  │     │     └─▶ Pinecone Vector DB (similarity search)
  │     ├─▶ web_search.py (web_search_node) [conditional for oss120b]
  │     │     ├─▶ tavily_service.py (web search)
  │     │     └─▶ jina_service.py (content extraction)
  │     ├─▶ generator.py (generator_node)
  │     │     ├─▶ llm/factory.py (LLMFactory)
  │     │     ├─▶ llm/oss20B.py (OSS 20B model)
  │     │     └─▶ llm/oss120B.py (OSS 120B model)
  │     └─▶ mindmap_node.py (mindmap_node) [conditional for oss120b]
  └─▶ Update UI with generated response
```

**Data Transformations:**
1. **chat.py**: Builds payload with messages, files, and query context
2. **state.py**: Maintains conversation state and document metadata
3. **retrieval.py**: Converts query to embedding, retrieves relevant chunks
4. **web_search.py**: Performs web search and extracts relevant content
5. **generator.py**: Builds RAG prompt, generates answer using LLM
6. **mindmap_node.py**: Generates structured mindmap data from response

### 3. Service Layer Dependencies

**Supporting Services:**
```
services/
  ├─▶ crawler_service.py (web crawling)
  ├─▶ embedding_service.py (embedding operations)
  ├─▶ jina_service.py (Jina AI integration)
  ├─▶ llm_service.py (LLM operations)
  ├─▶ pinecone_service.py (Pinecone operations)
  └─▶ reranker_service.py (result reranking)
```

**Utility Functions:**
```
utils/
  └─▶ file_store.py (file storage operations)
```

### 4. State Management Flow

**State Flow:**
```
state.py (ResearchState)
  ├─▶ Incoming: user query, uploaded files, messages
  ├─▶ Processing: retrieved_docs, web_search_results, answer
  ├─▶ Metadata: selected_llm, target_namespace, errors
  └─▶ Output: final answer, mindmap_data, analysis results
```

## Key Data Structures

### PDF Ingestion Data Flow
```python
# Input: PDF file path
# Output: Structured document data
{
    "full_text": str,
    "metadata": {
        "title": str,
        "author": str,
        "page_count": int,
        "filename": str,
        "source": str,
        "ingestion_timestamp": int
    },
    "chunks": list[dict],
    "chunk_count": int,
    "embedded_chunks": list[dict],
    "embedded_chunk_count": int,
    "pinecone_upsert_stats": dict
}
```

### Query Processing Data Flow
```python
# Input: User query + conversation history
# State updates through each node:
{
    "query": str,
    "messages": list[BaseMessage],
    "retrieved_docs": list[dict],
    "web_search_results": dict,
    "answer": str,
    "mindmap_data": dict,
    "selected_llm": str,
    "target_namespace": str
}
```

## External Service Integrations

1. **Pinecone**: Vector database for document storage and retrieval
2. **Tavily**: Web search API for real-time information
3. **Jina AI**: Content extraction from web pages
4. **LLM Models**: OSS 20B and OSS 120B for text generation

## File Dependencies Summary

### Core Application Files
- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `api_keys.txt.example` - API credentials template (copy to api_keys.txt for use)

### Ingestion Pipeline
- `src/research_gap_agent/Ingestion/pdf_ingestion.py` - Main ingestion orchestrator
- `src/research_gap_agent/Ingestion/parser.py` - PDF text extraction
- `src/research_gap_agent/Ingestion/semantic_chunker.py` - Document chunking
- `src/research_gap_agent/Ingestion/embedder.py` - Embedding generation
- `src/research_gap_agent/Ingestion/pinecone_upsert.py` - Vector database upload

### Graph and Nodes
- `src/research_gap_agent/graphs/basic_qa_graph.py` - LangGraph definition
- `src/research_gap_agent/nodes/retrieval.py` - Document retrieval
- `src/research_gap_agent/nodes/web_search.py` - Web search
- `src/research_gap_agent/nodes/generator.py` - Answer generation
- `src/research_gap_agent/nodes/mindmap_node.py` - Mindmap generation

### State and UI
- `src/research_gap_agent/state/state.py` - State management
- `src/research_gap_agent/ui/chat.py` - Chat interface utilities
- `src/research_gap_agent/ui/uploader.py` - File upload utilities

### Services
- `src/research_gap_agent/services/` - Various service integrations
- `src/research_gap_agent/llm/` - LLM model implementations

## Data Flow Summary

1. **Ingestion Flow**: PDF → parser → chunker → embedder → Pinecone
2. **Query Flow**: Query → retrieval → web_search → generator → mindmap
3. **State Flow**: State travels through all nodes, accumulating results
4. **UI Flow**: Streamlit manages user interaction and displays results

This architecture enables efficient document analysis, retrieval-augmented generation, and research gap discovery with both local PDF content and web-supplemented information.