# hybrid-search

A completely self-hosted, plug-and-play Python package for **hybrid search** (vector + keyword) and **Retrieval-Augmented Generation (RAG)**. 

Designed specifically for self-hosting on your own infrastructure without mandatory API dependencies, this library handles:
- **Semantic Chunking:** Automatically cleans HTML, processes markdown, and chunks your text intelligently.
- **Local Embeddings:** Free, entirely local vector embeddings on CPU/GPU using HuggingFace sentence-transformers.
- **Hybrid Retrieval:** True hybrid search using PostgreSQL (`pgvector` for vector similarity and `tsvector` for full-text keyword matching) combined via Reciprocal Rank Fusion (RRF).
- **RAG Answering:** Generate final answers using free, local HuggingFace LLMs natively (or extract precise context).
- **Delta Sync:** Keep your website database in sync with your vector database efficiently via our update manager.

## Requirements

1. **Python 3.9+**
2. **PostgreSQL 15+** with the **`pgvector`** extension installed and enabled on your database.
3. *(Optional)* A GPU with PyTorch/CUDA to drastically speed up local embeddings and LLM answer generation.

## Installation

Install the package directly from your local directory:

```bash
# Clone or navigate to the package directory
cd path/to/search_package

# Install the package and its dependencies
pip install .

# For development mode (editable)
pip install -e .
```

## Quick Start

### 1. Connecting the Engine

The main entry point is the `HybridSearchEngine`. You will need a standard PostgreSQL connection URI. `pgvector` will automatically create the required schema for your application.

```python
from dev.core.engine import HybridSearchEngine

engine = HybridSearchEngine(
    db_url="postgresql+psycopg://postgres:password@localhost:5432/mydb",
    collection_name="hybrid_search_docs",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2", # Default, local
    llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"            # Default, local (Set to None to disable)
)
```

### 2. Ingesting Data

You can chunk, embed, and store plain text or HTML/Markdown content dynamically. Metadata is stored securely as JSONB in PostgreSQL, allowing for lightning-fast filtering and rendering contexts.

```python
text_content = """
    <h1>Refund Policy</h1>
    <p>You can refund any item within 30 days of purchase, no questions asked.</p>
"""

metadata = {
    "url": "/policies/refund",
    "title": "Refund Policy"
}

# The engine automatically strips HTML and chunks the content!
chunks_stored = engine.ingest_text(text=text_content, metadata=metadata)
print(f"Stored {chunks_stored} chunks!")
```

Alternatively, you can ingest entire tables natively using an existing PostgreSQL or external database:
```python
engine.ingest_from_db(
    source_db_url="postgresql+psycopg://...",
    table_name="website_articles",
    content_col="body_text",
    meta_cols=["title", "url", "author"]
)
```

### 3. Searching 

You can perform either a lightning-fast vector search, or a robust hybrid search (Vector + Full-Text Keyword).

**Pure Vector Search**
```python
results = engine.search("How long do I have to refund?", top_k=5)
for r in results:
    print(r["score"], r["metadata"], r["content"])
```

**Hybrid Search (Recommended)**
Combines pgvector HNSW similarity + keyword match using Reciprocal Rank Fusion. Highly resilient for specific term lookups.
```python
results = engine.search_hybrid("30 day refund policy", top_k=5)
for r in results:
    print(r["score"], r["metadata"], r["content"])
```

### 4. RAG Q&A (Ask)

Pass the user's natural language question straight to the engine, and it retrieves passages and generates an answer locally using the HuggingFace LLM!

```python
response = engine.ask(
    query="What is the refund policy?",
    top_k=3,
    use_hybrid=True
)

print("Answer:", response["answer"])
print("Sources:", response["sources"])
```

### 5. Website Delta Sync Configuration

*(Refer to `sync_search.py` for full implementation examples.)*
If you have an active website, use the built-in `UpdateManager` to perform delta synchronizations. This checks `updated_at` timestamps on your database and only ingests modifying or newly added documents without wiping out your entire vector store.

```python
from dev.pipeline.update_manager import UpdateManager

tracker = UpdateManager(
    engine=engine,
    source_db_url="postgresql+psycopg://...",
    table_name="website_content",
    content_col="article_body",
    meta_cols=["title", "url"],
    last_sync_timestamp="2023-10-01 00:00:00"
)

processed = tracker.sync()
print(f"Synced {processed} new or updated documents.")
```
