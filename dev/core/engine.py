"""
engine.py
---------
The main entry point for the hybrid search package.

HybridSearchEngine provides:
    1. ingest_text()       — chunk + embed + store
    2. search()            — pure vector similarity search
    3. search_hybrid()     — vector search + cross-encoder reranking OR keyword search + vector search
    4. ask()               — full RAG: search + rerank + LLM answer
    5. ingest_from_db()    — pull from SQL DB and ingest
"""

import sqlalchemy as sa
from typing import List, Dict, Any, Optional, Literal

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

from .chunking import ChunkingManager


# --------------------------------------------------------------------------
# Optional LLM backends — imported lazily so the package works without them
# --------------------------------------------------------------------------

def _get_huggingface_llm(model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline
    
    # We use a lightweight local model default so it can run reasonably on CPU
    # If they have a GPU, they can configure device=0
    hf_pipeline = pipeline(
        "text-generation",
        model=model_id,
        max_new_tokens=256,
        temperature=0.1,
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


class HybridSearchEngine:
    """
    Plug-and-play hybrid search engine powered by:
        - HuggingFace sentence-transformers (free, local embeddings)
        - PostgreSQL + pgvector (vector storage + keyword search via RRF)
        - HuggingFace local LLM for Q&A (completely free/local)

    Quick start:
        engine = HybridSearchEngine(
            db_url="postgresql+psycopg://user:pass@localhost:5432/mydb"
        )
        engine.ask("What is the refund policy?")
    """

    def __init__(
        self,
        db_url: str,
        collection_name: str = "hybrid_search_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: Optional[str] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ):
        """
        Args:
            db_url:           PostgreSQL connection string.
                              Format: "postgresql+psycopg://user:pass@host:port/dbname"
            collection_name:  Name of the pgvector collection (creates its own table).
            embedding_model:  HuggingFace model for embeddings (free, runs locally).
                              Default: all-MiniLM-L6-v2 (fast, 384 dims, great quality)
            llm_model:        The HuggingFace model used to generate RAG answers locally.
                              Default: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (runs reasonably on CPU).
                              Set to None to disable answer generation and just return the context.
        """
        # SQLAlchemy URL format handling
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://")
            
        self.db_url = db_url
        self.collection_name = collection_name

        # 1. Load the embedding model (runs on CPU/GPU, no API key needed)
        print(f"[Engine] Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},  # Change to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},  # Cosine similarity
        )

        # 2. Connect to PostgreSQL + pgvector
        #    PGVector auto-creates the collection table if it doesn't exist.
        print(f"[Engine] Connecting to PostgreSQL (collection: {collection_name})")
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=db_url,
            use_jsonb=True,  # Stores metadata as JSONB for fast filtering
        )

        # 3. Semantic + Markdown chunker (shares the embedding model — no double-load)
        self.chunker = ChunkingManager(embedding_model=self.embeddings)

        # 4. LLM (lazy — only instantiated when ask() is called)
        self._llm_model = llm_model
        self._llm = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_text(self, text: str, metadata: dict) -> int:
        """
        Chunk, embed, and store a single document.

        Args:
            text:     Raw text content of the page/document.
            metadata: Dict with at least {"url": ..., "title": ...}
                      All keys are stored in pgvector's JSONB metadata column.

        Returns:
            Number of chunks stored.
        """
        use_markdown = self._looks_like_markdown(text)
        chunks: List[Document] = self.chunker.process_text(
            text=text,
            metadata=metadata,
            use_markdown_preprocessing=use_markdown,
            strip_html=True # strips html out before chunking!
        )

        if not chunks:
            print(f"[Engine] Warning: No chunks produced for {metadata.get('url', 'unknown')}")
            return 0

        self.vector_store.add_documents(chunks)
        return len(chunks)

    def ingest_from_db(
        self,
        source_db_url: str,
        table_name: str,
        content_col: str,
        meta_cols: List[str],
    ) -> None:
        """Pull content from any SQL database and ingest it (no delta-sync here)."""
        from ..adapters.sql_adapter import SQLAdapter
        adapter = SQLAdapter(source_db_url)
        rows = adapter.fetch_data(table_name, content_col, meta_cols)
        total = 0
        for row in rows:
            n = self.ingest_text(text=row["text"], metadata=row["metadata"])
            total += n
        print(f"[Engine] DB ingestion complete. {total} chunks stored.")


    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        unique_by_metadata: Optional[str] = "title"
    ) -> List[Dict[str, Any]]:
        """
        Pure vector similarity search (fast, no reranking).
        Deduplicates chunks so the same article doesn't show up 5 times.
        """
        # Fetch extra so we can filter duplicates
        results = self.vector_store.similarity_search_with_score(query, k=top_k * 5)
        
        formatted_results = []
        seen = set()
        
        for doc, score in results:
            if unique_by_metadata:
                val = doc.metadata.get(unique_by_metadata)
                if val:
                    if val in seen:
                        continue
                    seen.add(val)
                    
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            })
            if len(formatted_results) == top_k:
                break
            
        return formatted_results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        unique_by_metadata: Optional[str] = "title"
    ) -> List[Dict[str, Any]]:
        """
        True Hybrid Search using PostgreSQL (pgvector HNSW + tsvector Keyword Search).
        Combines semantic similarity and full-text keyword matching using Reciprocal Rank Fusion (RRF).
        """
        # 1. Embed the query to get the target vector
        query_embedding = self.embeddings.embed_query(query)

        # 2. Extract collection_id from vector store and run SQL
        with self.vector_store.session_maker() as session:
            # We must load the collection to find its ID in the DB
            collection = self.vector_store.get_collection(session)
            if not collection:
                return []
            collection_id = collection.uuid

            # 3. Hybrid Search Query (Reciprocal Rank Fusion in raw SQL)
            # RRF combines the ranks: score = 1 / (60 + rank)
            sql = sa.text('''
            WITH vector_search AS (
                SELECT id, document, cmetadata,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> cast(:query_embedding as vector)) AS rank
                FROM langchain_pg_embedding
                WHERE collection_id = :collection_id
                LIMIT :top_k
            ),
            text_search AS (
                SELECT id, document, cmetadata,
                       ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english', document), websearch_to_tsquery('english', :query_text)) DESC) AS rank
                FROM langchain_pg_embedding
                WHERE collection_id = :collection_id
                  AND to_tsvector('english', document) @@ websearch_to_tsquery('english', :query_text)
                LIMIT :top_k
            )
            SELECT
                COALESCE(v.id, t.id) AS id,
                COALESCE(v.document, t.document) AS document,
                COALESCE(v.cmetadata, t.cmetadata) AS cmetadata,
                COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + t.rank), 0.0) AS rrf_score
            FROM vector_search v
            FULL OUTER JOIN text_search t ON v.id = t.id
            ORDER BY rrf_score DESC
            LIMIT :top_k;
            ''')

            try:
                results = session.execute(sql, {
                    "query_embedding": str(query_embedding),
                    "query_text": query,
                    "collection_id": collection_id,
                    "top_k": top_k * 5  # fetch extra for deduplication
                }).fetchall()
            except sa.exc.ProgrammingError as e:
                # Fallback if there's an exact syntax issue
                print(f"[Engine] Hybrid SQL error: {e}. Falling back to normal vector search.")
                return self.search(query, top_k)

        # 4. Format the candidates and Deduplicate
        candidates = []
        seen = set()
        
        for rank, result in enumerate(results, start=1):
            if unique_by_metadata:
                val = result.cmetadata.get(unique_by_metadata)
                if val:
                    if val in seen:
                        continue
                    seen.add(val)
            
            # Reciprocal Rank Fusion (RRF) outputs raw math scores (max ~0.0327)
            # We normalize this to a beautiful 0.0 -> 1.0 (0% to 100%) confidence scale for users
            raw_score = float(result.rrf_score)
            max_possible_rrf = (1.0 / 61.0) * 2  # rank 1 in both searches
            normalized_score = min(raw_score / max_possible_rrf, 1.0)
                    
            candidates.append({
                "content": result.document,
                "metadata": result.cmetadata,
                "score": round(normalized_score, 4),
                "rank": len(candidates) + 1
            })
            
            if len(candidates) == top_k:
                break

        return candidates


    # ------------------------------------------------------------------
    # RAG answer generation
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve → generate answer.

        Args:
            query:        The user's natural language question.
            top_k:        Final passages to pass to the LLM.
            use_hybrid:   Whether to use Hybrid Search (Vector + Text).

        Returns:
            {
                "answer":  str,          # LLM-generated answer
                "sources": list[dict],   # Passages used (content + metadata)
                "query":   str,          # Original query
            }
        """
        # 1. Retrieve
        if use_hybrid:
            results = self.search_hybrid(query, top_k=top_k)
        else:
            results = self.search(query, top_k=top_k)

        if not results:
            return {"answer": "No relevant content found.", "sources": [], "query": query}

        # 2. Build context for the LLM
        context_parts = []
        for i, r in enumerate(results, start=1):
            title = r["metadata"].get("title", "")
            url = r["metadata"].get("url", "")
            header = f"[Source {i}]"
            if title:
                header += f" {title}"
            if url:
                header += f" ({url})"
            context_parts.append(f"{header}\n{r['content']}")

        context = "\n\n---\n\n".join(context_parts)

        # 3. Generate answer via LLM
        llm = self._get_llm()
        if llm is None:
            # No LLM configured — return raw context as answer
            return {
                "answer": context,
                "sources": results,
                "query": query,
            }

        prompt = self._build_rag_prompt(query, context)

        # Using HuggingFacePipeline which evaluates directly as a runnable
        response = llm.invoke(prompt)
        answer = response

        return {
            "answer": answer,
            "sources": results,
            "query": query,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_llm(self):
        """Lazy-load the local HuggingFace LLM on first ask()."""
        if self._llm is not None:
            return self._llm
        if not self._llm_model:
            return None
            
        print(f"[Engine] Loading HuggingFace LLM: {self._llm_model}...")
        self._llm = _get_huggingface_llm(self._llm_model)
        return self._llm

    @staticmethod
    def _build_rag_prompt(query: str, context: str) -> str:
        return f"""You are a helpful assistant. Answer the user's question using ONLY the sources provided below.
If the sources don't contain enough information to answer, say so honestly.
Be concise and cite the source number when referencing specific information.

--- SOURCES ---
{context}
--- END OF SOURCES ---

Question: {query}

Answer:"""

    @staticmethod
    def _looks_like_markdown(text: str) -> bool:
        """Heuristic: treat text as markdown if it has common markdown patterns."""
        markers = ["# ", "## ", "### ", "**", "```", "- [", "> "]
        return any(m in text[:2000] for m in markers)