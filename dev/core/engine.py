from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_core.documents import Document
from ..adapters.sql_adapter import SQLAdapter
from .chunking import ChunkingManager


class HybridSearchEngine:
    def __init__(self, db_url: str, collection_name: str = "my_docs"):
        """
        Initialize the engine with a local embedding model and Postgres connection.
        db_url example: "postgresql+psycopg://user:password@localhost:5432/dbname"
        """
        # 1. Load Local, Free Embeddings (Runs entirely on CPU/GPU)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 2. Connect to PostgreSQL (Requires pgvector extension installed)
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=db_url,
            use_jsonb=True, # Great for storing metadata
        )
        

        # 3. INITIALIZE OUR NEW ADVANCED CHUNKER
        # Notice how we pass the embeddings model directly into it!
        self.chunker = ChunkingManager(embedding_model=self.embeddings)

    
    def ingest_text(self, text: str, metadata: dict):
        """Chunks the text, embeds it, and saves to PostgreSQL."""
        
        documents = self.chunker.process_text(text=text, metadata=metadata)
        
        # 3. Insert into Postgres (Automatically embeds the text)
        self.vector_store.add_documents(documents)
        print(f"Successfully ingested {len(documents)} chunks.")

    def search(self, query: str, top_k: int = 5):
        """Performs a vector similarity search in Postgres."""
        # PGVector handles embedding the user's query and finding the closest matches
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score # Lower score = closer match depending on distance metric
            })
            
        return formatted_results
    
    def ingest_from_db(self, source_db_url: str, table_name: str, content_col: str, meta_cols: list):
        """
        A single command for the user to pull from their DB and push into the vector DB.
        """
        print(f"Connecting to source database to fetch {table_name}...")
        
        # 1. Initialize our new adapter
        adapter = SQLAdapter(source_db_url)
        
        # 2. Fetch the formatted data
        raw_documents = adapter.fetch_data(table_name, content_col, meta_cols)
        
        # 3. Loop through and ingest each one using our existing logic
        for doc in raw_documents:
            # We built this method in the previous step!
            self.ingest_text(text=doc["text"], metadata=doc["metadata"])
            
        print("Database ingestion complete!")