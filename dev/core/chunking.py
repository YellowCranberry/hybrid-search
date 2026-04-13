from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List, Dict, Any
from langchain_core.documents import Document

class ChunkingManager:
    def __init__(self, embedding_model):
        """
        Initializes the chunker. We pass in the embedding model from the main Engine
        so we don't have to load the model into memory twice.
        """
        # The Semantic Chunker uses math to find topic changes.
        # 'percentile' means it will split at the top X% of sudden shifts in meaning.
        self.semantic_chunker = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85 # 85 is a great default for blog posts/docs
        )
        
        # We also set up a Markdown Splitter as a pre-processor 
        # (Highly recommended for documentation)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )

    def process_text(self, text: str, metadata: dict, use_markdown_preprocessing: bool = True) -> List[Document]:
        """
        Takes raw text, applies smart chunking, and attaches metadata to EVERY chunk.
        """
        final_chunks = []

        if use_markdown_preprocessing:
            # Step 1: Split by Markdown headers first (Keeps H2 sections together)
            md_chunks = self.markdown_splitter.split_text(text)
            
            # Step 2: Apply semantic splitting to any sections that are still too long
            for md_chunk in md_chunks:
                # Merge the markdown metadata (like "Header 2": "CORS Setup") with the user's DB metadata
                combined_metadata = {**metadata, **md_chunk.metadata}
                
                semantic_sub_chunks = self.semantic_chunker.create_documents(
                    [md_chunk.page_content], 
                    metadatas=[combined_metadata]
                )
                final_chunks.extend(semantic_sub_chunks)
        else:
            # Pure semantic chunking (Great for plain text without formatting)
            final_chunks = self.semantic_chunker.create_documents([text], metadatas=[metadata])

        return final_chunks