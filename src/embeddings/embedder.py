"""
Embedder Module
Purpose: Convert text chunks into dense vector embeddings for semantic search.
Uses sentence-transformers to create embeddings that capture semantic meaning.

Logic:
1. Load pre-trained sentence-transformer model (e.g., all-MiniLM-L6-v2)
2. Take chunks from chunker.py
3. Batch process chunks for efficiency
4. Generate 384-dimensional embeddings
5. Store embeddings with chunk metadata
6. Handle caching to avoid re-embedding same texts
7. Return chunks with embeddings ready for vector store
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger
from src.utils.file_utils import load_json_config, save_json, ensure_directory_exists

logger = get_logger()


class TextEmbedder:
    """
    Creates vector embeddings from text chunks using sentence-transformers.
    Embeddings enable semantic search for RAG retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 8,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Name of sentence-transformer model to use
            batch_size: Number of texts to embed in one batch
            cache_dir: Directory to cache model files (uses HF_HOME if None)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_name}")
        
        # Load the model (will use HF_HOME/TRANSFORMERS_CACHE from env)
        try:
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
            logger.info(f"Model loaded successfully")
            logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedder initialized with batch_size={batch_size}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a batch of texts efficiently.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        try:
            logger.debug(f"Embedding batch of {len(texts)} texts")
            
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed all chunks and add embeddings to their data.
        
        Args:
            chunks: List of chunk dictionaries from chunker
        
        Returns:
            Chunks with embeddings added
        """
        logger.info(f"Starting embedding for {len(chunks)} chunks")
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Embed all texts
        embeddings = self.embed_batch(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            # Create a copy of the chunk
            embedded_chunk = chunk.copy()
            
            # Add embedding (convert to list for JSON serialization)
            embedded_chunk['embedding'] = embedding.tolist()
            embedded_chunk['embedding_model'] = self.model_name
            embedded_chunk['embedding_dim'] = self.embedding_dim
            
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"Embedding complete: {len(embedded_chunks)} chunks embedded")
        return embedded_chunks
    
    def embed_from_chunks_file(self, chunks_json_path: str) -> List[Dict[str, Any]]:
        """
        Load chunks from JSON and embed them.
        
        Args:
            chunks_json_path: Path to chunks JSON file from chunker
        
        Returns:
            Embedded chunks
        """
        logger.info(f"Loading chunks from: {chunks_json_path}")
        
        chunks_data = load_json_config(chunks_json_path)
        chunks = chunks_data['chunks']
        
        logger.info(f"Loaded {len(chunks)} chunks")
        
        embedded_chunks = self.embed_chunks(chunks)
        
        return embedded_chunks
    
    def save_embeddings(
        self,
        embedded_chunks: List[Dict[str, Any]],
        output_dir: str,
        pdf_name: str
    ) -> str:
        """
        Save embedded chunks to JSON file.
        
        Args:
            embedded_chunks: Chunks with embeddings
            output_dir: Directory to save embeddings
            pdf_name: Name of source PDF (for filename)
        
        Returns:
            Path to saved file
        """
        ensure_directory_exists(output_dir)
        
        # Create output structure
        output = {
            'metadata': {
                'pdf_name': pdf_name,
                'total_chunks': len(embedded_chunks),
                'embedding_model': self.model_name,
                'embedding_dim': self.embedding_dim
            },
            'embedded_chunks': embedded_chunks
        }
        
        output_filename = f"{pdf_name}_embeddings.json"
        output_path = f"{output_dir}/{output_filename}"
        
        save_json(output, output_path)
        logger.info(f"Saved {len(embedded_chunks)} embeddings to: {output_path}")
        
        return output_path
    
    def get_embedding_stats(self, embedded_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the embeddings.
        
        Args:
            embedded_chunks: Chunks with embeddings
        
        Returns:
            Dictionary with statistics
        """
        if not embedded_chunks:
            return {
                'total_embeddings': 0,
                'embedding_dim': 0,
                'model': self.model_name
            }
        
        # Calculate average embedding norm (as a sanity check)
        embeddings = [np.array(chunk['embedding']) for chunk in embedded_chunks]
        norms = [np.linalg.norm(emb) for emb in embeddings]
        
        stats = {
            'total_embeddings': len(embedded_chunks),
            'embedding_dim': self.embedding_dim,
            'model': self.model_name,
            'avg_embedding_norm': round(np.mean(norms), 4),
            'min_embedding_norm': round(np.min(norms), 4),
            'max_embedding_norm': round(np.max(norms), 4)
        }
        
        logger.info(f"Embedding stats: {stats}")
        return stats
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score (0-1, higher = more similar)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)


# ============================================
# Usage Example (for testing)
# ============================================
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    import time
    
    # Setup logger for testing
    setup_logger()
    
    # Example: Load chunks and embed them
    chunks_json_path = "data/processed_chunks/obl_test_pdf_rag1_chunks.json"
    
    # Create embedder
    embedder = TextEmbedder(
        model_name="all-MiniLM-L6-v2",
        batch_size=8
    )
    
    # Embed chunks
    print("\n" + "="*60)
    print("EMBEDDING CHUNKS")
    print("="*60)
    
    start_time = time.time()
    embedded_chunks = embedder.embed_from_chunks_file(chunks_json_path)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Embedded {len(embedded_chunks)} chunks in {elapsed:.2f} seconds")
    
    # Get statistics
    stats = embedder.get_embedding_stats(embedded_chunks)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show sample embedding
    print(f"\nFirst chunk:")
    print(f"  ID: {embedded_chunks[0]['chunk_id']}")
    print(f"  Text: {embedded_chunks[0]['text'][:100]}...")
    print(f"  Embedding shape: {len(embedded_chunks[0]['embedding'])}")
    print(f"  Embedding (first 5 dims): {embedded_chunks[0]['embedding'][:5]}")
    
    # Test similarity between first two chunks
    if len(embedded_chunks) >= 2:
        emb1 = np.array(embedded_chunks[0]['embedding'])
        emb2 = np.array(embedded_chunks[1]['embedding'])
        similarity = embedder.compute_similarity(emb1, emb2)
        print(f"\nSimilarity between chunk 0 and chunk 1: {similarity:.4f}")
    
    # Save embeddings
    output_path = embedder.save_embeddings(
        embedded_chunks=embedded_chunks,
        output_dir="data/embeddings",
        pdf_name="obl_test_pdf_rag1"
    )
    
    print(f"\n✅ Embeddings saved to: {output_path}")
