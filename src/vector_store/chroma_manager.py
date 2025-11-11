"""
Chroma Manager Module
Purpose: Manage ChromaDB vector store for storing and retrieving embeddings.
Handles insertion, querying, and persistence of vector embeddings for RAG.

Logic:
1. Initialize ChromaDB client with persistent storage
2. Create/load collection for storing embeddings
3. Add embedded chunks to the collection
4. Query collection for semantic similarity search
5. Return top-k most relevant chunks with metadata
6. Handle collection management (delete, reset, stats)
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
from src.utils.logger import get_logger
from src.utils.file_utils import load_json_config

logger = get_logger()


class ChromaManager:
    """
    Manages ChromaDB vector database for RAG retrieval.
    Stores embeddings and enables semantic search.
    """
    
    def __init__(
        self,
        persist_directory: str = "data/embeddings",
        collection_name: str = "pdf_embeddings"
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        logger.info(f"Initializing ChromaDB at: {persist_directory}")
        
        # Create persistent ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
        
        # Get or create collection
        self.collection = None
        logger.info(f"Collection name: {collection_name}")
    
    def get_or_create_collection(
        self, 
        embedding_dim: int = 384,
        distance_metric: str = "cosine"
    ) -> chromadb.Collection:
        """
        Get existing collection or create a new one.
        
        Args:
            embedding_dim: Dimension of embeddings (384 for all-MiniLM-L6-v2)
            distance_metric: Distance metric for similarity ("cosine", "l2", "ip")
        
        Returns:
            ChromaDB collection
        """
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
            logger.info(f"Collection count: {self.collection.count()}")
        except:
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection: {self.collection_name}")
            
            metadata = {
                "hnsw:space": distance_metric,  # cosine, l2, or ip (inner product)
                "dimension": embedding_dim
            }
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata=metadata
            )
            logger.info(f"Created new collection with {distance_metric} distance")
        
        return self.collection
    
    def add_chunks(
        self,
        embedded_chunks: List[Dict[str, Any]],
        batch_size: int = 100,
        skip_existing: bool = True
    ) -> int:
        """
        Add embedded chunks to the collection.
        
        Args:
            embedded_chunks: List of chunks with embeddings from embedder
            batch_size: Number of chunks to add per batch
        
        Returns:
            Number of chunks added
        """
        if not self.collection:
            logger.error("Collection not initialized. Call get_or_create_collection first.")
            raise ValueError("Collection not initialized")
        
        logger.info(f"Adding {len(embedded_chunks)} chunks to collection")
        
        total_added = 0
        
        # Process in batches
        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i:i+batch_size]
            
            # Prepare data for ChromaDB
            ids = [chunk['chunk_id'] for chunk in batch]

            # Optionally skip IDs that already exist in the collection
            if skip_existing:
                existing_ids = self.get_existing_ids(ids)
                if existing_ids:
                    original_len = len(ids)
                    # Filter batch to only new items
                    filtered = [c for c in batch if c['chunk_id'] not in existing_ids]
                    batch = filtered
                    ids = [chunk['chunk_id'] for chunk in batch]
                    logger.debug(f"Batch {i//batch_size + 1}: Skipping {original_len - len(ids)} existing chunks")

            if not batch:
                continue

            embeddings = [chunk['embedding'] for chunk in batch]
            documents = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                total_added += len(batch)
                logger.debug(f"Added batch {i//batch_size + 1}: {len(batch)} chunks")
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(f"Successfully added {total_added} chunks to collection")
        return total_added
    
    def add_from_embeddings_file(
        self,
        embeddings_json_path: str,
        batch_size: int = 100,
        skip_existing: bool = True
    ) -> int:
        """
        Load embeddings from JSON and add to collection.
        
        Args:
            embeddings_json_path: Path to embeddings JSON from embedder
            batch_size: Batch size for adding
        
        Returns:
            Number of chunks added
        """
        logger.info(f"Loading embeddings from: {embeddings_json_path}")
        
        embeddings_data = load_json_config(embeddings_json_path)
        embedded_chunks = embeddings_data['embedded_chunks']
        
        # Get embedding dimension from first chunk
        embedding_dim = len(embedded_chunks[0]['embedding'])
        
        # Ensure collection exists
        self.get_or_create_collection(embedding_dim=embedding_dim)
        
        # Add chunks
        count = self.add_chunks(embedded_chunks, batch_size, skip_existing=skip_existing)
        
        return count

    def get_existing_ids(self, ids: List[str], batch_size: int = 1000) -> set:
        """
        Return the subset of provided IDs that already exist in the collection.

        Args:
            ids: List of chunk IDs to check
            batch_size: Number of IDs to check per call

        Returns:
            Set of IDs that are already present
        """
        if not self.collection:
            raise ValueError("Collection not initialized")

        existing = set()
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i+batch_size]
            try:
                got = self.collection.get(ids=batch)
                # Some Chroma versions return ids under 'ids'; handle both possibilities
                returned_ids = []
                if isinstance(got, dict):
                    returned_ids = got.get('ids', [])
                # Normalize to flat list
                if isinstance(returned_ids, list):
                    # Some clients return a list (not nested) for get
                    existing.update(returned_ids)
            except Exception as e:
                logger.error(f"Error checking existing IDs: {e}")
                # Be conservative: don't mark any as existing for this batch
                continue
        return existing
    
    def query(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the collection for similar chunks.
        
        Args:
            query_text: Query text (for ChromaDB's built-in embedding, not used here)
            query_embedding: Pre-computed query embedding (from embedder)
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"page_number": 5})
        
        Returns:
            Dictionary with results: ids, documents, metadatas, distances
        """
        if not self.collection:
            logger.error("Collection not initialized")
            raise ValueError("Collection not initialized")
        
        logger.info(f"Querying collection for top {top_k} results")
        
        try:
            if query_embedding:
                # Use provided embedding (recommended for consistency with embedder)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter_metadata
                )
            else:
                # Fallback: let ChromaDB handle embedding (not recommended)
                logger.warning("No query embedding provided, using ChromaDB's built-in embedding")
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=top_k,
                    where=filter_metadata
                )
            
            logger.info(f"Query returned {len(results['ids'][0])} results")
            return results
        
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise
    
    def get_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
        
        Returns:
            Dictionary with chunk data
        """
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        try:
            result = self.collection.get(ids=[chunk_id])
            return result
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {e}")
            raise
    
    def delete_collection(self) -> None:
        """
        Delete the current collection.
        """
        if not self.collection:
            logger.warning("No collection to delete")
            return
        
        logger.info(f"Deleting collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            logger.info("Collection deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def reset_collection(self) -> None:
        """
        Reset the collection (delete and recreate).
        """
        logger.info(f"Resetting collection: {self.collection_name}")
        
        try:
            self.delete_collection()
            self.get_or_create_collection()
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection stats
        """
        if not self.collection:
            return {
                'collection_name': self.collection_name,
                'count': 0,
                'exists': False
            }
        
        try:
            count = self.collection.count()
            
            stats = {
                'collection_name': self.collection_name,
                'count': count,
                'exists': True,
                'persist_directory': self.persist_directory
            }
            
            logger.info(f"Collection stats: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise
    
    def list_all_collections(self) -> List[str]:
        """
        List all collections in the ChromaDB client.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            logger.info(f"Found {len(collection_names)} collections: {collection_names}")
            return collection_names
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise

    def list_distinct_metadata_values(self, field: str, page_size: int = 1000) -> List[Any]:
        """
        List distinct values for a metadata field by paging through the collection.

        Note: For large collections this may be expensive.

        Args:
            field: Metadata field name (e.g., 'pdf_name')
            page_size: Page size for iteration

        Returns:
            Sorted list of distinct metadata values
        """
        if not self.collection:
            return []

        try:
            total = self.collection.count()
            values = set()
            offset = 0
            while offset < total:
                got = self.collection.get(
                    include=["metadatas"],
                    limit=page_size,
                    offset=offset
                )
                metadatas = got.get('metadatas', []) if isinstance(got, dict) else []
                for md in metadatas:
                    if isinstance(md, dict) and field in md:
                        values.add(md[field])
                # If fewer than page_size returned, we're done
                returned = len(metadatas)
                if returned == 0:
                    break
                offset += returned
            return sorted(list(values))
        except Exception as e:
            logger.error(f"Error listing distinct metadata values for '{field}': {e}")
            return []

    def list_pdf_names(self) -> List[str]:
        """
        Convenience method to list distinct pdf_name values present in the collection.
        """
        return self.list_distinct_metadata_values('pdf_name')


# ============================================
# Usage Example (for testing)
# ============================================
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    from src.embeddings.embedder import TextEmbedder
    
    # Setup logger for testing
    setup_logger()
    
    print("\n" + "="*60)
    print("CHROMADB MANAGER TEST")
    print("="*60)
    
    # Initialize ChromaDB manager
    chroma = ChromaManager(
        persist_directory="data/embeddings",
        collection_name="pdf_embeddings"
    )
    
    # List existing collections
    print("\nExisting collections:")
    collections = chroma.list_all_collections()
    for col in collections:
        print(f"  - {col}")
    
    # Load embeddings and add to collection
    print("\n" + "="*60)
    print("ADDING EMBEDDINGS TO COLLECTION")
    print("="*60)
    
    embeddings_path = "data/embeddings/obl_test_pdf_rag1_embeddings.json"
    count = chroma.add_from_embeddings_file(embeddings_path)
    
    print(f"\n✓ Added {count} chunks to collection")
    
    # Get collection stats
    stats = chroma.get_collection_stats()
    print(f"\nCollection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test query
    print("\n" + "="*60)
    print("TESTING SEMANTIC SEARCH")
    print("="*60)
    
    # Create embedder to embed query
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    
    # Example query
    query_text = "What is attitude in IT teams?"
    print(f"\nQuery: \"{query_text}\"")
    
    # Embed the query
    query_embedding = embedder.embed_text(query_text)
    
    # Search
    results = chroma.query(
        query_text=query_text,
        query_embedding=query_embedding.tolist(),
        top_k=3
    )
    
    # Display results
    print(f"\nTop 3 results:")
    for i, (doc_id, document, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n--- Result {i+1} ---")
        print(f"ID: {doc_id}")
        print(f"Page: {metadata['page_number']}")
        print(f"Similarity Score: {1 - distance:.4f}")  # Convert distance to similarity
        print(f"Text: {document[:150]}...")
    
    print("\n✅ ChromaDB test complete!")
