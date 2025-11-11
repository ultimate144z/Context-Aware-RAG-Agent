"""Quick test to check pdf_embeddings collection"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vector_store.chroma_manager import ChromaManager

# Check pdf_embeddings collection
chroma = ChromaManager(
    collection_name="pdf_embeddings",
    persist_directory="data/embeddings"
)
chroma.get_or_create_collection(embedding_dim=384)

stats = chroma.get_collection_stats()
print(f"Collection: {stats['collection_name']}")
print(f"Total chunks: {stats['count']}")

if stats['count'] > 0:
    print("\n✅ Data found in 'pdf_embeddings' collection!")
    print("   The issue is that the system is looking at 'rag_documents' instead.")
else:
    print("\n❌ No data in 'pdf_embeddings' either!")
