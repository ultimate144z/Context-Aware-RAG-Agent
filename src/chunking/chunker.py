"""
Chunker Module
Purpose: Split extracted text into manageable chunks for embedding and retrieval.
Takes JSON output from cleaner and creates overlapping chunks with metadata.

Logic:
1. Load extracted JSON from cleaner (pages with metadata)
2. For each page, split text into ~400-word chunks with 50-word overlap
3. Preserve page-level metadata (page_number, pdf_name, extraction_method)
4. Add chunk-level metadata (chunk_id, word_count, position_in_page)
5. Handle edge cases (very short pages, last chunks)
6. Save chunks as JSON for embedding pipeline
7. Return list of chunk objects ready for embedding
"""

import re
import json
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger
from src.utils.file_utils import load_json_config, save_json, ensure_directory_exists

logger = get_logger()


class TextChunker:
    """
    Splits extracted text into overlapping chunks for RAG pipeline.
    Preserves metadata from extraction for citation and tracking.
    """
    
    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target number of words per chunk
            overlap: Number of overlapping words between consecutive chunks
            min_chunk_size: Minimum words for a chunk to be kept (avoid tiny chunks)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
        logger.info("Initialized TextChunker")
        logger.info(f"Settings: chunk_size={chunk_size}, overlap={overlap}, min_chunk_size={min_chunk_size}")
    
    def split_text_into_words(self, text: str) -> List[str]:
        """
        Split text into words while preserving whitespace context.
        
        Args:
            text: Text to split
        
        Returns:
            List of words
        """
        # Split on whitespace but keep meaningful tokens
        words = text.split()
        return words
    
    def create_chunks_from_words(
        self, 
        words: List[str], 
        chunk_size: int, 
        overlap: int
    ) -> List[str]:
        """
        Create overlapping chunks from a list of words.
        
        Args:
            words: List of words to chunk
            chunk_size: Number of words per chunk
            overlap: Number of overlapping words
        
        Returns:
            List of text chunks
        """
        if not words:
            return []
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            # Get chunk_size words starting from start_idx
            end_idx = start_idx + chunk_size
            chunk_words = words[start_idx:end_idx]
            
            # Join words back into text
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            # Move forward by (chunk_size - overlap) for next chunk
            # This creates the overlap
            start_idx += (chunk_size - overlap)
            
            # If we're at the end, break
            if end_idx >= len(words):
                break
        
        return chunks
    
    def chunk_page(
        self, 
        page_data: Dict[str, Any], 
        pdf_name: str
    ) -> List[Dict[str, Any]]:
        """
        Chunk a single page's text into overlapping chunks with metadata.
        
        Args:
            page_data: Dictionary with page text and metadata
            pdf_name: Name of the source PDF
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        page_num = page_data['page_number']
        text = page_data['text']
        
        # Split into words
        words = self.split_text_into_words(text)
        
        # If page is too short, return it as a single chunk
        if len(words) < self.min_chunk_size:
            logger.debug(f"Page {page_num}: Too short ({len(words)} words), keeping as single chunk")
            return [{
                'chunk_id': f"{pdf_name}_page{page_num}_chunk0",
                'text': text,
                'metadata': {
                    'pdf_name': pdf_name,
                    'page_number': page_num,
                    'chunk_index': 0,
                    'total_chunks_in_page': 1,
                    'word_count': len(words),
                    'char_count': len(text),
                    'extraction_method': page_data['extraction_method'],
                    'has_ocr': page_data['has_ocr']
                }
            }]
        
        # Create chunks
        chunk_texts = self.create_chunks_from_words(words, self.chunk_size, self.overlap)
        
        # Create chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            chunk_words = self.split_text_into_words(chunk_text)
            
            # Skip if chunk is too small (except if it's the last chunk)
            if len(chunk_words) < self.min_chunk_size and idx < len(chunk_texts) - 1:
                logger.debug(f"Page {page_num}, Chunk {idx}: Skipped (too small: {len(chunk_words)} words)")
                continue
            
            chunk_obj = {
                'chunk_id': f"{pdf_name}_page{page_num}_chunk{idx}",
                'text': chunk_text,
                'metadata': {
                    'pdf_name': pdf_name,
                    'page_number': page_num,
                    'chunk_index': idx,
                    'total_chunks_in_page': len(chunk_texts),
                    'word_count': len(chunk_words),
                    'char_count': len(chunk_text),
                    'extraction_method': page_data['extraction_method'],
                    'has_ocr': page_data['has_ocr']
                }
            }
            
            chunks.append(chunk_obj)
            logger.debug(f"Page {page_num}, Chunk {idx}: Created ({len(chunk_words)} words)")
        
        return chunks
    
    def chunk_document(self, extracted_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk an entire document (all pages) into chunks.
        
        Args:
            extracted_json: Output from cleaner.py (structured document with pages)
        
        Returns:
            List of all chunks from all pages with metadata
        """
        pdf_name = extracted_json['metadata']['pdf_name']
        pages = extracted_json['pages']
        
        logger.info(f"Starting chunking for document: {pdf_name}")
        logger.info(f"Processing {len(pages)} pages")
        
        all_chunks = []
        
        for page_data in pages:
            page_chunks = self.chunk_page(page_data, pdf_name)
            all_chunks.extend(page_chunks)
        
        logger.info(f"Chunking complete: {len(all_chunks)} total chunks created")
        
        return all_chunks
    
    def chunk_from_json_file(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load extracted JSON and chunk it.
        
        Args:
            json_path: Path to the extracted JSON file from cleaner
        
        Returns:
            List of chunks
        """
        logger.info(f"Loading extracted JSON from: {json_path}")
        
        extracted_json = load_json_config(json_path)
        chunks = self.chunk_document(extracted_json)
        
        return chunks
    
    def save_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        output_dir: str, 
        pdf_name: str
    ) -> str:
        """
        Save chunks to JSON file.
        
        Args:
            chunks: List of chunk objects
            output_dir: Directory to save chunks
            pdf_name: Name of source PDF (for filename)
        
        Returns:
            Path to saved file
        """
        ensure_directory_exists(output_dir)
        
        # Create output structure
        output = {
            'metadata': {
                'pdf_name': pdf_name,
                'total_chunks': len(chunks),
                'chunking_settings': {
                    'chunk_size': self.chunk_size,
                    'overlap': self.overlap,
                    'min_chunk_size': self.min_chunk_size
                }
            },
            'chunks': chunks
        }
        
        output_filename = f"{pdf_name}_chunks.json"
        output_path = f"{output_dir}/{output_filename}"
        
        save_json(output, output_path)
        logger.info(f"Saved {len(chunks)} chunks to: {output_path}")
        
        return output_path
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunking process.
        
        Args:
            chunks: List of chunk objects
        
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_words_per_chunk': 0,
                'avg_chars_per_chunk': 0,
                'total_words': 0,
                'total_chars': 0,
                'pages_processed': 0
            }
        
        total_words = sum(c['metadata']['word_count'] for c in chunks)
        total_chars = sum(c['metadata']['char_count'] for c in chunks)
        pages = set(c['metadata']['page_number'] for c in chunks)
        
        stats = {
            'total_chunks': len(chunks),
            'avg_words_per_chunk': round(total_words / len(chunks), 1),
            'avg_chars_per_chunk': round(total_chars / len(chunks), 1),
            'total_words': total_words,
            'total_chars': total_chars,
            'pages_processed': len(pages),
            'chunks_per_page': round(len(chunks) / len(pages), 1) if pages else 0
        }
        
        logger.info(f"Chunking stats: {stats}")
        return stats


# ============================================
# Usage Example (for testing)
# ============================================
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    
    # Setup logger for testing
    setup_logger()
    
    # Example: Load extracted JSON and chunk it
    json_path = "data/extracted_texts/obl_test_pdf_rag1_extracted.json"
    
    # Create chunker with settings from config
    chunker = TextChunker(
        chunk_size=400,
        overlap=50,
        min_chunk_size=100
    )
    
    # Chunk the document
    chunks = chunker.chunk_from_json_file(json_path)
    
    # Display results
    print("\n" + "="*60)
    print("CHUNKING RESULTS")
    print("="*60)
    
    stats = chunker.get_chunking_stats(chunks)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nFirst 3 chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['metadata']['page_number']}")
        print(f"Words: {chunk['metadata']['word_count']}")
        print(f"Text preview: {chunk['text'][:150]}...")
    
    # Save chunks
    output_path = chunker.save_chunks(
        chunks=chunks,
        output_dir="data/processed_chunks",
        pdf_name="obl_test_pdf_rag1"
    )
    
    print(f"\n✅ Chunks saved to: {output_path}")
