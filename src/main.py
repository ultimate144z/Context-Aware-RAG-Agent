"""
Main Orchestration Script
Purpose: Ties together all components of the RAG pipeline.
Provides CLI interface for processing PDFs and asking questions.

Complete Pipeline:
1. PDF Extraction (text + OCR)
2. Text Cleaning & Merging
3. Chunking (400 words with overlap)
4. Embedding (sentence-transformers)
5. Vector Storage (ChromaDB)
6. Interactive Q&A (retrieval + LLM generation)

Usage:
    python src/main.py --mode process --pdf path/to/document.pdf
    python src/main.py --mode query
    python src/main.py --mode interactive
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import setup_logger, get_logger
from src.utils.file_utils import (
    load_json_config, 
    load_yaml_config, 
    get_file_name_without_extension,
    ensure_directory_exists
)
from src.pdf_extraction.text_extractor import TextExtractor
from src.pdf_extraction.ocr_extractor import OCRExtractor
from src.pdf_extraction.cleaner import TextCleaner
from src.chunking.chunker import TextChunker
from src.embeddings.embedder import TextEmbedder
from src.vector_store.chroma_manager import ChromaManager
from src.qa_pipeline.retriever import Retriever
from src.qa_pipeline.ollama_llm import OllamaLLM


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.
    Handles PDF processing and question answering.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize RAG pipeline with configuration.
        
        Args:
            config_dir: Directory containing config files
        """
        # Initialize logger for this instance
        self.logger = get_logger("RAG_Pipeline")
        if self.logger is None:
            # If logger doesn't exist, create it
            self.logger = setup_logger("RAG_Pipeline", log_dir="logs", log_to_file=True)
        
        self.logger.info("Initializing RAG Pipeline...")
        
        # Load configurations
        self.paths_config = load_json_config(f"{config_dir}/paths_config.json")
        self.model_config = load_json_config(f"{config_dir}/model_config.json")
        self.settings = load_yaml_config(f"{config_dir}/settings.yaml")
        
        self.logger.info("✓ Loaded configurations")
        
        # Initialize components (lazy loading)
        self._text_cleaner = None
        self._chunker = None
        self._embedder = None
        self._chroma_manager = None
        self._retriever = None
        self._llm = None
    
    @property
    def text_cleaner(self):
        """Lazy load text cleaner."""
        if self._text_cleaner is None:
            self._text_cleaner = TextCleaner(
                clean_whitespace=self.settings['clean_whitespace'],
                remove_page_numbers=self.settings['remove_page_numbers']
            )
        return self._text_cleaner
    
    @property
    def chunker(self):
        """Lazy load chunker."""
        if self._chunker is None:
            self._chunker = TextChunker(
                chunk_size=self.settings['chunk_size'],
                overlap=self.settings['overlap'],
                min_chunk_size=self.settings['min_chunk_size']
            )
        return self._chunker
    
    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            self._embedder = TextEmbedder(
                model_name=self.model_config['embedding_model']['name'],
                batch_size=self.settings['embedding_batch_size']
            )
        return self._embedder
    
    @property
    def chroma_manager(self):
        """Lazy load ChromaDB manager."""
        if self._chroma_manager is None:
            self._chroma_manager = ChromaManager(
                persist_directory=self.model_config['vector_store']['persist_directory'],
                collection_name="pdf_embeddings"
            )
            self._chroma_manager.get_or_create_collection()
        return self._chroma_manager
    
    @property
    def retriever(self):
        """Lazy load retriever."""
        if self._retriever is None:
            self._retriever = Retriever(
                chroma_manager=self.chroma_manager,
                embedder=self.embedder,
                top_k=self.settings['top_k'],
                similarity_threshold=self.settings['similarity_threshold']
            )
        return self._retriever
    
    @property
    def llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            llm_config = self.model_config['llm']
            self._llm = OllamaLLM(
                model_name=llm_config['model_name'],
                host=self.paths_config['ollama']['host'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
                system_prompt=llm_config['system_prompt']
            )
        return self._llm
    
    def process_pdf(self, pdf_path: str) -> bool:
        """
        Process a PDF through the entire pipeline.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("="*60)
        self.logger.info(f"PROCESSING PDF: {pdf_path}")
        self.logger.info("="*60)
        
        if not os.path.exists(pdf_path):
            self.logger.error(f"PDF not found: {pdf_path}")
            return False
        
        pdf_name = get_file_name_without_extension(pdf_path)
        
        try:
            # Step 1: Extract text
            self.logger.info("\n[1/6] Extracting text from PDF...")
            text_extractor = TextExtractor(pdf_path)
            text_pages = text_extractor.extract_text()
            self.logger.info(f"✓ Extracted text from {len(text_pages)} pages")
            
            # Step 2: OCR for pages without text
            self.logger.info("\n[2/6] Checking for pages needing OCR...")
            ocr_pages_needed = text_extractor.get_pages_needing_ocr(
                text_pages,
                check_hybrid=True,
                hybrid_threshold=100
            )
            
            ocr_results = {}
            if ocr_pages_needed:
                if OCRExtractor.is_tesseract_installed():
                    self.logger.info(f"Running OCR on {len(ocr_pages_needed)} pages...")
                    ocr_extractor = OCRExtractor(
                        pdf_path=pdf_path,
                        temp_image_dir=self.paths_config['data']['temp_images'],
                        ocr_lang=self.settings['ocr_lang']
                    )
                    ocr_results = ocr_extractor.ocr_multiple_pages(ocr_pages_needed)
                    self.logger.info(f"✓ OCR completed")
                else:
                    self.logger.warning("Tesseract not installed, skipping OCR")
            else:
                self.logger.info("✓ No OCR needed")
            
            # Step 3: Clean and merge
            self.logger.info("\n[3/6] Cleaning and merging text...")
            merged_pages = self.text_cleaner.clean_and_merge(
                text_pages=text_pages,
                ocr_pages=ocr_results if ocr_results else None,
                merge_hybrid=True,
                output_format='json'
            )
            document = self.text_cleaner.create_structured_output(merged_pages, pdf_name)
            
            # Save extracted JSON
            extracted_path = self.text_cleaner.save_cleaned_text(
                cleaned_data=document,
                output_dir=self.paths_config['data']['extracted_texts'],
                pdf_name=pdf_name,
                output_format='json'
            )
            self.logger.info(f"✓ Saved extracted text: {extracted_path}")
            
            # Step 4: Chunk text
            self.logger.info("\n[4/6] Chunking text...")
            chunks = self.chunker.chunk_document(document)
            
            chunks_path = self.chunker.save_chunks(
                chunks=chunks,
                output_dir=self.paths_config['data']['processed_chunks'],
                pdf_name=pdf_name
            )
            self.logger.info(f"✓ Created {len(chunks)} chunks: {chunks_path}")
            
            # Step 5: Create embeddings
            self.logger.info("\n[5/6] Creating embeddings...")
            embedded_chunks = self.embedder.embed_chunks(chunks)
            
            embeddings_path = self.embedder.save_embeddings(
                embedded_chunks=embedded_chunks,
                output_dir=self.paths_config['data']['embeddings'],
                pdf_name=pdf_name
            )
            self.logger.info(f"✓ Generated embeddings: {embeddings_path}")
            
            # Step 6: Add to vector database
            self.logger.info("\n[6/6] Adding to vector database...")
            count = self.chroma_manager.add_chunks(embedded_chunks)
            self.logger.info(f"✓ Added {count} chunks to ChromaDB")
            
            self.logger.info("\n" + "="*60)
            self.logger.info(f"✅ PDF PROCESSING COMPLETE: {pdf_name}")
            self.logger.info("="*60)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}", exc_info=True)
            return False
    
    def ask_question(
        self, 
        question: str, 
        show_sources: bool = True, 
        pdf_scope: Optional[list] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> dict:
        """
        Ask a question and get an answer with sources and optional conversation context.
        
        Args:
            question: User question
            show_sources: Whether to include source information
            pdf_scope: Optional list of PDF names to limit search scope
            conversation_history: Optional list of previous Q&A pairs for context
                                 Format: [{"question": "...", "answer": "..."}, ...]
                                 System will use last 3 pairs for context
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        self.logger.info(f"\nQuery: {question}")
        if conversation_history:
            self.logger.info(f"Using conversation context: {len(conversation_history)} previous Q&A pairs")
        
        try:
            # Retrieve relevant context
            self.logger.info("Retrieving relevant chunks...")
            retriever_result = self.retriever.retrieve_and_format(question, filter_pdfs=pdf_scope)
            
            if retriever_result['num_chunks'] == 0:
                return {
                    'answer': "I couldn't find relevant information in the documents to answer this question.",
                    'sources': [],
                    'success': True
                }
            
            self.logger.info(f"✓ Retrieved {retriever_result['num_chunks']} chunks")
            
            # Limit conversation history to last 3 Q&A pairs to avoid token overflow
            limited_history = None
            if conversation_history and len(conversation_history) > 0:
                limited_history = conversation_history[-3:]  # Last 3 pairs only
                self.logger.debug(f"Limited conversation history to {len(limited_history)} pairs")
            
            # Generate answer with conversation context
            self.logger.info("Generating answer...")
            result = self.llm.generate_with_retrieval(
                question, 
                retriever_result,
                conversation_history=limited_history
            )
            
            if result['success']:
                self.logger.info("✓ Answer generated")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error answering question: {e}", exc_info=True)
            return {
                'answer': f"Error: {str(e)}",
                'sources': [],
                'success': False
            }

    def process_directory(self, dir_path: str, recursive: bool = True) -> dict:
        """
        Process all PDF files in a directory.

        Args:
            dir_path: Path to directory containing PDFs
            recursive: If True, search subdirectories as well

        Returns:
            Summary dict with counts
        """
        self.logger.info("="*60)
        self.logger.info(f"BATCH PROCESSING DIRECTORY: {dir_path}")
        self.logger.info("="*60)

        if not os.path.isdir(dir_path):
            self.logger.error(f"Directory not found: {dir_path}")
            return {"processed": 0, "succeeded": 0, "failed": 0}

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_paths = list(Path(dir_path).glob(pattern))
        processed = 0
        succeeded = 0
        failed = 0

        for pdf in pdf_paths:
            processed += 1
            ok = self.process_pdf(str(pdf))
            if ok:
                succeeded += 1
            else:
                failed += 1

        summary = {"processed": processed, "succeeded": succeeded, "failed": failed}
        self.logger.info(f"Batch processing complete: {summary}")
        return summary
    
    def interactive_mode(self):
        """
        Start interactive Q&A session.
        """
        print("\n" + "="*60)
        print("🤖 RAG INTERACTIVE MODE")
        print("="*60)
        print("\nAsk questions about your documents!")
        print("Commands:")
        print("  - Type your question to get an answer")
        print("  - 'stats' to see database statistics")
        print("  - 'quit' or 'exit' to stop")
        print("="*60 + "\n")
        
        # Check Ollama connection
        if not self.llm.check_connection():
            print("❌ Error: Ollama server is not running!")
            print("   Start Ollama with: ollama serve")
            return
        
        # Check database
        stats = self.chroma_manager.get_collection_stats()
        print(f"📊 Database: {stats['count']} chunks available\n")
        
        if stats['count'] == 0:
            print("⚠ Warning: No documents in database!")
            print("   Process a PDF first: python src/main.py --mode process --pdf path/to/file.pdf\n")
        
        while True:
            try:
                # Get user input
                question = input("💬 You: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'stats':
                    stats = self.chroma_manager.get_collection_stats()
                    print(f"\n📊 Database Statistics:")
                    print(f"   Collection: {stats['collection_name']}")
                    print(f"   Chunks: {stats['count']}")
                    print(f"   Location: {stats['persist_directory']}\n")
                    continue
                
                # Answer question
                print("\n🔍 Searching documents...")
                result = self.ask_question(question)
                
                print("\n🤖 Assistant:")
                print("-" * 60)
                print(result['answer'])
                print("-" * 60)
                
                if result.get('sources'):
                    print("\n📚 Sources:")
                    for source in result['sources']:
                        print(f"   - {source['pdf_name']}, Pages: {source['pages']}")
                
                print()
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Context-Aware RAG Agent - Process PDFs and answer questions"
    )
    
    parser.add_argument(
        '--mode',
        choices=['process', 'process_dir', 'query', 'interactive'],
        default='interactive',
        help='Operation mode: process PDF, process directory, single query, or interactive chat'
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        help='Path to PDF file (required for process mode)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Path to directory with PDFs (required for process_dir mode)'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='Question to ask (required for query mode)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config',
        help='Path to config directory'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    global logger
    logger = setup_logger("RAG_Pipeline", log_dir="logs", log_to_file=True)
    
    print("\n" + "="*60)
    print("🧠 CONTEXT-AWARE RAG AGENT")
    print("="*60)
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline(config_dir=args.config)
    except Exception as e:
        print(f"\n❌ Error initializing pipeline: {e}")
        return 1
    
    # Execute based on mode
    if args.mode == 'process':
        if not args.pdf:
            print("\n❌ Error: --pdf argument required for process mode")
            return 1
        
        success = pipeline.process_pdf(args.pdf)
        return 0 if success else 1
    elif args.mode == 'process_dir':
        if not args.dir:
            print("\n❌ Error: --dir argument required for process_dir mode")
            return 1
        summary = pipeline.process_directory(args.dir)
        print(f"\nProcessed: {summary['processed']} | Succeeded: {summary['succeeded']} | Failed: {summary['failed']}")
        return 0 if summary['failed'] == 0 else 1
    
    elif args.mode == 'query':
        if not args.question:
            print("\n❌ Error: --question argument required for query mode")
            return 1
        
        result = pipeline.ask_question(args.question)
        
        print("\n📝 Answer:")
        print(result['answer'])
        
        if result.get('sources'):
            print("\n📚 Sources:")
            for source in result['sources']:
                print(f"   - {source['pdf_name']}, Pages: {source['pages']}")
        
        return 0 if result['success'] else 1
    
    elif args.mode == 'interactive':
        pipeline.interactive_mode()
        return 0


if __name__ == "__main__":
    sys.exit(main())
