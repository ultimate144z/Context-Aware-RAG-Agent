"""
Ollama LLM Module
Purpose: Interface with local Ollama server to generate answers using retrieved context.
Implements the actual RAG (Retrieval-Augmented Generation) pattern.

How it links with previous steps:
- Step 2-3 (extraction/processing): Created the knowledge base
- Step 4 (retriever): Provides relevant context chunks
- This module: Generates grounded answers using context + user query
- Output: Factual answer with citations from the source documents

Logic:
1. Take user query and retrieved context from retriever
2. Build a prompt with system instructions, context, and query
3. Send prompt to local Ollama API (Mistral model)
4. Stream or batch response
5. Return answer with metadata (model used, tokens, sources)
6. Handle errors and timeouts gracefully
"""

import json
import requests
from typing import Dict, Any, Optional, List, Generator
from src.utils.logger import get_logger
from src.utils.file_utils import load_json_config

logger = get_logger()


class OllamaLLM:
    """
    Interface for local Ollama LLM server.
    Generates answers using retrieved context for RAG pipeline.
    """
    
    def __init__(
        self,
        model_name: str = "mistral",
        host: str = "http://localhost:11434",
        temperature: float = 0.4,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize Ollama LLM interface.
        
        Args:
            model_name: Name of Ollama model (e.g., "mistral", "llama2")
            host: Ollama server URL
            temperature: Randomness (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens in response
            system_prompt: System instruction for the model
        """
        self.model_name = model_name
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Default system prompt for RAG
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Always use the context to answer accurately. If the answer is not in the context, "
            "say 'I cannot find this information in the provided documents.' "
            "Always cite the page numbers when providing information."
        )
        
        logger.info(f"Initialized OllamaLLM: model={model_name}, host={host}")
        logger.info(f"Settings: temperature={temperature}, max_tokens={max_tokens}")
    
    def check_connection(self) -> bool:
        """
        Check if Ollama server is running and accessible.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server is accessible")
                return True
            else:
                logger.error(f"Ollama server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama server: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                logger.info(f"Available models: {models}")
                return models
            else:
                logger.error("Failed to list models")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def build_prompt(
        self, 
        query: str, 
        context: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build the RAG prompt with context, query, and optional conversation history.
        
        Args:
            query: Current user question
            context: Retrieved context from retriever
            conversation_history: Optional list of previous Q&A pairs
                                 Format: [{"question": "...", "answer": "..."}, ...]
        
        Returns:
            Formatted prompt string
        """
        # Build conversation context if provided
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            conversation_context = "\nPrevious conversation:\n"
            for i, qa_pair in enumerate(conversation_history, 1):
                conversation_context += f"Q{i}: {qa_pair['question']}\n"
                conversation_context += f"A{i}: {qa_pair['answer']}\n\n"
            conversation_context += "---\n\n"
        
        prompt = f"""Context from documents:
{context}

---
{conversation_context}
Based on the context above, please answer the following question. Cite the page numbers when possible.

Question: {query}

Answer:"""
        
        return prompt
    
    def generate(
        self,
        query: str,
        context: str,
        stream: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using Ollama with optional conversation memory.
        
        Args:
            query: User question
            context: Retrieved context
            stream: Whether to stream response (not implemented in this version)
            conversation_history: Optional list of previous Q&A pairs for context
        
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Generating answer for query: '{query[:50]}...'")
        
        # Build prompt with conversation history
        prompt = self.build_prompt(query, context, conversation_history)
        logger.debug(f"Prompt length: {len(prompt)} characters")
        if conversation_history:
            logger.debug(f"Including {len(conversation_history)} previous Q&A pairs")
        
        # Prepare request
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False  # We'll handle non-streaming for now
        }
        
        try:
            logger.debug("Sending request to Ollama...")
            response = requests.post(url, json=payload, timeout=120)  # 2 min timeout
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('response', '').strip()
                
                logger.info(f"Generated answer: {len(answer)} characters")
                
                return {
                    'answer': answer,
                    'model': self.model_name,
                    'prompt_tokens': data.get('prompt_eval_count', 0),
                    'response_tokens': data.get('eval_count', 0),
                    'total_duration_ms': data.get('total_duration', 0) / 1_000_000,  # Convert ns to ms
                    'success': True
                }
            else:
                logger.error(f"Ollama returned status {response.status_code}: {response.text}")
                return {
                    'answer': "Error: Failed to generate response from Ollama.",
                    'model': self.model_name,
                    'success': False,
                    'error': response.text
                }
        
        except requests.exceptions.Timeout:
            logger.error("Request to Ollama timed out")
            return {
                'answer': "Error: Request timed out. The query may be too complex.",
                'model': self.model_name,
                'success': False,
                'error': "Timeout"
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'model': self.model_name,
                'success': False,
                'error': str(e)
            }
    
    def generate_with_retrieval(
        self,
        query: str,
        retriever_result: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using retriever results with optional conversation memory.
        
        Args:
            query: User question
            retriever_result: Output from retriever.retrieve_and_format()
            conversation_history: Optional list of previous Q&A pairs for context
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        context = retriever_result['context']
        
        if not context:
            logger.warning("No context provided, answering without retrieval")
            return {
                'answer': "I don't have enough information to answer this question from the provided documents.",
                'sources': [],
                'num_chunks_used': 0,
                'success': True
            }
        
        # Generate answer with conversation history
        result = self.generate(query, context, conversation_history=conversation_history)
        
        # Add retrieval info
        result['sources'] = retriever_result['sources']
        result['num_chunks_used'] = retriever_result['num_chunks']
        result['chunks'] = retriever_result['chunks']
        
        return result
    
    def ask(
        self,
        query: str,
        context: str = "",
        include_stats: bool = False
    ) -> str:
        """
        Simple interface: ask a question and get the answer as a string.
        
        Args:
            query: User question
            context: Context (optional, can be empty for direct questions)
            include_stats: Whether to include statistics in response
        
        Returns:
            Answer string (or answer with stats)
        """
        result = self.generate(query, context)
        
        if include_stats:
            stats = (
                f"\n\n---\n"
                f"Model: {result['model']}\n"
                f"Tokens: {result.get('prompt_tokens', 0)} prompt + {result.get('response_tokens', 0)} response\n"
                f"Duration: {result.get('total_duration_ms', 0):.0f}ms"
            )
            return result['answer'] + stats
        
        return result['answer']


# ============================================
# Usage Example (for testing)
# ============================================
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    from src.embeddings.embedder import TextEmbedder
    from src.vector_store.chroma_manager import ChromaManager
    from src.qa_pipeline.retriever import Retriever
    
    # Setup logger
    setup_logger()
    
    print("\n" + "="*60)
    print("OLLAMA LLM TEST")
    print("="*60)
    
    # Initialize LLM
    llm = OllamaLLM(
        model_name="mistral",
        host="http://localhost:11434",
        temperature=0.4,
        max_tokens=512
    )
    
    # Check connection
    print("\nChecking Ollama connection...")
    if not llm.check_connection():
        print("❌ Error: Ollama server is not running!")
        print("   Start Ollama with: ollama serve")
        exit(1)
    
    print("✓ Ollama server is running")
    
    # List models
    models = llm.list_models()
    print(f"\nAvailable models: {models}")
    
    if "mistral" not in [m.split(':')[0] for m in models]:
        print("⚠ Warning: 'mistral' model not found. Run: ollama pull mistral")
    
    # Initialize retriever
    print("\nInitializing retriever...")
    chroma = ChromaManager(
        persist_directory="data/embeddings",
        collection_name="pdf_embeddings"
    )
    chroma.get_or_create_collection()
    
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    
    retriever = Retriever(
        chroma_manager=chroma,
        embedder=embedder,
        top_k=3,
        similarity_threshold=0.3
    )
    
    # Test RAG pipeline
    print("\n" + "="*60)
    print("RAG PIPELINE TEST")
    print("="*60)
    
    test_queries = [
        "What is attitude and why does it matter in IT teams?",
        "How can job satisfaction be measured?"
    ]
    
    for query in test_queries:
        print("\n" + "-"*60)
        print(f"Query: {query}")
        print("-"*60)
        
        # Step 1: Retrieve context
        print("\n[1/2] Retrieving relevant chunks...")
        retriever_result = retriever.retrieve_and_format(query, top_k=3)
        
        print(f"  ✓ Retrieved {retriever_result['num_chunks']} chunks")
        for source in retriever_result['sources']:
            print(f"    - {source['pdf_name']}, Pages: {source['pages']}")
        
        # Step 2: Generate answer
        print("\n[2/2] Generating answer with Ollama...")
        result = llm.generate_with_retrieval(query, retriever_result)
        
        if result['success']:
            print(f"\n✓ Answer generated ({result.get('response_tokens', 0)} tokens, {result.get('total_duration_ms', 0):.0f}ms)")
            print("\n" + "="*60)
            print("ANSWER:")
            print("="*60)
            print(result['answer'])
            print("\n" + "="*60)
            print("SOURCES:")
            print("="*60)
            for source in result['sources']:
                print(f"  - {source['pdf_name']}, Pages: {source['pages']}")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
    
    print("\n✅ Ollama LLM test complete!")
