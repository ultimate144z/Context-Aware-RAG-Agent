"""
Retriever Module
Purpose: Retrieve the most relevant chunks from ChromaDB for a given query.
Acts as the bridge between user questions and the vector database.

How it links with previous steps:
- Step 2 (extraction): Provides the source text that was chunked
- Step 3 (chunking/embedding): Created the chunks stored in ChromaDB
- Step 3 (chroma_manager): This module uses ChromaManager to query the DB
- Step 4 (ollama_llm): Retrieved chunks are passed as context to the LLM

Logic:
1. Take user query as input
2. Embed the query using the same embedder (all-MiniLM-L6-v2)
3. Query ChromaDB for top-k most similar chunks (cosine similarity)
4. Optionally filter by metadata (specific PDF, page range, etc.)
5. Format retrieved chunks into context string
6. Return chunks with metadata for citation
7. Provide relevance scores for filtering low-quality matches
"""

from typing import List, Dict, Any, Optional
import numpy as np
import re
from src.embeddings.embedder import TextEmbedder
from src.vector_store.chroma_manager import ChromaManager
from src.utils.logger import get_logger

logger = get_logger()


def extract_keywords(text: str) -> List[str]:
    """
    Extract important keywords from query using generic NLP patterns.
    Domain-agnostic: works for any PDF content.
    
    Focuses on:
    - Entities (emails, URLs, phone numbers, dates)
    - Numbers and measurements
    - Capitalized terms (proper nouns)
    - Question keywords
    """
    keywords = []
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    keywords.extend(emails)
    
    # URLs
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    keywords.extend(urls)
    
    # Phone numbers (various formats)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    keywords.extend(phones)
    
    # Numbers and measurements (percentages, currency, etc.)
    number_patterns = [
        r'\b\d+%\b',  # percentages
        r'\$\d+(?:\.\d{2})?\b',  # currency
        r'\b\d+(?:\.\d+)?(?:kg|km|m|cm|gb|mb|hrs?|mins?)\b',  # measurements
        r'\b\d+\b'  # plain numbers
    ]
    for pattern in number_patterns:
        keywords.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Dates (various formats)
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
    ]
    for pattern in date_patterns:
        keywords.extend(re.findall(pattern, text))
    
    # Capitalized words (likely proper nouns, acronyms)
    # Skip common question words
    skip_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'the', 'a', 'an'}
    words = text.split()
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)  # Remove punctuation
        if clean_word and len(clean_word) > 2:
            # Capitalized word or all caps (acronym)
            if clean_word[0].isupper() and clean_word.lower() not in skip_words:
                keywords.append(clean_word)
            # All caps acronyms (3+ letters)
            elif clean_word.isupper() and len(clean_word) >= 3:
                keywords.append(clean_word)
    
    # Important question terms (domain-agnostic)
    # These indicate factual queries
    factual_indicators = ['contact', 'address', 'location', 'price', 'cost', 'date', 'time', 'name', 'number', 'amount']
    text_lower = text.lower()
    for term in factual_indicators:
        if term in text_lower:
            keywords.append(term)
    
    return list(set(keywords))  # Remove duplicates


def compute_keyword_boost(chunk_text: str, keywords: List[str]) -> float:
    """
    Compute a boost score based on keyword presence in chunk.
    Domain-agnostic: prioritizes exact matches for any content.
    
    Returns: boost score (0.0 to 0.15)
    """
    if not keywords:
        return 0.0
    
    chunk_lower = chunk_text.lower()
    matches = 0
    
    for kw in keywords:
        kw_lower = kw.lower()
        # Exact match or word boundary match
        if kw_lower in chunk_lower:
            # Check if it's a whole word match (more weight)
            if re.search(r'\b' + re.escape(kw_lower) + r'\b', chunk_lower):
                matches += 1.5  # Higher weight for whole word
            else:
                matches += 1  # Partial match
    
    # Boost proportional to keyword matches (max 15% boost)
    boost = min(0.15, (matches / len(keywords)) * 0.15)
    return boost


def expand_query_generic(query: str) -> List[str]:
    """
    Generate query variations using domain-agnostic techniques.
    Works for ANY document type: academic, technical, business, legal, etc.
    
    Strategy:
    1. Original query
    2. Extract key entities and rephrase
    3. Question reformulation patterns
    4. Simplified versions
    
    Args:
        query: Original user query
    
    Returns:
        List of query variations
    """
    queries = [query]  # Always include original
    q_lower = query.lower().strip()
    
    # Pattern 1: Extract main nouns/entities and create focused query
    # Remove question words to get core content
    question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'does', 'do', 'did']
    words = q_lower.split()
    content_words = [w for w in words if w not in question_words and len(w) > 2]
    
    if len(content_words) >= 2:
        # Create a focused content query
        focused_query = ' '.join(content_words[:5])  # Take top 5 content words
        queries.append(focused_query)
    
    # Pattern 2: If asking about "X of Y", try "Y X"
    # Example: "price of product" -> "product price"
    of_pattern = r'(\w+)\s+of\s+(?:the\s+)?(\w+)'
    match = re.search(of_pattern, q_lower)
    if match:
        noun1, noun2 = match.groups()
        queries.append(f"{noun2} {noun1}")
    
    # Pattern 3: Question reformulation patterns
    reformulations = []
    
    # "What is X" -> "X information", "X details"
    what_is_pattern = r'what\s+is\s+(?:the\s+)?(.+?)(?:\?|$)'
    match = re.search(what_is_pattern, q_lower)
    if match:
        subject = match.group(1).strip()
        reformulations.extend([
            f"{subject} information",
            f"{subject} details",
            subject
        ])
    
    # "Can you X" -> just X
    can_you_pattern = r'can\s+you\s+(?:share|provide|give|tell|show)\s+(?:me\s+)?(?:the\s+)?(.+?)(?:\?|$)'
    match = re.search(can_you_pattern, q_lower)
    if match:
        content = match.group(1).strip()
        reformulations.append(content)
    
    # "How to X" -> "X method", "X process"
    how_to_pattern = r'how\s+to\s+(.+?)(?:\?|$)'
    match = re.search(how_to_pattern, q_lower)
    if match:
        action = match.group(1).strip()
        reformulations.extend([
            f"{action} method",
            f"{action} process",
            f"{action} steps"
        ])
    
    # "Where is X" -> "X location"
    where_is_pattern = r'where\s+is\s+(?:the\s+)?(.+?)(?:\?|$)'
    match = re.search(where_is_pattern, q_lower)
    if match:
        item = match.group(1).strip()
        reformulations.extend([
            f"{item} location",
            f"{item} address"
        ])
    
    queries.extend(reformulations)
    
    # Pattern 4: Remove common filler words for a simplified query
    filler_words = ['please', 'kindly', 'would', 'could', 'can', 'you', 'me', 'the', 'a', 'an', 'for', 'to']
    simplified_words = [w for w in words if w not in filler_words and len(w) > 2]
    if len(simplified_words) >= 2 and len(simplified_words) < len(words):
        simplified = ' '.join(simplified_words)
        queries.append(simplified)
    
    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        q_clean = q.strip()
        if q_clean and q_clean not in seen and len(q_clean) > 2:
            seen.add(q_clean)
            unique_queries.append(q_clean)
    
    return unique_queries[:8]  # Limit to 8 variations max


class Retriever:
    """
    Retrieves relevant chunks from vector database for RAG pipeline.
    Handles query embedding and similarity search with hybrid boosting.
    """
    
    def __init__(
        self,
        chroma_manager: ChromaManager,
        embedder: TextEmbedder,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        enable_caching: bool = True,
        cache_size: int = 100
    ):
        """
        Initialize the retriever.
        
        Args:
            chroma_manager: ChromaManager instance for querying database
            embedder: TextEmbedder instance for embedding queries
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score (0-1) to include results
            enable_caching: Whether to cache query results for speed
            cache_size: Maximum number of cached queries
        """
        self.chroma_manager = chroma_manager
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_caching = enable_caching
        
        # Simple LRU-like cache for query results
        self._query_cache = {}
        self._cache_size = cache_size
        self._cache_order = []  # Track access order
        
        logger.info("Initialized Retriever")
        logger.info(f"Settings: top_k={top_k}, similarity_threshold={similarity_threshold}, caching={'enabled' if enable_caching else 'disabled'}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        filter_pdfs: Optional[List[str]] = None,
        use_query_expansion: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User question/query text
            top_k: Number of chunks to retrieve (overrides default)
            filter_metadata: Optional metadata filters (e.g., {"pdf_name": "document"})
            filter_pdfs: Optional list of pdf_name values to restrict search
            use_query_expansion: Whether to try multiple query variations for better recall
        
        Returns:
            List of retrieved chunks with text, metadata, and similarity scores
        """
        # Input validation
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        query = query.strip()
        k = top_k if top_k is not None else self.top_k
        
        # Validate top_k
        if k <= 0:
            logger.warning(f"Invalid top_k value: {k}, using default")
            k = self.top_k
        
        # Check cache first
        cache_key = f"{query}|{k}|{filter_pdfs}|{use_query_expansion}"
        if self.enable_caching and cache_key in self._query_cache:
            logger.debug("Cache hit - returning cached results")
            return self._query_cache[cache_key]
        
        logger.info(f"Retrieving chunks for query: '{query[:50]}...'")
        logger.info(f"Base top_k: {k}")
        if filter_pdfs:
            logger.info(f"Filtering across {len(filter_pdfs)} PDF(s): {filter_pdfs}")
        
        # Step 1: Generate query variations if enabled
        queries_to_try = expand_query_generic(query) if use_query_expansion else [query]
        if len(queries_to_try) > 1:
            logger.debug(f"Query expansion: trying {len(queries_to_try)} variations")
        
        # Extract keywords for hybrid boosting
        keywords = extract_keywords(query)
        if keywords:
            logger.debug(f"Keywords for boosting: {keywords}")
        
        # Step 2: Embed and query with all variations, collect results
        all_results_map = {}  # chunk_id -> (chunk_data, best_distance)
        
        try:
            for query_variant in queries_to_try:
                logger.debug(f"Querying with: '{query_variant[:50]}...'")
                query_embedding = self.embedder.embed_text(query_variant)
                
                # Query ChromaDB
                # If multiple PDFs specified, query each separately for better distribution, then merge
                if filter_pdfs and len(filter_pdfs) > 1:
                    per_pdf_k = max(1, k // len(filter_pdfs))
                    for pdf in filter_pdfs:
                        pdf_filter = {"pdf_name": pdf}
                        try:
                            res = self.chroma_manager.query(
                                query_text=query_variant,
                                query_embedding=query_embedding.tolist(),
                                top_k=per_pdf_k,
                                filter_metadata=pdf_filter
                            )
                            # Collect results
                            for cid, doc, md, dist in zip(res['ids'][0], res['documents'][0], res['metadatas'][0], res['distances'][0]):
                                if cid not in all_results_map or dist < all_results_map[cid][1]:
                                    all_results_map[cid] = ((cid, doc, md, dist), dist)
                        except Exception as e:
                            logger.warning(f"Error querying PDF {pdf}: {e}")
                            continue
                else:
                    # Single filter or none
                    effective_filter = filter_metadata.copy() if filter_metadata else {}
                    if filter_pdfs:
                        effective_filter['pdf_name'] = filter_pdfs[0]
                    res = self.chroma_manager.query(
                        query_text=query_variant,
                        query_embedding=query_embedding.tolist(),
                        top_k=k,
                        filter_metadata=effective_filter if effective_filter else None
                    )
                    # Collect results
                    for cid, doc, md, dist in zip(res['ids'][0], res['documents'][0], res['metadatas'][0], res['distances'][0]):
                        if cid not in all_results_map or dist < all_results_map[cid][1]:
                            all_results_map[cid] = ((cid, doc, md, dist), dist)
        
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
        
        # Check if we got any results
        if not all_results_map:
            logger.warning("No results found for query")
            return []
        
        # Step 3: Apply keyword boosting to improve ranking
        if keywords:
            boosted_results = []
            for cid, (chunk_tuple, dist) in all_results_map.items():
                chunk_id, document, metadata, distance = chunk_tuple
                boost = compute_keyword_boost(document, keywords)
                # Reduce distance (improve similarity) based on keyword matches
                adjusted_distance = distance * (1.0 - boost)
                boosted_results.append(((chunk_id, document, metadata, adjusted_distance), adjusted_distance))
            all_results_map_items = boosted_results
        else:
            all_results_map_items = list(all_results_map.values())
        
        # Step 4: Sort merged results by (adjusted) distance and take top-k
        sorted_results = sorted(all_results_map_items, key=lambda x: x[1])[:k]
        results_tuples = [item[0] for item in sorted_results]
        
        # Step 5: Format results
        retrieved_chunks = []
        
        for idx, (chunk_id, document, metadata, distance) in enumerate(results_tuples):
            # Convert distance to similarity (for cosine distance: similarity = 1 - distance)
            similarity = 1.0 - distance
            
            # Apply a more lenient threshold for chunks with keyword matches
            effective_threshold = self.similarity_threshold
            if keywords:
                # Check if this chunk has keyword matches
                kw_boost = compute_keyword_boost(document, keywords)
                if kw_boost > 0.05:  # Has significant keyword match
                    # Lower threshold by up to 0.1 for keyword-rich chunks
                    effective_threshold = max(0.2, self.similarity_threshold - 0.1)
            
            # Filter by similarity threshold
            if similarity < effective_threshold:
                logger.debug(f"Chunk {idx}: Filtered out (similarity {similarity:.4f} < threshold {effective_threshold:.4f})")
                continue
            
            chunk_data = {
                'chunk_id': chunk_id,
                'text': document,
                'metadata': metadata,
                'similarity_score': round(similarity, 4),
                'rank': idx + 1
            }
            
            retrieved_chunks.append(chunk_data)
            logger.debug(f"Chunk {idx+1}: {metadata.get('pdf_name', 'unknown')} Page {metadata.get('page_number', '?')}, Score {similarity:.4f}")
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks (after filtering)")
        
        # Cache the results
        if self.enable_caching:
            self._update_cache(cache_key, retrieved_chunks)
        
        return retrieved_chunks
    
    def _update_cache(self, key: str, value: List[Dict[str, Any]]):
        """Update cache with LRU eviction policy."""
        if key in self._query_cache:
            # Move to end (most recent)
            self._cache_order.remove(key)
            self._cache_order.append(key)
        else:
            # Add new entry
            if len(self._query_cache) >= self._cache_size:
                # Evict oldest
                oldest = self._cache_order.pop(0)
                del self._query_cache[oldest]
            self._cache_order.append(key)
            self._query_cache[key] = value
    
    def clear_cache(self):
        """Clear the query cache."""
        self._query_cache.clear()
        self._cache_order.clear()
        logger.info("Query cache cleared")
    
    def format_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        include_metadata: bool = True,
        max_length: Optional[int] = None
    ) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            retrieved_chunks: List of retrieved chunks from retrieve()
            include_metadata: Whether to include page numbers and sources
            max_length: Optional max character limit for context
        
        Returns:
            Formatted context string
        """
        logger.debug(f"Formatting context from {len(retrieved_chunks)} chunks")
        
        context_parts = []
        total_chars = 0
        
        for chunk in retrieved_chunks:
            if include_metadata:
                # Format: [Source: PDF_name, Page X] Text...
                source_info = f"[Source: {chunk['metadata']['pdf_name']}, Page {chunk['metadata']['page_number']}]"
                chunk_text = f"{source_info}\n{chunk['text']}"
            else:
                chunk_text = chunk['text']
            
            # Check max length
            if max_length and (total_chars + len(chunk_text)) > max_length:
                logger.warning(f"Context length limit reached ({max_length} chars)")
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.debug(f"Formatted context: {total_chars} characters, {len(context_parts)} chunks")
        return context
    
    def get_unique_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique source information from retrieved chunks.
        
        Args:
            retrieved_chunks: List of retrieved chunks
        
        Returns:
            List of unique sources with PDF name and page numbers
        """
        sources = {}
        
        for chunk in retrieved_chunks:
            pdf_name = chunk['metadata']['pdf_name']
            page_num = chunk['metadata']['page_number']
            
            if pdf_name not in sources:
                sources[pdf_name] = {
                    'pdf_name': pdf_name,
                    'pages': set()
                }
            
            sources[pdf_name]['pages'].add(page_num)
        
        # Convert sets to sorted lists
        source_list = []
        for pdf_name, data in sources.items():
            source_list.append({
                'pdf_name': pdf_name,
                'pages': sorted(list(data['pages']))
            })
        
        return source_list
    
    def retrieve_and_format(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_metadata: bool = True,
        filter_pdfs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convenience method: retrieve chunks and format them in one call.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            include_metadata: Whether to include source info in context
        
        Args:
            filter_pdfs: Optional list of pdf_name values to restrict search scope.

        Returns:
            Dictionary with context, chunks, and sources
        """
        # Retrieve chunks
        chunks = self.retrieve(query, top_k=top_k, filter_pdfs=filter_pdfs)
        
        if not chunks:
            logger.warning("No relevant chunks found for query")
            return {
                'context': "",
                'chunks': [],
                'sources': [],
                'num_chunks': 0
            }
        
        # Format context
        context = self.format_context(chunks, include_metadata=include_metadata)
        
        # Get sources
        sources = self.get_unique_sources(chunks)
        
        return {
            'context': context,
            'chunks': chunks,
            'sources': sources,
            'num_chunks': len(chunks)
        }


# ============================================
# Usage Example (for testing)
# ============================================
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    
    # Setup logger
    setup_logger()
    
    print("\n" + "="*60)
    print("RETRIEVER TEST")
    print("="*60)
    
    # Initialize components
    print("\nInitializing ChromaDB and Embedder...")
    
    chroma = ChromaManager(
        persist_directory="data/embeddings",
        collection_name="pdf_embeddings"
    )
    chroma.get_or_create_collection()
    
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    
    # Create retriever
    retriever = Retriever(
        chroma_manager=chroma,
        embedder=embedder,
        top_k=3,
        similarity_threshold=0.3
    )
    
    # Test queries
    test_queries = [
        "What is attitude in IT teams?",
        "How does job satisfaction affect employees?",
        "What is organizational commitment?"
    ]
    
    for query in test_queries:
        print("\n" + "="*60)
        print(f"Query: '{query}'")
        print("="*60)
        
        # Retrieve and format
        result = retriever.retrieve_and_format(query, top_k=3)
        
        print(f"\nRetrieved {result['num_chunks']} chunks")
        
        # Show retrieved chunks
        print("\nTop Results:")
        for chunk in result['chunks']:
            print(f"\n  Rank {chunk['rank']}: Page {chunk['metadata']['page_number']}")
            print(f"  Similarity: {chunk['similarity_score']}")
            print(f"  Text: {chunk['text'][:100]}...")
        
        # Show sources
        print("\nSources:")
        for source in result['sources']:
            print(f"  - {source['pdf_name']}, Pages: {source['pages']}")
        
        # Show formatted context (truncated)
        print(f"\nFormatted Context (first 200 chars):")
        print(result['context'][:200] + "...")
    
    print("\n✅ Retriever test complete!")
