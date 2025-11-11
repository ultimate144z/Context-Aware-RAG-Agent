"""
Prompt Evaluator Module
Purpose: Test and compare different prompt templates for RAG to find the best one.
Helps optimize answer quality, accuracy, and citation behavior.

How it links with previous steps:
- Uses retriever to get context
- Uses ollama_llm to generate answers
- Compares multiple prompt strategies
- Evaluates which prompt produces the best results

Logic:
1. Define multiple prompt templates (different instruction styles)
2. For each template, generate answer for the same query+context
3. Compare results based on:
   - Answer length
   - Citation quality
   - Factual grounding
   - Response time
4. Optionally use human evaluation or heuristic scoring
5. Return best-performing template
"""

import time
from typing import Dict, Any, List, Optional
from src.qa_pipeline.ollama_llm import OllamaLLM
from src.utils.logger import get_logger

logger = get_logger()


class PromptEvaluator:
    """
    Evaluates different prompt templates for RAG pipeline.
    Helps find the best prompt structure for accurate, cited answers.
    """
    
    # Predefined prompt templates
    PROMPT_TEMPLATES = {
        "default": {
            "system": (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Always use the context to answer accurately. If the answer is not in the context, "
                "say 'I cannot find this information in the provided documents.' "
                "Always cite the page numbers when providing information."
            ),
            "user_template": (
                "Context from documents:\n{context}\n\n---\n\n"
                "Based on the context above, please answer the following question. "
                "Cite the page numbers when possible.\n\n"
                "Question: {query}\n\nAnswer:"
            )
        },
        
        "strict_citation": {
            "system": (
                "You are a precise assistant that MUST cite sources for every claim. "
                "Answer ONLY using information from the provided context. "
                "Format citations as (Page X). "
                "If information is not in the context, explicitly state: 'Not found in documents.'"
            ),
            "user_template": (
                "Context:\n{context}\n\n---\n\n"
                "Question: {query}\n\n"
                "Provide a detailed answer with citations in (Page X) format:"
            )
        },
        
        "concise": {
            "system": (
                "You are a concise assistant. Provide brief, accurate answers based on the context. "
                "Always cite page numbers. Keep answers under 3 sentences unless more detail is needed."
            ),
            "user_template": (
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Brief answer with page citations:"
            )
        },
        
        "detailed": {
            "system": (
                "You are a thorough assistant that provides comprehensive answers. "
                "Use all relevant information from the context. "
                "Structure your answer with clear points and cite sources as (Page X) for each point."
            ),
            "user_template": (
                "Context:\n{context}\n\n---\n\n"
                "Question: {query}\n\n"
                "Provide a detailed, well-structured answer with citations:"
            )
        },
        
        "conversational": {
            "system": (
                "You are a friendly, conversational assistant. "
                "Answer questions naturally while staying factual based on the provided context. "
                "Mention page numbers naturally in your response (e.g., 'According to page 5...')"
            ),
            "user_template": (
                "Here's some information from documents:\n{context}\n\n"
                "Now, the user asks: {query}\n\n"
                "Your friendly, informative response:"
            )
        }
    }
    
    def __init__(
        self,
        llm: OllamaLLM,
        templates: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        Initialize the prompt evaluator.
        
        Args:
            llm: OllamaLLM instance
            templates: Optional custom prompt templates (uses defaults if None)
        """
        self.llm = llm
        self.templates = templates or self.PROMPT_TEMPLATES
        
        logger.info(f"Initialized PromptEvaluator with {len(self.templates)} templates")
    
    def evaluate_prompt(
        self,
        query: str,
        context: str,
        template_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single prompt template.
        
        Args:
            query: User question
            context: Retrieved context
            template_name: Name of template to use
        
        Returns:
            Dictionary with answer and evaluation metrics
        """
        if template_name not in self.templates:
            logger.error(f"Template '{template_name}' not found")
            return None
        
        template = self.templates[template_name]
        
        logger.info(f"Evaluating template: {template_name}")
        
        # Build prompt using template
        prompt = template['user_template'].format(context=context, query=query)
        
        # Temporarily override LLM system prompt
        original_system = self.llm.system_prompt
        self.llm.system_prompt = template['system']
        
        # Generate answer
        start_time = time.time()
        result = self.llm.generate(query, context)
        elapsed = time.time() - start_time
        
        # Restore original system prompt
        self.llm.system_prompt = original_system
        
        if not result['success']:
            logger.error(f"Failed to generate answer for template {template_name}")
            return None
        
        # Compute heuristic scores
        answer = result['answer']
        
        # Count citations (look for "Page X" or "(Page X)" patterns)
        import re
        citation_matches = re.findall(r'\(?\bPage\s+\d+\b\)?', answer, re.IGNORECASE)
        num_citations = len(citation_matches)
        
        # Check if answer admits uncertainty
        uncertainty_phrases = [
            "cannot find",
            "not found",
            "don't have",
            "not mentioned",
            "doesn't say",
            "unclear"
        ]
        admits_uncertainty = any(phrase.lower() in answer.lower() for phrase in uncertainty_phrases)
        
        evaluation = {
            'template_name': template_name,
            'answer': answer,
            'metrics': {
                'answer_length_chars': len(answer),
                'answer_length_words': len(answer.split()),
                'num_citations': num_citations,
                'admits_uncertainty': admits_uncertainty,
                'response_time_ms': result.get('total_duration_ms', elapsed * 1000),
                'tokens_used': result.get('response_tokens', 0)
            },
            'success': True
        }
        
        logger.info(f"Template {template_name}: {evaluation['metrics']['answer_length_words']} words, {num_citations} citations")
        
        return evaluation
    
    def evaluate_all_templates(
        self,
        query: str,
        context: str,
        template_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all templates (or a subset) for a query.
        
        Args:
            query: User question
            context: Retrieved context
            template_names: Optional list of template names to test (tests all if None)
        
        Returns:
            List of evaluation results
        """
        templates_to_test = template_names or list(self.templates.keys())
        
        logger.info(f"Evaluating {len(templates_to_test)} templates for query: '{query[:50]}...'")
        
        results = []
        
        for template_name in templates_to_test:
            result = self.evaluate_prompt(query, context, template_name)
            if result:
                results.append(result)
        
        logger.info(f"Evaluation complete: {len(results)} templates tested")
        
        return results
    
    def rank_templates(
        self,
        evaluations: List[Dict[str, Any]],
        criteria: str = "balanced"
    ) -> List[Dict[str, Any]]:
        """
        Rank templates based on evaluation metrics.
        
        Args:
            evaluations: List of evaluation results
            criteria: Ranking criteria ("citations", "speed", "length", "balanced")
        
        Returns:
            Sorted list of evaluations (best first)
        """
        logger.info(f"Ranking templates by criteria: {criteria}")
        
        if criteria == "citations":
            # Prioritize templates with more citations
            ranked = sorted(evaluations, key=lambda x: x['metrics']['num_citations'], reverse=True)
        
        elif criteria == "speed":
            # Prioritize faster responses
            ranked = sorted(evaluations, key=lambda x: x['metrics']['response_time_ms'])
        
        elif criteria == "length":
            # Prioritize longer, more detailed answers
            ranked = sorted(evaluations, key=lambda x: x['metrics']['answer_length_words'], reverse=True)
        
        elif criteria == "balanced":
            # Balanced score: citations + moderate length - response time penalty
            def balanced_score(eval_result):
                metrics = eval_result['metrics']
                citation_score = metrics['num_citations'] * 10  # 10 points per citation
                length_score = min(metrics['answer_length_words'] / 10, 20)  # Cap at 20 points
                speed_penalty = metrics['response_time_ms'] / 1000  # 1 point per second
                return citation_score + length_score - speed_penalty
            
            ranked = sorted(evaluations, key=balanced_score, reverse=True)
        
        else:
            logger.warning(f"Unknown criteria '{criteria}', using default order")
            ranked = evaluations
        
        logger.info(f"Top template: {ranked[0]['template_name']}")
        
        return ranked
    
    def get_best_template(
        self,
        query: str,
        context: str,
        criteria: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Find the best template for a given query and context.
        
        Args:
            query: User question
            context: Retrieved context
            criteria: Ranking criteria
        
        Returns:
            Best template evaluation result
        """
        evaluations = self.evaluate_all_templates(query, context)
        
        if not evaluations:
            logger.error("No successful evaluations")
            return None
        
        ranked = self.rank_templates(evaluations, criteria)
        
        best = ranked[0]
        logger.info(f"Best template: {best['template_name']} (score based on {criteria})")
        
        return best
    
    def print_comparison(self, evaluations: List[Dict[str, Any]]) -> None:
        """
        Print a formatted comparison of evaluation results.
        
        Args:
            evaluations: List of evaluation results
        """
        print("\n" + "="*80)
        print("PROMPT TEMPLATE COMPARISON")
        print("="*80)
        
        for eval_result in evaluations:
            print(f"\n{'='*80}")
            print(f"Template: {eval_result['template_name'].upper()}")
            print(f"{'='*80}")
            
            metrics = eval_result['metrics']
            print(f"\nMetrics:")
            print(f"  Words: {metrics['answer_length_words']}")
            print(f"  Citations: {metrics['num_citations']}")
            print(f"  Response Time: {metrics['response_time_ms']:.0f}ms")
            print(f"  Admits Uncertainty: {metrics['admits_uncertainty']}")
            
            print(f"\nAnswer:")
            print(f"{eval_result['answer']}")


# ============================================
# Usage Example (for testing)
# ============================================
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    from src.embeddings.embedder import TextEmbedder
    from src.vector_store.chroma_manager import ChromaManager
    from src.qa_pipeline.retriever import Retriever
    from src.qa_pipeline.ollama_llm import OllamaLLM
    
    # Setup logger
    setup_logger()
    
    print("\n" + "="*60)
    print("PROMPT EVALUATOR TEST")
    print("="*60)
    
    # Initialize components
    print("\nInitializing RAG pipeline...")
    
    llm = OllamaLLM(model_name="mistral", temperature=0.4)
    
    if not llm.check_connection():
        print("❌ Error: Ollama server not running!")
        exit(1)
    
    chroma = ChromaManager(persist_directory="data/embeddings", collection_name="pdf_embeddings")
    chroma.get_or_create_collection()
    
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    
    retriever = Retriever(chroma_manager=chroma, embedder=embedder, top_k=3)
    
    # Create evaluator
    evaluator = PromptEvaluator(llm=llm)
    
    # Test query
    query = "What is attitude and why does it matter in IT teams?"
    
    print(f"\nTest Query: {query}")
    print("\nRetrieving context...")
    
    retriever_result = retriever.retrieve_and_format(query, top_k=3)
    context = retriever_result['context']
    
    print(f"✓ Retrieved {retriever_result['num_chunks']} chunks")
    
    # Evaluate all templates
    print("\n" + "="*60)
    print("EVALUATING ALL PROMPT TEMPLATES")
    print("="*60)
    
    evaluations = evaluator.evaluate_all_templates(query, context)
    
    # Print comparison
    evaluator.print_comparison(evaluations)
    
    # Rank templates
    print("\n" + "="*60)
    print("RANKING (Balanced Criteria)")
    print("="*60)
    
    ranked = evaluator.rank_templates(evaluations, criteria="balanced")
    
    for i, eval_result in enumerate(ranked, 1):
        metrics = eval_result['metrics']
        print(f"\n{i}. {eval_result['template_name']}")
        print(f"   Citations: {metrics['num_citations']}, Words: {metrics['answer_length_words']}, Time: {metrics['response_time_ms']:.0f}ms")
    
    print(f"\n🏆 Best template: {ranked[0]['template_name']}")
    
    print("\n✅ Prompt evaluator test complete!")
