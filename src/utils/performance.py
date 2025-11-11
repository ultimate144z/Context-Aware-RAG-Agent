"""
Performance monitoring utility for RAG pipeline.
Tracks retrieval latency and cache hit rate.
"""

import time
from typing import Optional
from functools import wraps


class PerformanceMonitor:
    """Monitor and log performance metrics for RAG operations."""
    
    def __init__(self):
        self.metrics = {
            'retrieval_times': [],
            'llm_times': [],
            'total_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'queries_processed': 0
        }
    
    def record_retrieval_time(self, duration: float):
        """Record a retrieval operation duration."""
        self.metrics['retrieval_times'].append(duration)
    
    def record_llm_time(self, duration: float):
        """Record an LLM generation duration."""
        self.metrics['llm_times'].append(duration)
    
    def record_total_time(self, duration: float):
        """Record total query processing time."""
        self.metrics['total_times'].append(duration)
        self.metrics['queries_processed'] += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics['cache_misses'] += 1
    
    def get_stats(self) -> dict:
        """Get current performance statistics."""
        stats = {
            'queries_processed': self.metrics['queries_processed'],
            'cache_hit_rate': 0.0,
            'avg_retrieval_time': 0.0,
            'avg_llm_time': 0.0,
            'avg_total_time': 0.0
        }
        
        total_cache = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache > 0:
            stats['cache_hit_rate'] = self.metrics['cache_hits'] / total_cache
        
        if self.metrics['retrieval_times']:
            stats['avg_retrieval_time'] = sum(self.metrics['retrieval_times']) / len(self.metrics['retrieval_times'])
        
        if self.metrics['llm_times']:
            stats['avg_llm_time'] = sum(self.metrics['llm_times']) / len(self.metrics['llm_times'])
        
        if self.metrics['total_times']:
            stats['avg_total_time'] = sum(self.metrics['total_times']) / len(self.metrics['total_times'])
        
        return stats
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'retrieval_times': [],
            'llm_times': [],
            'total_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'queries_processed': 0
        }


def timed(monitor: Optional[PerformanceMonitor] = None, metric_type: str = 'total'):
    """
    Decorator to time function execution.
    
    Args:
        monitor: PerformanceMonitor instance
        metric_type: Type of metric ('total', 'retrieval', 'llm')
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            
            if monitor:
                if metric_type == 'retrieval':
                    monitor.record_retrieval_time(duration)
                elif metric_type == 'llm':
                    monitor.record_llm_time(duration)
                else:
                    monitor.record_total_time(duration)
            
            return result
        return wrapper
    return decorator
