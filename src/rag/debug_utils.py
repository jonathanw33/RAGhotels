"""
Debug utilities for the RAG hotel system.
"""

import logging
import time
from typing import Any, Dict, List
from functools import wraps
import os

# Setup debug logger
logger = logging.getLogger("rag_debug")
logger.setLevel(logging.DEBUG)

# Add a file handler to save debug logs to a file
debug_dir = os.path.join("data", "logs")
os.makedirs(debug_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(debug_dir, "rag_debug.log"))
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def debug_log(message: str, level: int = logging.DEBUG) -> None:
    """Log a debug message."""
    logger.log(level, message)

def time_function(func):
    """Decorator to time a function and log the result."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        debug_log(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        debug_log(f"Completed {func.__name__} in {elapsed:.2f} seconds")
        return result
    return wrapper

class QueryDebugger:
    """Utility class to debug and log RAG query processing."""
    
    def __init__(self, query: str):
        """Initialize with the query."""
        self.query = query
        self.start_time = time.time()
        self.step_times = {}
        self.current_step = None
        self.step_start_time = None
        debug_log(f"Starting debug for query: {query}", logging.INFO)
    
    def start_step(self, step_name: str) -> None:
        """Start timing a new step."""
        if self.current_step:
            self.end_step()
        self.current_step = step_name
        self.step_start_time = time.time()
        debug_log(f"Starting step: {step_name}")
    
    def end_step(self) -> float:
        """End the current step and return the time taken."""
        if not self.current_step:
            return 0
        elapsed = time.time() - self.step_start_time
        self.step_times[self.current_step] = elapsed
        debug_log(f"Completed step: {self.current_step} in {elapsed:.2f} seconds")
        self.current_step = None
        return elapsed
    
    def log_retrieved_nodes(self, nodes: List[Any], max_nodes: int = 3) -> None:
        """Log information about retrieved nodes."""
        debug_log(f"Retrieved {len(nodes)} nodes")
        for i, node in enumerate(nodes[:max_nodes]):
            if hasattr(node, 'metadata') and node.metadata:
                hotel_name = node.metadata.get('Hotel_Name', 'Unknown')
                score = getattr(node, 'score', 'N/A')
                debug_log(f"  Node {i+1}: {hotel_name} (Score: {score})")
                
                # Log features if available
                features = [k for k, v in node.metadata.items() if k.startswith('has_') and v]
                if features:
                    debug_log(f"    Features: {', '.join(features[:5])}")
    
    def log_hotels(self, hotels: List[Dict[str, Any]], max_hotels: int = 3) -> None:
        """Log information about processed hotels."""
        debug_log(f"Processed {len(hotels)} hotels")
        for i, hotel in enumerate(hotels[:max_hotels]):
            debug_log(f"  Hotel {i+1}: {hotel.get('name', 'Unknown')} (Score: {hotel.get('score', 'N/A')})")
            
            # Log features
            features = [k.replace('has_', '') for k, v in hotel.get('features', {}).items() if v and k.startswith('has_')]
            if features:
                debug_log(f"    Features: {', '.join(features[:5])}")
            
            # Log sentiment
            sentiment = hotel.get('sentiment', {})
            if sentiment:
                debug_log(f"    Sentiment: {sentiment.get('compound', 0):.2f}")
    
    def finalize(self) -> Dict[str, float]:
        """Finalize the debugging session and return step times."""
        if self.current_step:
            self.end_step()
        total_time = time.time() - self.start_time
        debug_log(f"Query processing completed in {total_time:.2f} seconds", logging.INFO)
        debug_log("Step times:")
        for step, time_taken in self.step_times.items():
            debug_log(f"  {step}: {time_taken:.2f}s")
        
        # Return the timing information
        return {
            'total_time': total_time,
            'steps': self.step_times
        }
