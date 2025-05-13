# RAG Engine Module

This module contains the Retrieval Augmented Generation (RAG) engines that power the hotel recommendation system.

## Key Components

- `engine.py` - Basic RAG implementation
  - Integrates with LlamaIndex for vector search
  - Connects with Ollama for using Llama3
  - Processes queries and generates responses
  - Extracts hotel information from retrieved documents

- `advanced_engine.py` - Enhanced RAG with advanced search features
  - Implements hybrid search (vector + BM25)
  - Provides query expansion with synonyms and related terms
  - Handles misspelling correction
  - Supports metadata filtering and faceted search
  - Processes explicit filters from the UI

- `__init__.py` - Package initialization

## Core Classes

### `RAGEngine` (in engine.py)
Basic RAG implementation with:
- Vector-based semantic search
- Integration with Llama3 via Ollama
- Custom prompts for hotel recommendations
- Hotel information extraction

### `AdvancedRAGEngine` (in advanced_engine.py)
Enhanced RAG implementation that extends the basic engine with:
- `QueryExpander` for synonym and related term expansion
- `HybridRetriever` combining vector search and BM25
- Metadata filtering based on extracted or explicit filters
- Hotel ranking and sorting based on multiple criteria
- Support for filtering by rating, features, and sentiment

### Helper Classes
- `QueryExpander`: Handles query expansion with synonyms and misspelling correction
- `HybridRetriever`: Custom LlamaIndex retriever that combines multiple retrieval methods

## Usage Examples

```python
# Basic usage
from rag.engine import RAGEngine

rag_engine = RAGEngine()
result = rag_engine.query("I need a hotel in Paris with a spa")
print(result['response'])
print(f"Found {len(result['hotels'])} hotels")

# Advanced usage with filters
from rag.advanced_engine import AdvancedRAGEngine

advanced_engine = AdvancedRAGEngine()
filters = {
    "min_rating": 4.0,
    "required_features": ["pool", "breakfast"],
    "sort_by": "rating_high_to_low"
}
result = advanced_engine.query(
    "luxury hotel in New York", 
    use_query_expansion=True,
    filters=filters
)
print(result['response'])
print(f"Found {len(result['hotels'])} hotels")
```

## Customization

- Prompt templates can be modified for different recommendation styles
- The embedding model can be changed for different semantic understanding
- The LLM model can be replaced with alternatives from Ollama
- Filter criteria can be extended with additional metadata fields
