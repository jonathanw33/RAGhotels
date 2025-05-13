# Data Processing Module

This module handles loading, cleaning, processing, and embedding hotel review data.

## Key Components

- `process_data.py` - Data processing pipeline
  - Cleans and normalizes review text
  - Extracts hotel features from reviews
  - Performs sentiment analysis
  - Chunks reviews for optimal retrieval

- `generate_embeddings.py` - Embedding generation and vector DB setup
  - Sets up ChromaDB vector database
  - Generates embeddings for review chunks
  - Stores embeddings with metadata for retrieval

- `__init__.py` - Package initialization

## Data Processing Flow

1. Raw reviews are loaded from CSV files
2. Text cleaning and normalization is applied
3. Features are extracted from review text
4. Sentiment analysis is performed using NLTK VADER
5. Reviews are chunked by hotel and optimal size
6. Chunks are embedded using the specified model
7. Embeddings are stored in ChromaDB with metadata

## Usage

These scripts are typically run during setup:

```python
# Process raw data
python process_data.py

# Generate embeddings
python generate_embeddings.py
```

## Customization

- Embedding model can be changed in `generate_embeddings.py`
- Chunk size can be adjusted in `process_data.py`
- Additional features can be added to the feature extraction logic
- Sentiment analysis can be enhanced or replaced with custom implementation
