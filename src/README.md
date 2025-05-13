# Source Code (src) Folder

This folder contains all the source code for the Hotel Recommendation RAG System.

## Structure

- `data_processing/` - Scripts for data loading, cleaning, and embedding
- `rag/` - RAG engine implementation
- `ui/` - User interface implementation
- `main.py` - Main application entry point

## Key Components

### data_processing/

- `process_data.py` - Data cleaning, feature extraction, and sentiment analysis
  - Handles loading and processing hotel review datasets
  - Extracts features from review text (amenities, service quality, etc.)
  - Performs sentiment analysis using NLTK VADER
  - Chunks reviews for optimal retrieval

- `generate_embeddings.py` - Embedding generation and vector database setup
  - Sets up ChromaDB vector database
  - Generates embeddings using the specified model (BGE-M3 or E5-large-v2)
  - Stores embeddings with metadata for retrieval

### rag/

- `engine.py` - Basic RAG implementation
  - Integrates with LlamaIndex for vector search
  - Connects with Ollama for using Llama3
  - Processes queries and generates responses

- `advanced_engine.py` - Enhanced RAG with advanced search features
  - Implements hybrid search (vector + BM25)
  - Provides query expansion with synonyms
  - Supports metadata filtering and faceted search
  - Handles filtering by rating, features, etc.

### ui/

- `app.py` - Gradio UI implementation
  - Advanced filtering panel for refining results
  - Interactive map visualization with folium
  - Hotel comparison feature
  - Enhanced visualizations for features and sentiment

### Main Entry Point

- `main.py` - Application entry point that:
  - Initializes the RAG engine
  - Creates the Gradio UI
  - Launches the web server

## Development Notes

- Most components follow a modular design and can be extended or replaced
- Cross-module dependencies are minimized to enable component-level testing
- The system uses a plugin-based architecture allowing for easy extension
