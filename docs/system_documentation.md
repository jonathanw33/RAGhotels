# Hotel Recommendation RAG System Documentation

## Overview

This document provides detailed information about the Hotel Recommendation RAG System, its architecture, components, and usage. This system uses Retrieval Augmented Generation (RAG) to provide personalized hotel recommendations based on user preferences, review analysis, and location data.

## System Architecture

### Core Components

1. **RAG Framework**: LlamaIndex
   - Used for building the retrieval-augmented generation pipeline
   - Provides query processing and response generation

2. **Vector Database**: ChromaDB
   - Open-source, embedding database that runs locally
   - Stores vector embeddings of hotel review chunks for semantic search

3. **Embedding Model**: BGE-M3 (BAAI/bge-large-en-v1.5)
   - Open-source embedding model
   - Converts text data into vector representations for semantic search

4. **Large Language Model**: Llama3 via Ollama
   - Open-source model deployed locally using Ollama
   - Generates natural language responses based on retrieved context

5. **Data Processing**: Python with Pandas/NumPy/NLTK
   - Processes raw hotel review data
   - Performs sentiment analysis and feature extraction
   - Chunks reviews for optimal retrieval

6. **Frontend/Visualization**: Gradio and Folium
   - Gradio for the user interface
   - Folium for interactive map visualization of recommended hotels

### Data Flow

```
User Query → LlamaIndex → ChromaDB ↔ BGE Embeddings → Llama3 → Response
     ↑                                                     ↓
     └───────────────── Gradio UI / Folium Maps ──────────┘
```

1. User enters a natural language query through the Gradio UI
2. The query is processed and converted to a vector embedding
3. Vector search is performed in ChromaDB to retrieve relevant hotel reviews
4. Retrieved context is passed to Llama3 LLM along with the user query
5. Llama3 generates a response with hotel recommendations
6. Results are displayed in the UI with an interactive map

## Setup and Installation

### Prerequisites

- Python 3.9+
- Git
- [Ollama](https://ollama.ai/) for local LLM deployment
- Minimum Hardware: 16GB RAM, 4-core CPU, 50GB storage
- Recommended Hardware: 32GB RAM, 8-core CPU, 100GB storage, GPU

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd RAGhotels

# Create virtual environment
python -m venv hotel_rag_env
source hotel_rag_env/bin/activate  # On Windows: hotel_rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Llama3 model using Ollama (run in a separate terminal)
ollama pull llama3
```

### Data Setup

1. Place your hotel review datasets in the `data/raw/` directory:
   - `tripadvisor_hotel_reviews.csv`
   - `booking_hotel_reviews.csv`

2. Run the setup script:
```bash
python setup.py
```

Or run the data processing scripts individually:
```bash
python src/data_processing/process_data.py
python src/data_processing/generate_embeddings.py
```

### Running the Application

```bash
python src/main.py
```

Access the UI at http://localhost:7860

## System Components Details

### Data Processing Module

Located in `src/data_processing/process_data.py`

This module handles:
- Loading and cleaning hotel review datasets
- Sentiment analysis using NLTK's VADER
- Feature extraction based on keyword matching
- Chunking reviews by hotel for optimal retrieval

Key functions:
- `clean_text()`: Normalizes and cleans text data
- `extract_features()`: Identifies amenities and features mentioned in reviews
- `analyze_sentiment()`: Determines sentiment scores for reviews
- `chunk_reviews()`: Creates optimal chunks for vector embedding

### Embedding Generation Module

Located in `src/data_processing/generate_embeddings.py`

This module handles:
- Setting up the ChromaDB vector database
- Generating embeddings using the BGE model
- Storing embeddings and metadata in ChromaDB

### RAG Engine

Located in `src/rag/engine.py`

This module handles:
- Setting up the LlamaIndex framework
- Configuring the LLM (Llama3 via Ollama)
- Processing queries and generating responses
- Filtering hotels based on user preferences

Key components:
- Custom prompt templates for hotel recommendation
- Hybrid search combining semantic and metadata filtering
- Response generation with explanation of recommendations

### User Interface

Located in `src/ui/app.py`

This module handles:
- Building the Gradio UI for user interaction
- Creating interactive maps with Folium
- Displaying hotel recommendations with explanations

## File Structure

```
RAGhotels/
├── data/
│   ├── raw/                # Raw hotel review datasets
│   │   └── README.md       # Dataset format documentation
│   └── processed/          # Processed data and ChromaDB
├── src/
│   ├── __init__.py
│   ├── main.py             # Main application entry point
│   ├── data_processing/    # Data preprocessing scripts
│   │   ├── __init__.py
│   │   ├── process_data.py # Data cleaning and processing
│   │   └── generate_embeddings.py # Vector embedding generation
│   ├── rag/                # RAG implementation
│   │   ├── __init__.py
│   │   └── engine.py       # RAG engine with LlamaIndex
│   └── ui/                 # User interface implementation
│       ├── __init__.py
│       └── app.py          # Gradio UI implementation
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   └── test_rag.py         # Tests for the RAG engine
├── docs/                   # Documentation files
├── README.md               # Project overview and setup instructions
├── requirements.txt        # Project dependencies
└── setup.py                # Setup script for environment preparation
```

## Usage Guide

### Query Formatting

For best results, include specific information in your queries:

1. **Location**: Specify the city or region where you're looking for a hotel
   - Example: "hotels in Paris" or "accommodations near Central Park, New York"

2. **Price Range**: Indicate your budget
   - Example: "luxury", "budget-friendly", "affordable", "under $200 per night"

3. **Amenities**: Mention specific features you're looking for
   - Example: "with a pool", "spa facilities", "free breakfast", "pet-friendly"

4. **Traveler Type**: Specify who will be staying
   - Example: "for families", "business trip", "romantic getaway"

5. **Rating/Quality**: Indicate desired quality level
   - Example: "top-rated", "5-star", "with excellent reviews"

### Example Queries

- "I need a hotel in Paris with free breakfast, under $200 per night, and good for families."
- "Find me a luxury hotel in New York with excellent reviews about its spa facilities."
- "What are the best boutique hotels in Barcelona with rooftop views?"
- "Budget-friendly hotel in London near the city center with good public transport access?"
- "Pet-friendly hotel in Miami with beach access and high cleanliness ratings?"

## Performance Optimization

### For Lower-End Hardware

If running on a system with limited resources:

1. Use a smaller embedding model:
   - Change `embedding_model` in `src/rag/engine.py` to a smaller model like "BAAI/bge-small-en-v1.5"

2. Use a smaller LLM:
   - Use a smaller Llama3 model variant in Ollama, like "llama3:8b" instead of larger models

3. Reduce chunk sizes:
   - Modify the `chunk_size` parameter in `src/data_processing/process_data.py` to a smaller value (e.g., 256 instead of 512)

4. Limit the dataset size:
   - Process only a subset of reviews if dealing with very large datasets

## Troubleshooting

### Common Issues

1. **ChromaDB errors**:
   - Ensure you've run the data processing scripts
   - Check if the `data/processed/chroma_db` directory exists and has content

2. **Ollama connection errors**:
   - Ensure Ollama is running in the background
   - Check if the Llama3 model is downloaded (`ollama list`)

3. **Out of memory errors**:
   - Try reducing batch sizes in the embedding generation
   - Use a smaller LLM model
   - Process fewer reviews

4. **Slow performance**:
   - Consider using GPU acceleration if available
   - Reduce the number of retrieved documents (`similarity_top_k` in `src/rag/engine.py`)

## Future Enhancements

Potential improvements for future versions:

1. **Multi-modal capabilities**:
   - Process hotel images for visual understanding
   - Include visual features in recommendation criteria

2. **Fine-tuning**:
   - Fine-tune the embedding model on hotel-specific data
   - Train a specialized model for hotel feature extraction

3. **Conversational interface**:
   - Implement follow-up questions handling
   - Add memory to maintain context across multiple queries

4. **External API integration**:
   - Connect to hotel booking APIs for real-time pricing and availability
   - Integrate with maps APIs for better location context

5. **User feedback and ratings**:
   - Collect user feedback on recommendations
   - Implement a feedback loop to improve recommendations

## Contributing

Contributions to the Hotel Recommendation RAG System are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or questions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
