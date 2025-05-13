# Hotel RAG System

A Retrieval Augmented Generation (RAG) system for hotel recommendations, using vector search and large language models.

## Features

- Process hotel data from JSON or CSV formats
- Create semantic embeddings using BGE-large model
- Store and search vectors using ChromaDB
- Generate personalized hotel recommendations based on user queries
- Advanced search features including metadata filtering, query expansion, and hybrid search

## Setup

### Prerequisites

- Python 3.8+
- Ollama (for local LLM inference) - [Install Ollama](https://ollama.ai/)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/jonathanw33/RAGhotels.git
cd RAGhotels
```

2. Run the setup script:
```bash
run.bat setup
```

This will:
- Create a virtual environment 
- Install all required dependencies
- Set up the project structure

3. Download the large data files:
   - Due to GitHub's file size limitations, data files are not included in this repository
   - Place your hotel review datasets in the `data/raw/` directory:
     - `review.txt`: Contains hotel reviews
     - `offering.txt`: Contains hotel details

### Processing Data

Process the data and generate embeddings:

```bash
run.bat process-json
run.bat embeddings
```

## Usage

Run the application:

```bash
run.bat json
```

This launches a Gradio web interface where you can:
- Ask questions about hotels
- Get personalized recommendations
- See hotel locations on a map
- Filter results by various criteria

## System Architecture

The system consists of several components:

1. **Data Processing Pipeline**
   - Processes raw hotel data (JSON/CSV)
   - Performs text chunking for optimal retrieval
   - Generates sentiment scores for reviews

2. **Embedding Generation**
   - Uses BAAI/bge-large-en-v1.5 model for embeddings
   - Stores embeddings in ChromaDB

3. **RAG Engine**
   - Hybrid retrieval combining vector search and BM25
   - Query expansion for better matching
   - Metadata filtering for targeted results

4. **User Interface**
   - Gradio-based web interface
   - Map visualization for hotel locations
   - Filtering and sorting options

## License

[MIT License](LICENSE)
