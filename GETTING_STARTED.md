# Hotel Recommendation RAG System - Getting Started Guide

## Project Overview

This is a hotel recommendation system that uses Retrieval Augmented Generation (RAG) with advanced search features to provide personalized hotel recommendations based on user preferences, review analysis, and location data.

## What's Included

- Complete code structure for a RAG-based recommendation system
- Data processing pipeline for hotel reviews
- Vector database integration with ChromaDB
- LLM integration with Llama3 via Ollama
- Interactive UI with Gradio and map visualization with Folium
- **Advanced search features including hybrid search, query expansion, and enhanced visualization**
- Comprehensive documentation and tests

## Advanced UI Features

The system now includes several advanced UI features to enhance the user experience:

### Advanced Filtering Panel
- **Rating Filters**: Filter hotels by minimum star rating (0-5 stars)
- **Sentiment Filters**: Filter hotels by review sentiment score
- **Feature Filters**: Select specific amenities that hotels must have
- **Sorting Options**: Sort hotels by rating or sentiment score

### Hotel Comparison Tool
- Select multiple hotels to compare side-by-side
- Compare ratings, features, locations, and sentiment scores
- Visual indicators highlight differences between hotels

### Enhanced Visualizations
- Interactive map with color-coded markers and heatmap overlay
- Feature distribution chart with improved design
- Sentiment analysis visualization with detailed breakdown

To access these features:
1. Click the "Advanced Filtering" accordion at the top of the interface
2. Use the filters to refine your search results
3. Check the "Comparison" tab to select and compare hotels

For more details, see [Advanced UI Features Documentation](docs/advanced_ui_features.md)

## Quick Start Guide

### 1. Using Sample Data

A sample dataset is included at `data/raw/sample_hotel_reviews.csv`. To use it:

1. Rename it to either `tripadvisor_hotel_reviews.csv` or `booking_hotel_reviews.csv`:
   ```
   rename data\raw\sample_hotel_reviews.csv data\raw\tripadvisor_hotel_reviews.csv
   ```

2. Run the complete setup process:
   ```
   run.bat setup  # On Windows
   ./run.sh setup  # On Linux/macOS
   ```

### 2. Using Your Own Data

1. Place your hotel review datasets in the `data/raw/` directory:
   - `tripadvisor_hotel_reviews.csv`
   - `booking_hotel_reviews.csv`

2. Run the setup process:
   ```
   run.bat setup  # On Windows
   ./run.sh setup  # On Linux/macOS
   ```

### 3. Running the Application

Once setup is complete, start the application:
```
run.bat  # On Windows
./run.sh  # On Linux/macOS
```

The UI will be available at http://localhost:7860

## Step-by-Step Instructions

If you prefer to run the steps individually:

1. **Create and activate virtual environment**:
   ```
   python -m venv hotel_rag_env
   hotel_rag_env\Scripts\activate  # On Windows
   source hotel_rag_env/bin/activate  # On Linux/macOS
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Process data**:
   ```
   python src/data_processing/process_data.py
   ```

4. **Generate embeddings**:
   ```
   python src/data_processing/generate_embeddings.py
   ```

5. **Run the application**:
   ```
   python src/main.py
   ```

## Testing the Advanced Search Features

To test the advanced search features specifically:
```
run.bat test-advanced  # On Windows
./run.sh test-advanced  # On Linux/macOS
```

This will run tests for:
- Query expansion
- Misspelling correction
- Synonym expansion
- Hybrid search functionality

## Advanced Usage Tips

- **Query Format**: Include specific location, amenities, and price category for best results
- **Feature Combinations**: Try combining multiple features (e.g., "luxury hotel with spa and pool in New York")
- **Toggle Query Expansion**: Turn off query expansion if you want exact matches only
- **Examine All Tabs**: Check the Map, Features, and Sentiment tabs for additional insights
- **Try Different Phrasings**: The system can understand various ways of expressing the same preference

## Ollama Setup

This project requires Ollama to run Llama3 locally:

1. Download and install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Llama3 model:
   ```
   ollama pull llama3
   ```

## Hardware Requirements

- Minimum: 16GB RAM, 4-core CPU, 50GB storage
- Recommended: 32GB RAM, 8-core CPU, 100GB storage, GPU (optional but improves performance)

## Additional Documentation

For more detailed information, see:

- Full system documentation: `docs/system_documentation.md`
- Advanced search features: `docs/advanced_search_features.md`
- Data format documentation: `data/raw/README.md`
- Code documentation in each module

## Troubleshooting

If you encounter issues:

1. Check if Ollama is running and the Llama3 model is downloaded
2. Verify that the data processing steps completed successfully
3. Check if `data/processed/chroma_db` directory exists and contains data
4. Consult the logs for detailed error messages

### Common Issues

- **Slow responses**: Try using a smaller LLM model like "llama3:8b"
- **Out of memory errors**: Reduce the batch size in embedding generation
- **No results for specific query**: Try a more general query or enable query expansion
- **Map not showing**: Ensure hotel data includes latitude/longitude coordinates

## Support

For questions or issues, please refer to the documentation first. If problems persist, reach out to the development team.
