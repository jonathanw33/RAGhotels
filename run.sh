#!/bin/bash

echo "==================================="
echo "Hotel Recommendation RAG System"
echo "==================================="

if [ ! -d "hotel_rag_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv hotel_rag_env
    echo "Virtual environment created."
fi

echo "Activating virtual environment..."
source hotel_rag_env/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

if [ "$1" == "setup" ]; then
    echo "Running setup..."
    python setup.py
elif [ "$1" == "process-csv" ]; then
    echo "Processing CSV data..."
    python src/data_processing/process_data.py
elif [ "$1" == "process-json" ]; then
    echo "Processing JSON data..."
    python src/data_processing/process_json_data.py
elif [ "$1" == "embeddings" ]; then
    echo "Generating embeddings..."
    python src/data_processing/generate_embeddings.py
elif [ "$1" == "test" ]; then
    echo "Running tests..."
    python tests/test_rag.py
elif [ "$1" == "test-advanced" ]; then
    echo "Running advanced search tests..."
    python tests/test_advanced_search.py
elif [ "$1" == "advanced" ]; then
    echo "Starting application with advanced search features..."
    python src/main.py
elif [ "$1" == "json" ]; then
    echo "Starting application with JSON data format..."
    python src/main.py --data-format json
elif [ "$1" == "csv" ]; then
    echo "Starting application with CSV data format..."
    python src/main.py --data-format csv
else
    echo "Starting the application with advanced search features..."
    echo "If you encounter issues, try './run.sh setup' first."
    python src/main.py
fi
