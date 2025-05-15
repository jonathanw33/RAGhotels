#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Hotel Recommendation RAG System.
"""

import os
import logging
import argparse
import importlib.util
from src.ui.app import create_app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to start the Hotel Recommendation RAG System."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hotel Recommendation RAG System')
    parser.add_argument('--data-format', choices=['csv', 'json'], default='json',
                        help='Format of the raw data files (csv or json)')
    parser.add_argument('--use-debug', action='store_true',
                        help='Use the enhanced debugging RAG engine')
    args = parser.parse_args()
    
    logger.info(f"Starting Hotel Recommendation RAG System with {args.data_format.upper()} data format...")
    
    # Check if data exists
    if not os.path.exists("data/processed/chroma_db"):
        logger.warning("Processed data not found. Please run data processing scripts first.")
        print("Data not processed yet. Please follow these steps:")
        
        if args.data_format == 'json':
            print("1. Place your hotel review datasets (offering.txt and review.txt) in the 'data/raw/' directory")
            print("2. Run 'python src/data_processing/process_json_data.py'")
        else:
            print("1. Place your hotel review datasets in the 'data/raw/' directory")
            print("2. Run 'python src/data_processing/process_data.py'")
            
        print("3. Run 'python src/data_processing/generate_embeddings.py'")
        return
    
    # Initialize enhanced RAG engine
    logger.info("Initializing Advanced RAG engine...")
    try:
        # Choose which engine to use based on debug flag
        if args.use_debug:
            logger.info("Using enhanced debugging RAG engine")
            try:
                # Import the debug engine
                from src.rag.advanced_engine_debug import AdvancedRAGEngine
                logger.info("Debug engine imported successfully")
            except ImportError:
                logger.warning("Debug engine not found, falling back to standard engine")
                from src.rag.advanced_engine import AdvancedRAGEngine
        else:
            # Standard engine
            from src.rag.advanced_engine import AdvancedRAGEngine
        
        rag_engine = AdvancedRAGEngine()
        
        # Start the Gradio app
        logger.info("Starting Gradio UI...")
        app = create_app(rag_engine)
        app.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        logger.error(f"Error starting the application: {str(e)}")
        print(f"Error: {str(e)}")
        print("\nPossible causes:")
        print("1. Ollama is not running (https://ollama.ai/)")
        print("2. Required models are not downloaded (run 'ollama pull llama3')")
        print("3. Memory constraints (try using a smaller model variant)")
        return

if __name__ == "__main__":
    main()
