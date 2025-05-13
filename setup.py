#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script to prepare the environment and run the data processing steps.
"""

import os
import logging
import subprocess
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is set up correctly."""
    logger.info("Checking environment")
    
    # Check if Python version is correct
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        logger.error("Python 3.9+ is required")
        print("Error: Python 3.9+ is required")
        return False
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not hasattr(sys, 'base_prefix') or sys.prefix == sys.base_prefix:
        logger.warning("Virtual environment is not activated")
        print("Warning: It's recommended to run this script in a virtual environment")
    
    # Check if data directories exist
    if not os.path.exists("data/raw"):
        logger.warning("Raw data directory not found. Creating it...")
        os.makedirs("data/raw", exist_ok=True)
    
    if not os.path.exists("data/processed"):
        logger.warning("Processed data directory not found. Creating it...")
        os.makedirs("data/processed", exist_ok=True)
    
    # Check if Ollama is installed and running
    ollama_paths = [
        "ollama",  # Check PATH
        r"C:\Users\Jonathan Wiguna\AppData\Local\Programs\Ollama\ollama.exe",  # Known installation path
        r"C:\Program Files\Ollama\ollama.exe",  # Common installation path
    ]
    
    ollama_found = False
    for path in ollama_paths:
        try:
            subprocess.run([path, "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logger.info(f"Ollama is installed at {path}")
            ollama_found = True
            break
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    
    if not ollama_found:
        logger.warning("Ollama may not be installed or running")
        print("Warning: Ollama might not be installed or running.")
        print("Please install Ollama from https://ollama.ai/ and make sure it's running")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return False
    
    return True

def install_requirements():
    """Install requirements from requirements.txt."""
    logger.info("Installing requirements")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        logger.info("Requirements installed successfully")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Error installing requirements: {str(e)}")
        print(f"Error installing requirements: {str(e)}")
        return False

def check_data_files():
    """Check if data files exist in the raw directory."""
    logger.info("Checking for data files")
    
    # Check for CSV format files
    tripadvisor_path = os.path.join("data/raw", "tripadvisor_hotel_reviews.csv")
    booking_path = os.path.join("data/raw", "booking_hotel_reviews.csv")
    
    # Check for JSON format files
    offering_json_path = os.path.join("data/raw", "offering.txt")
    review_json_path = os.path.join("data/raw", "review.txt")
    
    # Sample file
    sample_path = os.path.join("data/raw", "sample_hotel_reviews.csv")
    
    # Check if any of the required files exist
    json_format = os.path.exists(offering_json_path) and os.path.exists(review_json_path)
    csv_format = os.path.exists(tripadvisor_path) or os.path.exists(booking_path)
    sample_exists = os.path.exists(sample_path)
    
    if json_format:
        logger.info("Found JSON format data files")
        print("Found JSON format data files:")
        print(f"- {offering_json_path}")
        print(f"- {review_json_path}")
        return True, "json"
    elif csv_format:
        logger.info("Found CSV format data files")
        if os.path.exists(tripadvisor_path):
            print(f"Found TripAdvisor dataset at {tripadvisor_path}")
        if os.path.exists(booking_path):
            print(f"Found Booking.com dataset at {booking_path}")
        return True, "csv"
    elif sample_exists:
        logger.info("Found sample data file")
        print(f"Found sample dataset at {sample_path}")
        print("You can rename it to tripadvisor_hotel_reviews.csv for processing")
        return True, "csv"
    else:
        logger.warning("No data files found")
        print("\nNo data files found. Please place one of the following in the data/raw directory:")
        print("- Option 1 (JSON format): offering.txt and review.txt")
        print("- Option 2 (CSV format): tripadvisor_hotel_reviews.csv or booking_hotel_reviews.csv")
        print("- Option 3: Use the included sample_hotel_reviews.csv by renaming it")
        
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return False, None
    
    return True, None

def run_step(script_path, description):
    """Run a step with the given script."""
    logger.info(f"Running {description}")
    print(f"\n=== Running {description} ===")
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            logger.info(f"{description} completed successfully")
            print(f"{description} completed successfully")
            return True
        else:
            logger.error(f"{description} failed with code {result.returncode}")
            print(f"{description} failed with code {result.returncode}")
            return False
    except subprocess.SubprocessError as e:
        logger.error(f"Error running {description}: {str(e)}")
        print(f"Error running {description}: {str(e)}")
        return False

def main():
    """Main function to set up the project."""
    logger.info("Starting setup")
    print("=== Hotel Recommendation RAG System Setup ===")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        print("Setup aborted due to environment check failure")
        return
    
    # Install requirements
    if not install_requirements():
        logger.error("Requirements installation failed")
        print("Setup aborted due to requirements installation failure")
        return
    
    # Check data files
    data_check_result, data_format = check_data_files()
    if not data_check_result:
        logger.error("Data files check failed")
        print("Setup aborted due to data files check failure")
        return
    
    # Run data processing based on data format
    if data_format == "json":
        print("\nProcessing JSON format data...")
        if not run_step("src/data_processing/process_json_data.py", "JSON Data Processing"):
            logger.error("JSON data processing failed")
            print("Setup aborted due to JSON data processing failure")
            return
    else:  # CSV format or None
        print("\nProcessing CSV format data...")
        if not run_step("src/data_processing/process_data.py", "CSV Data Processing"):
            logger.error("CSV data processing failed")
            print("Setup aborted due to CSV data processing failure")
            return
    
    # Run embedding generation
    if not run_step("src/data_processing/generate_embeddings.py", "Embedding Generation"):
        logger.error("Embedding generation failed")
        print("Setup aborted due to embedding generation failure")
        return
    
    # Success message
    logger.info("Setup completed successfully")
    print("\n=== Setup completed successfully ===")
    print("You can now run the application with: python src/main.py")
    if data_format == "json":
        print("Using JSON data format from offering.txt and review.txt")
    else:
        print("Using CSV data format from hotel review datasets")

if __name__ == "__main__":
    main()
