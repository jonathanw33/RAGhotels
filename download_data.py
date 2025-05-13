#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download large data files for the Hotel RAG system.
Users can run this script to download the necessary data files
after cloning the repository.
"""

import os
import sys
import argparse
import urllib.request
import hashlib
import shutil

def download_file(url, destination, expected_hash=None):
    """Download a file from a URL to a destination path."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading {url} to {destination}...")
    
    # Simple progress bar
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = 100.0 * downloaded / total_size if total_size > 0 else 0
        progress = min(int(percent / 2), 50)
        sys.stdout.write(f"\r[{'#' * progress}{' ' * (50-progress)}] {percent:.1f}%")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("\nDownload complete!")
        
        # Verify hash if provided
        if expected_hash:
            print("Verifying file integrity...")
            file_hash = hashlib.sha256(open(destination, 'rb').read()).hexdigest()
            if file_hash != expected_hash:
                print(f"WARNING: Hash mismatch! Expected {expected_hash}, got {file_hash}")
                return False
            else:
                print("File integrity verified.")
        
        return True
    except Exception as e:
        print(f"\nError downloading file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download large data files for the RAG hotel system.')
    parser.add_argument('--force', action='store_true', help='Force download even if files exist')
    args = parser.parse_args()
    
    # Create data directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Define data files to download
    # Replace these URLs with actual URLs where you've uploaded your data files
    data_files = [
        {
            "name": "review.txt",
            "url": "https://example.com/path/to/review.txt",  # Replace with actual URL
            "destination": "data/raw/review.txt",
            "hash": None  # Add SHA256 hash if available for verification
        },
        {
            "name": "offering.txt",
            "url": "https://example.com/path/to/offering.txt",  # Replace with actual URL
            "destination": "data/raw/offering.txt", 
            "hash": None  # Add SHA256 hash if available for verification
        }
    ]
    
    # Download each file
    for file_info in data_files:
        if os.path.exists(file_info["destination"]) and not args.force:
            print(f"{file_info['name']} already exists. Use --force to re-download.")
            continue
        
        success = download_file(
            file_info["url"], 
            file_info["destination"],
            file_info["hash"]
        )
        
        if not success:
            print(f"Failed to download {file_info['name']}.")
            print("You may need to download this file manually.")
    
    print("\nData download complete. You can now run the data processing scripts.")

if __name__ == "__main__":
    main()
