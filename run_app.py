#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct entry point for the Hotel Recommendation RAG System.
This script can be run directly from the project root.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main function from src.main
from src.main import main

if __name__ == "__main__":
    main()
