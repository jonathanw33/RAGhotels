#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Hotel Recommendation RAG System.
"""

import os
import logging
import unittest
from src.rag.engine import RAGEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestRAGSystem(unittest.TestCase):
    """Test cases for the RAG system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Check if ChromaDB exists
        if not os.path.exists("data/processed/chroma_db"):
            raise unittest.SkipTest(
                "ChromaDB not found. Please run the data processing and embedding generation scripts first."
            )
        
        # Initialize the RAG engine
        cls.rag_engine = RAGEngine()
    
    def test_query_with_location(self):
        """Test a query with a location."""
        query = "I'm looking for a hotel in Paris"
        result = self.rag_engine.query(query)
        
        self.assertIsNotNone(result)
        self.assertIn('response', result)
        self.assertIn('hotels', result)
        self.assertIsInstance(result['response'], str)
        self.assertIsInstance(result['hotels'], list)
        
        # Print the result for manual inspection
        print(f"\nTest query: {query}")
        print(f"Response: {result['response'][:100]}...")
        print(f"Number of hotels: {len(result['hotels'])}")
    
    def test_query_with_features(self):
        """Test a query with specific features."""
        query = "I need a luxury hotel with a spa and good breakfast"
        result = self.rag_engine.query(query)
        
        self.assertIsNotNone(result)
        self.assertIn('response', result)
        self.assertIn('hotels', result)
        
        # Print the result for manual inspection
        print(f"\nTest query: {query}")
        print(f"Response: {result['response'][:100]}...")
        print(f"Number of hotels: {len(result['hotels'])}")
    
    def test_query_with_location_and_features(self):
        """Test a query with both location and features."""
        query = "Find me a budget-friendly hotel in London near the city center"
        result = self.rag_engine.query(query)
        
        self.assertIsNotNone(result)
        self.assertIn('response', result)
        self.assertIn('hotels', result)
        
        # Print the result for manual inspection
        print(f"\nTest query: {query}")
        print(f"Response: {result['response'][:100]}...")
        print(f"Number of hotels: {len(result['hotels'])}")

def main():
    """Run the tests."""
    unittest.main()

if __name__ == "__main__":
    main()
