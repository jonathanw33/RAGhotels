#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Advanced Search Features in the Hotel Recommendation RAG System.
"""

import os
import logging
import unittest
from src.rag.advanced_engine import AdvancedRAGEngine, QueryExpander

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestAdvancedRAGSystem(unittest.TestCase):
    """Test cases for the advanced RAG system with enhanced search features."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Check if ChromaDB exists
        if not os.path.exists("data/processed/chroma_db"):
            raise unittest.SkipTest(
                "ChromaDB not found. Please run the data processing and embedding generation scripts first."
            )
        
        # Initialize the RAG engine
        cls.rag_engine = AdvancedRAGEngine()
        
        # Initialize the query expander for direct testing
        cls.query_expander = QueryExpander()
    
    def test_query_expansion(self):
        """Test query expansion functionality."""
        test_queries = [
            "luxury hotel in paris",
            "cheap hotel with pool",
            "family friendly resort with breakfast",
            "hotel near beach with wifi",
            "romantic getaway with spa"
        ]
        
        for query in test_queries:
            expanded = self.query_expander.expand_query(query)
            
            print(f"\nOriginal: {query}")
            print(f"Expanded to {len(expanded)} variations, including:")
            for i, exp in enumerate(expanded[:5]):  # Show first 5
                print(f"  {i+1}. {exp}")
            
            # Verify that expansion produces reasonable results
            self.assertGreater(len(expanded), 1, f"Query '{query}' wasn't expanded")
            self.assertIn(query, expanded, f"Original query not preserved in expansion")
    
    def test_hybrid_search(self):
        """Test hybrid search with a specific location and features."""
        query = "luxury hotel in new york with spa and breakfast"
        result = self.rag_engine.query(query, use_query_expansion=True)
        
        self.assertIn('response', result)
        self.assertIn('hotels', result)
        self.assertIn('expanded_queries', result)
        
        print(f"\nHybrid search test: {query}")
        print(f"Response: {result['response'][:100]}...")
        print(f"Number of expanded queries: {len(result['expanded_queries'])}")
        print(f"Number of hotels: {len(result['hotels'])}")
        
        # Verify that results contain expanded query data
        self.assertGreater(len(result['expanded_queries']), 1, "Query expansion not working")
    
    def test_misspelling_correction(self):
        """Test correction of common misspellings."""
        misspelled_queries = [
            "luxary hotl in paris",
            "buisness hotel with breakfst",
            "famly friendly hotel with swiming pool",
            "hotel with wiifi and restraunt"
        ]
        
        for query in misspelled_queries:
            processed = self.query_expander.preprocess_query(query)
            
            print(f"\nMisspelled: {query}")
            print(f"Corrected: {processed}")
            
            # Verify that misspellings were corrected
            self.assertNotEqual(query, processed, f"Query '{query}' wasn't corrected")
    
    def test_synonym_expansion(self):
        """Test synonym expansion for hotel domain terms."""
        domain_terms = ["hotel", "pool", "breakfast", "luxury", "beach", "family"]
        
        for term in domain_terms:
            if term in self.query_expander.synonyms:
                synonyms = self.query_expander.synonyms[term]
                
                print(f"\nTerm: {term}")
                print(f"Synonyms: {', '.join(synonyms)}")
                
                # Verify that synonyms exist
                self.assertGreater(len(synonyms), 0, f"No synonyms for '{term}'")

def main():
    """Run the tests."""
    unittest.main()

if __name__ == "__main__":
    main()
