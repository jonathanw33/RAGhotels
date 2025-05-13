#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG Engine for the Hotel Recommendation System.
"""

import os
import logging
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import pandas as pd
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.core import set_global_service_context

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define system prompts
SYSTEM_PROMPT = """You are a knowledgeable hotel recommendation assistant. 
Your job is to provide personalized hotel recommendations based on the user's preferences.
Use the retrieved information about hotels to match the user's requirements.
Always provide explanations for your recommendations based on review data.
Focus on aspects like location, amenities, price range, and traveler types."""

QUERY_PROMPT_TEMPLATE = """
You are an expert hotel recommendation system. Based on the user's preferences, 
you need to recommend hotels that match their criteria. 
Provide detailed explanations of why each hotel is a good match.

User preferences: {query_str}

Using the following hotel information, suggest up to 5 hotels that best match the user's needs.
For each hotel, explain why it's a good match, mentioning specific features from the reviews.
If location is mentioned, prioritize hotels in that location.
If specific amenities are requested, highlight those amenities in the recommendations.
If price range is specified, recommend hotels within that range.

Context information from hotel reviews and data:
{context_str}

Format your response as follows:
1. [Hotel Name] - [Location]
   Rating: [Average Score]
   Why it's a match: [Explanation with specific details from reviews]
   Key features: [List 3-5 key features from the reviews]

2. [Next Hotel]
   ...

If you can't find suitable hotels that match all criteria, explain what criteria
you were able to match and suggest alternatives.
"""

class RAGEngine:
    """RAG Engine for the Hotel Recommendation System."""
    
    def __init__(self, 
                 chroma_db_dir="data/processed/chroma_db",
                 embedding_model="BAAI/bge-large-en-v1.5",
                 llm_model="llama3"):
        """Initialize the RAG engine."""
        self.chroma_db_dir = chroma_db_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Check if the chroma_db directory exists
        if not os.path.exists(chroma_db_dir):
            raise FileNotFoundError(
                f"ChromaDB directory not found at {chroma_db_dir}. "
                f"Please run the data processing and embedding generation scripts first."
            )
        
        # Initialize the RAG components
        self.setup_rag_components()
    
    def setup_rag_components(self):
        """Set up the RAG components."""
        logger.info("Setting up RAG components")
        
        # Set up the embedding function for ChromaDB
        if "bge" in self.embedding_model.lower():
            logger.info(f"Using BGE embedding model: {self.embedding_model}")
            embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        elif "e5" in self.embedding_model.lower():
            logger.info(f"Using E5 embedding model: {self.embedding_model}")
            embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        else:
            logger.info(f"Using specified embedding model: {self.embedding_model}")
            embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        
        # Set up the LLM
        logger.info(f"Using LLM model: {self.llm_model}")
        llm = Ollama(model=self.llm_model, request_timeout=120.0)
        
        # Create the service context
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            system_prompt=SYSTEM_PROMPT
        )
        
        # Set the global service context
        set_global_service_context(service_context)
        
        # Set up ChromaDB client
        client = chromadb.PersistentClient(path=self.chroma_db_dir)
        chroma_collection = client.get_collection("hotel_reviews")
        
        # Create the vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create the index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=service_context,
        )
        
        # Create the query engine with custom prompt
        query_prompt = PromptTemplate(QUERY_PROMPT_TEMPLATE)
        self.query_engine = self.index.as_query_engine(
            text_qa_template=query_prompt,
            similarity_top_k=15,  # Retrieve more documents for better context
        )
        
        logger.info("RAG components set up successfully")
    
    def filter_hotels_by_metadata(self, query, top_k=20):
        """Filter hotels by metadata based on the query keywords."""
        client = chromadb.PersistentClient(path=self.chroma_db_dir)
        collection = client.get_collection("hotel_reviews")
        
        # Build where clauses based on the query keywords
        where_clauses = {}
        
        # Location filtering
        locations = [
            "paris", "london", "new york", "tokyo", "rome", "barcelona", 
            "amsterdam", "berlin", "singapore", "hong kong", "sydney", 
            "dubai", "bangkok", "istanbul", "miami", "los angeles",
            "chicago", "san francisco", "las vegas", "honolulu", "delhi"
        ]
        
        for location in locations:
            if location.lower() in query.lower():
                # We'll use a partial match in actual implementation
                # For now, just logging that we detected a location
                logger.info(f"Detected location: {location}")
        
        # Feature filtering based on keywords
        features = {
            "luxury": ["luxury", "luxurious", "upscale", "five star", "5 star"],
            "budget": ["budget", "cheap", "affordable", "inexpensive"],
            "family_friendly": ["family", "kid", "child", "children"],
            "beach": ["beach", "ocean", "sea"],
            "city_center": ["city center", "downtown", "central"],
            "pool": ["pool", "swimming"],
            "spa": ["spa", "massage", "wellness"],
            "breakfast": ["breakfast", "morning meal"]
        }
        
        detected_features = []
        for feature, keywords in features.items():
            for keyword in keywords:
                if keyword.lower() in query.lower():
                    detected_features.append(feature)
                    break
        
        logger.info(f"Detected features: {detected_features}")
        
        # For now, we won't directly filter by metadata in the query
        # Instead, we'll use these detected features to guide our LLM prompt
        # This is because the LLM can handle more nuanced matching
        
        # Return the list of detected features for use in the prompt
        return detected_features
    
    def query(self, user_query):
        """Run a query against the RAG engine."""
        logger.info(f"Running query: {user_query}")
        
        # First, filter hotels by metadata based on the query
        detected_features = self.filter_hotels_by_metadata(user_query)
        
        # Generate the response using the query engine
        response = self.query_engine.query(user_query)
        
        # Get the source nodes for additional information
        source_nodes = response.source_nodes
        
        # Extract hotel information from the source nodes
        hotels_info = []
        for node in source_nodes:
            if hasattr(node, 'metadata') and node.metadata:
                hotels_info.append({
                    'name': node.metadata.get('Hotel_Name', 'Unknown Hotel'),
                    'address': node.metadata.get('Hotel_Address', 'Unknown Location'),
                    'score': node.metadata.get('Average_Score', 0.0),
                    'latitude': node.metadata.get('Latitude', None),
                    'longitude': node.metadata.get('Longitude', None),
                    'source': node.metadata.get('source', 'unknown')
                })
        
        # Remove duplicates by name
        unique_hotels = []
        hotel_names = set()
        for hotel in hotels_info:
            if hotel['name'] not in hotel_names:
                unique_hotels.append(hotel)
                hotel_names.add(hotel['name'])
        
        return {
            'response': response.response,
            'hotels': unique_hotels[:5]  # Limit to top 5 hotels
        }

def main():
    """Test the RAG engine with a sample query."""
    logger.info("Testing RAG engine")
    
    try:
        rag_engine = RAGEngine()
        
        # Test query
        test_query = "I'm looking for a luxury hotel in Paris with a spa and good reviews about breakfast service."
        response = rag_engine.query(test_query)
        
        print(f"Response: {response['response']}")
        print("\nRecommended Hotels:")
        for hotel in response['hotels']:
            print(f"- {hotel['name']} ({hotel['address']}), Rating: {hotel['score']}")
        
        logger.info("RAG engine test completed successfully")
    
    except Exception as e:
        logger.error(f"Error testing RAG engine: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
