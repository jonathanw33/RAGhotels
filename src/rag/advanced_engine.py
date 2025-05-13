#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced RAG Engine with Advanced Search Features for the Hotel Recommendation System.
Fixed"""

import os
import logging
import re
import string
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from llama_index.core.indices.vector_store.base import VectorStoreIndex # Correct import
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
def download_nltk_resources():
    """Download required NLTK resources."""
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

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

class QueryExpander:
    """Class for expanding queries with synonyms, related terms, and spell correction."""
    
    def __init__(self):
        """Initialize the QueryExpander with necessary resources."""
        # Download NLTK resources if not already downloaded
        download_nltk_resources()
        
        # Initialize lemmatizer and stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Hotel domain specific synonym dictionary
        self.synonyms = {
            # Accommodation types
            "hotel": ["resort", "inn", "lodge", "motel", "accommodation"],
            "resort": ["hotel", "retreat", "spa resort", "vacation resort"],
            "inn": ["hotel", "bed and breakfast", "guesthouse", "lodging"],
            
            # Locations
            "downtown": ["city center", "central", "heart of the city", "urban"],
            "beach": ["oceanfront", "seafront", "coastal", "waterfront", "seaside"],
            "mountain": ["alpine", "hillside", "mountainside", "highlands"],
            
            # Amenities
            "pool": ["swimming pool", "infinity pool", "outdoor pool", "indoor pool"],
            "spa": ["wellness center", "massage", "jacuzzi", "hot tub", "sauna"],
            "gym": ["fitness center", "workout room", "exercise facility", "fitness suite"],
            "breakfast": ["morning meal", "continental breakfast", "breakfast buffet", "brunch"],
            "wifi": ["internet", "wireless internet", "connectivity", "online access"],
            
            # Price categories
            "luxury": ["high-end", "five-star", "premium", "upscale", "deluxe"],
            "budget": ["affordable", "economical", "inexpensive", "cheap", "low-cost"],
            "mid-range": ["moderate", "reasonably priced", "standard"],
            
            # Features
            "view": ["scenic view", "panorama", "overlook", "vista"],
            "quiet": ["peaceful", "serene", "tranquil", "silent"],
            "modern": ["contemporary", "updated", "sleek", "stylish"],
            "historic": ["heritage", "vintage", "classic", "traditional", "old-world"],
            
            # Guest types
            "family": ["kid-friendly", "child-friendly", "family-oriented", "for families"],
            "business": ["corporate", "for work", "professional", "business facilities"],
            "romantic": ["couples", "honeymoon", "intimate", "for couples"],
            "solo": ["individual traveler", "alone", "single traveler"]
        }
        
        # Common hotel locations 
        self.locations = [
            "new york", "las vegas", "london", "paris", "tokyo", "rome", "barcelona", 
            "dubai", "sydney", "hong kong", "berlin", "amsterdam", "madrid", "singapore", 
            "bangkok", "istanbul", "prague", "vienna", "venice", "florence", "miami",
            "san francisco", "los angeles", "chicago", "toronto", "vancouver", "montreal", 
            "mexico city", "rio de janeiro", "buenos aires", "cairo", "cape town", 
            "melbourne", "auckland", "bali", "seoul", "kyoto", "moscow", "st petersburg",
            "dublin", "edinburgh", "munich", "zurich", "geneva", "brussels", "stockholm",
            "copenhagen", "oslo", "helsinki", "athens", "santorini", "dubai", "abu dhabi"
        ]
        
        # Common misspellings
        self.common_misspellings = {
            "hotl": "hotel",
            "resor": "resort",
            "breakfst": "breakfast",
            "swiming": "swimming",
            "luxary": "luxury",
            "romanti": "romantic",
            "famly": "family",
            "buisness": "business",
            "restraunt": "restaurant",
            "resturant": "restaurant",
            "accomodation": "accommodation",
            "wiifi": "wifi",
            "wifii": "wifi",
            "fitnes": "fitness",
            "jaccuzi": "jacuzzi",
            "jaccuzzi": "jacuzzi"
        }
    
    def preprocess_query(self, query: str) -> str:
        """Clean and preprocess the query."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove punctuation
        query = query.translate(str.maketrans('', '', string.punctuation))
        
        # Correct common misspellings
        words = query.split()
        corrected_words = []
        
        for word in words:
            if word in self.common_misspellings:
                corrected_words.append(self.common_misspellings[word])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def expand_query(self, query: str) -> List[str]:
        """Expand the query with synonyms and related terms."""
        # Preprocess the query
        processed_query = self.preprocess_query(query)
        
        # Initialize expanded queries list with the original query
        expanded_queries = [query]
        
        # Tokenize
        tokens = word_tokenize(processed_query)
        
        # Remove stopwords and lemmatize
        filtered_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                lemmatized = self.lemmatizer.lemmatize(token)
                filtered_tokens.append(lemmatized)
        
        # Check for locations and add them separately
        detected_locations = []
        for location in self.locations:
            if location in processed_query:
                detected_locations.append(location)
        
        # Generate synonym expansions
        synonym_expansions = []
        
        # For each token, check if we have synonyms and create expanded versions
        for i, token in enumerate(filtered_tokens):
            if token in self.synonyms:
                for synonym in self.synonyms[token]:
                    # Create a new query with the synonym
                    new_tokens = filtered_tokens.copy()
                    new_tokens[i] = synonym
                    synonym_expansions.append(' '.join(new_tokens))
        
        # Add synonym expansions
        expanded_queries.extend(synonym_expansions)
        
        # Add location-specific expansions
        for location in detected_locations:
            expanded_queries.append(f"hotel in {location}")
            expanded_queries.append(f"accommodation in {location}")
        
        # Remove duplicates and return
        return list(set(expanded_queries))
        self.bm25_retriever = bm25_retriever
        self.chroma_collection = chroma_collection
        self.similarity_top_k = similarity_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k
        super().__init__()
        
        # Features we want to filter by
        self.feature_map = {
            "luxury": "has_luxury",
            "budget": "has_budget",
            "family": "has_family_friendly",
            "spa": "has_spa",
            "pool": "has_pool",
            "breakfast": "has_breakfast",
            "beach": "has_beach",
            "business": "has_business",
            "central": "has_city_center",
            "downtown": "has_city_center",
            "city center": "has_city_center",
            "wifi": "has_wifi",
            "parking": "has_parking",
            "pet": "has_pet_friendly",
            "airport": "has_airport_shuttle"
        }
        
        # Location keywords
        self.locations = [
            "new york", "las vegas", "london", "paris", "tokyo", "rome", "barcelona", 
            "dubai", "sydney", "hong kong", "berlin", "amsterdam", "madrid", "singapore", 
            "bangkok", "istanbul", "prague", "vienna", "venice", "florence", "miami",
            "san francisco", "los angeles", "chicago", "toronto", "vancouver", "montreal"
        ]
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve documents using hybrid search approach."""
        query_text = query_bundle.query_str
        logger.info(f"Hybrid retrieval for query: {query_text}")
        
        # Extract features and locations from the query
        detected_features = self._extract_features(query_text)
        detected_locations = self._extract_locations(query_text)
        
        logger.info(f"Detected features: {detected_features}")
        logger.info(f"Detected locations: {detected_locations}")
        
        # Get results from vector search
        vector_results = self.vector_retriever.retrieve(query_bundle)
        
        # Get results from BM25
        bm25_results = self.bm25_retriever.retrieve(query_bundle)
        
        # Combine results with reciprocal rank fusion
        combined_results = self._reciprocal_rank_fusion(
            [vector_results, bm25_results], 
            k=60  # Fusion parameter
        )
        
        # Apply metadata filtering if features detected
        if detected_features or detected_locations:
            filtered_results = self._filter_by_metadata(
                combined_results, 
                detected_features, 
                detected_locations
            )
            # If filtering removed too many results, fall back to combined results
            if len(filtered_results) < 5:
                logger.info("Filtered results too few, falling back to combined results")
                return combined_results[:self.rerank_top_k]
            else:
                logger.info(f"Returning {len(filtered_results)} filtered results")
                return filtered_results[:self.rerank_top_k]
        else:
            # No filtering needed
            logger.info(f"Returning {len(combined_results)} combined results without filtering")
            return combined_results[:self.rerank_top_k]
    
    def _extract_features(self, query_text: str) -> List[str]:
        """Extract feature keywords from the query."""
        features = []
        query_lower = query_text.lower()
        
        for keyword, feature_name in self.feature_map.items():
            if keyword in query_lower:
                # Extract the base feature name without the "has_" prefix
                base_feature = feature_name.replace("has_", "")
                if base_feature not in features:
                    features.append(base_feature)
        
        return features
    
    def _extract_locations(self, query_text: str) -> List[str]:
        """Extract location names from the query."""
        locations = []
        query_lower = query_text.lower()
        
        for location in self.locations:
            if location in query_lower:
                locations.append(location)
        
        return locations
    
    def _filter_by_metadata(
        self, 
        nodes: List[NodeWithScore],
        features: List[str],
        locations: List[str]
    ) -> List[NodeWithScore]:
        """Filter nodes by metadata based on detected features and locations."""
        if not features and not locations:
            return nodes
        
        filtered_nodes = []
        
        for node in nodes:
            # Check if node has metadata
            if not hasattr(node, 'metadata') or not node.metadata:
                continue
            
            # Check features
            feature_match = True
            for feature in features:
                feature_key = f"has_{feature}"
                # If the feature is explicitly requested but the hotel doesn't have it, it's not a match
                if feature_key in node.metadata and not node.metadata[feature_key]:
                    feature_match = False
                    break
            
            # Check location
            location_match = True
            if locations:
                # If locations are specified but none match, it's not a match
                location_match = False
                for location in locations:
                    # Check if location appears in hotel name or address
                    hotel_name = node.metadata.get('Hotel_Name', '').lower()
                    hotel_address = node.metadata.get('Hotel_Address', '').lower()
                    
                    if location in hotel_name or location in hotel_address:
                        location_match = True
                        break
            
            # Add node if it matches both feature and location criteria
            if feature_match and location_match:
                filtered_nodes.append(node)
        
        return filtered_nodes
    
    def _reciprocal_rank_fusion(
        self, 
        results_lists: List[List[NodeWithScore]], 
        k: int = 60
    ) -> List[NodeWithScore]:
        """Combine multiple result lists using reciprocal rank fusion."""
        # Create a dictionary to track the RRF score for each node
        rrf_scores = {}
        
        # Process each result list
        for results in results_lists:
            for rank, node in enumerate(results):
                node_id = node.node.node_id
                
                # Calculate RRF score for this occurrence
                rrf_score = 1.0 / (rank + k)
                
                # Add to the node's total score
                if node_id in rrf_scores:
                    rrf_scores[node_id]["score"] += rrf_score
                else:
                    rrf_scores[node_id] = {
                        "node": node,
                        "score": rrf_score
                    }
        
        # Sort nodes by their RRF score
        sorted_nodes = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Extract just the NodeWithScore objects
        result = [item["node"] for item in sorted_nodes]
        
        return result

class AdvancedRAGEngine:
    """Enhanced RAG Engine with Advanced Search Features."""
    
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
        
        # Initialize the query expander
        self.query_expander = QueryExpander()
        
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
        llm = Ollama(model=self.llm_model, request_timeout=180.0)
        
        # Configure settings instead of ServiceContext
        from llama_index.core import Settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.system_prompt = SYSTEM_PROMPT
        
        # Set up ChromaDB client
        client = chromadb.PersistentClient(path=self.chroma_db_dir)
        chroma_collection = client.get_collection("hotel_reviews")
        
        # Create the vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create the index using Settings instead of service_context
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )
        
        # Set up the vector retriever
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=15,
        )
        
        # Set up the BM25 retriever with safeguards for empty corpus
        try:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.index.docstore,
                similarity_top_k=15,
            )
            # Test if the retriever has a valid corpus
            if hasattr(bm25_retriever, 'corpus') and not bm25_retriever.corpus:
                logger.warning("BM25 corpus is empty, falling back to vector retriever only")
                self.hybrid_retriever = None
                self.query_engine = self.index.as_query_engine(
                    text_qa_template=PromptTemplate(QUERY_PROMPT_TEMPLATE),
                    similarity_top_k=15,
                )
                return
        except Exception as e:
            logger.warning(f"Error initializing BM25Retriever: {str(e)}, falling back to vector retriever only")
            self.hybrid_retriever = None
            self.query_engine = self.index.as_query_engine(
                text_qa_template=PromptTemplate(QUERY_PROMPT_TEMPLATE),
                similarity_top_k=15,
            )
            return
        
        # Create the hybrid retriever if BM25 initialization was successful
        
        # Create the hybrid retriever if BM25 initialization was successful
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            chroma_collection=chroma_collection,
            similarity_top_k=15,
            bm25_top_k=15,
            rerank_top_k=15
        )
        
        # Create the query engine with custom prompt
        query_prompt = PromptTemplate(QUERY_PROMPT_TEMPLATE)
        self.query_engine = self.index.as_query_engine(
            text_qa_template=query_prompt,
            retriever=self.hybrid_retriever,
        )
        
        logger.info("RAG components set up successfully")
    
    def expand_query(self, query: str) -> List[str]:
        """Expand the query with synonyms and related terms."""
        return self.query_expander.expand_query(query)
    
    def query(self, user_query: str, use_query_expansion: bool = True, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run an enhanced query against the RAG engine with optional explicit filters."""
        logger.info(f"Running query: {user_query}")
        
        # If the hybrid retriever is None (fallback mode), use the regular query engine
        if self.hybrid_retriever is None:
            logger.info("Using vector retriever only (hybrid retrieval not available)")
            response = self.query_engine.query(user_query)
            source_nodes = response.source_nodes
            
            # Skip the query expansion since we're in fallback mode
            expanded_queries = [user_query]
        elif use_query_expansion:
            # Expand the query
            expanded_queries = self.expand_query(user_query)
            logger.info(f"Expanded queries: {expanded_queries}")
            
            # Use the original query as the main query
            response = self.query_engine.query(user_query)
            
            # Get the source nodes and keep track of seen hotels
            source_nodes = response.source_nodes
            seen_hotels = set()
            
            # Process additional expanded queries if we don't have enough results
            if len(source_nodes) < 10:
                for exp_query in expanded_queries[:3]:  # Limit to first 3 expansions
                    if exp_query == user_query:
                        continue
                    
                    # Run the expanded query
                    exp_response = self.query_engine.query(exp_query)
                    exp_nodes = exp_response.source_nodes
                    
                    # Add unique hotels to results
                    for node in exp_nodes:
                        if hasattr(node, 'metadata') and node.metadata:
                            hotel_name = node.metadata.get('Hotel_Name', '')
                            if hotel_name and hotel_name not in seen_hotels:
                                source_nodes.append(node)
                                seen_hotels.add(hotel_name)
                                
                                # Stop if we have enough results
                                if len(source_nodes) >= 15:
                                    break
                    
                    # Stop if we have enough results
                    if len(source_nodes) >= 15:
                        break
        else:
            # Use only the original query
            response = self.query_engine.query(user_query)
            source_nodes = response.source_nodes
            expanded_queries = [user_query]
        
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
                    'source': node.metadata.get('source', 'unknown'),
                    # Add feature flags
                    'features': {k: v for k, v in node.metadata.items() if k.startswith('has_')},
                    # Add sentiment scores
                    'sentiment': {
                        'compound': node.metadata.get('sentiment_compound', 0.0),
                        'positive': node.metadata.get('sentiment_positive', 0.0),
                        'neutral': node.metadata.get('sentiment_neutral', 0.0),
                        'negative': node.metadata.get('sentiment_negative', 0.0)
                    }
                })
        
        # Apply explicit filters if provided
        if filters:
            filtered_hotels = []
            for hotel in hotels_info:
                # Check minimum rating filter
                if 'min_rating' in filters and hotel['score'] < filters['min_rating']:
                    continue
                
                # Check maximum rating filter (useful for budget options)
                if 'max_rating' in filters and hotel['score'] > filters['max_rating']:
                    continue
                
                # Check required features
                if 'required_features' in filters and filters['required_features']:
                    has_all_required = True
                    for feature in filters['required_features']:
                        feature_key = f"has_{feature}"
                        if feature_key not in hotel['features'] or not hotel['features'][feature_key]:
                            has_all_required = False
                            break
                    if not has_all_required:
                        continue
                
                # Check minimum sentiment
                if 'min_sentiment' in filters and hotel['sentiment']['compound'] < filters['min_sentiment']:
                    continue
                
                # Location filtering will be handled by the map visualization
                
                # Hotel passed all filters
                filtered_hotels.append(hotel)
            
            # Replace hotels_info with filtered results
            hotels_info = filtered_hotels
        
        # Remove duplicates by name
        unique_hotels = []
        hotel_names = set()
        for hotel in hotels_info:
            if hotel['name'] not in hotel_names:
                unique_hotels.append(hotel)
                hotel_names.add(hotel['name'])
        
        # Sort hotels by rating if requested
        if filters and 'sort_by' in filters:
            if filters['sort_by'] == 'rating_high_to_low':
                unique_hotels.sort(key=lambda x: x['score'], reverse=True)
            elif filters['sort_by'] == 'rating_low_to_high':
                unique_hotels.sort(key=lambda x: x['score'])
            elif filters['sort_by'] == 'sentiment_high_to_low':
                unique_hotels.sort(key=lambda x: x['sentiment']['compound'], reverse=True)
        
        # Select top hotels (use a larger number if we have filters)
        top_hotels = unique_hotels[:min(10, len(unique_hotels))]
        
        return {
            'response': response.response,
            'hotels': top_hotels[:5],  # Still limit to 5 for main display
            'all_hotels': unique_hotels,
            'expanded_queries': expanded_queries if use_query_expansion else [user_query],
            'filter_stats': {
                'total_before_filter': len(hotels_info),
                'total_after_filter': len(unique_hotels)
            } if filters else None
        }

def main():
    """Test the enhanced RAG engine with a sample query."""
    logger.info("Testing Enhanced RAG engine")
    
    try:
        # Download NLTK resources
        download_nltk_resources()
        
        # Initialize the RAG engine
        rag_engine = AdvancedRAGEngine()
        
        # Test query with expanded search
        test_query = "I'm looking for a 5-star hotel in Paris with a spa and good breakfast"
        response = rag_engine.query(test_query, use_query_expansion=True)
        
        print(f"Original query: {test_query}")
        print(f"Expanded queries: {response['expanded_queries']}")
        print(f"Response: {response['response'][:200]}...")
        print(f"Number of hotels: {len(response['hotels'])}")
        
        print("\nTop 5 Hotels:")
        for i, hotel in enumerate(response['hotels']):
            print(f"{i+1}. {hotel['name']} ({hotel['address']})")
            print(f"   Rating: {hotel['score']}")
            print(f"   Features: {', '.join([k.replace('has_', '') for k, v in hotel['features'].items() if v])}")
            print(f"   Sentiment: {hotel['sentiment']['compound']:.2f}")
            print()
        
        logger.info("Enhanced RAG engine test completed successfully")
    
    except Exception as e:
        logger.error(f"Error testing enhanced RAG engine: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
