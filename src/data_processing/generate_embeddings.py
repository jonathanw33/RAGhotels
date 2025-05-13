#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate embeddings and set up the vector database for hotel review chunks.
"""

import os
import pandas as pd
import logging
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
from typing import List, Union, Sequence

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom embedding function that doesn't require HuggingFace API key
class LocalSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Using sentence-transformers as embedding model."""

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(input)
        return embeddings.tolist()

class EmbeddingGenerator:
    """Class for generating embeddings and setting up the vector database."""
    
    def __init__(self, 
                 processed_data_dir="data/processed", 
                 chroma_db_dir="data/processed/chroma_db",
                 embedding_model="BAAI/bge-large-en-v1.5"):
        """Initialize the embedding generator."""
        self.processed_data_dir = processed_data_dir
        self.chroma_db_dir = chroma_db_dir
        self.embedding_model = embedding_model
        
        # Create the chroma_db directory if it doesn't exist
        if not os.path.exists(chroma_db_dir):
            os.makedirs(chroma_db_dir)
    
    def setup_chroma_client(self):
        """Set up and return the ChromaDB client."""
        logger.info(f"Setting up ChromaDB client at {self.chroma_db_dir}")
        
        # Set up the embedding function using our custom local implementation
        logger.info(f"Using local embedding model: {self.embedding_model}")
        ef = LocalSentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        
        # Create a persistent client
        client = chromadb.PersistentClient(path=self.chroma_db_dir)
        
        return client, ef
    
    def load_chunks(self):
        """Load the review chunks."""
        chunks_path = os.path.join(self.processed_data_dir, "chunks.csv")
        
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Chunks file not found at {chunks_path}. "
                f"Please run the data processing script first."
            )
        
        chunks_df = pd.read_csv(chunks_path)
        logger.info(f"Loaded {len(chunks_df)} chunks")
        
        return chunks_df
    
    def generate_embeddings(self):
        """Generate embeddings and store them in ChromaDB."""
        # Load the chunks
        try:
            chunks_df = self.load_chunks()
        except FileNotFoundError as e:
            logger.error(str(e))
            print(f"Error: {str(e)}")
            return False
        
        # Set up the ChromaDB client
        client, ef = self.setup_chroma_client()
        
        # Create or get the collection
        collection = client.get_or_create_collection(
            name="hotel_reviews",
            embedding_function=ef,
            metadata={"description": "Hotel review chunks for RAG system"}
        )
        
        # Check if the collection already has documents
        if collection.count() > 0:
            logger.info(f"Collection already has {collection.count()} documents")
            user_input = input("Collection already exists. Do you want to overwrite it? (y/n): ")
            if user_input.lower() != 'y':
                logger.info("Skipping embedding generation")
                return True
            else:
                logger.info("Clearing existing collection")
                collection.delete(where={})
        
        # Process chunks in batches to avoid memory issues
        batch_size = 100
        total_batches = int(np.ceil(len(chunks_df) / batch_size))
        
        for i in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(chunks_df))
            
            batch_df = chunks_df.iloc[start_idx:end_idx]
            
            # Prepare the batch data for ChromaDB
            ids = [f"chunk_{j}" for j in range(start_idx, end_idx)]
            documents = batch_df['chunk_text'].tolist()
            
            # Prepare metadata
            metadatas = []
            for _, row in batch_df.iterrows():
                metadata = {
                    'Hotel_Name': row['Hotel_Name'],
                    'Hotel_Address': row['Hotel_Address']
                }
                
                # Add average score if available
                if 'Average_Score' in row:
                    metadata['Average_Score'] = float(row['Average_Score'])
                elif 'Average_Rating' in row:
                    metadata['Average_Score'] = float(row['Average_Rating'])
                
                # Add source if available
                if 'source' in row:
                    metadata['source'] = row['source']
                
                # Add coordinates if available
                if 'Latitude' in row and 'Longitude' in row and not pd.isna(row['Latitude']) and not pd.isna(row['Longitude']):
                    metadata['Latitude'] = float(row['Latitude'])
                    metadata['Longitude'] = float(row['Longitude'])
                
                # Add feature flags
                for col in row.index:
                    if col.startswith('has_'):
                        metadata[col] = bool(row[col])
                
                # Add sentiment scores
                for sentiment_type in ['compound', 'pos', 'neg', 'neu']:
                    col = f'sentiment_{sentiment_type}'
                    if col in row and not pd.isna(row[col]):
                        metadata[col] = float(row[col])
                
                metadatas.append(metadata)
            
            # Add the batch to the collection
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added batch {i+1}/{total_batches} to ChromaDB")
        
        logger.info(f"Added a total of {collection.count()} documents to ChromaDB")
        return True
    
def main():
    """Main function to generate embeddings."""
    logger.info("Starting embedding generation")
    
    # Use BGE-M3 by default
    generator = EmbeddingGenerator(embedding_model="BAAI/bge-large-en-v1.5")
    success = generator.generate_embeddings()
    
    if success:
        logger.info("Embedding generation completed successfully")
        print("Embedding generation completed successfully!")
        print("You can now run the main application: python src/main.py")
    else:
        logger.error("Embedding generation failed")
        print("Embedding generation failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
