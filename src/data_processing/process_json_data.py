#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JSON data processor for the hotel review datasets.
"""

import os
import json
import logging
import nltk
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import ijson  # For streaming large JSON files

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
def download_nltk_resources():
    """Download required NLTK resources."""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    
    # Download punkt_tab if not already available
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading additional NLTK resources (punkt_tab)...")
        nltk.download('punkt')  # This should include the punkt_tab resources

class JSONDataProcessor:
    """Class for processing hotel review datasets in JSON format."""
    
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        """Initialize the data processor."""
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.sia = SentimentIntensityAnalyzer()
        
        # Create processed data directory if it doesn't exist
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
    
    def load_offerings(self):
        """Load the hotel offerings dataset."""
        logger.info("Loading offerings dataset from %s", self.raw_data_dir)
        
        offerings_path = os.path.join(self.raw_data_dir, "offering.txt")
        
        if not os.path.exists(offerings_path):
            raise FileNotFoundError(f"Offerings file not found at {offerings_path}")
        
        # Read offerings line by line (each line is a JSON object)
        offerings = []
        with open(offerings_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    offering = json.loads(line.strip())
                    offerings.append(offering)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing offering JSON: {e}")
                    continue
        
        logger.info(f"Loaded {len(offerings)} hotel offerings")
        return offerings
    
    def process_offerings(self, offerings):
        """Process hotel offerings data."""
        logger.info("Processing hotel offerings")
        
        processed_offerings = []
        
        for offering in tqdm(offerings, desc="Processing hotel offerings"):
            processed_offering = {
                'Hotel_ID': offering['id'],
                'Hotel_Name': offering['name'],
                'Hotel_Class': offering.get('hotel_class', None),
                'Region_ID': offering.get('region_id', None),
                'URL': offering.get('url', ''),
                'Type': offering.get('type', 'hotel')
            }
            
            # Extract address components
            address = offering.get('address', {})
            if address:
                processed_offering['Region'] = address.get('region', '')
                processed_offering['Street_Address'] = address.get('street-address', '')
                processed_offering['Postal_Code'] = address.get('postal-code', '')
                processed_offering['Locality'] = address.get('locality', '')
                
                # Construct full address
                processed_offering['Hotel_Address'] = (
                    f"{address.get('street-address', '')}, "
                    f"{address.get('locality', '')}, "
                    f"{address.get('region', '')} "
                    f"{address.get('postal-code', '')}"
                )
            else:
                processed_offering['Hotel_Address'] = ''
            
            # Extract coordinates (if available in your data)
            # If coordinates are in a different field, adjust accordingly
            processed_offering['Latitude'] = None
            processed_offering['Longitude'] = None
            
            processed_offerings.append(processed_offering)
        
        return pd.DataFrame(processed_offerings)
    
    def extract_features(self, review_text):
        """Extract features from review text."""
        features = {
            'amenities': False,
            'cleanliness': False,
            'service': False,
            'location': False,
            'value': False,
            'food': False,
            'room': False,
            'wifi': False,
            'family_friendly': False,
            'business': False,
            'luxury': False,
            'budget': False,
            'beach': False,
            'city_center': False,
            'pool': False,
            'spa': False,
            'parking': False,
            'pet_friendly': False,
            'airport_shuttle': False,
            'breakfast': False
        }
        
        # Simple keyword-based feature extraction
        keywords = {
            'amenities': ['amenity', 'amenities', 'facility', 'facilities'],
            'cleanliness': ['clean', 'cleanliness', 'dirty', 'dust', 'spotless'],
            'service': ['service', 'staff', 'helpful', 'friendly', 'reception'],
            'location': ['location', 'central', 'located', 'distance', 'far', 'close'],
            'value': ['value', 'price', 'expensive', 'cheap', 'affordable', 'worth'],
            'food': ['food', 'breakfast', 'dinner', 'lunch', 'meal', 'restaurant'],
            'room': ['room', 'bed', 'bathroom', 'shower', 'spacious', 'small'],
            'wifi': ['wifi', 'internet', 'connection', 'online'],
            'family_friendly': ['family', 'kid', 'child', 'children'],
            'business': ['business', 'work', 'meeting', 'conference'],
            'luxury': ['luxury', 'luxurious', 'elegant', 'upscale', 'fancy'],
            'budget': ['budget', 'cheap', 'affordable', 'inexpensive'],
            'beach': ['beach', 'ocean', 'sea', 'sand', 'shore'],
            'city_center': ['city center', 'downtown', 'central', 'heart of'],
            'pool': ['pool', 'swimming', 'swim'],
            'spa': ['spa', 'massage', 'wellness', 'sauna'],
            'parking': ['parking', 'park', 'garage'],
            'pet_friendly': ['pet', 'dog', 'cat', 'animal'],
            'airport_shuttle': ['airport', 'shuttle', 'transfer', 'pickup'],
            'breakfast': ['breakfast', 'morning meal', 'brunch']
        }
        
        if not isinstance(review_text, str):
            return features
        
        review_text = review_text.lower()
        
        for feature, kws in keywords.items():
            for kw in kws:
                if kw in review_text:
                    features[feature] = True
                    break
        
        return features
    
    def process_reviews_by_hotel(self, hotel_id, reviews, hotel_info):
        """Process reviews for a specific hotel."""
        if not reviews:
            return None
            
        logger.info(f"Processing {len(reviews)} reviews for hotel {hotel_id}")
        
        processed_reviews = []
        all_features = {}
        all_sentiments = {
            'compound': [],
            'neg': [],
            'neu': [],
            'pos': []
        }
        
        for review in tqdm(reviews, desc=f"Processing reviews for hotel {hotel_id}", leave=False):
            # Clean review text
            review_text = review.get('text', '')
            
            # Extract features from review text
            features = self.extract_features(review_text)
            
            # Analyze sentiment
            sentiment = self.sia.polarity_scores(review_text)
            
            # Extract ratings
            ratings = review.get('ratings', {})
            overall_rating = ratings.get('overall', None)
            
            # Create processed review
            processed_review = {
                'Review_ID': review.get('id'),
                'Hotel_ID': hotel_id,
                'Review': review_text,
                'Title': review.get('title', ''),
                'Rating': overall_rating,
                'Date': review.get('date', ''),
                'Date_Stayed': review.get('date_stayed', ''),
                'Author': review.get('author', {}).get('username', ''),
                'Author_Location': review.get('author', {}).get('location', ''),
                'Via_Mobile': review.get('via_mobile', False),
                'Helpful_Votes': review.get('num_helpful_votes', 0)
            }
            
            # Add individual ratings
            for key, value in ratings.items():
                processed_review[f'Rating_{key}'] = value
            
            # Add sentiment scores
            for key, value in sentiment.items():
                processed_review[f'Sentiment_{key}'] = value
                all_sentiments[key].append(value)
            
            # Add features
            for key, value in features.items():
                processed_review[f'Has_{key}'] = value
                
                # Update all_features
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(value)
            
            processed_reviews.append(processed_review)
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'Hotel_ID': hotel_id,
            'Hotel_Name': hotel_info.get('Hotel_Name', ''),
            'Hotel_Address': hotel_info.get('Hotel_Address', ''),
            'Average_Rating': np.mean([r.get('Rating', 0) for r in processed_reviews if r.get('Rating')]),
            'Review_Count': len(processed_reviews),
            'Latest_Review_Date': max([r.get('Date', '') for r in processed_reviews], default='')
        }
        
        # Add average sentiment scores
        for key, values in all_sentiments.items():
            if values:
                aggregate_metrics[f'Average_Sentiment_{key}'] = np.mean(values)
            else:
                aggregate_metrics[f'Average_Sentiment_{key}'] = 0
        
        # Add feature presence percentages
        for key, values in all_features.items():
            if values:
                aggregate_metrics[f'Percentage_Has_{key}'] = np.mean([1 if v else 0 for v in values]) * 100
                aggregate_metrics[f'Has_{key}'] = np.mean([1 if v else 0 for v in values]) > 0.3  # If at least 30% mention it
            else:
                aggregate_metrics[f'Percentage_Has_{key}'] = 0
                aggregate_metrics[f'Has_{key}'] = False
        
        # Add hotel info to aggregate metrics
        for key, value in hotel_info.items():
            if key not in aggregate_metrics:
                aggregate_metrics[key] = value
        
        return {
            'reviews': pd.DataFrame(processed_reviews),
            'aggregate': aggregate_metrics
        }
    
    def chunk_reviews_by_hotel(self, processed_data, chunk_size=512):
        """Chunk review text by hotel for optimal retrieval."""
        logger.info(f"Chunking reviews into segments of approximately {chunk_size} tokens")
        
        chunks = []
        
        for hotel_id, hotel_data in processed_data.items():
            if not hotel_data or 'reviews' not in hotel_data or 'aggregate' not in hotel_data:
                continue
                
            hotel_info = hotel_data['aggregate']
            reviews_df = hotel_data['reviews']
            
            # Combine all reviews for this hotel
            all_reviews = " ".join(reviews_df['Review'].dropna().tolist())
            
            # Simple chunking by splitting the text
            try:
                sentences = nltk.sent_tokenize(all_reviews)
            except LookupError:
                # Fallback if punkt_tab is not available - use a simple sentence splitter
                logger.warning("NLTK punkt_tab not available, using fallback sentence tokenization")
                sentences = []
                for sentence in re.split(r'(?<=[.!?])\s+', all_reviews):
                    if sentence.strip():
                        sentences.append(sentence.strip())
            
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                # Rough token count (words + punctuation)
                sentence_size = len(sentence.split())
                
                if current_size + sentence_size > chunk_size:
                    # Create a chunk
                    chunk_text = " ".join(current_chunk)
                    chunk_data = hotel_info.copy()
                    chunk_data['chunk_text'] = chunk_text
                    chunks.append(chunk_data)
                    
                    # Reset for next chunk
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            # Add the last chunk if it has content
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_data = hotel_info.copy()
                chunk_data['chunk_text'] = chunk_text
                chunks.append(chunk_data)
        
        chunks_df = pd.DataFrame(chunks)
        logger.info(f"Created {len(chunks_df)} chunks from reviews")
        
        return chunks_df
    
    def process_streaming(self, max_hotels=None, reviews_per_hotel=None):
        """Process hotel and review data with streaming for large files."""
        # Download NLTK resources
        download_nltk_resources()
        
        # Load and process offerings (hotels)
        offerings = self.load_offerings()
        offerings_df = self.process_offerings(offerings)
        
        # Create a lookup dictionary for hotels by ID
        hotels_by_id = {str(row['Hotel_ID']): row.to_dict() for _, row in offerings_df.iterrows()}
        
        # Create a directory to store processed reviews by hotel
        hotel_reviews_dir = os.path.join(self.processed_data_dir, "hotel_reviews")
        os.makedirs(hotel_reviews_dir, exist_ok=True)
        
        # Track hotels with reviews
        hotels_with_reviews = set()
        
        # Reviews file path
        reviews_path = os.path.join(self.raw_data_dir, "review.txt")
        
        if not os.path.exists(reviews_path):
            raise FileNotFoundError(f"Reviews file not found at {reviews_path}")
        
        # Group reviews by hotel
        reviews_by_hotel = {}
        hotels_processed = 0
        
        # Process reviews line by line (streaming for large file)
        with open(reviews_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Reading reviews")):
                try:
                    review = json.loads(line.strip())
                    hotel_id = str(review.get('offering_id', ''))
                    
                    # Skip if hotel not in our offerings data
                    if hotel_id not in hotels_by_id:
                        continue
                    
                    # Add to reviews_by_hotel
                    if hotel_id not in reviews_by_hotel:
                        reviews_by_hotel[hotel_id] = []
                    
                    reviews_by_hotel[hotel_id].append(review)
                    hotels_with_reviews.add(hotel_id)
                    
                    # Process hotel once we have enough reviews or at end of file
                    if reviews_per_hotel and len(reviews_by_hotel[hotel_id]) >= reviews_per_hotel:
                        hotel_info = hotels_by_id[hotel_id]
                        processed_data = self.process_reviews_by_hotel(
                            hotel_id, 
                            reviews_by_hotel[hotel_id], 
                            hotel_info
                        )
                        
                        # Save processed data for this hotel
                        if processed_data:
                            processed_data['reviews'].to_csv(
                                os.path.join(hotel_reviews_dir, f"hotel_{hotel_id}_reviews.csv"),
                                index=False
                            )
                            hotels_processed += 1
                        
                        # Clear memory
                        del reviews_by_hotel[hotel_id]
                        
                        # Check if we've reached the max hotels limit
                        if max_hotels and hotels_processed >= max_hotels:
                            logger.info(f"Reached maximum number of hotels to process: {max_hotels}")
                            break
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing review JSON at line {line_num}: {e}")
                    continue
        
        # Process any remaining hotels
        remaining_hotels = set(reviews_by_hotel.keys())
        logger.info(f"Processing remaining {len(remaining_hotels)} hotels")
        
        for hotel_id in tqdm(remaining_hotels, desc="Processing remaining hotels"):
            hotel_info = hotels_by_id[hotel_id]
            processed_data = self.process_reviews_by_hotel(
                hotel_id, 
                reviews_by_hotel[hotel_id], 
                hotel_info
            )
            
            # Save processed data for this hotel
            if processed_data:
                processed_data['reviews'].to_csv(
                    os.path.join(hotel_reviews_dir, f"hotel_{hotel_id}_reviews.csv"),
                    index=False
                )
                hotels_processed += 1
        
        # Now collect all the hotel aggregates and create chunks
        logger.info("Collecting hotel aggregates and creating chunks")
        aggregates = []
        
        for hotel_id in tqdm(hotels_with_reviews, desc="Collecting hotel aggregates"):
            # Skip if we don't have processed data for this hotel
            hotel_reviews_path = os.path.join(hotel_reviews_dir, f"hotel_{hotel_id}_reviews.csv")
            if not os.path.exists(hotel_reviews_path):
                continue
                
            # Load the processed reviews
            hotel_reviews = pd.read_csv(hotel_reviews_path)
            
            # Create the aggregate data if we have reviews
            if not hotel_reviews.empty:
                hotel_info = hotels_by_id[hotel_id]
                
                # Calculate aggregate metrics
                aggregate = {
                    'Hotel_ID': hotel_id,
                    'Hotel_Name': hotel_info.get('Hotel_Name', ''),
                    'Hotel_Address': hotel_info.get('Hotel_Address', ''),
                    'Latitude': hotel_info.get('Latitude', None),
                    'Longitude': hotel_info.get('Longitude', None),
                    'Average_Rating': hotel_reviews['Rating'].mean() if 'Rating' in hotel_reviews.columns else None,
                    'Review_Count': len(hotel_reviews)
                }
                
                # Add sentiment scores
                for key in ['compound', 'positive', 'neutral', 'negative']:
                    col = f'Sentiment_{key}'
                    if col in hotel_reviews.columns:
                        aggregate[f'sentiment_{key}'] = hotel_reviews[col].mean()
                
                # Add features
                feature_cols = [col for col in hotel_reviews.columns if col.startswith('Has_')]
                for col in feature_cols:
                    feature_name = col.lower()
                    aggregate[feature_name] = hotel_reviews[col].mean() > 0.3  # If >30% of reviews mention it
                
                aggregates.append(aggregate)
        
        # Save aggregates
        aggregates_df = pd.DataFrame(aggregates)
        aggregates_df.to_csv(os.path.join(self.processed_data_dir, "hotel_aggregates.csv"), index=False)
        
        # Create chunks for embedding
        logger.info("Creating review chunks for embedding")
        
        all_chunks = []
        
        for hotel_id in tqdm(hotels_with_reviews, desc="Chunking reviews by hotel"):
            # Skip if we don't have processed data for this hotel
            hotel_reviews_path = os.path.join(hotel_reviews_dir, f"hotel_{hotel_id}_reviews.csv")
            if not os.path.exists(hotel_reviews_path):
                continue
                
            # Load the processed reviews
            hotel_reviews = pd.read_csv(hotel_reviews_path)
            
            if not hotel_reviews.empty:
                # Get hotel info
                hotel_info = next((agg for agg in aggregates if agg['Hotel_ID'] == hotel_id), 
                                  hotels_by_id.get(hotel_id, {}))
                
                # Combine all reviews
                all_review_text = " ".join(hotel_reviews['Review'].dropna().tolist())
                
                # Simple chunking by sentences
                chunk_size = 512  # tokens
                try:
                    sentences = nltk.sent_tokenize(all_review_text)
                except LookupError:
                    # Fallback if punkt_tab is not available - use a simple sentence splitter
                    logger.warning("NLTK punkt_tab not available, using fallback sentence tokenization")
                    sentences = []
                    for sentence in re.split(r'(?<=[.!?])\s+', all_review_text):
                        if sentence.strip():
                            sentences.append(sentence.strip())
                
                current_chunk = []
                current_size = 0
                
                for sentence in sentences:
                    # Rough token count
                    sentence_size = len(sentence.split())
                    
                    if current_size + sentence_size > chunk_size:
                        # Create a chunk
                        chunk_text = " ".join(current_chunk)
                        
                        chunk = {
                            'Hotel_ID': hotel_id,
                            'Hotel_Name': hotel_info.get('Hotel_Name', ''),
                            'Hotel_Address': hotel_info.get('Hotel_Address', ''),
                            'Latitude': hotel_info.get('Latitude', None),
                            'Longitude': hotel_info.get('Longitude', None),
                            'chunk_text': chunk_text
                        }
                        
                        # Add sentiment and feature fields
                        for key, value in hotel_info.items():
                            if key.startswith('sentiment_') or key.startswith('has_'):
                                chunk[key] = value
                        
                        all_chunks.append(chunk)
                        
                        # Reset chunk
                        current_chunk = [sentence]
                        current_size = sentence_size
                    else:
                        current_chunk.append(sentence)
                        current_size += sentence_size
                
                # Add the last chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    
                    chunk = {
                        'Hotel_ID': hotel_id,
                        'Hotel_Name': hotel_info.get('Hotel_Name', ''),
                        'Hotel_Address': hotel_info.get('Hotel_Address', ''),
                        'Latitude': hotel_info.get('Latitude', None),
                        'Longitude': hotel_info.get('Longitude', None),
                        'chunk_text': chunk_text
                    }
                    
                    # Add sentiment and feature fields
                    for key, value in hotel_info.items():
                        if key.startswith('sentiment_') or key.startswith('has_'):
                            chunk[key] = value
                    
                    all_chunks.append(chunk)
        
        # Save chunks
        chunks_df = pd.DataFrame(all_chunks)
        chunks_df.to_csv(os.path.join(self.processed_data_dir, "chunks.csv"), index=False)
        
        # Save the offerings data
        offerings_df.to_csv(os.path.join(self.processed_data_dir, "offerings.csv"), index=False)
        
        logger.info("Data processing completed successfully")
        logger.info(f"Processed {hotels_processed} hotels with reviews")
        logger.info(f"Created {len(chunks_df)} chunks for embedding")
        
        return {
            'hotels_processed': hotels_processed,
            'chunks_created': len(chunks_df)
        }

def main():
    """Main function to process JSON hotel data."""
    logger.info("Starting JSON data processing")
    
    processor = JSONDataProcessor()
    
    # Process with limits for testing (remove limits for full processing)
    # For testing: limit to 20 hotels with max 100 reviews each
    # For production: set max_hotels=None, reviews_per_hotel=None
    result = processor.process_streaming(max_hotels=20, reviews_per_hotel=100)
    
    logger.info(f"JSON data processing completed: {result}")
    print("Data processing completed successfully!")
    print(f"Processed {result['hotels_processed']} hotels")
    print(f"Created {result['chunks_created']} chunks for embedding")
    print("You can now generate embeddings with: python src/data_processing/generate_embeddings.py")

if __name__ == "__main__":
    main()
