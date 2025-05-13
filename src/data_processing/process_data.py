#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing module for the hotel review datasets.
"""

import os
import pandas as pd
import nltk
import logging
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import numpy as np

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

class DataProcessor:
    """Class for processing hotel review datasets."""
    
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        """Initialize the data processor."""
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.sia = SentimentIntensityAnalyzer()
        
        # Create processed data directory if it doesn't exist
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
    
    def load_datasets(self):
        """Load the hotel review datasets."""
        logger.info("Loading datasets from %s", self.raw_data_dir)
        
        tripadvisor_path = os.path.join(self.raw_data_dir, "tripadvisor_hotel_reviews.csv")
        booking_path = os.path.join(self.raw_data_dir, "booking_hotel_reviews.csv")
        
        # Check if datasets exist
        if not (os.path.exists(tripadvisor_path) and os.path.exists(booking_path)):
            raise FileNotFoundError(
                f"Dataset files not found in {self.raw_data_dir}. "
                f"Please ensure tripadvisor_hotel_reviews.csv and booking_hotel_reviews.csv exist."
            )
        
        # Load datasets
        try:
            tripadvisor_df = pd.read_csv(tripadvisor_path)
            booking_df = pd.read_csv(booking_path)
            
            logger.info(f"Loaded TripAdvisor dataset with {len(tripadvisor_df)} reviews")
            logger.info(f"Loaded Booking.com dataset with {len(booking_df)} reviews")
            
            return tripadvisor_df, booking_df
        
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise
    
    def clean_text(self, text):
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower()
        text = ' '.join(text.split())  # Remove extra spaces
        
        return text
    
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
        
        for feature, kws in keywords.items():
            for kw in kws:
                if kw in review_text.lower():
                    features[feature] = True
                    break
        
        return features
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of review text."""
        if not isinstance(text, str):
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        
        return self.sia.polarity_scores(text)
    
    def process_tripadvisor_data(self, df):
        """Process TripAdvisor dataset."""
        logger.info("Processing TripAdvisor dataset")
        
        # Clone the dataframe to avoid modification warnings
        processed_df = df.copy()
        
        # Clean review text
        processed_df['cleaned_review'] = processed_df['Review'].apply(self.clean_text)
        
        # Extract features
        tqdm.pandas(desc="Extracting features")
        processed_df['features'] = processed_df['cleaned_review'].progress_apply(self.extract_features)
        
        # Extract sentiment
        tqdm.pandas(desc="Analyzing sentiment")
        processed_df['sentiment'] = processed_df['cleaned_review'].progress_apply(self.analyze_sentiment)
        
        # Add source column
        processed_df['source'] = 'tripadvisor'
        
        # Extract the sentiment scores into separate columns
        processed_df['sentiment_compound'] = processed_df['sentiment'].apply(lambda x: x['compound'])
        processed_df['sentiment_positive'] = processed_df['sentiment'].apply(lambda x: x['pos'])
        processed_df['sentiment_neutral'] = processed_df['sentiment'].apply(lambda x: x['neu'])
        processed_df['sentiment_negative'] = processed_df['sentiment'].apply(lambda x: x['neg'])
        
        # Extract features into separate columns
        for feature in next(iter(processed_df['features']), {}):
            processed_df[f'has_{feature}'] = processed_df['features'].apply(lambda x: x.get(feature, False))
        
        return processed_df
    
    def process_booking_data(self, df):
        """Process Booking.com dataset."""
        logger.info("Processing Booking.com dataset")
        
        # Clone the dataframe to avoid modification warnings
        processed_df = df.copy()
        
        # Clean review text
        processed_df['cleaned_review'] = processed_df['Review'].apply(self.clean_text)
        
        # Extract features
        tqdm.pandas(desc="Extracting features")
        processed_df['features'] = processed_df['cleaned_review'].progress_apply(self.extract_features)
        
        # Extract sentiment
        tqdm.pandas(desc="Analyzing sentiment")
        processed_df['sentiment'] = processed_df['cleaned_review'].progress_apply(self.analyze_sentiment)
        
        # Add source column
        processed_df['source'] = 'booking'
        
        # Extract the sentiment scores into separate columns
        processed_df['sentiment_compound'] = processed_df['sentiment'].apply(lambda x: x['compound'])
        processed_df['sentiment_positive'] = processed_df['sentiment'].apply(lambda x: x['pos'])
        processed_df['sentiment_neutral'] = processed_df['sentiment'].apply(lambda x: x['neu'])
        processed_df['sentiment_negative'] = processed_df['sentiment'].apply(lambda x: x['neg'])
        
        # Extract features into separate columns
        for feature in next(iter(processed_df['features']), {}):
            processed_df[f'has_{feature}'] = processed_df['features'].apply(lambda x: x.get(feature, False))
        
        return processed_df
    
    def chunk_reviews(self, df, chunk_size=512):
        """Chunk review text into smaller segments for embedding."""
        logger.info(f"Chunking reviews into segments of approximately {chunk_size} tokens")
        
        chunks = []
        
        # Group by hotel
        for hotel_name, hotel_df in tqdm(df.groupby('Hotel_Name'), desc="Chunking reviews by hotel"):
            # Combine all reviews for this hotel
            all_reviews = " ".join(hotel_df['cleaned_review'].dropna().tolist())
            
            # Get basic hotel info
            hotel_info = {
                'Hotel_Name': hotel_name,
                'Hotel_Address': hotel_df['Hotel_Address'].iloc[0] if 'Hotel_Address' in hotel_df.columns else "",
                'Average_Score': hotel_df['Rating'].mean() if 'Rating' in hotel_df.columns else 0,
                'Latitude': hotel_df['Latitude'].iloc[0] if 'Latitude' in hotel_df.columns else np.nan,
                'Longitude': hotel_df['Longitude'].iloc[0] if 'Longitude' in hotel_df.columns else np.nan,
                'source': hotel_df['source'].iloc[0]
            }
            
            # Aggregate features
            feature_cols = [col for col in hotel_df.columns if col.startswith('has_')]
            for col in feature_cols:
                hotel_info[col] = hotel_df[col].mean() > 0.3  # If at least 30% of reviews mention it
            
            # Sentiment averages
            hotel_info['sentiment_compound'] = hotel_df['sentiment_compound'].mean()
            hotel_info['sentiment_positive'] = hotel_df['sentiment_positive'].mean()
            hotel_info['sentiment_neutral'] = hotel_df['sentiment_neutral'].mean()
            hotel_info['sentiment_negative'] = hotel_df['sentiment_negative'].mean()
            
            # Simple chunking by splitting the text
            sentences = nltk.sent_tokenize(all_reviews)
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
        logger.info(f"Created {len(chunks_df)} chunks from {len(df)} reviews")
        
        return chunks_df
    
    def process(self):
        """Process the datasets and create chunks."""
        # Download NLTK resources
        download_nltk_resources()
        
        # Load datasets
        try:
            tripadvisor_df, booking_df = self.load_datasets()
        except FileNotFoundError as e:
            logger.error(str(e))
            print(f"Error: {str(e)}")
            print("Please place the dataset files in the data/raw directory.")
            return False
        
        # Process datasets
        processed_tripadvisor = self.process_tripadvisor_data(tripadvisor_df)
        processed_booking = self.process_booking_data(booking_df)
        
        # Combine datasets
        logger.info("Combining datasets")
        combined_df = pd.concat([processed_tripadvisor, processed_booking], ignore_index=True)
        
        # Create chunks for embeddings
        chunks_df = self.chunk_reviews(combined_df)
        
        # Save processed data
        logger.info("Saving processed data")
        processed_tripadvisor.to_csv(os.path.join(self.processed_data_dir, "processed_tripadvisor.csv"), index=False)
        processed_booking.to_csv(os.path.join(self.processed_data_dir, "processed_booking.csv"), index=False)
        combined_df.to_csv(os.path.join(self.processed_data_dir, "processed_combined.csv"), index=False)
        chunks_df.to_csv(os.path.join(self.processed_data_dir, "chunks.csv"), index=False)
        
        logger.info("Data processing completed successfully")
        return True

def main():
    """Main function to process data."""
    logger.info("Starting data processing")
    
    processor = DataProcessor()
    success = processor.process()
    
    if success:
        logger.info("Data processing completed successfully")
        print("Data processing completed successfully!")
        print("You can now run the main application: python src/main.py")
    else:
        logger.error("Data processing failed")
        print("Data processing failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
