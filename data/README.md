# Data Folder

This folder contains all the data-related files for the Hotel Recommendation RAG System.

## Structure

- `raw/` - Contains raw hotel review datasets
  - `tripadvisor_hotel_reviews.csv` - TripAdvisor hotel reviews dataset
  - `booking_hotel_reviews.csv` - Booking.com hotel reviews dataset
  - `sample_hotel_reviews.csv` - Sample dataset for testing purposes
  - `README.md` - Documentation on data format requirements

- `processed/` - Contains processed data and vector database
  - `processed_tripadvisor.csv` - Processed TripAdvisor reviews
  - `processed_booking.csv` - Processed Booking.com reviews
  - `processed_combined.csv` - Combined dataset of all processed reviews
  - `chunks.csv` - Chunked reviews ready for embedding
  - `chroma_db/` - ChromaDB vector database containing embedded reviews

## Key Files

- `sample_hotel_reviews.csv`: A small sample dataset with 21 reviews for 7 hotels that can be used for testing the system without the full datasets.

## Data Processing Flow

1. Raw reviews → Cleaning → Feature extraction → Sentiment analysis → `processed_*.csv`
2. Processed reviews → Chunking → `chunks.csv`
3. Chunks → Embedding → `chroma_db/`

## Important Notes

- The system expects the review datasets to have specific columns as documented in `raw/README.md`
- If you don't have the official datasets, you can rename `sample_hotel_reviews.csv` to `tripadvisor_hotel_reviews.csv` for testing
- The ChromaDB vector database will be created automatically during setup
