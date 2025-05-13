# Hotel Review Datasets

This directory should contain the hotel review datasets used for the recommendation system.

## Supported Data Formats

The system supports two data formats:

### 1. JSON Format (TripAdvisor Data)

Required files:
- `offering.txt` - JSON file with hotel details (one JSON object per line)
- `review.txt` - JSON file with hotel reviews (one JSON object per line)

#### Hotel Offerings Format (offering.txt)
Each line contains a JSON object with the following structure:
```json
{
  "hotel_class": 4.0,
  "region_id": 60763,
  "url": "http://www.tripadvisor.com/Hotel_Review-...",
  "phone": "",
  "details": null,
  "address": {
    "region": "NY",
    "street-address": "123 Main St",
    "postal-code": "10001",
    "locality": "New York City"
  },
  "type": "hotel",
  "id": 12345,
  "name": "Hotel Name"
}
```

#### Hotel Reviews Format (review.txt)
Each line contains a JSON object with the following structure:
```json
{
  "ratings": {
    "service": 5.0,
    "cleanliness": 5.0,
    "overall": 5.0,
    "value": 5.0,
    "location": 5.0,
    "sleep_quality": 5.0,
    "rooms": 5.0
  },
  "title": "Great Stay",
  "text": "This hotel was amazing...",
  "author": {
    "username": "Traveler123",
    "location": "New York, NY"
  },
  "date_stayed": "December 2020",
  "offering_id": 12345,
  "date": "January 2, 2021",
  "id": 9876543,
  "via_mobile": false
}
```

### 2. CSV Format

Required files (at least one):
1. `tripadvisor_hotel_reviews.csv` - TripAdvisor hotel reviews
2. `booking_hotel_reviews.csv` - Booking.com hotel reviews

#### TripAdvisor Dataset Format
- `Hotel_Name`: Name of the hotel
- `Hotel_Address`: Address of the hotel
- `Review`: The text of the review
- `Rating`: Rating given by the reviewer (numeric, typically 1-5)
- `Reviewer_Nationality`: Nationality of the reviewer (optional)
- `Review_Date`: Date of the review (optional)
- `Latitude`: Latitude coordinates of the hotel (optional but recommended for mapping)
- `Longitude`: Longitude coordinates of the hotel (optional but recommended for mapping)

#### Booking.com Dataset Format
- `Hotel_Name`: Name of the hotel
- `Hotel_Address`: Address of the hotel
- `Review`: The text of the review
- `Rating`: Rating given by the reviewer (numeric, typically 1-10, will be normalized to 1-5)
- `Reviewer_Nationality`: Nationality of the reviewer (optional)
- `Review_Date`: Date of the review (optional)
- `Latitude`: Latitude coordinates of the hotel (optional but recommended for mapping)
- `Longitude`: Longitude coordinates of the hotel (optional but recommended for mapping)

### 3. Sample Data

A sample dataset is provided for testing purposes:
- `sample_hotel_reviews.csv` - Small dataset with sample hotel reviews

## Data Processing

After placing the datasets in this directory, run the appropriate data processing script:

### For JSON Format:
```
python src/data_processing/process_json_data.py
python src/data_processing/generate_embeddings.py
```

### For CSV Format:
```
python src/data_processing/process_data.py
python src/data_processing/generate_embeddings.py
```

### Automatic Setup:
```
python setup.py
```

This will automatically detect your data format and run the appropriate processing scripts.
