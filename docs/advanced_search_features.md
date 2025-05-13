# Advanced Search Features - Implementation Details

This document provides details about the advanced search features implemented in the Hotel Recommendation RAG System.

## 1. Hybrid Search

The system implements a sophisticated hybrid search approach that combines multiple retrieval mechanisms:

### Vector Search
- Uses embedding-based semantic search to find contextually relevant results
- Leverages the BGE embedding model for high-quality semantic understanding
- Captures nuanced meaning beyond keyword matching

### BM25 Lexical Search
- Implements the BM25 algorithm for keyword-based retrieval
- Provides complementary results that might be missed by semantic search
- Particularly effective for exact feature matching

### Reciprocal Rank Fusion
- Combines results from both search methods using RRF algorithm
- Ranks documents based on their positions in multiple result lists
- Produces a unified ranking that balances semantic and lexical relevance

### Metadata Filtering
- Dynamically extracts features and locations from user queries
- Filters results based on hotel attributes in the vector database
- Implements fallback mechanisms to prevent over-filtering

## 2. Query Expansion

The system automatically expands queries to improve recall:

### Synonym Expansion
- Maintains a domain-specific synonym dictionary for hotel terminology
- Expands terms like "luxury" → "high-end", "five-star", "premium", etc.
- Generates multiple query variations to capture different expressions of the same intent

### Location Detection
- Identifies location mentions in queries
- Adds location-specific variations to capture more relevant results
- Supports over 50 major cities worldwide

### Misspelling Correction
- Automatically corrects common misspellings in hotel-related terminology
- Handles variations like "breakfst" → "breakfast", "wiifi" → "wifi", etc.
- Increases robustness to user input variations

### Query Preprocessing
- Removes stopwords and normalizes text
- Lemmatizes terms to handle different word forms
- Maintains important context while reducing noise

## 3. Faceted Search

The system implements dynamic filtering based on detected attributes:

### Feature Extraction
- Automatically identifies requested features in natural language queries
- Maps user expressions to internal feature representations
- Supports 20+ hotel features including amenities, price categories, and guest types

### Faceted Filtering
- Filters hotels based on extracted features
- Dynamically adjusts filtering based on available results
- Implements "soft filtering" to avoid empty result sets

### Feature Highlighting
- Highlights matching features in recommendation results
- Provides visual indicators of feature matches in the UI
- Improves user understanding of why hotels were recommended

## 4. Enhanced Visualization

The system includes multiple visualization components:

### Interactive Map
- Displays hotel locations on an interactive map
- Uses color coding to indicate hotel ratings
- Includes popup information with key hotel details

### Feature Chart
- Visualizes the distribution of features across recommended hotels
- Helps users understand the availability of desired amenities
- Provides at-a-glance feature comparison

### Sentiment Analysis Visualization
- Shows sentiment breakdown from review analysis
- Compares positive, neutral, and negative sentiment across hotels
- Provides additional context beyond numerical ratings

## Technical Implementation

The advanced search features are implemented in:
- `src/rag/advanced_engine.py`: Core search functionality
- `src/ui/app.py`: Enhanced visualization interface

### Custom Components

1. **HybridRetriever**: Custom LlamaIndex retriever that combines multiple search methods
2. **QueryExpander**: Specialized component for query enhancement and expansion
3. **EnhancedHotelRecommendationUI**: Extended UI with visualization capabilities

### Performance Considerations

- Query expansion is optional and can be toggled in the UI
- The system caches results for similar queries to improve performance
- Visualization generation is designed to minimize processing overhead

## Usage Examples

**Original Query**: "hotel in paris with breakfast"

**Expanded to**:
- "hotel in paris with breakfast"
- "accommodation in paris with breakfast"
- "inn in paris with breakfast"
- "hotel in paris with morning meal"
- "resort in paris with breakfast buffet"

**Feature Extraction**:
- Location: "paris"
- Amenities: "breakfast"

**Result Filtering**:
- Hotels in Paris location
- Hotels with breakfast amenity
- Ranked by combination of semantic relevance and feature match

## Future Enhancements

Potential improvements to the advanced search functionality:

1. **Personalized Ranking**: Adjust result ranking based on user preferences
2. **Conversational Refinement**: Allow users to refine search through follow-up questions
3. **Multi-criteria Optimization**: Balance multiple preferences with weighted ranking
4. **Time-aware Filtering**: Consider seasonality and time-specific factors
5. **User Feedback Loop**: Incorporate feedback to improve future search results
