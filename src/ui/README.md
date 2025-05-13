# UI Module

This module contains the user interface implementation for the Hotel Recommendation RAG System.

## Key Components

- `app.py` - Main UI implementation using Gradio
  - Advanced filtering panel for refining results
  - Interactive map visualization with folium
  - Hotel comparison feature
  - Enhanced visualizations for features and sentiment
  - Query expansion toggle
  - Hotel selection interface

- `__init__.py` - Package initialization

## Core Classes

### `AdvancedHotelRecommendationUI` (in app.py)

The main UI class that:
- Builds the Gradio interface
- Creates visualizations (maps, charts)
- Handles user interactions
- Processes filters and search criteria
- Generates comparison views

## Key UI Features

1. **Advanced Filtering Panel**
   - Rating filters (minimum rating slider)
   - Sentiment filters (minimum sentiment score)
   - Feature filters (checkboxes for common hotel amenities)
   - Sorting options (rating high-to-low, rating low-to-high, sentiment high-to-low)

2. **Hotel Comparison Tool**
   - Hotel selection interface with checkboxes
   - Side-by-side comparison view for selected hotels
   - Visual indicators for feature presence/absence
   - Sentiment analysis comparison
   - Estimated price ranges based on hotel features

3. **Enhanced Visualizations**
   - Interactive map with color-coded markers and heatmap
   - Feature distribution chart
   - Sentiment analysis visualization
   - Rating comparison chart

4. **Chat Interface**
   - Natural language query input
   - Query expansion toggle
   - Formatted response display

## Key Methods

- `build_interface()`: Creates the complete Gradio interface
- `process_query()`: Processes user queries with filters
- `create_map()`: Generates interactive folium maps
- `create_feature_chart()`: Creates feature distribution charts
- `create_sentiment_chart()`: Creates sentiment analysis charts
- `create_hotel_selection_html()`: Generates HTML for hotel selection
- `create_comparison_html()`: Generates HTML for hotel comparison

## Customization

- CSS styles can be modified in the `css` property
- Chart designs can be customized in the chart creation methods
- Filter options can be extended in the `build_interface()` method
- Additional visualizations can be added as new tabs
