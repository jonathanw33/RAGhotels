# Advanced UI Features

The Hotel Recommendation RAG System includes several advanced UI features to enhance usability and provide a more powerful user experience.

## Advanced Filtering Panel

![Advanced Filtering Panel](images/filtering_panel.png)

The Advanced Filtering Panel allows users to refine their hotel search results with specific criteria:

### Rating Filters
- **Minimum Rating**: Filter hotels by minimum star rating (0-5 stars)
- Useful for finding only top-rated hotels regardless of other features

### Sentiment Filters
- **Minimum Sentiment Score**: Filter hotels by review sentiment
- Higher values prioritize hotels with more positive reviews
- Scale ranges from 0 (include all hotels) to 1 (only very positive reviews)

### Feature Filters
- **Required Features**: Select specific amenities that hotels must have
- Enables multi-criteria filtering (e.g., hotels with pool AND spa AND breakfast)
- Common hotel features are provided as checkbox options

### Sorting Options
- **Rating (High to Low)**: Sort hotels by highest rating first
- **Rating (Low to High)**: Sort hotels by lowest rating first
- **Sentiment (High to Low)**: Sort hotels by most positive sentiment first

## Hotel Comparison Tool

![Hotel Comparison](images/comparison_tool.png)

The Hotel Comparison feature allows users to select multiple hotels and view them side-by-side:

### Selection Interface
- Select up to 5 hotels from search results
- Each hotel card shows key information for easy selection
- Checkboxes allow for intuitive selection

### Comparison Table
- Side-by-side comparison of all selected hotels
- Compare ratings, features, location, and sentiment scores
- Visual indicators highlight differences (checkmarks and color coding)
- Estimated price comparison based on features and ratings

## Enhanced Visualizations

The system includes several improved visualization features:

### Interactive Map
- **Color-coded markers**: Hotels are color-coded by rating
- **Heatmap overlay**: Shows density of hotels in different areas
- **Measurement tool**: Calculate distances between locations
- **Enhanced popups**: Detailed hotel information in map popups
- **Selected hotel highlighting**: Compared hotels are highlighted in purple

### Feature Distribution Chart
- Visual representation of common features across search results
- Percentage-based visualization for better understanding
- Enhanced design with gradient colors and data labels

### Sentiment Analysis Visualization
- Breakdown of sentiment scores across recommended hotels
- Stacked bar chart showing positive, neutral and negative sentiment components
- Color-coded for easy interpretation

## Usage Tips

1. **Start with a natural language query** in the search box
2. **Refine results** using the Advanced Filtering Panel
3. **Select hotels** for comparison by checking the boxes in the Comparison tab
4. **Click "Compare Selected Hotels"** to generate a side-by-side comparison
5. **Explore the Map tab** to see hotel locations and geographical distribution
6. **Use the Features tab** to understand what amenities are available
7. **Check the Sentiment tab** to see review sentiment analysis

## Implementation Details

The Advanced UI is implemented in `src/ui/app.py` using the Gradio framework. Key components include:

- `AdvancedHotelRecommendationUI` class: Main UI implementation
- Custom HTML generation for hotel comparison and selection
- JavaScript integration for interactive selection
- Enhanced visualization methods for better data presentation
- Filter processing and application logic
