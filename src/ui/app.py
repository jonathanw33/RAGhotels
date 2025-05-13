#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced UI for the Hotel Recommendation RAG System with enhanced features:
- Advanced Filtering Panel: Filter hotels by rating, features, and sentiment
- Hotel Comparison: Side-by-side comparison of selected hotels
- Enhanced Visualizations: Interactive maps with better visual indicators
- Dynamic Feature Filtering: Filter by specific amenities and hotel features
"""

import os
import logging
import gradio as gr
import folium
from folium.plugins import MarkerCluster, HeatMap
import tempfile
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedHotelRecommendationUI:
    """Advanced UI for the Hotel Recommendation System with enhanced features."""
    
    def __init__(self, rag_engine):
        """Initialize the UI with the RAG engine."""
        self.rag_engine = rag_engine
        self.theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
        )
        
        # Define common hotel features for filtering
        self.common_features = [
            "luxury", "budget", "family_friendly", "pool", "spa", 
            "breakfast", "beach", "business", "city_center", "wifi", 
            "parking", "pet_friendly", "airport_shuttle"
        ]
        
        self.css = """
        .hotel-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f9f9f9;
            transition: all 0.3s ease;
        }
        .hotel-card:hover {
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .hotel-name {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
            color: #1565c0;
        }
        .hotel-address {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .hotel-rating {
            font-size: 16px;
            font-weight: bold;
            color: #ff9800;
            margin-bottom: 5px;
        }
        .hotel-features {
            margin-top: 10px;
        }
        .feature-tag {
            display: inline-block;
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .visualization-container {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: white;
        }
        .expanded-queries {
            margin-top: 10px;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #f5f5f5;
            font-size: 14px;
        }
        .query-expansion-toggle {
            margin-top: 10px;
        }
        .filter-panel {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        }
        .filter-section {
            margin-bottom: 15px;
        }
        .filter-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: #1565c0;
        }
        .filter-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 10px;
        }
        .hotel-comparison {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        .comparison-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #1565c0;
        }
        /* Star rating styling */
        .star-rating {
            color: #ffab00;
            font-size: 18px;
        }
        .star-empty {
            color: #e0e0e0;
        }
        /* Toggle button styling */
        .toggle-button {
            background-color: #bbdefb;
            border-radius: 15px;
            padding: 5px 15px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .toggle-button:hover {
            background-color: #90caf9;
        }
        .toggle-button.active {
            background-color: #1976d2;
            color: white;
        }
        """
    
    def create_map(self, hotels):
        """Create a folium map with the recommended hotels."""
        # Filter hotels with valid coordinates
        valid_hotels = [h for h in hotels if h['latitude'] is not None and h['longitude'] is not None]
        
        if not valid_hotels:
            return None
        
        # Calculate the center of the map
        center_lat = sum(h['latitude'] for h in valid_hotels) / len(valid_hotels)
        center_lon = sum(h['longitude'] for h in valid_hotels) / len(valid_hotels)
        
        # Create the map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add a marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each hotel
        for i, hotel in enumerate(valid_hotels):
            # Create a popup with hotel information
            feature_str = ""
            for feature_name, has_feature in hotel.get('features', {}).items():
                if has_feature and feature_name.startswith('has_'):
                    # Clean up the feature name (remove 'has_' prefix)
                    clean_name = feature_name[4:].replace('_', ' ').title()
                    feature_str += f"<span class='feature-tag'>{clean_name}</span> "
            
            # Create color based on rating
            color = 'red'
            if hotel['score'] >= 4.5:
                color = 'green'
            elif hotel['score'] >= 3.5:
                color = 'blue'
            elif hotel['score'] >= 2.5:
                color = 'orange'
            
            popup_text = f"""
                <div class='hotel-card'>
                    <div class='hotel-name'>{hotel['name']}</div>
                    <div class='hotel-address'>{hotel['address']}</div>
                    <div class='hotel-rating'>Rating: {hotel['score']:.1f}/5</div>
                    <div class='hotel-features'>{feature_str}</div>
                </div>
            """
            
            # Add the marker
            folium.Marker(
                location=[hotel['latitude'], hotel['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{i+1}. {hotel['name']}",
                icon=folium.Icon(color=color, icon='hotel', prefix='fa')
            ).add_to(marker_cluster)
        
        # Add a heatmap layer for hotel density
        heat_data = [[hotel['latitude'], hotel['longitude'], min(1.0, hotel['score']/5.0)] for hotel in valid_hotels]
        HeatMap(heat_data, radius=15).add_to(m)
        
        # Save the map to a temporary file
        _, temp_path = tempfile.mkstemp(suffix='.html')
        m.save(temp_path)
        
        return temp_path
    
    def create_feature_chart(self, hotels):
        """Create a chart of hotel features."""
        if not hotels:
            return None
        
        # Extract feature counts
        feature_counts = {}
        for hotel in hotels:
            for feature_name, has_feature in hotel.get('features', {}).items():
                if has_feature and feature_name.startswith('has_'):
                    # Clean up the feature name (remove 'has_' prefix)
                    clean_name = feature_name[4:].replace('_', ' ').title()
                    feature_counts[clean_name] = feature_counts.get(clean_name, 0) + 1
        
        # Sort features by count
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create the chart
        if sorted_features:
            labels = [f[0] for f in sorted_features]
            values = [f[1] for f in sorted_features]
            
            plt.figure(figsize=(8, 4))
            plt.barh(labels, values, color='skyblue')
            plt.xlabel('Number of Hotels')
            plt.title('Top Features in Recommended Hotels')
            plt.tight_layout()
            
            # Save the chart to a temporary file
            _, temp_path = tempfile.mkstemp(suffix='.png')
            plt.savefig(temp_path)
            plt.close()
            
            return temp_path
        
        return None
    
    def create_sentiment_chart(self, hotels):
        """Create a chart of hotel sentiment scores."""
        if not hotels:
            return None
        
        # Extract sentiment scores
        sentiment_data = []
        hotel_names = []
        
        for hotel in hotels[:5]:  # Limit to top 5
            sentiment = hotel.get('sentiment', {})
            sentiment_data.append([
                sentiment.get('positive', 0),
                sentiment.get('neutral', 0),
                sentiment.get('negative', 0)
            ])
            hotel_names.append(hotel['name'][:20] + "..." if len(hotel['name']) > 20 else hotel['name'])
        
        if sentiment_data:
            sentiment_data = np.array(sentiment_data)
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Create the stacked bar chart
            width = 0.6
            bottom = np.zeros(len(hotel_names))
            
            for i, sentiment_type in enumerate(['Positive', 'Neutral', 'Negative']):
                p = ax.bar(hotel_names, sentiment_data[:, i], width, label=sentiment_type, bottom=bottom)
                bottom += sentiment_data[:, i]
            
            ax.set_title('Sentiment Analysis of Hotel Reviews')
            ax.legend(loc='upper right')
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the chart to a temporary file
            _, temp_path = tempfile.mkstemp(suffix='.png')
            plt.savefig(temp_path)
            plt.close()
            
            return temp_path
        
        return None
    
    def create_expanded_queries_html(self, expanded_queries):
        """Create HTML for expanded queries."""
        if not expanded_queries or len(expanded_queries) <= 1:
            return ""
        
        html = "<div class='expanded-queries'><strong>Query Expansions:</strong><br>"
        for i, query in enumerate(expanded_queries[:5]):  # Limit to first 5
            html += f"{i+1}. {query}<br>"
        html += "</div>"
        
        return html
    
    def create_hotel_selection_html(self, hotels):
        """Create HTML for hotel selection for comparison."""
        if not hotels:
            return "<div>No hotels available for comparison</div>"
        
        # Create a JavaScript handler to manage selections
        js = """
        <script>
        function toggleHotelSelection(checkbox) {
            // Get all checkboxes
            var checkboxes = document.querySelectorAll('input[type="checkbox"][name^="hotel_"]');
            
            // Count selected checkboxes
            var selectedCount = 0;
            var selectedIndices = [];
            
            checkboxes.forEach(function(cb) {
                if (cb.checked) {
                    selectedCount++;
                    selectedIndices.push(cb.value);
                }
            });
            
            // Limit to 5 selections
            if (selectedCount > 5 && checkbox.checked) {
                checkbox.checked = false;
                alert("You can select a maximum of 5 hotels for comparison");
            }
            
            // Update hidden input with selected indices
            document.getElementById('selected_hotels').value = selectedIndices.join(',');
        }
        </script>
        """
        
        # Create the HTML for hotel selection
        html = js + "<div><h3>Select Hotels to Compare (max 5)</h3>"
        
        # Hidden input to store selected indices
        html += '<input type="hidden" id="selected_hotels" name="selected_hotels" value="">'
        
        for i, hotel in enumerate(hotels):
            name = hotel['name']
            if len(name) > 40:
                name = name[:37] + "..."
                
            # Create star rating string
            stars = "★" * int(hotel['score']) + "☆" * (5 - int(hotel['score']))
            
            # Extract key features
            features = []
            for feature, has_feature in hotel.get('features', {}).items():
                if has_feature and feature.startswith('has_'):
                    # Clean up the feature name (remove 'has_' prefix)
                    clean_name = feature[4:].replace('_', ' ').title()
                    features.append(clean_name)
            
            # Limit to top 3 features for display
            feature_str = ", ".join(features[:3])
            if len(features) > 3:
                feature_str += "..."
            
            html += f"""
            <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #f9f9f9;">
                <div style="display: flex; align-items: center;">
                    <input type="checkbox" id="hotel_{i}" name="hotel_{i}" value="{i}" 
                           onchange="toggleHotelSelection(this)" style="margin-right: 10px; transform: scale(1.2);">
                    <div style="flex-grow: 1;">
                        <div class="hotel-name">{name}</div>
                        <div style="color: #ff9800;">{stars} ({hotel['score']:.1f}/5)</div>
                        <div style="color: #666; margin-top: 5px; font-size: 14px;">{hotel['address']}</div>
                        <div style="margin-top: 5px; font-size: 14px;">
                            <strong>Features:</strong> {feature_str}
                        </div>
                    </div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def create_comparison_html(self, hotels):
        """Create HTML table for hotel comparison."""
        if not hotels or len(hotels) < 2:
            return "<p>Select at least two hotels to compare</p>"
        
        # Create comparison table
        html = """
        <div style="margin-top: 20px;">
            <h3 style="margin-bottom: 15px; color: #1565c0;">Hotel Comparison</h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; border: 1px solid #ddd;">
        """
        
        # Table header with hotel names
        html += "<tr style='background-color: #e3f2fd;'><th style='border: 1px solid #ddd; padding: 12px; text-align: left;'>Feature</th>"
        for hotel in hotels:
            name = hotel['name']
            if len(name) > 30:
                name = name[:27] + "..."
            html += f"<th style='border: 1px solid #ddd; padding: 12px; text-align: left;'>{name}</th>"
        html += "</tr>"
        
        # Rating row
        html += "<tr><td style='border: 1px solid #ddd; padding: 12px; font-weight: bold; background-color: #f9f9f9;'>Rating</td>"
        for hotel in hotels:
            rating = hotel['score']
            stars = "★" * int(rating) + "☆" * (5 - int(rating))
            html += f"<td style='border: 1px solid #ddd; padding: 12px;'><span style='color: #ff9800;'>{stars}</span> {rating:.1f}/5</td>"
        html += "</tr>"
        
        # Address row
        html += "<tr><td style='border: 1px solid #ddd; padding: 12px; font-weight: bold; background-color: #f9f9f9;'>Location</td>"
        for hotel in hotels:
            address = hotel['address']
            if len(address) > 50:
                address = address[:47] + "..."
            html += f"<td style='border: 1px solid #ddd; padding: 12px;'>{address}</td>"
        html += "</tr>"
        
        # Sentiment rows
        sentiment_types = [
            ("Positive Sentiment", "positive"),
            ("Neutral Sentiment", "neutral"),
            ("Negative Sentiment", "negative"),
            ("Overall Sentiment", "compound")
        ]
        
        for label, key in sentiment_types:
            html += f"<tr><td style='border: 1px solid #ddd; padding: 12px; font-weight: bold; background-color: #f9f9f9;'>{label}</td>"
            for hotel in hotels:
                value = hotel['sentiment'][key]
                color = ""
                if key == "compound":
                    color = "#4caf50" if value > 0.2 else "#ff9800" if value > -0.2 else "#f44336"
                elif key == "positive":
                    color = "#4caf50"
                elif key == "negative":
                    color = "#f44336"
                
                html += f"<td style='border: 1px solid #ddd; padding: 12px; color: {color}; font-weight: bold;'>{value:.2f}</td>"
            html += "</tr>"
        
        # Feature rows
        all_features = set()
        for hotel in hotels:
            for feature, has_feature in hotel.get('features', {}).items():
                if has_feature and feature.startswith('has_'):
                    all_features.add(feature)
        
        # Sort features
        sorted_features = sorted(all_features)
        
        for feature in sorted_features:
            # Clean up the feature name
            feature_name = feature[4:].replace('_', ' ').title()
            
            html += f"<tr><td style='border: 1px solid #ddd; padding: 12px; font-weight: bold; background-color: #f9f9f9;'>{feature_name}</td>"
            for hotel in hotels:
                has_feature = hotel.get('features', {}).get(feature, False)
                icon = "✓" if has_feature else "✗"
                color = "#4caf50" if has_feature else "#f44336"
                html += f"<td style='border: 1px solid #ddd; padding: 12px; color: {color}; text-align: center; font-weight: bold;'>{icon}</td>"
            html += "</tr>"
        
        # Estimated price row
        html += "<tr><td style='border: 1px solid #ddd; padding: 12px; font-weight: bold; background-color: #f9f9f9;'>Est. Price Range</td>"
        for hotel in hotels:
            # Base price on rating and luxury features
            base_price = hotel['score'] * 30
            
            # Adjust for luxury vs budget
            if hotel['features'].get('has_luxury', False):
                base_price *= 1.5
            if hotel['features'].get('has_budget', False):
                base_price *= 0.7
            
            # Adjust for additional features
            feature_price_factors = {
                'has_spa': 1.2,
                'has_pool': 1.1,
                'has_beach': 1.15,
                'has_city_center': 1.1,
                'has_breakfast': 1.05
            }
            
            for feature, factor in feature_price_factors.items():
                if hotel['features'].get(feature, False):
                    base_price *= factor
            
            # Create a price range
            min_price = max(40, int(base_price * 0.8))
            max_price = int(base_price * 1.2)
            
            html += f"<td style='border: 1px solid #ddd; padding: 12px;'>${min_price} - ${max_price}</td>"
        html += "</tr>"
        
        html += """
                </table>
            </div>
        </div>
        """
        return html
    
    def process_query(self, query, use_query_expansion, filters, history):
        """Process the user query and return the response with optional filters."""
        logger.info(f"Processing query: {query}")
        logger.info(f"Applied filters: {filters}")
        
        try:
            start_time = time.time()
            
            # Process filters
            applied_filters = {}
            
            # Minimum rating filter
            if filters.get("min_rating"):
                applied_filters["min_rating"] = filters["min_rating"]
            
            # Required features
            required_features = []
            for feature in self.common_features:
                if filters.get(f"feature_{feature}", False):
                    required_features.append(feature)
            
            if required_features:
                applied_filters["required_features"] = required_features
            
            # Sentiment filter
            if filters.get("min_sentiment"):
                applied_filters["min_sentiment"] = (filters["min_sentiment"] * 2) - 1  # Convert 0-1 to -1 to 1 scale
            
            # Sorting
            if filters.get("sort_by"):
                applied_filters["sort_by"] = filters["sort_by"]
            
            # Get recommendations from the RAG engine with filters
            result = self.rag_engine.query(
                query, 
                use_query_expansion=use_query_expansion,
                filters=applied_filters if applied_filters else None
            )
            
            response_text = result['response']
            hotels = result.get('hotels', [])
            all_hotels = result.get('all_hotels', [])
            expanded_queries = result.get('expanded_queries', [])
            filter_stats = result.get('filter_stats', None)
            
            # Create a map if hotels have coordinates
            map_path = self.create_map(all_hotels)
            
            # Create feature and sentiment charts
            feature_chart_path = self.create_feature_chart(all_hotels)
            sentiment_chart_path = self.create_sentiment_chart(hotels)
            
            # Create expanded queries HTML
            expanded_queries_html = self.create_expanded_queries_html(expanded_queries)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Add processing time and filter info to response
            processing_info = f"\n\n*Query processed in {processing_time:.2f} seconds"
            if use_query_expansion:
                processing_info += f" with query expansion ({len(expanded_queries)} variations)"
            
            if filter_stats:
                processing_info += f"\nFilters applied: {len(applied_filters)} filters"
                if "min_rating" in applied_filters:
                    processing_info += f", minimum rating: {applied_filters['min_rating']}/5"
                if "required_features" in applied_filters:
                    if len(applied_filters["required_features"]) > 0:
                        feature_list = [f.replace('_', ' ').title() for f in applied_filters["required_features"]]
                        processing_info += f", required features: {', '.join(feature_list)}"
                
                if filter_stats.get('total_before_filter') != filter_stats.get('total_after_filter'):
                    processing_info += f"\nResults: {filter_stats.get('total_after_filter')} hotels (filtered from {filter_stats.get('total_before_filter')})"
            
            processing_info += "*"
            
            response_text += processing_info
            
            if expanded_queries_html:
                response_text += "\n\n" + expanded_queries_html
            
            # Return the response text, map, and charts
            return response_text, map_path, feature_chart_path, sentiment_chart_path, all_hotels
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error: {str(e)}", None, None, None, []
    
    def build_interface(self):
        """Build and return the Gradio interface."""
        logger.info("Building enhanced Gradio interface with Advanced Filtering")
        
        # Create the chat interface
        with gr.Blocks(theme=self.theme, css=self.css) as interface:
            gr.Markdown("# 🏨 Hotel Recommendation System")
            gr.Markdown("""
                Enter your hotel preferences in natural language, and I'll recommend hotels based on reviews and features.
                
                The system uses advanced search features:
                - **Hybrid Search**: Combines semantic and keyword search for better results
                - **Query Expansion**: Automatically expands your query with synonyms and related terms
                - **Dynamic Filtering**: Filters hotels based on features mentioned in your query
                
                Example queries:
                - "I need a hotel in Paris with free breakfast, under $200 per night, and good for families."
                - "Find me a luxury hotel in New York with excellent reviews about its spa facilities."
                - "What are the best boutique hotels in Barcelona with rooftop views?"
            """)
            
            # State for storing all hotels
            all_hotels_state = gr.State([])
            
            # Advanced filtering panel
            with gr.Accordion("Advanced Filtering", open=False):
                with gr.Box(elem_classes=["filter-panel"]):
                    gr.Markdown("## 🔍 Filter Your Results", elem_classes=["filter-title"])
                    
                    with gr.Row():
                        # Rating filter
                        with gr.Column(scale=1):
                            gr.Markdown("### Rating", elem_classes=["filter-section"])
                            min_rating = gr.Slider(
                                minimum=0, 
                                maximum=5, 
                                value=0, 
                                step=0.5, 
                                label="Minimum Rating"
                            )
                        
                        # Sentiment filter
                        with gr.Column(scale=1):
                            gr.Markdown("### Sentiment", elem_classes=["filter-section"])
                            min_sentiment = gr.Slider(
                                minimum=0,
                                maximum=1, 
                                value=0,
                                step=0.1,
                                label="Minimum Sentiment Score"
                            )
                    
                    # Feature filters
                    gr.Markdown("### Required Features", elem_classes=["filter-section"])
                    
                    feature_checkboxes = {}
                    with gr.Row(elem_classes=["filter-row"]):
                        for i, feature in enumerate(self.common_features):
                            display_name = feature.replace('_', ' ').title()
                            feature_checkboxes[f"feature_{feature}"] = gr.Checkbox(
                                label=display_name, 
                                value=False,
                            )
                            
                            # Start a new row every 4 features
                            if (i + 1) % 4 == 0 and i < len(self.common_features) - 1:
                                gr.Row(elem_classes=["filter-row"])
                    
                    # Sorting options
                    gr.Markdown("### Sort Results", elem_classes=["filter-section"])
                    sort_by = gr.Radio(
                        choices=[
                            "rating_high_to_low", 
                            "rating_low_to_high", 
                            "sentiment_high_to_low"
                        ],
                        value="rating_high_to_low",
                        label="Sort By",
                        info="Choose how to sort the hotel results"
                    )
                    
                    # Reset filters button
                    reset_filters = gr.Button("Reset Filters")
            
            with gr.Row():
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        layout="bubble",
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="What kind of hotel are you looking for?",
                            show_label=False,
                            scale=9,
                        )
                        submit = gr.Button("Submit", scale=1)
                    
                    with gr.Row():
                        use_query_expansion = gr.Checkbox(
                            label="Enable Query Expansion",
                            value=True,
                            info="Automatically expand your query with synonyms and related terms",
                        )
                
                with gr.Column(scale=4):
                    with gr.Tab("Map"):
                        map_output = gr.HTML(label="Hotel Locations")
                    
                    with gr.Tab("Features"):
                        feature_chart = gr.Image(label="Hotel Features")
                    
                    with gr.Tab("Sentiment"):
                        sentiment_chart = gr.Image(label="Review Sentiment")
                    
                    with gr.Tab("Comparison"):
                        hotel_comparison = gr.HTML(label="Select hotels to compare")
                        compare_button = gr.Button("Compare Selected Hotels")
                        comparison_result = gr.HTML(label="Comparison Results")
            
            # Set up interactions
            def user(message, history):
                """Handle user messages."""
                return "", history + [[message, None]]
            
            def get_filter_values(min_rating, min_sentiment, sort_by, **kwargs):
                """Collect all filter values."""
                filters = {
                    "min_rating": min_rating if min_rating > 0 else None,
                    "min_sentiment": min_sentiment if min_sentiment > 0 else None,
                    "sort_by": sort_by
                }
                
                # Add feature filters
                for key, value in kwargs.items():
                    if key.startswith("feature_") and value:
                        filters[key] = value
                
                return filters
            
            def bot(history, use_expansion, min_rating, min_sentiment, sort_by, **kwargs):
                """Handle bot responses with filters."""
                message = history[-1][0]
                
                # Collect all filters
                filters = get_filter_values(min_rating, min_sentiment, sort_by, **kwargs)
                
                response, map_path, feature_path, sentiment_path, all_hotels = self.process_query(
                    message, use_expansion, filters, history
                )
                
                if map_path:
                    # Read the HTML content
                    with open(map_path, 'r', encoding='utf-8') as f:
                        map_html = f.read()
                    
                    # Clean up the file
                    os.remove(map_path)
                else:
                    map_html = "<div>No location data available for the recommended hotels.</div>"
                
                # Update the charts
                if feature_path:
                    feature_img = feature_path
                else:
                    feature_img = None
                
                if sentiment_path:
                    sentiment_img = sentiment_path
                else:
                    sentiment_img = None
                
                history[-1][1] = response
                
                # Create hotel selection HTML for comparison
                comparison_html = self.create_hotel_selection_html(all_hotels[:10])
                
                return history, map_html, feature_img, sentiment_img, comparison_html, all_hotels
            
            def reset_filter_values():
                """Reset all filter values."""
                result = {"min_rating": 0, "min_sentiment": 0, "sort_by": "rating_high_to_low"}
                for feature in self.common_features:
                    result[f"feature_{feature}"] = False
                return result
            
            def compare_hotels(selected_indices, all_hotels):
                """Compare selected hotels."""
                if not selected_indices or not all_hotels:
                    return "Please select hotels to compare"
                
                # Parse selected indices (they come as a string like "0,2,3")
                try:
                    if isinstance(selected_indices, str):
                        indices = [int(idx) for idx in selected_indices.split(",") if idx.strip()]
                    else:
                        indices = []
                    
                    # Get selected hotels
                    selected_hotels = [all_hotels[idx] for idx in indices if idx < len(all_hotels)]
                    
                    if not selected_hotels or len(selected_hotels) < 2:
                        return "Please select at least two hotels to compare"
                    
                    # Create comparison table
                    return self.create_comparison_html(selected_hotels)
                except Exception as e:
                    logger.error(f"Error comparing hotels: {str(e)}")
                    return f"Error comparing hotels: {str(e)}"
            
            # Set up event handlers for filters
            filter_inputs = [min_rating, min_sentiment, sort_by]
            filter_inputs.extend(list(feature_checkboxes.values()))
            
            # Set up event handlers
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, 
                [chatbot, use_query_expansion, min_rating, min_sentiment, sort_by] + list(feature_checkboxes.values()),
                [chatbot, map_output, feature_chart, sentiment_chart, hotel_comparison, all_hotels_state]
            )
            
            submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, 
                [chatbot, use_query_expansion, min_rating, min_sentiment, sort_by] + list(feature_checkboxes.values()),
                [chatbot, map_output, feature_chart, sentiment_chart, hotel_comparison, all_hotels_state]
            )
            
            # Handle hotel comparison
            compare_button.click(
                compare_hotels, 
                [hotel_comparison, all_hotels_state], 
                [comparison_result]
            )
            
            # Handle filter reset
            reset_filters.click(
                reset_filter_values,
                [],
                [min_rating, min_sentiment, sort_by] + list(feature_checkboxes.values())
            )
            
            # Add an example queries section
            with gr.Accordion("Example Queries", open=False):
                examples = gr.Examples(
                    examples=[
                        "I need a luxury hotel in Paris with a good spa and breakfast included.",
                        "Find me a budget-friendly hotel in London near the city center with good reviews.",
                        "What are the best family-friendly hotels in New York with pool facilities?",
                        "I want a beachfront hotel in Miami with high ratings for cleanliness.",
                        "Recommend me a boutique hotel in Rome with good location and breakfast.",
                        "Looking for a hotel with excellent service in Tokyo for a business trip.",
                        "I need a quiet hotel in Barcelona with a rooftop bar and city views.",
                        "Where can I find a romantic hotel in Venice with canal views?",
                        "Suggest a hotel in Las Vegas with a casino and multiple restaurants.",
                        "I want a ski resort in Whistler with a fireplace in the room."
                    ],
                    inputs=msg
                )
        
        return interface

def create_app(rag_engine):
    """Create and return the Gradio app."""
    ui = AdvancedHotelRecommendationUI(rag_engine)
    interface = ui.build_interface()
    return interface

def main():
    """Main function to run the UI."""
    logger.info("Starting the advanced UI")
    
    try:
        # Import here to avoid circular imports
        from src.rag.advanced_engine import AdvancedRAGEngine
        
        # Initialize the RAG engine
        rag_engine = AdvancedRAGEngine()
        
        # Create the UI
        app = create_app(rag_engine)
        app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    
    except Exception as e:
        logger.error(f"Error starting the UI: {str(e)}")
        print(f"Error: {str(e)}")
        print("\nPossible causes:")
        print("1. Ollama is not running (https://ollama.ai/)")
        print("2. Required models are not downloaded (run 'ollama pull llama3')")
        print("3. Memory constraints (try using a smaller model variant)")
        return

if __name__ == "__main__":
    main()
