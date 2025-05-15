        except Exception as e:
            logger.warning(f"Error initializing BM25Retriever: {str(e)}, falling back to vector retriever only")
            self.hybrid_retriever = None
            self.query_engine = self.index.as_query_engine(
                text_qa_template=PromptTemplate(QUERY_PROMPT_TEMPLATE),
                similarity_top_k=15,
            )
            return
        
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
        debug_log("RAG components set up successfully", logging.INFO)
    
    def expand_query(self, query: str) -> List[str]:
        """Expand the query with synonyms and related terms."""
        return self.query_expander.expand_query(query)
    
    def query(self, user_query: str, use_query_expansion: bool = True, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run an enhanced query against the RAG engine with optional explicit filters."""
        logger.info(f"Running query: {user_query}")
        
        # Initialize query debugger
        debugger = QueryDebugger(user_query)
        
        # Define supported locations
        debugger.start_step("location_detection")
        supported_locations = [
            "new york", "manhattan", "brooklyn", "queens", "bronx", "staten island",
            "los angeles", "houston", "chicago", "miami", "las vegas", "san francisco",
            "boston", "seattle", "washington", "washington dc", "atlanta", "dallas",
            "philadelphia", "san diego", "austin", "denver", "phoenix", "orlando",
            "new orleans", "portland", "nashville", "san antonio", "baltimore"
        ]
        
        # Extract locations from the query
        detected_locations = self._extract_locations(user_query.lower())
        debug_log(f"Detected locations: {detected_locations}")
        
        # Check if any detected locations are not supported
        unsupported_locations = []
        for loc in detected_locations:
            if not any(loc in supp_loc or supp_loc in loc for supp_loc in supported_locations):
                unsupported_locations.append(loc)
        
        debugger.end_step()
        
        # If unsupported locations are detected, return an informative message
        if unsupported_locations:
            logger.info(f"Unsupported locations detected: {unsupported_locations}")
            debugger.start_step("generate_location_error")
            response_message = f"I'm sorry, but I don't have information about hotels in {', '.join(unsupported_locations)}. "
            response_message += "Currently, I only have data for hotels in select U.S. cities including New York, Los Angeles, "
            response_message += "Houston, Chicago, Miami, Las Vegas, and other major US cities. "
            response_message += "Would you like recommendations for any of these locations instead?"
            debugger.end_step()
            debugger.finalize()
            
            return {
                'response': response_message,
                'hotels': [],
                'all_hotels': [],
                'expanded_queries': [user_query],
                'filter_stats': None
            }
        
        # If the hybrid retriever is None (fallback mode), use the regular query engine
        if self.hybrid_retriever is None:
            logger.info("Using vector retriever only (hybrid retrieval not available)")
            
            # Track time for vector retrieval
            debugger.start_step("vector_retrieval")
            response = self.query_engine.query(user_query)
            source_nodes = response.source_nodes
            debugger.end_step()
            
            # Log information about retrieved nodes
            debugger.log_retrieved_nodes(source_nodes)
            
            # Skip the query expansion since we're in fallback mode
            expanded_queries = [user_query]
        elif use_query_expansion:
            # Expand the query
            debugger.start_step("query_expansion")
            expanded_queries = self.expand_query(user_query)
            debug_log(f"Expanded queries: {expanded_queries}")
            debugger.end_step()
            
            # Use the original query as the main query
            debugger.start_step("initial_retrieval")
            response = self.query_engine.query(user_query)
            source_nodes = response.source_nodes
            seen_hotels = set()
            debugger.end_step()
            
            # Log information about retrieved nodes
            debugger.log_retrieved_nodes(source_nodes)
            
            # Process additional expanded queries if we don't have enough results
            if len(source_nodes) < 10:
                debug_log(f"Not enough results ({len(source_nodes)}), trying expanded queries...")
                
                debugger.start_step("expanded_retrieval")
                for exp_query in expanded_queries[:3]:  # Limit to first 3 expansions
                    if exp_query == user_query:
                        continue
                    
                    # Run the expanded query
                    debug_log(f"Trying expanded query: {exp_query}")
                    exp_response = self.query_engine.query(exp_query)
                    exp_nodes = exp_response.source_nodes
                    
                    # Add unique hotels to results
                    added = 0
                    for node in exp_nodes:
                        if hasattr(node, 'metadata') and node.metadata:
                            hotel_name = node.metadata.get('Hotel_Name', '')
                            if hotel_name and hotel_name not in seen_hotels:
                                source_nodes.append(node)
                                seen_hotels.add(hotel_name)
                                added += 1
                                
                                # Stop if we have enough results
                                if len(source_nodes) >= 15:
                                    break
                    
                    debug_log(f"Added {added} unique hotels from expanded query")
                    
                    # Stop if we have enough results
                    if len(source_nodes) >= 15:
                        debug_log("Reached target number of results, stopping expansion")
                        break
                
                debugger.end_step()
        else:
            # Use only the original query
            debugger.start_step("direct_retrieval")
            response = self.query_engine.query(user_query)
            source_nodes = response.source_nodes
            debugger.end_step()
            
            # Log information about retrieved nodes
            debugger.log_retrieved_nodes(source_nodes)
            
            expanded_queries = [user_query]
        
        # Extract hotel information from the source nodes
        debugger.start_step("hotel_extraction")
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
        debugger.end_step()
        
        debug_log(f"Extracted information for {len(hotels_info)} hotels")
        
        # If no hotels were found, return an appropriate message
        if not hotels_info:
            logger.info("No hotels found matching the query criteria")
            debugger.start_step("generate_empty_response")
            response_message = "I couldn't find any hotels matching your specific criteria. "
            response_message += "Please try broadening your search or specifying different amenities or locations. "
            response_message += "Currently, the system has data primarily for major US cities."
            debugger.end_step()
            debugger.finalize()
            
            return {
                'response': response_message,
                'hotels': [],
                'all_hotels': [],
                'expanded_queries': expanded_queries if use_query_expansion else [user_query],
                'filter_stats': None
            }
        
        # Apply explicit filters if provided
        if filters:
            debugger.start_step("filtering")
            debug_log(f"Applying filters: {filters}")
            
            filtered_hotels = []
            for hotel in hotels_info:
                # Check minimum rating filter
                if 'min_rating' in filters and filters['min_rating'] and hotel['score'] < filters['min_rating']:
                    continue
                
                # Check maximum rating filter (useful for budget options)
                if 'max_rating' in filters and filters['max_rating'] and hotel['score'] > filters['max_rating']:
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
                if 'min_sentiment' in filters and filters['min_sentiment'] and hotel['sentiment']['compound'] < filters['min_sentiment']:
                    continue
                
                # Location filtering will be handled by the map visualization
                
                # Hotel passed all filters
                filtered_hotels.append(hotel)
            
            debug_log(f"Filtered from {len(hotels_info)} to {len(filtered_hotels)} hotels")
            
            # Replace hotels_info with filtered results
            hotels_info = filtered_hotels
            debugger.end_step()
        
        # Remove duplicates by name
        debugger.start_step("deduplication")
        unique_hotels = []
        hotel_names = set()
        for hotel in hotels_info:
            if hotel['name'] not in hotel_names:
                unique_hotels.append(hotel)
                hotel_names.add(hotel['name'])
        
        debug_log(f"Reduced from {len(hotels_info)} to {len(unique_hotels)} unique hotels")
        debugger.end_step()
        
        # Sort hotels by rating if requested
        if filters and 'sort_by' in filters:
            debugger.start_step("sorting")
            if filters['sort_by'] == 'rating_high_to_low':
                unique_hotels.sort(key=lambda x: x['score'], reverse=True)
                debug_log("Sorted by rating (high to low)")
            elif filters['sort_by'] == 'rating_low_to_high':
                unique_hotels.sort(key=lambda x: x['score'])
                debug_log("Sorted by rating (low to high)")
            elif filters['sort_by'] == 'sentiment_high_to_low':
                unique_hotels.sort(key=lambda x: x['sentiment']['compound'], reverse=True)
                debug_log("Sorted by sentiment (high to low)")
            debugger.end_step()
        
        # Select top hotels (use a larger number if we have filters)
        top_hotels = unique_hotels[:min(10, len(unique_hotels))]
        
        # Log detailed information about top hotels
        debugger.log_hotels(top_hotels)
        
        # Finalize debugging
        timing_info = debugger.finalize()
        
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
    debug_log("Testing Enhanced RAG engine", logging.INFO)
    
    try:
        # Download NLTK resources
        download_nltk_resources()
        
        # Initialize the RAG engine
        rag_engine = AdvancedRAGEngine()
        
        # Test query with expanded search
        test_query = "I'm looking for a 5-star hotel in New York with a spa and good breakfast"
        debug_log(f"Running test query: '{test_query}'", logging.INFO)
        
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
        debug_log("Enhanced RAG engine test completed successfully", logging.INFO)
    
    except Exception as e:
        logger.error(f"Error testing enhanced RAG engine: {str(e)}")
        debug_log(f"Error testing enhanced RAG engine: {str(e)}", logging.ERROR)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
