import requests
import json
import time
from typing import Dict, Any
from datetime import datetime

def test_movie_recommendation_api():
    """Test various endpoints of the movie recommendation API"""
    
    # API configuration
    BASE_URL = "http://localhost:8000"
    
    def make_request(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict:
        """Make HTTP request to API"""
        url = f"{BASE_URL}/{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {endpoint}: {str(e)}")
            return None
    
    # Test queries
    test_queries = [
        {
            "query": "I want a movie like White Christmas",
            "user_context": {
                "preferred_genres": ["musical", "family"]
            }
        },
        {
            "query": "I enjoyed that one. Do you have any other similar recommendations?",
            "user_context": {
                "preferred_genres": ["musical", "family"]
            }
        },
        {
            "query": "I prefer something more action-packed",
            "user_context": {
                "preferred_genres": ["action", "thriller"]
            }
        }
    ]
    
    print("\nTesting Movie Recommendation API")
    print("=" * 50)
    
    # Test recommendation endpoint
    print("\nTesting recommendations:")
    for i, query_data in enumerate(test_queries, 1):
        print(f"\nTest Query {i}:")
        print(f"Query: {query_data['query']}")
        
        result = make_request("recommend/", "POST", query_data)
        if result:
            print(f"Response: {result['response']}")
            print(f"Timestamp: {result['timestamp']}")
        
        # Short delay between requests
        time.sleep(1)
    
    # Test history endpoint
    print("\nTesting conversation history:")
    history = make_request("history/")
    if history:
        print("\nConversation History:")
        for message in history['history']:
            print(f"{message['role'].title()}: {message['content']}")
    
    # Test history clearing
    print("\nTesting history clearing:")
    result = make_request("history/", "DELETE")
    if result:
        print(f"Result: {result['message']}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_movie_recommendation_api()