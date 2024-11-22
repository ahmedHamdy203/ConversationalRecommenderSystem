import requests
import json
from datetime import datetime
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleMovieRecommenderTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_cases = [
            {
                "name": "Musical Lover",
                "input": "I really enjoyed White Christmas because of the musical numbers and holiday spirit. I love classic musicals with great performances.",
                "expected_genre": "musical"
            },
            {
                "name": "Action Fan",
                "input": "The Bourne Identity was amazing! I love action movies with good plots and intense chase scenes.",
                "expected_genre": "action"
            },
            {
                "name": "Quality Focus",
                "input": "I'm looking for well-made films with strong performances. I didn't like The Screaming Skull because of its poor production quality.",
                "expected_genre": "drama"
            },
            {
                "name": "General Request",
                "input": "Can you recommend a good movie? I enjoy well-crafted stories with good character development.",
                "expected_genre": "any"
            }
        ]
    
    def make_request(self, user_input: str) -> Dict[str, Any]:
        """Make a single recommendation request"""
        try:
            url = f"{self.base_url}/recommend/"
            payload = {"user_input": user_input}
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
    
    def save_results(self, results: list):
        """Save test results to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recommendation_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": timestamp,
                    "results": results
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def run_tests(self):
        """Run all test cases"""
        logger.info("Starting recommendation tests")
        results = []
        
        for test_case in self.test_cases:
            logger.info(f"\nTesting: {test_case['name']}")
            logger.info(f"Input: {test_case['input']}")
            
            # Make request
            response = self.make_request(test_case['input'])
            
            # Log and store result
            if "error" in response:
                logger.error(f"Test failed: {response['error']}")
                test_result = {
                    "test_case": test_case,
                    "status": "failed",
                    "error": response['error']
                }
            else:
                logger.info(f"Recommendation: {response['recommendation']}")
                test_result = {
                    "test_case": test_case,
                    "status": "success",
                    "response": response
                }
            
            results.append(test_result)
            logger.info("-" * 80)
        
        # Save all results
        self.save_results(results)
        logger.info("\nAll tests completed!")

def main():
    """Main test execution"""
    try:
        # Create and run tester
        tester = SimpleMovieRecommenderTester()
        tester.run_tests()
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")

if __name__ == "__main__":
    main()