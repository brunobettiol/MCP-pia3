import requests
import json

def debug_service_scoring():
    """Debug service scoring for home modification queries"""
    
    # Test specific query that should match Seniors' Choice
    query = "lighting installation grab bars"
    
    print(f"Debugging query: {query}")
    print("="*50)
    
    try:
        # Get all services first
        response = requests.get("http://localhost:8000/api/v1/services/ai/recommend", 
                              params={"query": query, "limit": 10})
        result = response.json()
        
        print("All recommendations:")
        for i, service in enumerate(result.get('recommendations', [])):
            print(f"{i+1}. {service['provider_name']} ({service['provider_type']}) - Score: {service['score']}")
        
        print("\n" + "="*50)
        
        # Get debug info
        debug_response = requests.get("http://localhost:8000/api/v1/services/ai/recommend/debug", 
                                    params={"query": query})
        debug_result = debug_response.json()
        
        print("Debug info for top result:")
        print(json.dumps(debug_result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_service_scoring() 