import requests
import json

def test_home_modification():
    """Test home modification queries"""
    queries = [
        "Install grab bars and improve lighting",
        "lighting installation grab bars",
        "home safety modifications",
        "handrails and lighting"
    ]
    
    for query in queries:
        print(f"\n--- Testing: {query} ---")
        try:
            response = requests.get("http://localhost:8000/api/v1/services/ai/recommend/debug", 
                                  params={"query": query})
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            if result.get("provider_type") == "home_modification":
                print("✅ GOOD: Found home modification service")
            else:
                print("❌ BAD: Wrong service type or no service found")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_home_modification() 