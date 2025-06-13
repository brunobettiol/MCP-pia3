import requests
import json

def test_product_quality():
    """Test product recommendation quality"""
    print("Testing Product Recommendations...")
    
    # Test medication management query
    url = "http://localhost:8000/api/v1/products/ai/recommend/debug"
    params = {"query": "Establish a medication management system"}
    
    try:
        response = requests.get(url, params=params)
        result = response.json()
        
        print(f"Product Query: {params['query']}")
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if "message" in result and "No products found" in result["message"]:
            print("✅ GOOD: No irrelevant products recommended")
        else:
            print("❌ BAD: Still recommending irrelevant products")
            
    except Exception as e:
        print(f"Error testing products: {e}")

def test_service_quality():
    """Test service recommendation quality"""
    print("\nTesting Service Recommendations...")
    
    # Test medication management query
    url = "http://localhost:8000/api/v1/services/ai/recommend/debug"
    params = {"query": "Establish a medication management system"}
    
    try:
        response = requests.get(url, params=params)
        result = response.json()
        
        print(f"Service Query: {params['query']}")
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if "message" in result and "No providers found" in result["message"]:
            print("✅ GOOD: No irrelevant services recommended")
        elif result.get("threshold_met") == False:
            print("✅ GOOD: Service found but below threshold (won't be returned to users)")
        else:
            print("❌ BAD: Still recommending irrelevant services")
            
    except Exception as e:
        print(f"Error testing services: {e}")

def test_blog_quality():
    """Test blog recommendation quality"""
    print("\nTesting Blog Recommendations...")
    
    # Test medication management query
    url = "http://localhost:8000/api/v1/blogs/ai/recommend/debug"
    params = {"query": "Establish a medication management system"}
    
    try:
        response = requests.get(url, params=params)
        result = response.json()
        
        print(f"Blog Query: {params['query']}")
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if "message" in result and "No blogs found" in result["message"]:
            print("✅ GOOD: No irrelevant blogs recommended")
        elif result.get("threshold_met") == False:
            print("✅ GOOD: Blog found but below threshold (won't be returned to users)")
        else:
            print("❌ BAD: Still recommending irrelevant blogs")
            
    except Exception as e:
        print(f"Error testing blogs: {e}")

def test_legitimate_queries():
    """Test that legitimate queries still work"""
    print("\n" + "="*50)
    print("Testing Legitimate Queries...")
    print("="*50)
    
    # Test home modification query
    print("\n--- Testing Home Modification Query ---")
    try:
        response = requests.get("http://localhost:8000/api/v1/services/ai/recommend/debug", 
                              params={"query": "Install grab bars and improve lighting"})
        result = response.json()
        print(f"Service Query: Install grab bars and improve lighting")
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get("provider_type") == "home_modification":
            print("✅ GOOD: Correctly recommended home modification service")
        else:
            print("❌ BAD: Failed to recommend appropriate service")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test in-home care query
    print("\n--- Testing In-Home Care Query ---")
    try:
        response = requests.get("http://localhost:8000/api/v1/services/ai/recommend/debug", 
                              params={"query": "Need a caregiver for personal care assistance"})
        result = response.json()
        print(f"Service Query: Need a caregiver for personal care assistance")
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get("provider_type") == "in_home_care":
            print("✅ GOOD: Correctly recommended in-home care service")
        else:
            print("❌ BAD: Failed to recommend appropriate service")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_product_quality()
    test_service_quality()
    test_blog_quality()
    test_legitimate_queries() 