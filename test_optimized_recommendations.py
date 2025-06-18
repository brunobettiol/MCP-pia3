import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from app.services.blog_service import BlogService
from app.services.service_service import ServiceService
from app.services.product_recommendation_service import ProductRecommendationService


async def test_35_questions():
    """Test all 35 specific eldercare questions with optimized recommendation systems"""
    
    # Initialize services
    blog_service = BlogService()
    service_service = ServiceService()
    product_service = ProductRecommendationService()
    
    # The 35 specific questions organized by category
    test_questions = {
        "SAFE Category Tasks": [
            "Ensure handrails are installed on both sides of stairs",
            "Install grab bars in the bathroom near the toilet and shower",
            "Improve lighting in all areas, especially hallways and stairs",
            "Assess fall risks and implement prevention strategies",
            "Provide appropriate mobility aids like a cane, walker, or wheelchair",
            "Remove tripping hazards like loose rugs, clutter, or electrical cords",
            "Ensure the bedroom is accessible without using stairs",
            "Evaluate neighborhood safety and accessibility for walking",
            "Post emergency numbers visibly near phones",
            "Install a personal emergency response system",
            "Evaluate in-home care support needs and consider professional help"
        ],
        "Functional Ability Tasks": [
            "Ensure safe bathing practices and provide assistance if needed",
            "Provide assistance with dressing or undressing if necessary",
            "Ensure safe movement from bed or chair without help",
            "Provide assistance with toilet use if needed"
        ],
        "HEALTHY Category Tasks": [
            "Manage chronic medical conditions effectively",
            "Organize and manage daily medications",
            "Implement a medication management system",
            "Schedule regular check-ups with primary care and specialists",
            "Ensure regular, balanced meals for proper nutrition",
            "Assist with meal preparation and grocery shopping if needed",
            "Establish an exercise routine or regular physical activity",
            "Monitor cognitive health and address memory issues",
            "Improve sleep quality and address any issues",
            "Schedule regular dental, vision, and hearing check-ups"
        ],
        "Emotional Health Tasks": [
            "Address feelings of depression or hopelessness",
            "Encourage participation in enjoyable activities",
            "Reduce feelings of loneliness or isolation",
            "Ensure vaccinations are up to date"
        ],
        "PREPARED Category Tasks": [
            "Establish advance directives like a living will or healthcare proxy",
            "Set up a durable power of attorney for finances",
            "Create a will or trust",
            "Discuss end-of-life care preferences with family",
            "Review and update insurance coverage",
            "Develop a financial plan for potential long-term care needs",
            "Consider living arrangement options for future needs",
            "Implement a system for managing bills and financial matters",
            "Organize important documents for easy access",
            "Create a communication plan for family care decisions"
        ]
    }
    
    print("=" * 80)
    print("TESTING OPTIMIZED RECOMMENDATION SYSTEMS FOR 35 ELDERCARE QUESTIONS")
    print("=" * 80)
    
    total_questions = 0
    successful_recommendations = {"blogs": 0, "services": 0, "products": 0}
    
    for category, questions in test_questions.items():
        print(f"\n{'=' * 60}")
        print(f"CATEGORY: {category}")
        print(f"{'=' * 60}")
        
        for i, question in enumerate(questions, 1):
            total_questions += 1
            print(f"\n{i}. {question}")
            print("-" * 60)
            
            # Test Blog Recommendations
            try:
                blog_recommendation = blog_service.get_best_recommendation(question)
                if blog_recommendation:
                    blog = blog_service.get_blog_by_source_id(blog_recommendation)
                    if blog:
                        successful_recommendations["blogs"] += 1
                        print(f"✅ BLOG: {blog.title[:60]}...")
                        print(f"   Category: {blog.category}, Score: Good match")
                    else:
                        print(f"❌ BLOG: Found ID {blog_recommendation} but couldn't retrieve blog")
                else:
                    print("❌ BLOG: No recommendation found")
            except Exception as e:
                print(f"❌ BLOG: Error - {e}")
            
            # Test Service Recommendations
            try:
                service_recommendation = service_service.get_best_recommendation(question)
                if service_recommendation:
                    service = service_service.get_provider_by_id(service_recommendation)
                    if service:
                        successful_recommendations["services"] += 1
                        print(f"✅ SERVICE: {service.name[:50]}...")
                        print(f"   Type: {service.type}, Score: Good match")
                    else:
                        print(f"❌ SERVICE: Found ID {service_recommendation} but couldn't retrieve service")
                else:
                    print("❌ SERVICE: No recommendation found")
            except Exception as e:
                print(f"❌ SERVICE: Error - {e}")
            
            # Test Product Recommendations
            try:
                product_recommendation = await product_service.get_best_recommendation(question)
                if product_recommendation:
                    # For products, we get the handle, so we'll just show it
                    successful_recommendations["products"] += 1
                    print(f"✅ PRODUCT: {product_recommendation}")
                    print(f"   Score: Good match")
                else:
                    print("❌ PRODUCT: No recommendation found")
            except Exception as e:
                print(f"❌ PRODUCT: Error - {e}")
    
    # Summary Report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Questions Tested: {total_questions}")
    print(f"Blog Recommendations: {successful_recommendations['blogs']}/{total_questions} ({successful_recommendations['blogs']/total_questions*100:.1f}%)")
    print(f"Service Recommendations: {successful_recommendations['services']}/{total_questions} ({successful_recommendations['services']/total_questions*100:.1f}%)")
    print(f"Product Recommendations: {successful_recommendations['products']}/{total_questions} ({successful_recommendations['products']/total_questions*100:.1f}%)")
    
    overall_success = sum(successful_recommendations.values()) / (total_questions * 3) * 100
    print(f"Overall Success Rate: {overall_success:.1f}%")
    
    if overall_success >= 80:
        print("🎉 EXCELLENT: Recommendation system is working very well!")
    elif overall_success >= 60:
        print("✅ GOOD: Recommendation system is working well!")
    elif overall_success >= 40:
        print("⚠️  FAIR: Recommendation system needs some improvement")
    else:
        print("❌ POOR: Recommendation system needs significant improvement")


async def test_sample_questions():
    """Test a few sample questions to verify the optimization is working"""
    
    # Initialize services
    blog_service = BlogService()
    service_service = ServiceService()
    product_service = ProductRecommendationService()
    
    sample_questions = [
        "Install grab bars in bathroom",
        "Medication management system",
        "Fall prevention strategies",
        "End-of-life planning",
        "Exercise routine for seniors"
    ]
    
    print("\n" + "=" * 80)
    print("DETAILED TESTING OF SAMPLE QUESTIONS")
    print("=" * 80)
    
    for question in sample_questions:
        print(f"\nQUESTION: {question}")
        print("-" * 60)
        
        # Blog recommendations with scores
        try:
            blog, score = blog_service.recommend_best_blog_with_score(question)
            if blog and score > 0:
                print(f"BLOG: {blog.title}")
                print(f"  Category: {blog.category}")
                print(f"  Score: {score:.2f}")
                print(f"  Summary: {blog.summary[:100]}...")
            else:
                print("BLOG: No recommendation found")
        except Exception as e:
            print(f"BLOG: Error - {e}")
        
        # Service recommendations with scores
        try:
            service, score = service_service.recommend_best_provider_with_score(question)
            if service and score > 0:
                print(f"SERVICE: {service.name}")
                print(f"  Type: {service.type}")
                print(f"  Score: {score:.2f}")
                print(f"  Description: {service.description[:100]}...")
            else:
                print("SERVICE: No recommendation found")
        except Exception as e:
            print(f"SERVICE: Error - {e}")
        
        # Product recommendations with scores
        try:
            product, score = await product_service.recommend_best_product_with_score(question)
            if product and score > 0:
                print(f"PRODUCT: {product.title}")
                print(f"  Handle: {product.handle}")
                print(f"  Score: {score:.2f}")
                print(f"  Tags: {', '.join(product.tags[:5])}")
            else:
                print("PRODUCT: No recommendation found")
        except Exception as e:
            print(f"PRODUCT: Error - {e}")


async def main():
    """Main test function"""
    print("Starting comprehensive test of optimized recommendation systems...")
    
    # Test sample questions first for detailed analysis
    await test_sample_questions()
    
    # Test all 35 questions for coverage analysis
    await test_35_questions()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main()) 