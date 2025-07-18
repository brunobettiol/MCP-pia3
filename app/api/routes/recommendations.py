from fastapi import APIRouter

from app.services.blog_service import BlogService
from app.services.file_product_service import FileProductService
from app.services.service_service import ServiceService

router = APIRouter()

@router.get("/combined-recommendations", response_model=dict)
async def get_combined_recommendations(prompt: str):
    # Fetch product recommendations
    product_service = FileProductService()
    products = product_service.search_products(prompt)

    # Fetch service recommendations
    service_service = ServiceService()
    services = service_service.recommend_best_provider_with_score(prompt)

    # Fetch blog recommendations
    blog_service = BlogService()
    blogs = blog_service.get_recommendations(prompt)

    return {
        "products": products,
        "services": services,
        "blogs": blogs
    }
