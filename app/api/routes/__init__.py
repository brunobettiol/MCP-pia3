from fastapi import APIRouter
from app.api.routes.product import router as product_router
from app.api.routes.health import router as health_router

router = APIRouter()

router.include_router(health_router, prefix="/health", tags=["health"])
router.include_router(product_router, prefix="/products", tags=["products"]) 