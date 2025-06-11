from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from app.models.service import ServiceList, ServiceProvider, ServiceResponse
from app.services.service_service import ServiceService

router = APIRouter()
service_service = ServiceService()

@router.get("/", response_model=ServiceList)
def get_all_providers():
    """Get all service providers"""
    return service_service.get_all_providers()

@router.get("/search", response_model=ServiceList)
def search_providers(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
):
    """Search service providers using semantic similarity"""
    return service_service.search_providers(query, limit)

@router.get("/types", response_model=List[str])
def get_service_types():
    """Get all available service types"""
    return service_service.get_service_types()

@router.get("/areas", response_model=List[str])
def get_service_areas():
    """Get all available service areas"""
    return service_service.get_service_areas()

@router.get("/type/{service_type}", response_model=ServiceList)
def get_providers_by_type(service_type: str):
    """Get providers by service type"""
    return service_service.get_providers_by_type(service_type)

@router.get("/region/{region}", response_model=ServiceList)
def get_providers_by_region(region: str):
    """Get providers by service region"""
    return service_service.get_providers_by_region(region)

@router.get("/area/{area}", response_model=ServiceList)
def get_providers_by_area(area: str):
    """Get providers by service area"""
    return service_service.get_providers_by_area(area)

@router.get("/recommendations", response_model=ServiceList)
def get_recommendations(
    query: str = Query(..., description="Query for recommendations"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of recommendations")
):
    """Get service provider recommendations based on query"""
    return service_service.get_recommendations(query, limit)

@router.get("/ai/recommend")
def get_ai_recommendation(query: str = Query(..., description="Query for AI recommendation")):
    """Get single best service provider recommendation ID for AI/MCP"""
    provider_id = service_service.get_best_recommendation(query)
    if not provider_id:
        raise HTTPException(status_code=404, detail="No relevant service provider found")
    return {"provider_id": provider_id}

@router.get("/ai/recommend/debug")
def get_ai_recommendation_debug(query: str = Query(..., description="Query for AI recommendation debug")):
    """Get best service provider recommendation with score for debugging"""
    provider, score = service_service.recommend_best_provider_with_score(query)
    if not provider:
        return {"message": "No providers found", "score": 0.0}
    return {
        "provider_id": provider.id,
        "provider_name": provider.name,
        "provider_type": provider.type,
        "score": float(score),
        "threshold_met": bool(score >= 1.2)
    }

@router.get("/statistics")
def get_service_statistics():
    """Get service provider statistics"""
    return service_service.get_statistics()

@router.get("/{provider_id}", response_model=ServiceProvider)
def get_provider_by_id(provider_id: str):
    """Get a specific service provider by ID"""
    provider = service_service.get_provider_by_id(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Service provider not found")
    return provider 