from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from app.models.blog import BlogList, BlogEntry, BlogResponse
from app.services.blog_service import BlogService

router = APIRouter()
blog_service = BlogService()

@router.get("/", response_model=BlogList)
def get_all_blogs():
    """Get all blogs"""
    return blog_service.get_all_blogs()

@router.get("/search", response_model=BlogList)
def search_blogs(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
):
    """Search blogs using semantic similarity"""
    return blog_service.search_blogs(query, limit)

@router.get("/categories", response_model=List[str])
def get_categories():
    """Get all available blog categories"""
    return blog_service.get_categories()

@router.get("/subcategories", response_model=List[str])
def get_subcategories():
    """Get all available blog subcategories"""
    return blog_service.get_subcategories()

@router.get("/category/{category}", response_model=BlogList)
def get_blogs_by_category(category: str):
    """Get blogs by category"""
    return blog_service.get_blogs_by_category(category)

@router.get("/subcategory/{subcategory}", response_model=BlogList)
def get_blogs_by_subcategory(subcategory: str):
    """Get blogs by subcategory"""
    return blog_service.get_blogs_by_subcategory(subcategory)

@router.get("/tag/{tag}", response_model=BlogList)
def get_blogs_by_tag(tag: str):
    """Get blogs by tag"""
    return blog_service.get_blogs_by_tag(tag)

@router.get("/recommendations", response_model=BlogList)
def get_recommendations(
    query: str = Query(..., description="Query for recommendations"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of recommendations")
):
    """Get blog recommendations based on query"""
    return blog_service.get_recommendations(query, limit)

@router.get("/ai/recommend/sources")
def get_ai_recommendation(query: str = Query(..., description="Query for AI recommendation")):
    """Get single best blog recommendation source_id for AI/MCP"""
    source_id = blog_service.get_best_recommendation(query)
    if not source_id:
        raise HTTPException(status_code=404, detail="No relevant blog found")
    return {"source_id": source_id}

@router.get("/ai/recommend/debug")
def get_ai_recommendation_debug(query: str = Query(..., description="Query for AI recommendation debug")):
    """Get best blog recommendation with score for debugging"""
    blog, score = blog_service.recommend_best_blog_with_score(query)
    if not blog:
        return {"message": "No blogs found", "score": 0.0}
    return {
        "source_id": blog.source_id,
        "title": blog.title,
        "category": blog.category,
        "subcategory": blog.subcategory,
        "score": float(score),
        "threshold_met": bool(score >= 0.5)
    }

@router.get("/statistics")
def get_blog_statistics():
    """Get blog statistics"""
    return blog_service.get_statistics()

@router.get("/{source_id}", response_model=BlogEntry)
def get_blog_by_source_id(source_id: str):
    """Get a specific blog by source ID"""
    blog = blog_service.get_blog_by_source_id(source_id)
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    return blog 