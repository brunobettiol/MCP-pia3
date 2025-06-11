from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from loguru import logger

from app.services.shopify_service import ShopifyService
from app.services.product_recommendation_service import ProductRecommendationService
from app.models.product import Product, ProductList, ProductResponse
from app.models.common import GenericResponse

router = APIRouter()

@router.get(
    "/",
    response_model=ProductResponse,
    summary="Listar todos os produtos",
    description="Retorna todos os produtos disponíveis na loja Shopify, incluindo descrição completa e tags."
)
async def get_all_products():
    """
    Endpoint para listar todos os produtos da loja Shopify.
    
    Returns:
        ProductResponse: Resposta contendo a lista de produtos.
    """
    try:
        service = ShopifyService()
        products = await service.get_all_products()
        
        return ProductResponse(
            success=True,
            data=ProductList(products=products, total=len(products)),
            message="Produtos recuperados com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao listar produtos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar produtos: {str(e)}")

@router.get("/search", response_model=ProductResponse)
async def search_products(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
):
    """Search products using semantic similarity"""
    try:
        service = ProductRecommendationService()
        products = await service.search_products(query, limit)
        
        return ProductResponse(
            success=True,
            data=ProductList(products=products, total=len(products)),
            message=f"Found {len(products)} products matching '{query}'"
        )
    except Exception as e:
        logger.error(f"Error searching products: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.get("/recommendations", response_model=ProductResponse)
async def get_recommendations(
    query: str = Query(..., description="Query for recommendations"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of recommendations")
):
    """Get product recommendations based on query"""
    try:
        service = ProductRecommendationService()
        products = await service.get_recommendations(query, limit)
        
        return ProductResponse(
            success=True,
            data=ProductList(products=products, total=len(products)),
            message=f"Found {len(products)} recommended products for '{query}'"
        )
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@router.get("/ai/recommend/handle")
async def get_ai_recommendation(query: str = Query(..., description="Query for AI recommendation")):
    """Get single best product recommendation handle for AI/MCP"""
    try:
        service = ProductRecommendationService()
        handle = await service.get_best_recommendation(query)
        if not handle:
            raise HTTPException(status_code=404, detail="No relevant product found")
        return {"handle": handle}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AI recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI recommendation error: {str(e)}")

@router.get("/ai/recommend/debug")
async def get_ai_recommendation_debug(query: str = Query(..., description="Query for AI recommendation debug")):
    """Get best product recommendation with score for debugging"""
    try:
        service = ProductRecommendationService()
        product, score = await service.recommend_best_product_with_score(query)
        if not product:
            return {"message": "No products found", "score": 0.0}
        
        return {
            "handle": product.handle,
            "title": product.title,
            "price": product.price,
            "currency": product.currency,
            "available": product.available,
            "tags": product.tags,
            "score": float(score),
            "threshold_met": bool(score >= 1.5)
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@router.get("/statistics")
async def get_product_statistics():
    """Get product statistics"""
    try:
        service = ProductRecommendationService()
        stats = await service.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

@router.get(
    "/{handle}",
    response_model=GenericResponse[Product],
    summary="Buscar produto por handle",
    description="Retorna um produto específico pelo seu handle (slug), incluindo descrição completa e tags."
)
async def get_product_by_handle(handle: str):
    """
    Endpoint para buscar um produto específico pelo handle.
    
    Args:
        handle: O handle (slug) do produto.
        
    Returns:
        GenericResponse[Product]: Resposta contendo o produto encontrado.
    """
    try:
        service = ShopifyService()
        product = await service.get_product_by_handle(handle)
        
        if not product:
            raise HTTPException(status_code=404, detail=f"Produto não encontrado: {handle}")
        
        return GenericResponse(
            success=True,
            data=product,
            message="Produto encontrado com sucesso"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao buscar produto {handle}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar produto: {str(e)}")

@router.get(
    "/ai/format",
    response_model=GenericResponse[List[dict]],
    summary="Formatar produtos para IA",
    description="Retorna produtos formatados para uso em chatbots de IA."
)
async def get_products_for_ai(
    query: Optional[str] = Query(None, description="Optional query to filter products"), 
    limit: Optional[int] = Query(250, description="Número máximo de produtos a retornar")
):
    """
    Endpoint para obter produtos formatados para uso em chatbots de IA.
    
    Args:
        query: Optional query to filter products semantically.
        limit: Número máximo de produtos a retornar.
        
    Returns:
        GenericResponse[List[dict]]: Resposta contendo os produtos formatados.
    """
    try:
        if query:
            # Use recommendation service for semantic filtering
            rec_service = ProductRecommendationService()
            products = await rec_service.search_products(query, limit)
        else:
            # Use regular service for all products
            service = ShopifyService()
            products = await service.get_all_products()
            products = products[:limit]
        
        # Format products for AI
        service = ShopifyService()
        formatted_products = service.format_products_for_ai(products)
        
        return GenericResponse(
            success=True,
            data=formatted_products,
            message=f"Produtos formatados com sucesso. Total: {len(formatted_products)}"
        )
    except Exception as e:
        logger.error(f"Erro ao formatar produtos para IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao formatar produtos: {str(e)}") 