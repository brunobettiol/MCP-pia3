from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from loguru import logger

from app.services.file_product_service import FileProductService
from app.models.product import Product, ProductList, ProductResponse
from app.models.common import GenericResponse

router = APIRouter()

@router.get(
    "/",
    response_model=ProductResponse,
    summary="Listar todos os produtos",
    description="Retorna todos os produtos disponíveis do arquivo CSV, incluindo descrição completa e tags."
)
async def get_all_products():
    """
    Endpoint para listar todos os produtos do arquivo CSV.
    
    Returns:
        ProductResponse: Resposta contendo a lista de produtos.
    """
    try:
        service = FileProductService()
        products = service.get_all_products()
        
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
        service = FileProductService()
        products = service.search_products(query, limit)
        
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
        service = FileProductService()
        products = service.get_recommendations(query, limit)
        
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
    """Get comprehensive product recommendation information for AI/MCP"""
    try:
        service = FileProductService()
        product, score = service.get_best_recommendation(query)
        
        # Define threshold for quality recommendations
        threshold = 2.0
        threshold_met = bool(score >= threshold)
        
        # Only return product if threshold is met
        if not product or not threshold_met:
            raise HTTPException(status_code=404, detail="No relevant product found")
        
        # Generate the correct URL based on product source
        product_url = service.get_product_url(product)
        
        return {
            "handle": product.handle,
            "title": product.title,
            "description": product.description,
            "price": product.price,
            "currency": product.currency,
            "available": product.available,
            "tags": product.tags,
            "images": product.images,
            "variants": product.variants,
            "url": product_url,
            "score": float(score),
            "threshold_met": threshold_met
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AI recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI recommendation error: {str(e)}")

@router.get("/ai/recommend/debug")
async def get_ai_recommendation_debug(query: str = Query(..., description="Query for AI recommendation debug")):
    """Get best product recommendation with score for debugging"""
    try:
        service = FileProductService()
        product, score = service.get_best_recommendation(query)
        if not product:
            return {"message": "No products found", "score": 0.0}
        
        # Generate the correct URL based on product source
        product_url = service.get_product_url(product)
        
        return {
            "handle": product.handle,
            "title": product.title,
            "price": product.price,
            "currency": product.currency,
            "available": product.available,
            "tags": product.tags,
            "score": float(score),
            "threshold_met": bool(score >= 2.0),
            "url": product_url,
            "source": "WooCommerce" if product.handle.startswith('woo-') else "Shopify"
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@router.get("/statistics")
async def get_product_statistics():
    """Get product statistics"""
    try:
        service = FileProductService()
        stats = service.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

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
        service = FileProductService()
        
        if query:
            products = service.search_products(query, limit)
        else:
            products = service.get_all_products()[:limit]
        
        # Format products for AI (simplified format)
        formatted_products = []
        for product in products:
            product_url = service.get_product_url(product)
            
            formatted_products.append({
                "handle": product.handle,
                "title": product.title,
                "description": product.description,
                "price": f"${product.price:.2f}",
                "currency": product.currency,
                "available": product.available,
                "tags": ", ".join(product.tags),
                "url": product_url,
                "source": "WooCommerce" if product.handle.startswith('woo-') else "Shopify"
            })
        
        return GenericResponse(
            success=True,
            data=formatted_products,
            message=f"Produtos formatados com sucesso. Total: {len(formatted_products)}"
        )
    except Exception as e:
        logger.error(f"Erro ao formatar produtos para IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao formatar produtos: {str(e)}") 