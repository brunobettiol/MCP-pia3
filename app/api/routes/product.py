from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from loguru import logger

from app.services.shopify_service import ShopifyService
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
async def get_products_for_ai(limit: Optional[int] = Query(250, description="Número máximo de produtos a retornar")):
    """
    Endpoint para obter produtos formatados para uso em chatbots de IA.
    
    Args:
        limit: Número máximo de produtos a retornar.
        
    Returns:
        GenericResponse[List[dict]]: Resposta contendo os produtos formatados.
    """
    try:
        service = ShopifyService()
        products = await service.get_all_products()
        
        # Limitar número de produtos
        products = products[:limit]
        
        # Formatar produtos para IA
        formatted_products = service.format_products_for_ai(products)
        
        return GenericResponse(
            success=True,
            data=formatted_products,
            message=f"Produtos formatados com sucesso. Total: {len(formatted_products)}"
        )
    except Exception as e:
        logger.error(f"Erro ao formatar produtos para IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao formatar produtos: {str(e)}") 