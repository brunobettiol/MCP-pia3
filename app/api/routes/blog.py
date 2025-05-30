from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from loguru import logger

from app.services.blog_service import BlogService
from app.models.blog import BlogEntry, BlogList, BlogResponse, BlogRecommendation
from app.models.common import GenericResponse

router = APIRouter()

@router.get(
    "/",
    response_model=BlogResponse,
    summary="Listar todos os blogs",
    description="Retorna todos os blogs disponíveis."
)
async def get_all_blogs():
    """
    Endpoint para listar todos os blogs.
    
    Returns:
        BlogResponse: Resposta contendo a lista de blogs.
    """
    try:
        service = BlogService()
        blogs = await service.get_all_blogs()
        
        return BlogResponse(
            success=True,
            data=BlogList(blogs=blogs, total=len(blogs)),
            message="Blogs recuperados com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao listar blogs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar blogs: {str(e)}")

@router.get(
    "/category/{category}",
    response_model=BlogResponse,
    summary="Buscar blogs por categoria",
    description="Retorna blogs de uma categoria específica."
)
async def get_blogs_by_category(category: str):
    """
    Endpoint para buscar blogs por categoria.
    
    Args:
        category: A categoria dos blogs.
        
    Returns:
        BlogResponse: Resposta contendo a lista de blogs da categoria.
    """
    try:
        service = BlogService()
        blogs = await service.get_blogs_by_category(category)
        
        return BlogResponse(
            success=True,
            data=BlogList(blogs=blogs, total=len(blogs)),
            message=f"Blogs da categoria '{category}' recuperados com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao buscar blogs por categoria: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar blogs: {str(e)}")

@router.get(
    "/tag/{tag}",
    response_model=BlogResponse,
    summary="Buscar blogs por tag",
    description="Retorna blogs que contêm uma tag específica."
)
async def get_blogs_by_tag(tag: str):
    """
    Endpoint para buscar blogs por tag.
    
    Args:
        tag: A tag a ser buscada.
        
    Returns:
        BlogResponse: Resposta contendo a lista de blogs com a tag.
    """
    try:
        service = BlogService()
        blogs = await service.get_blogs_by_tag(tag)
        
        return BlogResponse(
            success=True,
            data=BlogList(blogs=blogs, total=len(blogs)),
            message=f"Blogs com a tag '{tag}' recuperados com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao buscar blogs por tag: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar blogs: {str(e)}")

@router.get(
    "/search",
    response_model=BlogResponse,
    summary="Buscar blogs por texto",
    description="Busca blogs que contenham o texto especificado no título ou conteúdo."
)
async def search_blogs(query: str = Query(..., description="Texto a ser buscado")):
    """
    Endpoint para buscar blogs por texto.
    
    Args:
        query: O texto a ser buscado.
        
    Returns:
        BlogResponse: Resposta contendo a lista de blogs encontrados.
    """
    try:
        service = BlogService()
        blogs = await service.search_blogs(query)
        
        return BlogResponse(
            success=True,
            data=BlogList(blogs=blogs, total=len(blogs)),
            message=f"Blogs contendo '{query}' recuperados com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao buscar blogs por texto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar blogs: {str(e)}")

@router.get(
    "/recommend",
    response_model=GenericResponse[List[BlogEntry]],
    summary="Recomendar blogs por consulta",
    description="Recomenda blogs com base em uma consulta de texto."
)
async def recommend_blogs(
    query: str = Query(..., description="Consulta de texto"),
    limit: int = Query(3, description="Número máximo de blogs a retornar")
):
    """
    Endpoint para recomendar blogs com base em uma consulta.
    
    Args:
        query: A consulta de texto.
        limit: Número máximo de blogs a retornar.
        
    Returns:
        GenericResponse[List[BlogEntry]]: Resposta contendo os blogs recomendados.
    """
    try:
        service = BlogService()
        blogs = await service.recommend_blogs(query, limit)
        
        return GenericResponse(
            success=True,
            data=blogs,
            message=f"Blogs recomendados para '{query}' recuperados com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao recomendar blogs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao recomendar blogs: {str(e)}")

@router.get(
    "/recommend/product/{product_handle}",
    response_model=GenericResponse[List[BlogEntry]],
    summary="Recomendar blogs para um produto",
    description="Recomenda blogs relacionados a um produto específico."
)
async def recommend_blogs_for_product(
    product_handle: str,
    limit: int = Query(3, description="Número máximo de blogs a retornar")
):
    """
    Endpoint para recomendar blogs relacionados a um produto.
    
    Args:
        product_handle: O handle do produto.
        limit: Número máximo de blogs a retornar.
        
    Returns:
        GenericResponse[List[BlogEntry]]: Resposta contendo os blogs recomendados.
    """
    try:
        service = BlogService()
        blogs = await service.recommend_blogs_for_product(product_handle, limit)
        
        return GenericResponse(
            success=True,
            data=blogs,
            message=f"Blogs recomendados para o produto '{product_handle}' recuperados com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao recomendar blogs para produto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao recomendar blogs: {str(e)}")

@router.get(
    "/ai/recommend",
    response_model=GenericResponse[List[dict]],
    summary="Recomendar blogs para IA",
    description="Recomenda blogs formatados para uso em chatbots de IA."
)
async def recommend_blogs_for_ai(
    query: str = Query(..., description="Consulta de texto"),
    limit: int = Query(3, description="Número máximo de blogs a retornar")
):
    """
    Endpoint para recomendar blogs formatados para uso em chatbots de IA.
    
    Args:
        query: A consulta de texto.
        limit: Número máximo de blogs a retornar.
        
    Returns:
        GenericResponse[List[dict]]: Resposta contendo os blogs recomendados formatados.
    """
    try:
        service = BlogService()
        blogs = await service.recommend_blogs(query, limit)
        
        # Formatar blogs para IA
        formatted_blogs = []
        for blog in blogs:
            formatted_blog = {
                "title": blog.title,
                "content": blog.content,
                "author": blog.author,
                "category": blog.category,
                "tags": ", ".join(blog.tags),
                "created_at": blog.created_at.isoformat()
            }
            formatted_blogs.append(formatted_blog)
        
        return GenericResponse(
            success=True,
            data=formatted_blogs,
            message=f"Blogs recomendados para IA recuperados com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao recomendar blogs para IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao recomendar blogs: {str(e)}")

@router.get(
    "/ai/recommend/sources",
    response_model=GenericResponse[List[str]],
    summary="Recomendar fontes de blogs para IA",
    description="Recomenda source_ids de blogs com base em uma consulta de texto para uso em chatbots de IA."
)
async def recommend_blog_sources_for_ai(
    query: str = Query(..., description="Consulta de texto"),
    limit: int = Query(3, description="Número máximo de blogs a retornar")
):
    """
    Endpoint para recomendar apenas os source_ids de blogs para uso em chatbots de IA.
    
    Args:
        query: A consulta de texto.
        limit: Número máximo de blogs a retornar.
        
    Returns:
        GenericResponse[List[str]]: Resposta contendo os source_ids dos blogs recomendados.
    """
    try:
        service = BlogService()
        blogs = await service.recommend_blogs(query, limit)
        
        # Extrair apenas os source_ids
        source_ids = [blog.source_id for blog in blogs]
        
        return GenericResponse(
            success=True,
            data=source_ids,
            message=f"Source IDs de blogs recomendados recuperados com sucesso"
        )
    except Exception as e:
        logger.error(f"Erro ao recomendar source_ids de blogs para IA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao recomendar blogs: {str(e)}") 