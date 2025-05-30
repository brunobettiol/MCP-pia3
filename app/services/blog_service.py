import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger
import re
from pathlib import Path

from app.models.blog import BlogEntry
from app.core.config import settings

class BlogService:
    """Serviço para gerenciar blogs e recomendações."""
    
    def __init__(self):
        """Inicializa o serviço de blogs."""
        self.blog_data_path = os.path.join(settings.BASE_DIR, "data", "blogs")
        self.blogs = []
        self.loaded = False
    
    async def load_blogs(self) -> None:
        """
        Carrega todos os blogs dos arquivos JSON.
        """
        if self.loaded:
            return
            
        try:
            logger.info("Carregando dados de blogs...")
            self.blogs = []
            
            # Garantir que o diretório existe
            os.makedirs(self.blog_data_path, exist_ok=True)
            
            # Carregar cada arquivo JSON na pasta de blogs
            for file_name in os.listdir(self.blog_data_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.blog_data_path, file_name)
                    logger.info(f"Carregando arquivo de blog: {file_path}")
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                        category_blogs = json.load(file)
                        
                        for blog_data in category_blogs:
                            # Converter string de data para objeto datetime
                            if isinstance(blog_data.get('created_at'), str):
                                blog_data['created_at'] = datetime.fromisoformat(
                                    blog_data['created_at'].replace('Z', '+00:00')
                                )
                            
                            blog = BlogEntry(**blog_data)
                            self.blogs.append(blog)
            
            logger.info(f"Total de blogs carregados: {len(self.blogs)}")
            self.loaded = True
            
        except Exception as e:
            logger.error(f"Erro ao carregar blogs: {str(e)}")
            raise
    
    async def get_all_blogs(self) -> List[BlogEntry]:
        """
        Retorna todos os blogs carregados.
        
        Returns:
            Lista de todos os blogs.
        """
        await self.load_blogs()
        return self.blogs
    
    async def get_blogs_by_category(self, category: str) -> List[BlogEntry]:
        """
        Retorna blogs de uma categoria específica.
        
        Args:
            category: A categoria dos blogs.
            
        Returns:
            Lista de blogs da categoria especificada.
        """
        await self.load_blogs()
        return [blog for blog in self.blogs if blog.category.lower() == category.lower()]
    
    async def get_blogs_by_tag(self, tag: str) -> List[BlogEntry]:
        """
        Retorna blogs que contêm uma tag específica.
        
        Args:
            tag: A tag a ser buscada.
            
        Returns:
            Lista de blogs com a tag especificada.
        """
        await self.load_blogs()
        return [blog for blog in self.blogs if tag.lower() in [t.lower() for t in blog.tags]]
    
    async def search_blogs(self, query: str) -> List[BlogEntry]:
        """
        Busca blogs por texto no título ou conteúdo.
        
        Args:
            query: O texto a ser buscado.
            
        Returns:
            Lista de blogs que correspondem à consulta.
        """
        await self.load_blogs()
        query = query.lower()
        return [
            blog for blog in self.blogs 
            if query in blog.title.lower() or query in blog.content.lower()
        ]
    
    async def recommend_blogs_for_product(self, product_handle: str, limit: int = 3) -> List[BlogEntry]:
        """
        Recomenda blogs relacionados a um produto específico.
        
        Args:
            product_handle: O handle do produto.
            limit: Número máximo de blogs a retornar.
            
        Returns:
            Lista de blogs recomendados.
        """
        from app.services.shopify_service import ShopifyService
        
        await self.load_blogs()
        
        try:
            # Obter informações do produto
            shopify_service = ShopifyService()
            product = await shopify_service.get_product_by_handle(product_handle)
            
            if not product:
                logger.warning(f"Produto não encontrado: {product_handle}")
                return []
            
            # Criar consulta baseada no título e tags do produto
            search_terms = [product.title] + product.tags
            search_query = " ".join(search_terms)
            
            # Encontrar blogs relevantes
            relevant_blogs = await self.recommend_blogs(search_query, limit)
            return relevant_blogs
            
        except Exception as e:
            logger.error(f"Erro ao recomendar blogs para produto: {str(e)}")
            return []
    
    async def recommend_blogs(self, query: str, limit: int = 3) -> List[BlogEntry]:
        """
        Recomenda blogs com base em uma consulta de texto.
        
        Args:
            query: A consulta de texto.
            limit: Número máximo de blogs a retornar.
            
        Returns:
            Lista de blogs recomendados.
        """
        await self.load_blogs()
        
        # Implementação simples de recomendação baseada em palavras-chave
        # Em um sistema de produção, você pode querer usar um modelo de ML ou embeddings
        
        query = query.lower()
        query_words = set(re.findall(r'\w+', query))
        
        # Calcular pontuação para cada blog
        blog_scores = []
        
        for blog in self.blogs:
            score = 0
            
            # Verificar título
            title_words = set(re.findall(r'\w+', blog.title.lower()))
            title_match = len(query_words.intersection(title_words))
            score += title_match * 3  # Título tem peso maior
            
            # Verificar conteúdo
            content_words = set(re.findall(r'\w+', blog.content.lower()))
            content_match = len(query_words.intersection(content_words))
            score += content_match
            
            # Verificar tags
            tag_words = set()
            for tag in blog.tags:
                tag_words.update(set(re.findall(r'\w+', tag.lower())))
            
            tag_match = len(query_words.intersection(tag_words))
            score += tag_match * 2  # Tags têm peso médio
            
            blog_scores.append((blog, score))
        
        # Ordenar por pontuação e limitar resultados
        blog_scores.sort(key=lambda x: x[1], reverse=True)
        recommended_blogs = [blog for blog, score in blog_scores[:limit] if score > 0]
        
        return recommended_blogs 