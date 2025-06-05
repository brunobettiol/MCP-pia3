import json
import os
from typing import List, Dict, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.models.blog import BlogEntry, BlogList, BlogResponse
from app.core.config import settings


class BlogService:
    def __init__(self):
        """Initialize the blog service with TF-IDF indexing"""
        self.blogs: List[BlogEntry] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._load_blogs()
        self._build_search_index()
    
    def _load_blogs(self):
        """Load all blog entries from JSON files"""
        blogs_dir = os.path.join(settings.BASE_DIR, "data", "blogs")
        if not os.path.exists(blogs_dir):
            return
        
        for filename in os.listdir(blogs_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(blogs_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            blog = BlogEntry(**item)
                            self.blogs.append(blog)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _build_search_index(self):
        """Build TF-IDF index for all blog content"""
        if not self.blogs:
            return
        
        # Combine title, content, and tags for each blog
        self.processed_content = []
        for blog in self.blogs:
            # Combine all text fields with appropriate weights
            combined_text = f"{blog.title} {blog.title} {blog.title} " + \
                           f"{' '.join(blog.tags)} {' '.join(blog.tags)} " + \
                           f"{blog.content}"
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Initialize TF-IDF vectorizer with optimized parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit vocabulary size
            stop_words='english',  # Remove common English stop words
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=1,  # Minimum document frequency
            max_df=0.9,  # Maximum document frequency (filter out very common terms)
            sublinear_tf=True,  # Apply sublinear tf scaling
            norm='l2'  # Normalize vectors
        )
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)

    def get_all_blogs(self) -> BlogList:
        """Get all blogs"""
        return BlogList(blogs=self.blogs, total=len(self.blogs))

    def search_blogs(self, query: str, limit: int = 10) -> BlogList:
        """Search blogs using TF-IDF similarity"""
        if not query or not self.blogs or self.tfidf_matrix is None:
            return BlogList(blogs=[], total=0)
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter results with minimum similarity threshold
        min_similarity = 0.1  # Adjust threshold as needed
        filtered_results = []
        
        for idx in sorted_indices:
            if similarities[idx] >= min_similarity and len(filtered_results) < limit:
                filtered_results.append(self.blogs[idx])
            elif len(filtered_results) >= limit:
                break
        
        return BlogList(blogs=filtered_results, total=len(filtered_results))

    def get_blogs_by_category(self, category: str) -> BlogList:
        """Get blogs filtered by category"""
        filtered_blogs = [blog for blog in self.blogs if blog.category.lower() == category.lower()]
        return BlogList(blogs=filtered_blogs, total=len(filtered_blogs))

    def get_blog_by_source_id(self, source_id: str) -> Optional[BlogEntry]:
        """Get a specific blog by source ID"""
        for blog in self.blogs:
            if blog.source_id == source_id:
                return blog
        return None

    def get_recommendations(self, query: str, limit: int = 5) -> BlogList:
        """Get blog recommendations based on query using TF-IDF similarity"""
        if not query or not self.blogs or self.tfidf_matrix is None:
            return BlogList(blogs=[], total=0)
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter results with minimum similarity threshold
        min_similarity = 0.15  # Slightly higher threshold for recommendations
        recommended_blogs = []
        
        for idx in sorted_indices:
            if similarities[idx] >= min_similarity and len(recommended_blogs) < limit:
                recommended_blogs.append(self.blogs[idx])
            elif len(recommended_blogs) >= limit:
                break
        
        return BlogList(blogs=recommended_blogs, total=len(recommended_blogs))

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best blog recommendation source_id based on query"""
        if not query or not self.blogs or self.tfidf_matrix is None:
            return None
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Find the best match
        best_idx = similarities.argmax()
        best_similarity = similarities[best_idx]
        
        # Return source_id only if similarity meets minimum threshold
        min_similarity = 0.1
        if best_similarity >= min_similarity:
            return self.blogs[best_idx].source_id
        
        return None

    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        categories = set(blog.category for blog in self.blogs)
        return sorted(list(categories))

    def get_statistics(self) -> Dict:
        """Get blog statistics"""
        categories = {}
        for blog in self.blogs:
            categories[blog.category] = categories.get(blog.category, 0) + 1
        
        return {
            "total_blogs": len(self.blogs),
            "categories": categories,
            "authors": list(set(blog.author for blog in self.blogs))
        }

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

    async def recommend_best_blog_with_score(self, query: str) -> tuple[BlogEntry, float] | tuple[None, 0]:
        """
        Recomenda o melhor blog com base em uma consulta de texto e retorna o score.
        
        Args:
            query: A consulta de texto.
            
        Returns:
            Tupla contendo o melhor blog e seu score, ou (None, 0) se nenhum for encontrado.
        """
        await self.load_blogs()
        
        query = query.lower()
        query_words = set(re.findall(r'\w+', query))
        
        # Calcular pontuação para cada blog
        best_blog = None
        best_score = 0
        
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
            
            if score > best_score:
                best_score = score
                best_blog = blog
        
        return (best_blog, best_score) if best_blog else (None, 0) 