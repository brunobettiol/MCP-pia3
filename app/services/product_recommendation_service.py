import re
import asyncio
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.models.product import Product
from app.services.shopify_service import ShopifyService


class ProductRecommendationService:
    def __init__(self):
        """Initialize the product recommendation service with TF-IDF indexing"""
        self.products: List[Product] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._loaded = False
    
    async def _ensure_loaded(self):
        """Ensure products are loaded before processing"""
        if not self._loaded:
            await self._load_products()
            self._build_search_index()
            self._loaded = True
    
    async def _load_products(self):
        """Load all products from Shopify"""
        try:
            shopify_service = ShopifyService()
            self.products = await shopify_service.get_all_products()
            print(f"Loaded {len(self.products)} products for recommendations")
        except Exception as e:
            print(f"Error loading products: {e}")
            self.products = []
    
    def _strip_html(self, html_content: str) -> str:
        """Strip HTML tags from content and return clean text"""
        if not html_content:
            return ""
        
        # Remove HTML tags using regex
        clean_text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Replace common HTML entities
        clean_text = clean_text.replace('&nbsp;', ' ')
        clean_text = clean_text.replace('&amp;', '&')
        clean_text = clean_text.replace('&lt;', '<')
        clean_text = clean_text.replace('&gt;', '>')
        clean_text = clean_text.replace('&quot;', '"')
        clean_text = clean_text.replace('&#39;', "'")
        
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text

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
        """Build TF-IDF index for all product content"""
        if not self.products:
            return
        
        # Combine title, description, tags for each product
        self.processed_content = []
        for product in self.products:
            # Combine all text fields with appropriate weights
            # Title gets the highest weight (3x), tags get medium weight (2x), description gets base weight
            combined_text = f"{product.title} {product.title} {product.title} " + \
                           f"{' '.join(product.tags)} {' '.join(product.tags)} " + \
                           f"{self._strip_html(product.description or '')}"
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Initialize TF-IDF vectorizer with optimized parameters
        # Adjust parameters based on number of documents
        num_docs = len(self.processed_content)
        max_features = min(5000, num_docs * 100)  # Reasonable vocabulary size
        min_df = 1  # Always include words that appear at least once
        max_df = max(0.95, 1.0 - (1.0 / num_docs))  # Ensure max_df > min_df
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features if max_features > 0 else None,
            stop_words='english',  # Remove common English stop words
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,  # Apply sublinear tf scaling
            norm='l2'  # Normalize vectors
        )
        
        # Build TF-IDF matrix
        if self.processed_content:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
            except Exception as e:
                print(f"Error building TF-IDF matrix: {e}")
                # Fallback to simpler configuration
                self.tfidf_vectorizer = TfidfVectorizer(
                    stop_words=None,
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=1.0
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)

    async def recommend_best_product_with_score(self, query: str) -> Tuple[Optional[Product], float]:
        """Get the best product recommendation with TF-IDF similarity score"""
        await self._ensure_loaded()
        
        if not query or not self.products or self.tfidf_matrix is None:
            return None, 0.0
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Find the best match
        best_idx = similarities.argmax()
        best_similarity = similarities[best_idx]
        
        # Convert cosine similarity to a more intuitive score (0-10)
        score = best_similarity * 10
        
        # Set minimum relevance threshold (adjust as needed)
        min_relevance_score = 1.5  # Equivalent to 15% cosine similarity - balanced threshold
        
        if score >= min_relevance_score:
            return self.products[best_idx], score
        
        return None, 0.0

    async def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best product recommendation handle based on query with threshold"""
        product, score = await self.recommend_best_product_with_score(query)
        
        if product and score >= 1.5:  # Minimum threshold - balanced for good quality
            return product.handle
        
        return None

    async def get_recommendations(self, query: str, limit: int = 5) -> List[Product]:
        """Get product recommendations based on query using TF-IDF similarity"""
        await self._ensure_loaded()
        
        if not query or not self.products or self.tfidf_matrix is None:
            return []
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter results with minimum similarity threshold
        min_similarity = 0.2  # Balanced threshold for recommendations (20% cosine similarity)
        recommended_products = []
        
        for idx in sorted_indices:
            if similarities[idx] >= min_similarity and len(recommended_products) < limit:
                recommended_products.append(self.products[idx])
            elif len(recommended_products) >= limit:
                break
        
        return recommended_products

    async def search_products(self, query: str, limit: int = 10) -> List[Product]:
        """Search products using TF-IDF similarity"""
        await self._ensure_loaded()
        
        if not query or not self.products or self.tfidf_matrix is None:
            return []
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter results with minimum similarity threshold
        min_similarity = 0.15  # Balanced threshold for search (15% cosine similarity)
        filtered_results = []
        
        for idx in sorted_indices:
            if similarities[idx] >= min_similarity and len(filtered_results) < limit:
                filtered_results.append(self.products[idx])
            elif len(filtered_results) >= limit:
                break
        
        return filtered_results

    async def get_statistics(self) -> Dict:
        """Get product statistics"""
        await self._ensure_loaded()
        
        if not self.products:
            return {
                "total_products": 0,
                "available_products": 0,
                "average_price": 0,
                "currencies": [],
                "total_tags": 0
            }
        
        available_count = sum(1 for p in self.products if p.available)
        prices = [p.price for p in self.products if p.price > 0]
        avg_price = sum(prices) / len(prices) if prices else 0
        currencies = list(set(p.currency for p in self.products))
        all_tags = set()
        for product in self.products:
            all_tags.update(product.tags)
        
        return {
            "total_products": len(self.products),
            "available_products": available_count,
            "average_price": round(avg_price, 2),
            "currencies": currencies,
            "total_tags": len(all_tags)
        } 