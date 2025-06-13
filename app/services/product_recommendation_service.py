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
        """Initialize the product recommendation service with intelligent keyword-based matching"""
        self.products: List[Product] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._loaded = False
        
        # Domain-specific keyword mappings for better product matching
        self.product_keywords = {
            'mobility': [
                'walker', 'wheelchair', 'cane', 'rollator', 'mobility scooter',
                'walking stick', 'crutches', 'mobility aid', 'walking aid',
                'transport chair', 'knee walker', 'mobility device'
            ],
            'bathroom_safety': [
                'grab bars', 'shower seat', 'bath bench', 'toilet safety',
                'raised toilet seat', 'commode', 'shower chair', 'bath rail',
                'safety rail', 'bathroom safety', 'shower grab bar', 'tub rail',
                'toilet rail', 'shower handle', 'bath handle', 'non-slip mat'
            ],
            'home_safety': [
                'handrails', 'stair rail', 'ramp', 'threshold ramp', 'door ramp',
                'safety light', 'motion sensor light', 'night light', 'LED light',
                'lighting', 'lights', 'illumination', 'bright light', 'lamp',
                'flashlight', 'emergency light', 'pathway light', 'step light'
            ],
            'daily_living': [
                'reacher', 'grabber', 'dressing aid', 'sock aid', 'shoe horn',
                'button hook', 'zipper pull', 'jar opener', 'can opener',
                'kitchen aid', 'eating utensils', 'adaptive utensils',
                'large grip', 'easy grip', 'ergonomic', 'arthritis aid'
            ],
            'medical_equipment': [
                'blood pressure monitor', 'thermometer', 'pulse oximeter',
                'nebulizer', 'CPAP', 'oxygen concentrator', 'hospital bed',
                'medical alert', 'pill dispenser', 'medication organizer',
                'first aid', 'medical supplies', 'health monitor'
            ],
            'comfort': [
                'cushion', 'pillow', 'support cushion', 'back support',
                'seat cushion', 'lumbar support', 'orthopedic pillow',
                'memory foam', 'gel cushion', 'pressure relief',
                'comfort pad', 'positioning aid'
            ],
            'exercise': [
                'exercise equipment', 'resistance band', 'therapy ball',
                'balance pad', 'yoga mat', 'stretching aid', 'fitness',
                'physical therapy', 'rehabilitation', 'strength training',
                'balance training', 'flexibility', 'workout'
            ]
        }
    
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

    def _calculate_product_category_score(self, query: str, product: Product) -> float:
        """Calculate product category relevance score based on domain keywords"""
        query_lower = query.lower()
        max_score = 0.0
        
        # Check each product category and its keywords
        for category, keywords in self.product_keywords.items():
            # Calculate keyword match score for this category
            keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
            if keyword_matches > 0:
                # Check if product matches this category based on title, description, or tags
                product_text = f"{product.title} {self._strip_html(product.description or '')} {' '.join(product.tags)}".lower()
                category_matches = sum(1 for keyword in keywords if keyword in product_text)
                
                if category_matches > 0:
                    # Score based on both query and product matching this category
                    category_score = ((keyword_matches / len(keywords)) * (category_matches / len(keywords))) * 10
                    max_score = max(max_score, category_score)
        
        return max_score

    def _calculate_direct_keyword_score(self, query: str, product: Product) -> float:
        """Calculate direct keyword matching score"""
        query_lower = query.lower()
        score = 0.0
        
        # Check title for direct matches (highest weight)
        title_words = product.title.lower().split()
        query_words = query_lower.split()
        title_matches = sum(1 for word in query_words if any(word in title_word for title_word in title_words))
        if title_matches > 0:
            score += (title_matches / len(query_words)) * 6.0
        
        # Check tags for matches
        for tag in product.tags:
            if tag.lower() in query_lower:
                score += 3.0
        
        # Check description for matches
        if product.description:
            description_text = self._strip_html(product.description).lower()
            description_matches = sum(1 for word in query_words if word in description_text)
            if description_matches > 0:
                score += (description_matches / len(query_words)) * 2.0
        
        # Key phrases that should boost relevance
        key_phrases = [
            'grab bars', 'handrails', 'lighting', 'lights', 'walker', 'wheelchair',
            'safety', 'mobility', 'bathroom', 'shower', 'toilet', 'medical',
            'exercise', 'therapy', 'support', 'aid', 'assistance'
        ]
        
        product_text = f"{product.title} {self._strip_html(product.description or '')} {' '.join(product.tags)}".lower()
        for phrase in key_phrases:
            if phrase in query_lower and phrase in product_text:
                score += 2.0
        
        return min(score, 10.0)  # Cap at 10

    def _build_search_index(self):
        """Build TF-IDF index for all product content with title and tags emphasis"""
        if not self.products:
            return
        
        # Combine title, description, tags for each product with emphasis on title and tags
        self.processed_content = []
        for product in self.products:
            # Emphasize title and tags much more than description
            combined_text = f"{product.title} " * 8 + \
                           f"{' '.join(product.tags)} " * 5 + \
                           f"{self._strip_html(product.description or '')}"
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Handle small datasets - skip TF-IDF if too few products
        if len(self.processed_content) < 2:
            print(f"Skipping TF-IDF for small dataset ({len(self.processed_content)} products)")
            self.tfidf_matrix = None
            self.tfidf_vectorizer = None
            return
        
        # Initialize TF-IDF vectorizer with optimized parameters
        # Adjust parameters based on number of documents
        num_docs = len(self.processed_content)
        max_features = min(1000, num_docs * 20)  # Even smaller vocabulary for small datasets
        min_df = 1  # Always include words that appear at least once
        max_df = min(0.95, max(0.5, 1.0 - (2.0 / num_docs)))  # Ensure max_df > min_df and reasonable
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features if max_features > 0 else None,
            stop_words=None,  # Don't remove stop words for small datasets
            ngram_range=(1, 2),  # Simpler ngrams for small datasets
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,  # Apply sublinear tf scaling
            norm='l2'  # Normalize vectors
        )
        
        # Build TF-IDF matrix
        if self.processed_content:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
                print(f"Built TF-IDF matrix for {len(self.processed_content)} products")
            except Exception as e:
                print(f"Error building TF-IDF matrix: {e}")
                # Fallback - disable TF-IDF for small datasets
                self.tfidf_matrix = None
                self.tfidf_vectorizer = None

    async def recommend_best_product_with_score(self, query: str) -> Tuple[Optional[Product], float]:
        """Get the best product recommendation with hybrid scoring"""
        await self._ensure_loaded()
        
        if not query or not self.products:
            return None, 0.0
        
        best_product = None
        best_score = 0.0
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores
            category_score = self._calculate_product_category_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score - adjust weights based on whether TF-IDF is available
            if self.tfidf_matrix is not None:
                # Normal scoring with TF-IDF
                combined_score = (category_score * 0.5) + (direct_keyword_score * 0.4) + (tfidf_score * 0.1)
            else:
                # Fallback scoring without TF-IDF
                combined_score = (category_score * 0.6) + (direct_keyword_score * 0.4)
            
            if combined_score > best_score:
                best_score = combined_score
                best_product = product
        
        return best_product, best_score

    async def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best product recommendation handle based on query with threshold"""
        product, score = await self.recommend_best_product_with_score(query)
        
        # Set minimum relevance threshold - very low for better coverage
        min_relevance_score = 0.3  # Very permissive threshold for product matching
        
        if product and score >= min_relevance_score:
            return product.handle
        
        return None

    async def get_recommendations(self, query: str, limit: int = 5) -> List[Product]:
        """Get product recommendations based on query using hybrid scoring"""
        await self._ensure_loaded()
        
        if not query or not self.products:
            return []
        
        scored_products = []
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores
            category_score = self._calculate_product_category_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score - adjust weights based on whether TF-IDF is available
            if self.tfidf_matrix is not None:
                # Normal scoring with TF-IDF
                combined_score = (category_score * 0.4) + (direct_keyword_score * 0.4) + (tfidf_score * 0.2)
            else:
                # Fallback scoring without TF-IDF
                combined_score = (category_score * 0.5) + (direct_keyword_score * 0.5)
            
            if combined_score > 0.5:  # Lowered threshold for recommendations
                scored_products.append((product, combined_score))
        
        # Sort by score and return top results
        scored_products.sort(key=lambda x: x[1], reverse=True)
        recommended_products = [product for product, score in scored_products[:limit]]
        
        return recommended_products

    async def search_products(self, query: str, limit: int = 10) -> List[Product]:
        """Search products using hybrid scoring"""
        await self._ensure_loaded()
        
        if not query or not self.products:
            return []
        
        scored_products = []
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores
            category_score = self._calculate_product_category_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score - adjust weights based on whether TF-IDF is available
            if self.tfidf_matrix is not None:
                # Normal scoring with TF-IDF
                combined_score = (category_score * 0.3) + (direct_keyword_score * 0.4) + (tfidf_score * 0.3)
            else:
                # Fallback scoring without TF-IDF
                combined_score = (category_score * 0.4) + (direct_keyword_score * 0.6)
            
            if combined_score > 0.3:  # Lower threshold for search
                scored_products.append((product, combined_score))
        
        # Sort by score and return top results
        scored_products.sort(key=lambda x: x[1], reverse=True)
        filtered_results = [product for product, score in scored_products[:limit]]
        
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