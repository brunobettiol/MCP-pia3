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
        """Initialize the product recommendation service optimized for eldercare queries"""
        self.products: List[Product] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._loaded = False
        
        # Eldercare-focused keyword mappings based on your question categories
        self.eldercare_keywords = {
            # SAFE Category - Home Safety & Fall Prevention
            'stairs_safety': [
                'stairs', 'stair', 'handrails', 'handrail', 'stair rail', 'stairway', 
                'steps', 'step', 'railing', 'banister', 'stair safety'
            ],
            'bathroom_safety': [
                'grab bars', 'grab bar', 'bathroom', 'shower', 'toilet', 'bath', 
                'shower seat', 'bath bench', 'toilet safety', 'raised toilet seat',
                'commode', 'shower chair', 'bath rail', 'safety rail', 'tub rail',
                'toilet rail', 'shower handle', 'bath handle', 'non-slip mat',
                'shower grab bar', 'bathroom safety'
            ],
            'lighting': [
                'lighting', 'lights', 'light', 'illumination', 'bright light', 
                'LED light', 'motion sensor light', 'night light', 'lamp',
                'flashlight', 'emergency light', 'pathway light', 'step light',
                'hallway light', 'stair light', 'adequate lighting'
            ],
            'fall_prevention': [
                'fall', 'falls', 'fall prevention', 'tripping hazards', 'trip', 
                'slip', 'balance', 'stability', 'safety', 'accident prevention',
                'injury prevention', 'fall risk', 'fall safety'
            ],
            'mobility_aids': [
                'cane', 'walker', 'wheelchair', 'rollator', 'mobility scooter',
                'walking stick', 'crutches', 'mobility aid', 'walking aid',
                'transport chair', 'knee walker', 'mobility device', 'mobility'
            ],
            'emergency_response': [
                'emergency', 'personal emergency response', 'medical alert',
                'emergency system', 'alert system', 'emergency button',
                'help button', 'panic button', 'emergency response'
            ],
            
            # HEALTHY Category - Health & Medical Management
            'medication_management': [
                'medication', 'pill', 'prescription', 'medicine', 'drug',
                'pill dispenser', 'medication organizer', 'pill box', 'pill reminder',
                'medication alarm', 'pill sorter', 'weekly pill organizer',
                'daily pill organizer', 'medication tracker', 'pill container',
                'medicine organizer', 'prescription organizer', 'medication box',
                'pill case', 'medicine box', 'medication dispenser', 'pill planner',
                'medication management', 'pill organization', 'medication system'
            ],
            'health_monitoring': [
                'blood pressure monitor', 'thermometer', 'pulse oximeter',
                'health monitor', 'medical equipment', 'health screening',
                'vital signs', 'blood pressure', 'temperature', 'pulse',
                'oxygen saturation', 'health check', 'medical device'
            ],
            'chronic_conditions': [
                'chronic', 'diabetes', 'arthritis', 'heart disease', 'COPD',
                'hypertension', 'chronic pain', 'chronic condition',
                'disease management', 'condition management'
            ],
            
            # Functional Ability - Daily Living Aids
            'bathing_assistance': [
                'bathing', 'bath', 'shower', 'shower seat', 'bath bench',
                'shower chair', 'bathing aid', 'shower assistance',
                'bath assistance', 'bathing independently'
            ],
            'dressing_aids': [
                'dressing', 'dress', 'undress', 'dressing aid', 'sock aid',
                'shoe horn', 'button hook', 'zipper pull', 'clothing aid',
                'dressing assistance', 'adaptive clothing'
            ],
            'transfer_aids': [
                'transfer', 'bed', 'chair', 'transfer aid', 'lift', 'lifting',
                'transfer board', 'transfer belt', 'gait belt', 'mobility transfer'
            ],
            'toilet_assistance': [
                'toilet', 'toileting', 'commode', 'raised toilet seat',
                'toilet safety', 'toilet aid', 'toilet assistance',
                'bathroom assistance', 'toileting aid'
            ],
            
            # General Daily Living
            'daily_living': [
                'reacher', 'grabber', 'jar opener', 'can opener', 'kitchen aid',
                'eating utensils', 'adaptive utensils', 'large grip', 'easy grip',
                'ergonomic', 'arthritis aid', 'daily living aid', 'independence'
            ],
            'comfort_support': [
                'cushion', 'pillow', 'support cushion', 'back support',
                'seat cushion', 'lumbar support', 'orthopedic pillow',
                'memory foam', 'gel cushion', 'pressure relief',
                'comfort pad', 'positioning aid'
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

    def _calculate_eldercare_relevance_score(self, query: str, product: Product) -> float:
        """Calculate eldercare relevance score based on domain-specific keywords"""
        query_lower = query.lower()
        product_text = f"{product.title} {self._strip_html(product.description or '')} {' '.join(product.tags)}".lower()
        
        total_score = 0.0
        
        # Check each eldercare category
        for category, keywords in self.eldercare_keywords.items():
            query_matches = sum(1 for keyword in keywords if keyword in query_lower)
            product_matches = sum(1 for keyword in keywords if keyword in product_text)
            
            if query_matches > 0 and product_matches > 0:
                # Score based on relevance strength
                category_score = min(query_matches * product_matches * 2.0, 10.0)
                total_score += category_score
        
        return total_score

    def _calculate_direct_keyword_score(self, query: str, product: Product) -> float:
        """Calculate direct keyword matching score"""
        query_lower = query.lower()
        score = 0.0
        
        # Split query into meaningful words (filter out very short words)
        query_words = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
        
        if not query_words:
            return 0.0
        
        # Check title for matches (highest weight)
        title_words = product.title.lower().split()
        title_matches = 0
        for word in query_words:
            if any(word in title_word for title_word in title_words):
                title_matches += 1
        
        if title_matches > 0:
            score += (title_matches / len(query_words)) * 8.0
        
        # Check tags for matches
        tag_matches = 0
        for tag in product.tags:
            tag_lower = tag.lower()
            for word in query_words:
                if word in tag_lower:
                    tag_matches += 1
        
        if tag_matches > 0:
            score += min(tag_matches / len(query_words), 1.0) * 6.0
        
        # Check description for matches
        if product.description:
            description_text = self._strip_html(product.description).lower()
            desc_matches = 0
            for word in query_words:
                if word in description_text:
                    desc_matches += 1
            
            if desc_matches > 0:
                score += (desc_matches / len(query_words)) * 3.0
        
        return score

    def _build_search_index(self):
        """Build TF-IDF index for all product content"""
        if not self.products:
            return
        
        # Combine title, description, tags for each product with emphasis on title and tags
        self.processed_content = []
        for product in self.products:
            # Emphasize title and tags more than description
            combined_text = f"{product.title} " * 5 + \
                           f"{' '.join(product.tags)} " * 3 + \
                           f"{self._strip_html(product.description or '')}"
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Handle small datasets
        if len(self.processed_content) < 2:
            print(f"Skipping TF-IDF for small dataset ({len(self.processed_content)} products)")
            self.tfidf_matrix = None
            self.tfidf_vectorizer = None
            return
        
        # Initialize TF-IDF vectorizer with reasonable parameters
        num_docs = len(self.processed_content)
        max_features = min(2000, num_docs * 50)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features if max_features > 0 else None,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm='l2'
        )
        
        # Build TF-IDF matrix
        if self.processed_content:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
                print(f"Built TF-IDF matrix for {len(self.processed_content)} products")
            except Exception as e:
                print(f"Error building TF-IDF matrix: {e}")
                self.tfidf_matrix = None
                self.tfidf_vectorizer = None

    async def recommend_best_product_with_score(self, query: str) -> Tuple[Optional[Product], float]:
        """Get the best product recommendation with optimized scoring"""
        await self._ensure_loaded()
        
        if not query or not self.products:
            return None, 0.0
        
        best_product = None
        best_score = 0.0
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores
            eldercare_score = self._calculate_eldercare_relevance_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with balanced weights
            if self.tfidf_matrix is not None:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.3) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.4)
            
            if combined_score > best_score:
                best_score = combined_score
                best_product = product
        
        return best_product, best_score

    async def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best product recommendation handle with reasonable threshold"""
        product, score = await self.recommend_best_product_with_score(query)
        
        # Much more reasonable threshold for eldercare queries
        min_relevance_score = 2.0  # Lowered from 5.0 to actually return results
        
        if product and score >= min_relevance_score:
            return product.handle
        
        return None

    async def get_recommendations(self, query: str, limit: int = 5) -> List[Product]:
        """Get product recommendations with optimized scoring"""
        await self._ensure_loaded()
        
        if not query or not self.products:
            return []
        
        scored_products = []
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores
            eldercare_score = self._calculate_eldercare_relevance_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with balanced weights
            if self.tfidf_matrix is not None:
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.4) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.5)
            
            # Much lower threshold to actually return recommendations
            if combined_score > 1.0:  # Lowered from 2.0
                scored_products.append((product, combined_score))
        
        # Sort by score and return top results
        scored_products.sort(key=lambda x: x[1], reverse=True)
        recommended_products = [product for product, score in scored_products[:limit]]
        
        return recommended_products

    async def search_products(self, query: str, limit: int = 10) -> List[Product]:
        """Search products with optimized scoring"""
        await self._ensure_loaded()
        
        if not query or not self.products:
            return []
        
        scored_products = []
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores
            eldercare_score = self._calculate_eldercare_relevance_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with balanced weights
            if self.tfidf_matrix is not None:
                combined_score = (eldercare_score * 0.3) + (direct_keyword_score * 0.4) + (tfidf_score * 0.3)
            else:
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.6)
            
            # Lower threshold for search to return more results
            if combined_score > 0.5:  # Lowered from 1.5
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