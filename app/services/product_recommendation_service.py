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
        """Initialize the product recommendation service optimized for the 35 specific eldercare questions"""
        self.products: List[Product] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._loaded = False
        
        # Manual mapping for the 35 specific questions to exact product categories/keywords
        self.question_product_mapping = {
            # SAFE Category Tasks (Questions 1-11)
            "handrails installed both sides stairs": ["handrail", "stair rail", "railing", "banister", "stair safety"],
            "install grab bars bathroom toilet shower": ["grab bar", "safety bar", "bathroom safety", "shower grab", "toilet grab"],
            "improve lighting areas hallways stairs": ["LED light", "motion sensor", "night light", "pathway light", "stair light", "hallway light"],
            "assess fall risks implement prevention strategies": ["fall prevention", "safety assessment", "fall alert", "balance aid"],
            "provide appropriate mobility aids cane walker wheelchair": ["cane", "walker", "wheelchair", "rollator", "mobility scooter", "walking stick"],
            "remove tripping hazards loose rugs clutter electrical cords": ["non-slip mat", "cord cover", "cable management", "rug gripper"],
            "ensure bedroom accessible without using stairs": ["bed rail", "bed assist", "bedroom safety", "transfer aid"],
            "evaluate neighborhood safety accessibility walking": ["reflective vest", "safety light", "walking aid", "visibility gear"],
            "post emergency numbers visibly near phones": ["emergency list", "phone labels", "large print", "contact organizer"],
            "install personal emergency response system": ["medical alert", "emergency button", "personal alarm", "help button"],
            "evaluate home care support needs consider professional help": ["care assessment", "monitoring device", "health tracker"],
            
            # Functional Ability Tasks (Questions 12-15)
            "ensure safe bathing practices provide assistance needed": ["shower chair", "bath bench", "shower head", "bath mat", "shower caddy"],
            "provide assistance dressing undressing necessary": ["dressing aid", "sock aid", "shoe horn", "button hook", "zipper pull"],
            "ensure safe movement bed chair without help": ["transfer board", "pivot disc", "bed rail", "chair cushion", "lift cushion"],
            "provide assistance toilet use needed": ["raised toilet seat", "toilet safety frame", "commode", "toilet paper aid"],
            
            # HEALTHY Category Tasks (Questions 1-10)
            "manage chronic medical conditions effectively": ["pill organizer", "medication tracker", "health monitor", "blood pressure monitor"],
            "organize manage daily medications": ["pill dispenser", "medication organizer", "pill box", "weekly organizer", "pill reminder"],
            "implement medication management system": ["automatic pill dispenser", "smart pill box", "medication alarm", "pill tracker"],
            "schedule regular checkups primary care specialists": ["appointment book", "calendar", "reminder system", "health planner"],
            "ensure regular balanced meals proper nutrition": ["meal planner", "portion control", "nutrition guide", "kitchen scale"],
            "assist meal preparation grocery shopping needed": ["kitchen aid", "jar opener", "can opener", "cutting board", "easy grip utensils"],
            "establish exercise routine regular physical activity": ["exercise equipment", "resistance band", "balance pad", "pedometer", "fitness tracker"],
            "monitor cognitive health address memory issues": ["memory aid", "brain games", "cognitive trainer", "reminder device"],
            "improve sleep quality address issues": ["sleep tracker", "white noise", "sleep mask", "comfortable pillow", "mattress pad"],
            "schedule regular dental vision hearing checkups": ["magnifying glass", "reading light", "dental care", "vision aid"],
            
            # Emotional Health Tasks (Questions 11-14)
            "address feelings depression hopelessness": ["mood tracker", "light therapy", "relaxation aid", "comfort item"],
            "encourage participation enjoyable activities": ["hobby supplies", "puzzle", "game", "craft kit", "entertainment"],
            "reduce feelings loneliness isolation": ["communication device", "video call", "social connector", "companion robot"],
            "ensure vaccinations up date": ["vaccination record", "health tracker", "medical organizer"],
            
            # PREPARED Category Tasks (Questions 1-10)
            "establish advance directives living will healthcare proxy": ["document organizer", "legal forms", "file system", "important papers"],
            "set durable power attorney finances": ["financial organizer", "document folder", "legal kit", "planning guide"],
            "create will trust": ["estate planning", "document storage", "legal organizer", "will kit"],
            "discuss end life care preferences family": ["communication aid", "planning guide", "discussion starter", "family planner"],
            "review update insurance coverage": ["insurance organizer", "policy folder", "coverage tracker", "benefits guide"],
            "develop financial plan potential long term care needs": ["financial planner", "budget tracker", "savings calculator", "planning software"],
            "consider living arrangement options future needs": ["housing guide", "community finder", "planning checklist", "decision aid"],
            "implement system managing bills financial matters": ["bill organizer", "checkbook", "payment tracker", "budget planner"],
            "organize important documents easy access": ["file organizer", "document storage", "filing system", "paper organizer"],
            "create communication plan family care decisions": ["family planner", "communication board", "decision tree", "planning guide"]
        }
        
        # Eldercare-focused keyword mappings optimized for the 35 specific questions
        self.eldercare_keywords = {
            # SAFE Category - Home Safety & Fall Prevention
            'handrails_stair_safety': [
                'handrail', 'handrails', 'stair rail', 'railing', 'banister', 'stair safety',
                'grab rail', 'safety rail', 'stair support', 'stairway rail', 'both sides'
            ],
            'grab_bars_bathroom': [
                'grab bars', 'grab bar', 'safety bars', 'bathroom safety', 'shower grab bars',
                'toilet grab bars', 'bath safety', 'shower safety', 'bathroom grab', 'safety rail'
            ],
            'lighting_improvement': [
                'LED light', 'motion sensor light', 'night light', 'pathway light', 'stair light',
                'hallway light', 'bright light', 'automatic light', 'sensor light', 'lighting'
            ],
            'fall_prevention_products': [
                'fall prevention', 'fall alert', 'balance aid', 'stability aid', 'fall detector',
                'safety monitor', 'fall alarm', 'balance trainer', 'stability trainer'
            ],
            'mobility_aids_products': [
                'cane', 'walker', 'wheelchair', 'rollator', 'mobility scooter', 'walking stick',
                'walking aid', 'mobility aid', 'walking frame', 'transport chair', 'knee walker'
            ],
            'hazard_removal_products': [
                'non-slip mat', 'anti-slip', 'cord cover', 'cable management', 'rug gripper',
                'rug pad', 'safety mat', 'floor safety', 'slip resistant'
            ],
            'bedroom_accessibility': [
                'bed rail', 'bed assist', 'bedroom safety', 'transfer aid', 'bed support',
                'bed handle', 'bed mobility', 'bedroom aid', 'bed safety'
            ],
            'outdoor_safety': [
                'reflective vest', 'safety light', 'visibility gear', 'walking safety',
                'outdoor safety', 'reflective gear', 'safety vest', 'walking light'
            ],
            'emergency_communication': [
                'emergency list', 'phone labels', 'large print', 'contact organizer',
                'emergency contacts', 'phone aid', 'communication aid'
            ],
            'emergency_response_products': [
                'medical alert', 'emergency button', 'personal alarm', 'help button',
                'medical alarm', 'emergency system', 'alert system', 'panic button'
            ],
            'care_monitoring': [
                'care monitor', 'health tracker', 'monitoring device', 'wellness tracker',
                'activity monitor', 'safety monitor', 'care alert'
            ],
            
            # Functional Ability Products
            'bathing_products': [
                'shower chair', 'bath bench', 'shower seat', 'bath seat', 'shower stool',
                'bath stool', 'shower head', 'bath mat', 'shower caddy', 'bath aid'
            ],
            'dressing_aids': [
                'dressing aid', 'sock aid', 'shoe horn', 'button hook', 'zipper pull',
                'dressing stick', 'clothing aid', 'adaptive clothing', 'easy dress'
            ],
            'transfer_mobility_products': [
                'transfer board', 'pivot disc', 'transfer disc', 'sliding board',
                'transfer belt', 'gait belt', 'mobility transfer', 'transfer aid'
            ],
            'toilet_assistance_products': [
                'raised toilet seat', 'toilet safety frame', 'commode', 'toilet rail',
                'toilet support', 'toilet aid', 'toilet paper aid', 'bathroom aid'
            ],
            
            # HEALTHY Category Products
            'chronic_condition_management': [
                'pill organizer', 'medication tracker', 'health monitor', 'blood pressure monitor',
                'glucose monitor', 'pulse oximeter', 'thermometer', 'health tracking'
            ],
            'medication_organization': [
                'pill dispenser', 'medication organizer', 'pill box', 'weekly organizer',
                'daily organizer', 'pill reminder', 'medication box', 'pill container'
            ],
            'medication_systems': [
                'automatic pill dispenser', 'smart pill box', 'medication alarm',
                'pill tracker', 'medication system', 'pill management', 'med dispenser'
            ],
            'health_scheduling': [
                'appointment book', 'calendar', 'reminder system', 'health planner',
                'medical calendar', 'appointment planner', 'health organizer'
            ],
            'nutrition_products': [
                'meal planner', 'portion control', 'nutrition guide', 'kitchen scale',
                'measuring cups', 'portion plates', 'diet tracker', 'nutrition tracker'
            ],
            'kitchen_aids': [
                'jar opener', 'can opener', 'bottle opener', 'easy grip utensils',
                'adaptive utensils', 'kitchen aid', 'ergonomic utensils', 'kitchen tools'
            ],
            'exercise_products': [
                'exercise equipment', 'resistance band', 'balance pad', 'pedometer',
                'fitness tracker', 'exercise bike', 'yoga mat', 'balance trainer'
            ],
            'cognitive_aids': [
                'memory aid', 'brain games', 'cognitive trainer', 'reminder device',
                'memory book', 'cognitive games', 'brain training', 'memory support'
            ],
            'sleep_products': [
                'sleep tracker', 'white noise', 'sound machine', 'sleep mask',
                'comfortable pillow', 'mattress pad', 'sleep aid', 'bedtime comfort'
            ],
            'health_screening_aids': [
                'magnifying glass', 'reading light', 'dental care', 'vision aid',
                'hearing aid', 'reading glasses', 'magnifier', 'vision support'
            ],
            
            # Emotional Health Products
            'mood_support': [
                'mood tracker', 'light therapy', 'SAD light', 'therapy light',
                'relaxation aid', 'comfort item', 'stress relief', 'mood light'
            ],
            'activity_engagement': [
                'hobby supplies', 'puzzle', 'jigsaw puzzle', 'game', 'board game',
                'craft kit', 'entertainment', 'activity book', 'brain games'
            ],
            'social_connection': [
                'communication device', 'video call', 'tablet', 'social connector',
                'companion robot', 'pet robot', 'communication aid', 'video phone'
            ],
            'health_tracking': [
                'vaccination record', 'health tracker', 'medical organizer',
                'health journal', 'medical record', 'health log'
            ],
            
            # PREPARED Category Products
            'document_organization': [
                'document organizer', 'legal forms', 'file system', 'important papers',
                'filing cabinet', 'document folder', 'paper organizer', 'file box'
            ],
            'financial_organization': [
                'financial organizer', 'document folder', 'legal kit', 'planning guide',
                'financial planner', 'budget tracker', 'money organizer'
            ],
            'estate_planning_products': [
                'estate planning', 'document storage', 'legal organizer', 'will kit',
                'estate kit', 'legal forms', 'planning documents'
            ],
            'communication_planning': [
                'communication aid', 'planning guide', 'discussion starter', 'family planner',
                'conversation starter', 'planning book', 'communication board'
            ],
            'insurance_organization': [
                'insurance organizer', 'policy folder', 'coverage tracker', 'benefits guide',
                'insurance folder', 'policy organizer', 'benefits tracker'
            ],
            'financial_planning_tools': [
                'financial planner', 'budget tracker', 'savings calculator', 'planning software',
                'budget book', 'financial calculator', 'retirement planner'
            ],
            'housing_planning': [
                'housing guide', 'community finder', 'planning checklist', 'decision aid',
                'housing planner', 'living guide', 'senior living guide'
            ],
            'bill_management': [
                'bill organizer', 'checkbook', 'payment tracker', 'budget planner',
                'bill tracker', 'expense tracker', 'money management', 'bill folder'
            ],
            'file_organization': [
                'file organizer', 'document storage', 'filing system', 'paper organizer',
                'file box', 'document box', 'storage system', 'organization system'
            ],
            'family_planning_tools': [
                'family planner', 'communication board', 'decision tree', 'planning guide',
                'family organizer', 'care planner', 'family calendar'
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

    def _calculate_exact_question_match_score(self, query: str, product: Product) -> float:
        """Calculate exact match score for the 35 specific questions"""
        query_lower = self._preprocess_text(query)
        product_text = f"{product.title} {self._strip_html(product.description or '')} {' '.join(product.tags)}".lower()
        
        # Check for direct question mapping first
        for question_key, product_keywords in self.question_product_mapping.items():
            # Calculate similarity between query and question key
            question_words = set(question_key.split())
            query_words = set(query_lower.split())
            
            # Calculate overlap
            overlap = len(question_words.intersection(query_words))
            if overlap > 0:
                similarity = overlap / max(len(question_words), len(query_words))
                if similarity > 0.3:  # Threshold for considering it a match
                    # Check if product contains any of the mapped keywords
                    keyword_matches = sum(1 for keyword in product_keywords if keyword.lower() in product_text)
                    if keyword_matches > 0:
                        return 70.0 + (similarity * 30.0) + (keyword_matches * 5.0)  # Score 70-100+
        
        return 0.0

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
                category_score = min(query_matches * product_matches * 4.0, 20.0)
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
            score += (title_matches / len(query_words)) * 20.0
        
        # Check tags for matches
        tag_matches = 0
        for tag in product.tags:
            tag_lower = tag.lower()
            for word in query_words:
                if word in tag_lower:
                    tag_matches += 1
        
        if tag_matches > 0:
            score += min(tag_matches / len(query_words), 1.0) * 15.0
        
        # Check description for matches
        if product.description:
            description_text = self._strip_html(product.description).lower()
            desc_matches = 0
            for word in query_words:
                if word in description_text:
                    desc_matches += 1
            
            if desc_matches > 0:
                score += (desc_matches / len(query_words)) * 10.0
        
        return score

    def _build_search_index(self):
        """Build TF-IDF index for all product content"""
        if not self.products:
            return
        
        # Combine title, description, tags for each product with emphasis on title and tags
        self.processed_content = []
        for product in self.products:
            # Emphasize title and tags more than description
            combined_text = f"{product.title} " * 8 + \
                           f"{' '.join(product.tags)} " * 6 + \
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
        """Get the best product recommendation with optimized scoring for the 35 specific questions"""
        await self._ensure_loaded()
        
        if not query or not self.products:
            return None, 0.0
        
        best_product = None
        best_score = 0.0
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores with priority on exact question matching
            exact_question_score = self._calculate_exact_question_match_score(query, product)
            eldercare_score = self._calculate_eldercare_relevance_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 15  # Scale to 0-15
            
            # Combined score with heavy emphasis on exact question matching
            if exact_question_score > 0:
                # If we have an exact question match, prioritize it heavily
                combined_score = exact_question_score + (eldercare_score * 0.2) + (direct_keyword_score * 0.1)
            elif self.tfidf_matrix is not None and tfidf_score > 1.0:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.3) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.4)
            
            if combined_score > best_score:
                best_score = combined_score
                best_product = product
        
        return best_product, best_score

    async def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best product recommendation handle with reasonable threshold for the 35 specific questions"""
        product, score = await self.recommend_best_product_with_score(query)
        
        # Higher threshold due to exact question matching
        min_relevance_score = 8.0
        
        if product and score >= min_relevance_score:
            return product.handle
        
        return None

    async def get_recommendations(self, query: str, limit: int = 5) -> List[Product]:
        """Get product recommendations with optimized scoring for the 35 specific questions"""
        await self._ensure_loaded()
        
        if not query or not self.products:
            return []
        
        scored_products = []
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores with priority on exact question matching
            exact_question_score = self._calculate_exact_question_match_score(query, product)
            eldercare_score = self._calculate_eldercare_relevance_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 15  # Scale to 0-15
            
            # Combined score with heavy emphasis on exact question matching
            if exact_question_score > 0:
                # If we have an exact question match, prioritize it heavily
                combined_score = exact_question_score + (eldercare_score * 0.2) + (direct_keyword_score * 0.1)
            elif self.tfidf_matrix is not None and tfidf_score > 0.8:
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.4) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.5)
            
            # Lower threshold to return recommendations
            if combined_score > 3.0:
                scored_products.append((product, combined_score))
        
        # Sort by score and return top results
        scored_products.sort(key=lambda x: x[1], reverse=True)
        recommended_products = [product for product, score in scored_products[:limit]]
        
        return recommended_products

    async def search_products(self, query: str, limit: int = 10) -> List[Product]:
        """Search products with optimized scoring for the 35 specific questions"""
        await self._ensure_loaded()
        
        if not query or not self.products:
            return []
        
        scored_products = []
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores with priority on exact question matching
            exact_question_score = self._calculate_exact_question_match_score(query, product)
            eldercare_score = self._calculate_eldercare_relevance_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 15  # Scale to 0-15
            
            # Combined score with heavy emphasis on exact question matching
            if exact_question_score > 0:
                # If we have an exact question match, prioritize it heavily
                combined_score = exact_question_score + (eldercare_score * 0.3) + (direct_keyword_score * 0.2)
            elif self.tfidf_matrix is not None and tfidf_score > 0.5:
                combined_score = (eldercare_score * 0.3) + (direct_keyword_score * 0.4) + (tfidf_score * 0.3)
            else:
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.6)
            
            # Lower threshold for search to return more results
            if combined_score > 1.5:
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