import json
import os
from typing import List, Dict, Optional, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.models.service import ServiceProvider, ServiceList, ServiceResponse
from app.core.config import settings


class ServiceService:
    def __init__(self):
        """Initialize the service service with intelligent keyword-based matching"""
        self.providers: List[ServiceProvider] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._load_services()
        self._build_search_index()
        
        # Domain-specific keyword mappings for better service matching
        self.service_type_keywords = {
            'home_modification': [
                'lighting', 'lights', 'grab bars', 'handrails', 'ramps', 'stair lift',
                'bathroom modification', 'shower modification', 'accessibility',
                'home safety', 'fall prevention', 'mobility access', 'door widening',
                'threshold ramps', 'safety rails', 'home adaptation'
            ],
            'in_home_care': [
                'caregiver', 'home care', 'personal care', 'companion care',
                'elderly care', 'senior care', 'home health aide', 'care assistant',
                'daily living assistance', 'meal preparation', 'medication reminders',
                'transportation assistance', 'housekeeping', 'respite care'
            ],
            'physical_therapy': [
                'physical therapy', 'mobility', 'balance', 'strength', 'rehabilitation',
                'exercise', 'movement', 'walking', 'gait training', 'fall prevention',
                'muscle strengthening', 'range of motion', 'therapeutic exercise',
                'recovery', 'injury rehabilitation', 'pain management'
            ],
            'occupational_therapy': [
                'occupational therapy', 'daily living skills', 'adaptive equipment',
                'home safety assessment', 'cognitive rehabilitation', 'fine motor skills',
                'activities of daily living', 'ADL training', 'equipment training',
                'home modification assessment', 'functional assessment'
            ],
            'medical_equipment': [
                'medical equipment', 'durable medical equipment', 'DME', 'mobility aids',
                'wheelchairs', 'walkers', 'hospital beds', 'oxygen equipment',
                'CPAP', 'medical supplies', 'prosthetics', 'orthotics'
            ],
            'elder_law': [
                'elder law', 'estate planning', 'wills', 'power of attorney',
                'guardianship', 'medicaid planning', 'legal documents', 'advance directives',
                'living will', 'trust', 'probate', 'elder abuse', 'legal advice'
            ],
            'financial_planning': [
                'financial planning', 'retirement planning', 'investment', 'insurance',
                'long term care insurance', 'medicare', 'social security', 'benefits',
                'financial advisor', 'wealth management', 'estate planning'
            ],
            'insurance': [
                'insurance', 'health insurance', 'medicare', 'medicaid', 'long term care',
                'disability insurance', 'life insurance', 'insurance broker',
                'insurance agent', 'coverage', 'benefits', 'claims'
            ],
            'palliative_care': [
                'palliative care', 'comfort care', 'pain management', 'symptom management',
                'quality of life', 'end of life care', 'hospice', 'terminal illness',
                'chronic illness', 'advanced illness', 'supportive care'
            ],
            'hospice': [
                'hospice', 'end of life care', 'terminal care', 'comfort care',
                'palliative care', 'bereavement', 'grief support', 'dying process',
                'final care', 'compassionate care', 'spiritual care'
            ],
            'grief_counseling': [
                'grief counseling', 'bereavement', 'loss', 'mourning', 'grief support',
                'counseling', 'therapy', 'emotional support', 'coping', 'healing',
                'grief recovery', 'support groups', 'mental health'
            ],
            'assisted_living': [
                'assisted living', 'senior living', 'retirement community', 'care facility',
                'independent living', 'memory care', 'senior housing', 'residential care',
                'adult care', 'senior community', 'care home'
            ],
            'nursing_home': [
                'nursing home', 'skilled nursing', 'long term care facility', 'care facility',
                'nursing facility', 'convalescent home', 'rehabilitation facility',
                'extended care', 'residential care', 'institutional care'
            ],
            'transportation': [
                'transportation', 'medical transportation', 'senior transportation',
                'wheelchair accessible', 'door to door', 'medical appointments',
                'non-emergency transport', 'mobility transport', 'accessible vehicle'
            ],
            'meal_service': [
                'meal delivery', 'meals on wheels', 'senior meals', 'nutrition',
                'food delivery', 'meal preparation', 'dietary services', 'nutrition counseling',
                'meal planning', 'senior nutrition', 'home delivered meals'
            ],
            'geriatric_medicine': [
                'geriatric medicine', 'geriatrician', 'senior health', 'elderly care',
                'age-related health', 'senior medical care', 'geriatric care',
                'aging', 'senior wellness', 'elderly health', 'geriatric assessment'
            ],
            'at_home_tech': [
                'technology', 'smart home', 'medical alert', 'emergency response',
                'monitoring system', 'telehealth', 'remote monitoring', 'safety technology',
                'home automation', 'assistive technology', 'digital health'
            ],
            'medication_management': [
                'medication management', 'pill organization', 'medication reminders',
                'prescription management', 'medication adherence', 'pill dispensing',
                'medication monitoring', 'pharmacy services', 'medication review',
                'drug interaction', 'medication safety', 'prescription delivery',
                'medication synchronization', 'pill packaging', 'medication counseling'
            ]
        }
    
    def _load_services(self):
        """Load all service providers from JSON files"""
        services_dir = os.path.join(settings.BASE_DIR, "data", "services")
        if not os.path.exists(services_dir):
            return
        
        for filename in os.listdir(services_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(services_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Handle the "providers" array structure
                        if 'providers' in data:
                            providers_data = data['providers']
                        else:
                            providers_data = data if isinstance(data, list) else [data]
                        
                        for item in providers_data:
                            provider = ServiceProvider(**item)
                            self.providers.append(provider)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Replace underscores with spaces for better word matching
        text = text.replace('_', ' ')
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _calculate_service_type_score(self, query: str, provider: ServiceProvider) -> float:
        """Calculate service type relevance score based on domain keywords and descriptions"""
        query_lower = query.lower()
        max_score = 0.0
        
        # Check each service type and its keywords
        for service_type, keywords in self.service_type_keywords.items():
            if provider.type == service_type or service_type in provider.types:
                # Calculate keyword match score for this service type
                keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
                if keyword_matches > 0:
                    # Also check if the provider's descriptions contain relevant keywords for this service type
                    description_text = f"{provider.description} {provider.detailedDescription}".lower()
                    description_keyword_matches = sum(1 for keyword in keywords if keyword in description_text)
                    
                    # Score based on both query matching and provider description relevance
                    query_relevance = (keyword_matches / len(keywords)) * 8  # Increased from 5
                    description_relevance = (description_keyword_matches / len(keywords)) * 3  # Reduced from 5
                    type_score = query_relevance + description_relevance
                    max_score = max(max_score, type_score)
                else:
                    # Even if no keyword matches, give some score for correct service type
                    # This helps legitimate services with generic descriptions
                    if self._is_query_relevant_to_service_type(query_lower, service_type):
                        max_score = max(max_score, 4.0)  # Increased base score for correct type
        
        # NEGATIVE SCORING: Penalize completely mismatched service types
        # If query is clearly about one domain but provider is in a completely different domain
        medication_keywords = ['medication', 'pill', 'prescription', 'drug', 'medicine']
        home_safety_keywords = ['lighting', 'lights', 'grab bars', 'handrails', 'safety modification']
        care_keywords = ['caregiver', 'personal care', 'home care', 'companion care']
        
        query_is_medication = any(keyword in query_lower for keyword in medication_keywords)
        query_is_home_safety = any(keyword in query_lower for keyword in home_safety_keywords)
        query_is_care = any(keyword in query_lower for keyword in care_keywords)
        
        # Apply penalties for mismatched types
        if query_is_medication and provider.type in ['insurance', 'elder_law', 'financial_planning']:
            max_score -= 10.0  # Heavy penalty
        elif query_is_home_safety and provider.type in ['insurance', 'elder_law', 'financial_planning', 'in_home_care']:
            max_score -= 8.0  # Heavy penalty
        elif query_is_care and provider.type in ['insurance', 'elder_law', 'financial_planning', 'home_modification']:
            max_score -= 6.0  # Moderate penalty
        
        return max_score
    
    def _is_query_relevant_to_service_type(self, query: str, service_type: str) -> bool:
        """Check if a query is relevant to a service type even without exact keyword matches"""
        home_modification_indicators = ['lighting', 'lights', 'grab bars', 'handrails', 'ramps', 'safety', 'modification', 'install', 'home improvement', 'improve']
        in_home_care_indicators = ['caregiver', 'care', 'assistance', 'help', 'support', 'companion']
        physical_therapy_indicators = ['therapy', 'rehabilitation', 'mobility', 'balance', 'exercise', 'movement']
        
        if service_type == 'home_modification':
            return any(indicator in query for indicator in home_modification_indicators)
        elif service_type == 'in_home_care':
            return any(indicator in query for indicator in in_home_care_indicators)
        elif service_type == 'physical_therapy':
            return any(indicator in query for indicator in physical_therapy_indicators)
        
        return False

    def _calculate_direct_keyword_score(self, query: str, provider: ServiceProvider) -> float:
        """Calculate direct keyword matching score with heavy emphasis on descriptions and context awareness"""
        query_lower = query.lower()
        score = 0.0
        
        # Check provider type directly (medium weight)
        provider_type_readable = provider.type.replace('_', ' ')
        if provider_type_readable in query_lower:
            score += 3.0
        
        # Check descriptions for direct matches (HIGHEST WEIGHT - these are always filled)
        description_text = f"{provider.description} {provider.detailedDescription}".lower()
        query_words = query_lower.split()
        
        # CONTEXT-AWARE WORD MATCHING: Only count matches that make semantic sense
        meaningful_matches = 0
        total_words = len(query_words)
        
        for word in query_words:
            if word in description_text:
                # Check if this is a meaningful match based on context
                if self._is_meaningful_match(word, query_lower, provider.type, description_text):
                    meaningful_matches += 1
        
        if meaningful_matches > 0:
            score += (meaningful_matches / total_words) * 8.0  # Very high weight for meaningful matches
        
        # Phrase matching in descriptions (only relevant phrases)
        relevant_phrases = self._get_relevant_phrases_for_query(query_lower)
        
        for phrase in relevant_phrases:
            if phrase in query_lower and phrase in description_text:
                score += 4.0  # High bonus for phrase matches in descriptions
        
        # Check name for matches (lower weight)
        if any(word in provider.name.lower() for word in query_words):
            score += 2.0
        
        return min(score, 15.0)  # Increased cap to allow for higher description scores
    
    def _is_meaningful_match(self, word: str, query: str, provider_type: str, description: str) -> bool:
        """Check if a word match is semantically meaningful given the context"""
        # Common words that can be misleading
        generic_words = ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        if word in generic_words:
            return False
        
        # Context-specific checks
        if word == 'management':
            # Only meaningful if it's in the right context
            if provider_type == 'medication_management':
                return True
            elif provider_type in ['insurance', 'financial_planning', 'elder_law']:
                # Check if it's about medication management specifically
                return 'medication' in description or 'pill' in description or 'prescription' in description
            return False
        
        if word == 'system':
            # Only meaningful for certain provider types
            if provider_type in ['at_home_tech', 'medication_management', 'home_modification']:
                return True
            elif provider_type in ['insurance', 'financial_planning', 'elder_law']:
                return False  # Insurance "systems" are not relevant to medication systems
            return True
        
        if word == 'establish':
            # This is often too generic
            return len(word) > 4  # Only count longer, more specific words
        
        # For other words, check if they're domain-specific
        medication_words = ['medication', 'pill', 'prescription', 'drug', 'medicine', 'dosage', 'pharmacy']
        if word in medication_words:
            return provider_type in ['medication_management', 'geriatric_medicine', 'in_home_care', 'at_home_tech']
        
        return True  # Default to meaningful for other words
    
    def _get_relevant_phrases_for_query(self, query: str) -> List[str]:
        """Get phrases that are relevant to the specific query"""
        all_phrases = [
            'home modification', 'lighting', 'grab bars', 'handrails', 'safety',
            'physical therapy', 'occupational therapy', 'medical equipment',
            'caregiver', 'home care', 'elder law', 'financial planning',
            'insurance', 'palliative care', 'hospice', 'grief counseling',
            'in-home care', 'assisted living', 'nursing home', 'transportation',
            'meal service', 'geriatric medicine', 'at-home tech', 'medication management'
        ]
        
        # Only return phrases that are actually relevant to the query
        relevant_phrases = []
        for phrase in all_phrases:
            if any(word in query for word in phrase.split()):
                relevant_phrases.append(phrase)
        
        return relevant_phrases

    def _build_search_index(self):
        """Build TF-IDF index for all service provider content with heavy emphasis on descriptions"""
        if not self.providers:
            return
        
        # Combine all text fields for each provider with HEAVY emphasis on descriptions
        self.processed_content = []
        for provider in self.providers:
            # Emphasize descriptions much more since they're always filled and most relevant
            service_type_text = provider.type.replace('_', ' ')
            combined_text = f"{provider.description} " * 15 + \
                           f"{provider.detailedDescription} " * 12 + \
                           f"{service_type_text} " * 8 + \
                           f"{provider.name} " * 5 + \
                           f"{' '.join(provider.types)} " * 3 + \
                           f"{' '.join(provider.services)} " * 2 + \
                           f"{' '.join(provider.specialties)} " * 2
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Initialize TF-IDF vectorizer with optimized parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,  # Smaller vocabulary for better focus
            stop_words='english',  # Remove common English stop words
            ngram_range=(1, 3),  # Include trigrams for better phrase matching
            min_df=1,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
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

    def get_all_providers(self) -> ServiceList:
        """Get all service providers"""
        return ServiceList(providers=self.providers, total=len(self.providers))

    def search_providers(self, query: str, limit: int = 10) -> ServiceList:
        """Search providers using hybrid scoring (service type + TF-IDF)"""
        if not query or not self.providers:
            return ServiceList(providers=[], total=0)
        
        scored_providers = []
        
        for i, provider in enumerate(self.providers):
            # Calculate multiple scores
            service_type_score = self._calculate_service_type_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with weights - MORE emphasis on direct keyword matching (descriptions)
            combined_score = (service_type_score * 0.3) + (direct_keyword_score * 0.5) + (tfidf_score * 0.2)
            
            if combined_score > 0.5:  # Minimum threshold
                scored_providers.append((provider, combined_score))
        
        # Sort by score and return top results
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        filtered_results = [provider for provider, score in scored_providers[:limit]]
        
        return ServiceList(providers=filtered_results, total=len(filtered_results))

    def get_providers_by_type(self, service_type: str) -> ServiceList:
        """Get providers filtered by service type"""
        filtered_providers = [provider for provider in self.providers 
                            if provider.type.lower() == service_type.lower()]
        return ServiceList(providers=filtered_providers, total=len(filtered_providers))

    def get_provider_by_id(self, provider_id: str) -> Optional[ServiceProvider]:
        """Get a specific provider by ID"""
        for provider in self.providers:
            if provider.id == provider_id:
                return provider
        return None

    def get_recommendations(self, query: str, limit: int = 5) -> ServiceList:
        """Get service provider recommendations based on query using hybrid scoring"""
        if not query or not self.providers:
            return ServiceList(providers=[], total=0)
        
        scored_providers = []
        
        for i, provider in enumerate(self.providers):
            # Calculate multiple scores
            service_type_score = self._calculate_service_type_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with weights favoring direct keyword matching (descriptions)
            combined_score = (service_type_score * 0.3) + (direct_keyword_score * 0.6) + (tfidf_score * 0.1)
            
            if combined_score > 4.0:  # MUCH higher threshold for recommendations
                scored_providers.append((provider, combined_score))
        
        # Sort by score and return top results
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        recommended_providers = [provider for provider, score in scored_providers[:limit]]
        
        return ServiceList(providers=recommended_providers, total=len(recommended_providers))

    def recommend_best_provider_with_score(self, query: str) -> Tuple[Optional[ServiceProvider], float]:
        """Get the best service provider recommendation with hybrid scoring"""
        if not query or not self.providers:
            return None, 0.0
        
        best_provider = None
        best_score = 0.0
        
        for i, provider in enumerate(self.providers):
            # Calculate multiple scores
            service_type_score = self._calculate_service_type_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with HEAVY emphasis on direct keyword matching (descriptions)
            combined_score = (service_type_score * 0.2) + (direct_keyword_score * 0.7) + (tfidf_score * 0.1)
            
            if combined_score > best_score:
                best_score = combined_score
                best_provider = provider
        
        return best_provider, best_score

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best service provider recommendation ID based on query with threshold"""
        provider, score = self.recommend_best_provider_with_score(query)
        
        # Set minimum relevance threshold - MUCH HIGHER for quality
        min_relevance_score = 5.0  # Require very strong service type and keyword matching
        
        if provider and score >= min_relevance_score:
            return provider.id
        
        return None

    def get_service_types(self) -> List[str]:
        """Get all unique service types"""
        types = set(provider.type for provider in self.providers)
        return sorted(list(types))

    def get_service_areas(self) -> List[str]:
        """Get all unique service areas"""
        areas = set()
        for provider in self.providers:
            areas.update(provider.serviceAreas)
        return sorted(list(areas))

    def get_statistics(self) -> Dict:
        """Get service provider statistics"""
        types = {}
        regions = {}
        
        for provider in self.providers:
            types[provider.type] = types.get(provider.type, 0) + 1
            for region in provider.serviceRegions:
                regions[region] = regions.get(region, 0) + 1
        
        return {
            "total_providers": len(self.providers),
            "service_types": types,
            "service_regions": regions,
            "featured_count": len([p for p in self.providers if p.featured])
        }

    def get_providers_by_region(self, region: str) -> ServiceList:
        """Get providers filtered by service region"""
        filtered_providers = [provider for provider in self.providers 
                            if region.lower() in [r.lower() for r in provider.serviceRegions]]
        return ServiceList(providers=filtered_providers, total=len(filtered_providers))

    def get_providers_by_area(self, area: str) -> ServiceList:
        """Get providers filtered by service area"""
        filtered_providers = [provider for provider in self.providers 
                            if area.lower() in [a.lower() for a in provider.serviceAreas]]
        return ServiceList(providers=filtered_providers, total=len(filtered_providers))

    def recommend_services(self, query: str, limit: int = 5) -> List[Tuple[ServiceProvider, float]]:
        """Recommend services based on query with improved scoring"""
        if not self.providers:
            return []
        
        scored_providers = []
        
        for provider in self.providers:
            # Calculate different score components
            service_type_score = self._calculate_service_type_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score (if available)
            tfidf_score = 0.0
            if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer is not None:
                try:
                    query_vector = self.tfidf_vectorizer.transform([query])
                    provider_index = self.providers.index(provider)
                    if provider_index < len(self.tfidf_matrix):
                        similarity = cosine_similarity(query_vector, self.tfidf_matrix[provider_index:provider_index+1])
                        tfidf_score = float(similarity[0][0]) * 10  # Scale up TF-IDF
                except:
                    tfidf_score = 0.0
            
            # Combine scores with weights
            total_score = (
                service_type_score * 0.4 +      # 40% weight to service type matching
                direct_keyword_score * 0.4 +    # 40% weight to direct keyword matching  
                tfidf_score * 0.2               # 20% weight to TF-IDF
            )
            
            # Debug logging for home_modification services
            if provider.type == 'home_modification':
                print(f"DEBUG - {provider.name}:")
                print(f"  Service Type Score: {service_type_score}")
                print(f"  Direct Keyword Score: {direct_keyword_score}")
                print(f"  TF-IDF Score: {tfidf_score}")
                print(f"  Total Score: {total_score}")
            
            # Apply minimum threshold
            if total_score >= 1.0:  # Temporarily lowered threshold
                scored_providers.append((provider, total_score))
        
        # Sort by score (descending) and return top results
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        return scored_providers[:limit] 