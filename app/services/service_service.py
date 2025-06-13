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
        
        # Domain-specific keyword mappings for better service type matching
        self.service_type_keywords = {
            'home_modification': [
                'lighting', 'lights', 'handrails', 'grab bars', 'ramps', 'stair lift', 
                'bathroom modification', 'kitchen modification', 'accessibility', 
                'home safety', 'install', 'installation', 'modification', 'renovation',
                'safety equipment', 'home improvement', 'barrier removal', 'doorway widening',
                'flooring', 'non-slip', 'threshold', 'lever handles', 'shower seat',
                'walk-in tub', 'comfort height toilet', 'motion sensor lights'
            ],
            'in_home_care': [
                'caregiver', 'companion', 'personal care', 'bathing', 'dressing',
                'meal preparation', 'medication reminder', 'housekeeping', 'transportation',
                'companionship', 'assistance', 'daily living', 'ADL', 'home care',
                'care services', 'elderly care', 'senior care'
            ],
            'physical_therapy': [
                'physical therapy', 'PT', 'mobility', 'strength', 'balance', 'exercise',
                'rehabilitation', 'recovery', 'movement', 'walking', 'gait training',
                'fall prevention', 'muscle strength', 'range of motion', 'therapy'
            ],
            'occupational_therapy': [
                'occupational therapy', 'OT', 'daily activities', 'adaptive equipment',
                'cognitive training', 'memory', 'safety assessment', 'home evaluation',
                'functional assessment', 'independence', 'life skills'
            ],
            'medical_equipment': [
                'wheelchair', 'walker', 'cane', 'hospital bed', 'oxygen', 'CPAP',
                'medical supplies', 'mobility equipment', 'durable medical equipment',
                'DME', 'lift chair', 'scooter', 'commode', 'shower chair'
            ],
            'geriatric_medicine': [
                'doctor', 'physician', 'medical care', 'health', 'medication management',
                'chronic conditions', 'geriatrician', 'primary care', 'medical',
                'healthcare', 'treatment', 'diagnosis', 'prescription'
            ],
            'financial_advisor': [
                'financial planning', 'retirement', 'investment', 'insurance planning',
                'estate planning', 'financial advice', 'money management', 'savings',
                'financial security', 'wealth management'
            ],
            'elder_law': [
                'legal', 'attorney', 'lawyer', 'estate planning', 'will', 'trust',
                'power of attorney', 'guardianship', 'medicaid planning', 'probate',
                'legal advice', 'elder law', 'legal documents'
            ],
            'insurance': [
                'insurance', 'medicare', 'medicaid', 'health insurance', 'long term care insurance',
                'life insurance', 'coverage', 'benefits', 'claims', 'policy'
            ],
            'transportation': [
                'transportation', 'rides', 'medical transport', 'appointments',
                'driving', 'shuttle', 'mobility transport', 'wheelchair transport'
            ],
            'meal_service': [
                'meals', 'nutrition', 'food delivery', 'meal planning', 'cooking',
                'dietary', 'nutritional support', 'meal preparation', 'food services'
            ],
            'grief_counseling': [
                'grief', 'counseling', 'therapy', 'bereavement', 'loss', 'emotional support',
                'mental health', 'counselor', 'therapist', 'support groups'
            ],
            'palliative_care': [
                'palliative', 'hospice', 'end of life', 'comfort care', 'pain management',
                'symptom management', 'terminal care', 'quality of life'
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
                    query_relevance = (keyword_matches / len(keywords)) * 5
                    description_relevance = (description_keyword_matches / len(keywords)) * 5
                    type_score = query_relevance + description_relevance
                    max_score = max(max_score, type_score)
        
        return max_score

    def _calculate_direct_keyword_score(self, query: str, provider: ServiceProvider) -> float:
        """Calculate direct keyword matching score with heavy emphasis on descriptions"""
        query_lower = query.lower()
        score = 0.0
        
        # Check provider type directly (medium weight)
        provider_type_readable = provider.type.replace('_', ' ')
        if provider_type_readable in query_lower:
            score += 3.0
        
        # Check descriptions for direct matches (HIGHEST WEIGHT - these are always filled)
        description_text = f"{provider.description} {provider.detailedDescription}".lower()
        query_words = query_lower.split()
        
        # Word-by-word matching in descriptions
        description_matches = sum(1 for word in query_words if word in description_text)
        if description_matches > 0:
            score += (description_matches / len(query_words)) * 8.0  # Very high weight
        
        # Phrase matching in descriptions
        key_phrases = [
            'home modification', 'lighting', 'grab bars', 'handrails', 'safety',
            'physical therapy', 'occupational therapy', 'medical equipment',
            'caregiver', 'home care', 'elder law', 'financial planning',
            'insurance', 'palliative care', 'hospice', 'grief counseling',
            'in-home care', 'assisted living', 'nursing home', 'transportation',
            'meal service', 'geriatric medicine', 'at-home tech'
        ]
        
        for phrase in key_phrases:
            if phrase in query_lower and phrase in description_text:
                score += 4.0  # High bonus for phrase matches in descriptions
        
        # Check name for matches (lower weight)
        if any(word in provider.name.lower() for word in query_words):
            score += 2.0
        
        return min(score, 15.0)  # Increased cap to allow for higher description scores

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
            
            if combined_score > 1.0:  # Higher threshold for recommendations
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
        
        # Set minimum relevance threshold - higher for quality
        min_relevance_score = 2.0  # Require meaningful service type or keyword matching
        
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