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
        """Initialize the service service optimized for eldercare queries"""
        self.providers: List[ServiceProvider] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._load_services()
        self._build_search_index()
        
        # Eldercare-focused keyword mappings based on your question categories
        self.eldercare_service_keywords = {
            # SAFE Category - Home Safety & Modifications
            'home_modification': [
                'home modification', 'home safety', 'safety modification', 'accessibility',
                'grab bars', 'handrails', 'ramps', 'stair rail', 'bathroom modification',
                'lighting installation', 'safety equipment', 'fall prevention',
                'home improvement', 'accessibility modification', 'safety installation',
                'grab bar installation', 'handrail installation', 'lighting upgrade',
                'bathroom safety', 'stair safety', 'home accessibility'
            ],
            'lighting_services': [
                'lighting', 'lighting installation', 'electrical', 'LED installation',
                'motion sensor lights', 'night lights', 'pathway lighting',
                'stair lighting', 'hallway lighting', 'emergency lighting',
                'bright lighting', 'adequate lighting', 'lighting upgrade',
                'electrical services', 'lighting contractor'
            ],
            'handyman_services': [
                'handyman', 'home repair', 'maintenance', 'installation services',
                'home services', 'repair services', 'general contractor',
                'home maintenance', 'property maintenance', 'fix', 'install',
                'home improvement', 'contractor', 'skilled trades'
            ],
            
            # HEALTHY Category - Health & Medical Services
            'home_healthcare': [
                'home healthcare', 'home health', 'nursing services', 'medical care',
                'healthcare services', 'in-home care', 'skilled nursing',
                'medical services', 'health services', 'nursing care',
                'home nursing', 'medical assistance', 'healthcare provider',
                'health aide', 'medical support'
            ],
            'medication_services': [
                'medication management', 'pharmacy services', 'prescription delivery',
                'medication delivery', 'pill organization', 'medication reminder',
                'pharmaceutical services', 'medication assistance', 'drug management',
                'prescription services', 'medication support', 'pharmacy delivery'
            ],
            'physical_therapy': [
                'physical therapy', 'occupational therapy', 'rehabilitation',
                'therapy services', 'PT', 'OT', 'therapeutic services',
                'mobility therapy', 'exercise therapy', 'rehabilitation services',
                'therapeutic exercise', 'recovery services', 'therapy'
            ],
            'mental_health_services': [
                'mental health', 'counseling', 'therapy', 'psychological services',
                'mental health services', 'behavioral health', 'psychiatric services',
                'emotional support', 'mental wellness', 'counseling services',
                'psychotherapy', 'mental health support'
            ],
            
            # PREPARED Category - Planning & Legal Services
            'legal_services': [
                'legal services', 'attorney', 'lawyer', 'legal advice',
                'estate planning', 'will preparation', 'power of attorney',
                'advance directives', 'legal documents', 'elder law',
                'estate attorney', 'legal planning', 'legal assistance',
                'legal counsel', 'legal consultation'
            ],
            'financial_services': [
                'financial planning', 'financial advisor', 'financial services',
                'insurance services', 'retirement planning', 'financial consultation',
                'financial assistance', 'insurance agent', 'financial planner',
                'investment services', 'financial counseling', 'benefits assistance'
            ],
            'care_coordination': [
                'care coordination', 'care management', 'case management',
                'care planning', 'geriatric care management', 'care services',
                'elder care coordination', 'care navigator', 'care consultant',
                'care manager', 'senior care coordination'
            ],
            
            # Caregiver Support Services
            'caregiver_support': [
                'caregiver support', 'respite care', 'caregiver services',
                'family support', 'caregiver assistance', 'caregiver relief',
                'support services', 'caregiver resources', 'respite services',
                'caregiver education', 'caregiver training', 'support groups'
            ],
            
            # Daily Living Support
            'personal_care': [
                'personal care', 'companion care', 'caregiving', 'home care',
                'personal assistance', 'daily living assistance', 'care services',
                'companion services', 'personal care services', 'home companion',
                'care aide', 'personal care aide', 'home care aide'
            ],
            'housekeeping_services': [
                'housekeeping', 'cleaning services', 'home cleaning',
                'domestic services', 'house cleaning', 'cleaning',
                'maid services', 'residential cleaning', 'home maintenance cleaning'
            ],
            'meal_services': [
                'meal delivery', 'meal services', 'nutrition services',
                'food delivery', 'meal preparation', 'cooking services',
                'dietary services', 'meal planning', 'nutrition support',
                'food services', 'meal assistance'
            ],
            'transportation_services': [
                'transportation', 'medical transportation', 'senior transportation',
                'transport services', 'ride services', 'medical transport',
                'transportation services', 'mobility services', 'travel assistance',
                'transportation assistance', 'senior rides'
            ],
            
            # Emergency Services
            'emergency_services': [
                'emergency services', 'medical alert', 'emergency response',
                'alert services', 'emergency monitoring', 'safety monitoring',
                'personal emergency response', 'emergency assistance',
                'safety services', 'monitoring services'
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
                        
                        # Handle different JSON structures
                        if isinstance(data, list):
                            for item in data:
                                provider = ServiceProvider(**item)
                                self.providers.append(provider)
                        elif isinstance(data, dict):
                            if 'providers' in data:
                                for item in data['providers']:
                                    provider = ServiceProvider(**item)
                                    self.providers.append(provider)
                            else:
                                # Single provider object
                                provider = ServiceProvider(**data)
                                self.providers.append(provider)
                                
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _calculate_eldercare_service_relevance_score(self, query: str, provider: ServiceProvider) -> float:
        """Calculate eldercare service relevance score based on domain-specific keywords"""
        query_lower = query.lower()
        provider_text = f"{provider.name} {provider.description} {provider.detailedDescription} {provider.type} {' '.join(provider.types)} {' '.join(provider.serviceAreas)}".lower()
        
        total_score = 0.0
        
        # Check each eldercare service category
        for category, keywords in self.eldercare_service_keywords.items():
            query_matches = sum(1 for keyword in keywords if keyword in query_lower)
            provider_matches = sum(1 for keyword in keywords if keyword in provider_text)
            
            if query_matches > 0 and provider_matches > 0:
                # Score based on relevance strength
                category_score = min(query_matches * provider_matches * 2.0, 15.0)
                total_score += category_score
        
        # Bonus for service type matching
        if provider.type:
            service_type_keywords = {
                'home_modification': ['modification', 'safety', 'accessibility', 'installation'],
                'healthcare': ['health', 'medical', 'nursing', 'care'],
                'legal': ['legal', 'attorney', 'lawyer', 'law'],
                'financial': ['financial', 'insurance', 'planning'],
                'personal_care': ['care', 'companion', 'assistance', 'support'],
                'transportation': ['transportation', 'transport', 'ride', 'travel']
            }
            
            provider_type = provider.type.lower()
            for service_type, type_keywords in service_type_keywords.items():
                if any(keyword in provider_type for keyword in type_keywords):
                    if any(keyword in query_lower for keyword in type_keywords):
                        total_score += 8.0
                        break
        
        return total_score

    def _calculate_direct_keyword_score(self, query: str, provider: ServiceProvider) -> float:
        """Calculate direct keyword matching score"""
        query_lower = query.lower()
        score = 0.0
        
        # Split query into meaningful words (filter out very short words)
        query_words = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
        
        if not query_words:
            return 0.0
        
        # Check provider name for matches (highest weight)
        name_words = provider.name.lower().split()
        name_matches = 0
        for word in query_words:
            if any(word in name_word for name_word in name_words):
                name_matches += 1
        
        if name_matches > 0:
            score += (name_matches / len(query_words)) * 12.0
        
        # Check service type for matches
        if provider.type:
            type_words = provider.type.lower().split()
            type_matches = 0
            for word in query_words:
                if any(word in type_word for type_word in type_words):
                    type_matches += 1
            
            if type_matches > 0:
                score += (type_matches / len(query_words)) * 10.0
        
        # Check service types (multiple) for matches
        types_matches = 0
        for service_type in provider.types:
            type_words = service_type.lower().split()
            for word in query_words:
                if any(word in type_word for type_word in type_words):
                    types_matches += 1
        
        if types_matches > 0:
            score += min(types_matches / len(query_words), 1.0) * 8.0
        
        # Check description for matches
        if provider.description:
            desc_words = provider.description.lower().split()
            desc_matches = 0
            for word in query_words:
                if any(word in desc_word for desc_word in desc_words):
                    desc_matches += 1
            
            if desc_matches > 0:
                score += (desc_matches / len(query_words)) * 6.0
        
        # Check detailed description for matches
        if provider.detailedDescription:
            detailed_desc_words = provider.detailedDescription.lower().split()
            detailed_desc_matches = 0
            for word in query_words:
                if any(word in detailed_word for detailed_word in detailed_desc_words):
                    detailed_desc_matches += 1
            
            if detailed_desc_matches > 0:
                score += (detailed_desc_matches / len(query_words)) * 4.0
        
        # Check service areas for matches
        area_matches = 0
        for area in provider.serviceAreas:
            area_words = area.lower().split()
            for word in query_words:
                if any(word in area_word for area_word in area_words):
                    area_matches += 1
        
        if area_matches > 0:
            score += min(area_matches / len(query_words), 1.0) * 3.0
        
        return score

    def _build_search_index(self):
        """Build TF-IDF index for all service provider content"""
        if not self.providers:
            return
        
        # Combine name, type, description, service areas for each provider with strategic emphasis
        self.processed_content = []
        for provider in self.providers:
            # Strategic emphasis: name and type get highest weight
            combined_text = f"{provider.name} " * 6 + \
                           f"{provider.type} " * 5 + \
                           f"{' '.join(provider.types)} " * 4 + \
                           f"{provider.description} " * 3 + \
                           f"{provider.detailedDescription or ''} " * 2 + \
                           f"{' '.join(provider.serviceAreas)}"
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Handle small datasets
        if len(self.processed_content) < 2:
            print(f"Skipping TF-IDF for small dataset ({len(self.processed_content)} providers)")
            self.tfidf_matrix = None
            self.tfidf_vectorizer = None
            return
        
        # Initialize TF-IDF vectorizer with reasonable parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
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
                print(f"Built TF-IDF matrix for {len(self.processed_content)} service providers")
            except Exception as e:
                print(f"Error building TF-IDF matrix: {e}")
                self.tfidf_matrix = None
                self.tfidf_vectorizer = None

    def get_all_providers(self) -> ServiceList:
        """Get all service providers"""
        return ServiceList(providers=self.providers, total=len(self.providers))

    def search_providers(self, query: str, limit: int = 10) -> ServiceList:
        """Search providers using optimized scoring"""
        if not query or not self.providers:
            return ServiceList(providers=[], total=0)
        
        scored_providers = []
        
        for i, provider in enumerate(self.providers):
            # Calculate scores
            eldercare_score = self._calculate_eldercare_service_relevance_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with balanced weights
            if tfidf_score > 0.3:
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.4) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.5)
            
            # Lower threshold for search to return more results
            if combined_score > 0.5:  # Much lower threshold
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
        """Get service provider recommendations with optimized scoring"""
        if not query or not self.providers:
            return ServiceList(providers=[], total=0)
        
        scored_providers = []
        
        for i, provider in enumerate(self.providers):
            # Calculate scores
            eldercare_score = self._calculate_eldercare_service_relevance_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with emphasis on eldercare matching for recommendations
            if tfidf_score > 0.5:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.3) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.4)
            
            # Lower threshold for recommendations to ensure results
            if combined_score > 2.0:  # Much lower threshold
                scored_providers.append((provider, combined_score))
        
        # Sort by score and return top results
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        recommended_providers = [provider for provider, score in scored_providers[:limit]]
        
        return ServiceList(providers=recommended_providers, total=len(recommended_providers))

    def recommend_best_provider_with_score(self, query: str) -> Tuple[Optional[ServiceProvider], float]:
        """Get the best service provider recommendation with optimized scoring"""
        if not query or not self.providers:
            return None, 0.0
        
        best_provider = None
        best_score = 0.0
        
        for i, provider in enumerate(self.providers):
            # Calculate scores
            eldercare_score = self._calculate_eldercare_service_relevance_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with heavy emphasis on eldercare matching
            if tfidf_score > 0.5:
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.2) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.7) + (direct_keyword_score * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_provider = provider
        
        return best_provider, best_score

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best service provider recommendation ID with reasonable threshold"""
        provider, score = self.recommend_best_provider_with_score(query)
        
        # Much more reasonable threshold for eldercare queries
        min_relevance_score = 2.0  # Lowered from 5.0 to actually return results
        
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