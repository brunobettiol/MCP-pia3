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
        """Initialize the service service with TF-IDF indexing"""
        self.providers: List[ServiceProvider] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._load_services()
        self._build_search_index()
    
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

    def _build_search_index(self):
        """Build TF-IDF index for all service provider content"""
        if not self.providers:
            return
        
        # Combine all text fields for each provider
        self.processed_content = []
        for provider in self.providers:
            # Combine all searchable text fields with appropriate weights
            combined_text = f"{provider.name} {provider.name} {provider.name} " + \
                           f"{provider.type} {provider.type} " + \
                           f"{' '.join(provider.types)} " + \
                           f"{provider.description} {provider.description} " + \
                           f"{provider.detailedDescription} " + \
                           f"{' '.join(provider.serviceAreas)} " + \
                           f"{' '.join(provider.serviceRegions)} " + \
                           f"{' '.join(provider.services)} {' '.join(provider.services)} " + \
                           f"{' '.join(provider.specialties)} {' '.join(provider.specialties)} " + \
                           f"{' '.join(provider.credentials)} " + \
                           f"{' '.join(provider.languages)}"
            
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

    def get_all_providers(self) -> ServiceList:
        """Get all service providers"""
        return ServiceList(providers=self.providers, total=len(self.providers))

    def search_providers(self, query: str, limit: int = 10) -> ServiceList:
        """Search providers using TF-IDF similarity"""
        if not query or not self.providers or self.tfidf_matrix is None:
            return ServiceList(providers=[], total=0)
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter results with minimum similarity threshold
        min_similarity = 0.35  # Extremely high threshold for search
        filtered_results = []
        
        for idx in sorted_indices:
            if similarities[idx] >= min_similarity and len(filtered_results) < limit:
                filtered_results.append(self.providers[idx])
            elif len(filtered_results) >= limit:
                break
        
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
        """Get service provider recommendations based on query using TF-IDF similarity"""
        if not query or not self.providers or self.tfidf_matrix is None:
            return ServiceList(providers=[], total=0)
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter results with minimum similarity threshold
        min_similarity = 0.4  # Extremely high threshold for recommendations
        recommended_providers = []
        
        for idx in sorted_indices:
            if similarities[idx] >= min_similarity and len(recommended_providers) < limit:
                recommended_providers.append(self.providers[idx])
            elif len(recommended_providers) >= limit:
                break
        
        return ServiceList(providers=recommended_providers, total=len(recommended_providers))

    def recommend_best_provider_with_score(self, query: str) -> Tuple[Optional[ServiceProvider], float]:
        """Get the best service provider recommendation with similarity score"""
        if not query or not self.providers or self.tfidf_matrix is None:
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
        
        return self.providers[best_idx], score

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best service provider recommendation ID based on query with threshold"""
        provider, score = self.recommend_best_provider_with_score(query)
        
        # Set minimum relevance threshold (adjust as needed)
        min_relevance_score = 3.0  # Equivalent to 30% cosine similarity - extremely high threshold
        
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