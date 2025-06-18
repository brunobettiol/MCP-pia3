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
        """Initialize the service service optimized for the 35 specific eldercare questions"""
        self.providers: List[ServiceProvider] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._load_services()
        self._build_search_index()
        
        # Manual mapping for the 35 specific questions to exact service provider matches
        self.question_service_mapping = {
            # SAFE Category Tasks (Questions 1-11)
            "handrails installed both sides stairs": ["handyman", "home_modification", "contractor"],
            "install grab bars bathroom toilet shower": ["handyman", "home_modification", "contractor", "accessibility"],
            "improve lighting areas hallways stairs": ["electrical", "handyman", "contractor", "lighting"],
            "assess fall risks implement prevention strategies": ["occupational_therapy", "physical_therapy", "home_assessment"],
            "provide appropriate mobility aids cane walker wheelchair": ["medical_equipment", "dme", "mobility"],
            "remove tripping hazards loose rugs clutter electrical cords": ["handyman", "organizing", "home_modification"],
            "ensure bedroom accessible without using stairs": ["home_modification", "contractor", "accessibility"],
            "evaluate neighborhood safety accessibility walking": ["assessment", "safety_evaluation"],
            "post emergency numbers visibly near phones": ["organizing", "safety_planning"],
            "install personal emergency response system": ["medical_alert", "emergency_system", "safety_monitoring"],
            "evaluate home care support needs consider professional help": ["in_home_care", "care_coordination", "assessment"],
            
            # Functional Ability Tasks (Questions 12-15)
            "ensure safe bathing practices provide assistance needed": ["in_home_care", "personal_care", "bathing_assistance"],
            "provide assistance dressing undressing necessary": ["in_home_care", "personal_care", "daily_living"],
            "ensure safe movement bed chair without help": ["physical_therapy", "occupational_therapy", "mobility"],
            "provide assistance toilet use needed": ["in_home_care", "personal_care", "toileting_assistance"],
            
            # HEALTHY Category Tasks (Questions 1-10)
            "manage chronic medical conditions effectively": ["home_healthcare", "nursing", "chronic_care", "medical_management"],
            "organize manage daily medications": ["pharmacy", "medication_management", "nursing"],
            "implement medication management system": ["pharmacy", "medication_management", "nursing"],
            "schedule regular checkups primary care specialists": ["care_coordination", "transportation", "scheduling"],
            "ensure regular balanced meals proper nutrition": ["meal_delivery", "nutrition", "meal_preparation"],
            "assist meal preparation grocery shopping needed": ["meal_delivery", "grocery_delivery", "personal_care"],
            "establish exercise routine regular physical activity": ["physical_therapy", "fitness", "exercise_program"],
            "monitor cognitive health address memory issues": ["memory_care", "cognitive_assessment", "neuropsychology"],
            "improve sleep quality address issues": ["sleep_specialist", "medical_care", "home_healthcare"],
            "schedule regular dental vision hearing checkups": ["transportation", "care_coordination", "scheduling"],
            
            # Emotional Health Tasks (Questions 11-14)
            "address feelings depression hopelessness": ["mental_health", "counseling", "therapy", "psychiatry"],
            "encourage participation enjoyable activities": ["companion_care", "activity_programs", "social_services"],
            "reduce feelings loneliness isolation": ["companion_care", "social_services", "community_programs"],
            "ensure vaccinations up date": ["home_healthcare", "nursing", "medical_care"],
            
            # PREPARED Category Tasks (Questions 1-10)
            "establish advance directives living will healthcare proxy": ["elder_law", "legal_services", "estate_planning"],
            "set durable power attorney finances": ["elder_law", "legal_services", "financial_planning"],
            "create will trust": ["elder_law", "legal_services", "estate_planning"],
            "discuss end life care preferences family": ["counseling", "family_therapy", "end_of_life_planning"],
            "review update insurance coverage": ["insurance_agent", "benefits_counselor", "financial_planning"],
            "develop financial plan potential long term care needs": ["financial_planning", "long_term_care_planning", "insurance"],
            "consider living arrangement options future needs": ["senior_living_advisor", "care_coordination", "housing_specialist"],
            "implement system managing bills financial matters": ["financial_management", "bill_paying_service", "money_management"],
            "organize important documents easy access": ["organizing_services", "document_management", "legal_services"],
            "create communication plan family care decisions": ["family_counseling", "care_coordination", "mediation"]
        }
        
        # Eldercare-focused keyword mappings based on the 35 specific questions
        self.eldercare_service_keywords = {
            # SAFE Category - Home Safety & Modifications
            'handrails_installation': [
                'handrails', 'handrail installation', 'stair rail', 'railing installation',
                'banister', 'stair safety', 'grab rail', 'safety rail installation',
                'handyman', 'contractor', 'home modification', 'accessibility'
            ],
            'grab_bars_installation': [
                'grab bars', 'grab bar installation', 'bathroom safety', 'shower grab bars',
                'toilet grab bars', 'safety bars', 'bathroom modification', 'accessibility',
                'handyman', 'contractor', 'home modification', 'bathroom contractor'
            ],
            'lighting_services': [
                'lighting installation', 'electrical', 'LED installation', 'lighting contractor',
                'motion sensor lights', 'pathway lighting', 'stair lighting', 'hallway lighting',
                'electrical services', 'lighting upgrade', 'bright lighting', 'electrician'
            ],
            'fall_risk_assessment': [
                'fall risk assessment', 'occupational therapy', 'physical therapy', 'safety assessment',
                'home safety evaluation', 'fall prevention', 'mobility assessment',
                'therapy services', 'OT', 'PT', 'safety evaluation'
            ],
            'mobility_equipment': [
                'mobility aids', 'medical equipment', 'DME', 'wheelchair', 'walker', 'cane',
                'rollator', 'mobility scooter', 'durable medical equipment', 'mobility devices',
                'assistive devices', 'mobility equipment rental'
            ],
            'hazard_removal': [
                'handyman', 'home organization', 'decluttering', 'organizing services',
                'home modification', 'safety modification', 'hazard removal',
                'home maintenance', 'contractor', 'home improvement'
            ],
            'bedroom_modification': [
                'home modification', 'accessibility modification', 'bedroom modification',
                'contractor', 'home improvement', 'accessibility services',
                'barrier removal', 'home adaptation'
            ],
            'safety_evaluation': [
                'safety assessment', 'home evaluation', 'safety consultation',
                'risk assessment', 'safety planning', 'evaluation services'
            ],
            'emergency_planning': [
                'emergency planning', 'safety planning', 'organizing services',
                'emergency preparedness', 'safety consultation'
            ],
            'emergency_response_systems': [
                'medical alert', 'emergency response', 'personal emergency response',
                'alert system', 'medical alarm', 'emergency monitoring',
                'safety monitoring', 'help button', 'emergency services'
            ],
            'care_assessment': [
                'care assessment', 'care coordination', 'in-home care', 'care evaluation',
                'care planning', 'home care assessment', 'care consultation',
                'geriatric care management', 'care services'
            ],
            
            # Functional Ability Support
            'bathing_assistance': [
                'bathing assistance', 'personal care', 'in-home care', 'caregiving',
                'daily living assistance', 'ADL assistance', 'shower assistance',
                'personal care aide', 'home health aide', 'companion care'
            ],
            'dressing_assistance': [
                'dressing assistance', 'personal care', 'in-home care', 'caregiving',
                'daily living assistance', 'ADL assistance', 'clothing assistance',
                'personal care aide', 'home health aide'
            ],
            'mobility_assistance': [
                'mobility assistance', 'transfer assistance', 'physical therapy',
                'occupational therapy', 'mobility training', 'movement assistance',
                'PT', 'OT', 'therapy services', 'rehabilitation'
            ],
            'toileting_assistance': [
                'toileting assistance', 'personal care', 'in-home care', 'caregiving',
                'bathroom assistance', 'ADL assistance', 'personal care aide',
                'home health aide', 'incontinence care'
            ],
            
            # HEALTHY Category - Health & Medical Services
            'chronic_care_management': [
                'chronic care', 'disease management', 'home healthcare', 'nursing services',
                'medical management', 'chronic condition management', 'skilled nursing',
                'home health', 'medical care', 'health management'
            ],
            'medication_management_services': [
                'medication management', 'pharmacy services', 'pill organization',
                'medication delivery', 'prescription management', 'medication adherence',
                'nursing services', 'medication review', 'pharmaceutical services'
            ],
            'healthcare_coordination': [
                'care coordination', 'healthcare coordination', 'medical transportation',
                'appointment scheduling', 'healthcare management', 'care planning',
                'geriatric care management', 'health services coordination'
            ],
            'nutrition_services': [
                'meal delivery', 'nutrition services', 'meal preparation', 'dietary services',
                'food delivery', 'meal planning', 'nutrition counseling',
                'grocery delivery', 'cooking services', 'meal services'
            ],
            'exercise_services': [
                'physical therapy', 'exercise programs', 'fitness services', 'rehabilitation',
                'activity programs', 'movement therapy', 'PT', 'fitness training',
                'exercise therapy', 'wellness programs'
            ],
            'cognitive_services': [
                'memory care', 'cognitive assessment', 'neuropsychology', 'dementia care',
                'memory services', 'cognitive therapy', 'brain health', 'memory support',
                'cognitive evaluation', 'mental health services'
            ],
            'sleep_services': [
                'sleep specialist', 'sleep medicine', 'medical care', 'home healthcare',
                'sleep consultation', 'health services', 'medical services'
            ],
            'health_screenings': [
                'transportation services', 'medical transportation', 'healthcare coordination',
                'appointment services', 'care coordination', 'health services'
            ],
            
            # Emotional Health Services
            'mental_health_services': [
                'mental health', 'counseling', 'therapy', 'psychiatry', 'psychology',
                'emotional support', 'behavioral health', 'mental health counseling',
                'psychotherapy', 'depression counseling', 'grief counseling'
            ],
            'social_engagement': [
                'companion care', 'social services', 'activity programs', 'companionship',
                'social engagement', 'community programs', 'recreational services',
                'social activities', 'companion services'
            ],
            'companionship_services': [
                'companion care', 'companionship', 'social services', 'friendship services',
                'social support', 'isolation prevention', 'community engagement',
                'social connection', 'companion services'
            ],
            'vaccination_services': [
                'home healthcare', 'nursing services', 'medical care', 'health services',
                'vaccination services', 'immunization', 'preventive care'
            ],
            
            # PREPARED Category - Planning & Legal Services
            'legal_planning_services': [
                'elder law', 'legal services', 'estate planning', 'advance directives',
                'legal planning', 'attorney services', 'legal consultation',
                'estate attorney', 'elder law attorney', 'legal documents'
            ],
            'financial_legal_services': [
                'elder law', 'legal services', 'financial planning', 'estate planning',
                'power of attorney', 'legal documents', 'financial legal services',
                'attorney services', 'legal consultation'
            ],
            'estate_planning_services': [
                'estate planning', 'elder law', 'legal services', 'will preparation',
                'trust services', 'estate attorney', 'legal planning',
                'inheritance planning', 'estate documents'
            ],
            'end_of_life_counseling': [
                'counseling', 'family therapy', 'end-of-life planning', 'grief counseling',
                'family counseling', 'emotional support', 'therapy services',
                'bereavement counseling', 'spiritual counseling'
            ],
            'insurance_services': [
                'insurance agent', 'insurance services', 'benefits counselor', 'insurance planning',
                'medicare counseling', 'insurance consultation', 'benefits planning',
                'insurance review', 'coverage planning'
            ],
            'financial_planning_services': [
                'financial planning', 'financial advisor', 'long-term care planning',
                'retirement planning', 'financial consultation', 'financial services',
                'financial counseling', 'care financing', 'financial management'
            ],
            'living_arrangement_planning': [
                'senior living advisor', 'care coordination', 'housing specialist',
                'senior housing consultant', 'living arrangement planning',
                'care placement', 'senior living placement', 'housing services'
            ],
            'financial_management_services': [
                'financial management', 'bill paying service', 'money management',
                'financial organization', 'bill management', 'financial services',
                'daily money management', 'financial assistance'
            ],
            'document_organization_services': [
                'organizing services', 'document management', 'file organization',
                'paperwork organization', 'record management', 'document services',
                'organizing consultant', 'administrative services'
            ],
            'family_communication_services': [
                'family counseling', 'care coordination', 'mediation services',
                'family therapy', 'communication facilitation', 'family planning',
                'care planning', 'family consultation'
            ]
        }
        
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

    def _calculate_exact_question_match_score(self, query: str, provider: ServiceProvider) -> float:
        """Calculate exact match score for the 35 specific questions with stricter matching"""
        query_lower = self._preprocess_text(query)
        provider_type = provider.type.lower()
        provider_text = f"{provider.name} {provider.description} {provider.detailedDescription}".lower()
        
        # Check for direct question mapping first
        for question_key, service_types in self.question_service_mapping.items():
            # Check if provider type matches any of the expected service types
            type_match = any(service_type in provider_type for service_type in service_types)
            
            # Also check if provider text contains service type keywords
            text_match = any(service_type.replace('_', ' ') in provider_text for service_type in service_types)
            
            if type_match or text_match:
                # Calculate similarity between query and question key
                question_words = set(question_key.split())
                query_words = set(query_lower.split())
                
                # Calculate overlap - require stronger overlap for exact matching
                overlap = len(question_words.intersection(query_words))
                if overlap > 0:
                    similarity = overlap / max(len(question_words), len(query_words))
                    if similarity > 0.4:  # Increased threshold from 0.3 to 0.4
                        # Higher base score for exact matches
                        base_score = 70.0 if type_match else 50.0  # Prefer type matches
                        return base_score + (similarity * 30.0)  # Score between 70-100 or 50-80
        
        return 0.0

    def _calculate_eldercare_service_relevance_score(self, query: str, provider: ServiceProvider) -> float:
        """Calculate eldercare service relevance score with stricter matching to prevent unrelated recommendations"""
        query_lower = query.lower()
        provider_text = f"{provider.name} {provider.description} {provider.detailedDescription} {provider.type} {' '.join(provider.types)} {' '.join(provider.serviceAreas)}".lower()
        
        total_score = 0.0
        
        # First, check if the query contains eldercare-related terms
        eldercare_query_terms = [
            'senior', 'elderly', 'aging', 'elder', 'geriatric', 'care', 'caregiver',
            'health', 'medical', 'therapy', 'nursing', 'medication', 'chronic',
            'safety', 'fall', 'mobility', 'grab', 'handrail', 'bathroom', 'shower', 'toilet',
            'legal', 'attorney', 'will', 'trust', 'insurance', 'financial', 'planning',
            'home', 'modification', 'accessible', 'assistance', 'help', 'support',
            'bathing', 'dressing', 'toileting', 'emergency', 'alert', 'monitoring'
        ]
        
        query_has_eldercare_terms = any(term in query_lower for term in eldercare_query_terms)
        
        # If query doesn't contain eldercare terms, return very low score
        if not query_has_eldercare_terms:
            return 0.0
        
        # Check each eldercare service category
        for category, keywords in self.eldercare_service_keywords.items():
            query_matches = sum(1 for keyword in keywords if keyword in query_lower)
            provider_matches = sum(1 for keyword in keywords if keyword in provider_text)
            
            if query_matches > 0 and provider_matches > 0:
                # Score based on relevance strength - require stronger matches
                category_score = min(query_matches * provider_matches * 2.5, 15.0)  # Reduced multiplier
                total_score += category_score
        
        # Bonus for service type matching with eldercare context
        if provider.type:
            eldercare_service_types = {
                'in_home_care': ['care', 'health', 'assistance', 'support', 'elderly', 'senior'],
                'home_modification': ['modification', 'safety', 'accessibility', 'home', 'bathroom'],
                'physical_therapy': ['therapy', 'physical', 'mobility', 'exercise', 'rehabilitation'],
                'elder_law': ['legal', 'attorney', 'law', 'estate', 'planning', 'will'],
                'insurance': ['insurance', 'coverage', 'benefits', 'medicare', 'medicaid'],
                'financial_advisor': ['financial', 'planning', 'money', 'retirement', 'advisor'],
                'geriatric_medicine': ['medical', 'doctor', 'physician', 'geriatric', 'health'],
                'hospice': ['hospice', 'end-of-life', 'palliative', 'comfort', 'terminal'],
                'palliative_care': ['palliative', 'comfort', 'pain', 'end-of-life', 'hospice'],
                'grief_counseling': ['grief', 'counseling', 'bereavement', 'loss', 'support']
            }
            
            provider_type = provider.type.lower()
            for service_type, type_keywords in eldercare_service_types.items():
                if service_type in provider_type or any(keyword in provider_type for keyword in type_keywords):
                    if any(keyword in query_lower for keyword in type_keywords):
                        total_score += 8.0  # Reduced from 10.0
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
            score += (name_matches / len(query_words)) * 15.0
        
        # Check service type for matches
        if provider.type:
            type_words = provider.type.lower().split()
            type_matches = 0
            for word in query_words:
                if any(word in type_word for type_word in type_words):
                    type_matches += 1
            
            if type_matches > 0:
                score += (type_matches / len(query_words)) * 12.0
        
        # Check service types (multiple) for matches
        types_matches = 0
        for service_type in provider.types:
            type_words = service_type.lower().split()
            for word in query_words:
                if any(word in type_word for type_word in type_words):
                    types_matches += 1
        
        if types_matches > 0:
            score += min(types_matches / len(query_words), 1.0) * 10.0
        
        # Check description for matches
        if provider.description:
            desc_words = provider.description.lower().split()
            desc_matches = 0
            for word in query_words:
                if any(word in desc_word for desc_word in desc_words):
                    desc_matches += 1
            
            if desc_matches > 0:
                score += (desc_matches / len(query_words)) * 8.0
        
        # Check detailed description for matches
        if provider.detailedDescription:
            detailed_desc_words = provider.detailedDescription.lower().split()
            detailed_desc_matches = 0
            for word in query_words:
                if any(word in detailed_word for detailed_word in detailed_desc_words):
                    detailed_desc_matches += 1
            
            if detailed_desc_matches > 0:
                score += (detailed_desc_matches / len(query_words)) * 6.0
        
        # Check service areas for matches
        area_matches = 0
        for area in provider.serviceAreas:
            area_words = area.lower().split()
            for word in query_words:
                if any(word in area_word for area_word in area_words):
                    area_matches += 1
        
        if area_matches > 0:
            score += min(area_matches / len(query_words), 1.0) * 4.0
        
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
        """Search providers using optimized scoring for the 35 specific questions"""
        if not query or not self.providers:
            return ServiceList(providers=[], total=0)
        
        scored_providers = []
        
        for i, provider in enumerate(self.providers):
            # Calculate scores with priority on exact question matching
            exact_question_score = self._calculate_exact_question_match_score(query, provider)
            eldercare_score = self._calculate_eldercare_service_relevance_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 12  # Scale to 0-12
            
            # Combined score with heavy emphasis on exact question matching
            if exact_question_score > 0:
                # If we have an exact question match, prioritize it heavily
                combined_score = exact_question_score + (eldercare_score * 0.3) + (direct_keyword_score * 0.2)
            elif tfidf_score > 0.4:
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.4) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.5)
            
            # Lower threshold for search to return more results
            if combined_score > 1.0:
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
        """Get service provider recommendations with optimized scoring for the 35 specific questions"""
        if not query or not self.providers:
            return ServiceList(providers=[], total=0)
        
        scored_providers = []
        
        for i, provider in enumerate(self.providers):
            # Calculate scores with priority on exact question matching
            exact_question_score = self._calculate_exact_question_match_score(query, provider)
            eldercare_score = self._calculate_eldercare_service_relevance_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 12  # Scale to 0-12
            
            # Combined score with heavy emphasis on exact question matching
            if exact_question_score > 0:
                # If we have an exact question match, prioritize it heavily
                combined_score = exact_question_score + (eldercare_score * 0.2) + (direct_keyword_score * 0.1)
            elif tfidf_score > 0.6:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.3) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.4)
            
            # Lower threshold for recommendations to ensure results
            if combined_score > 3.0:
                scored_providers.append((provider, combined_score))
        
        # Sort by score and return top results
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        recommended_providers = [provider for provider, score in scored_providers[:limit]]
        
        return ServiceList(providers=recommended_providers, total=len(recommended_providers))

    def recommend_best_provider_with_score(self, query: str) -> Tuple[Optional[ServiceProvider], float]:
        """Get the best service provider recommendation with optimized scoring for the 35 specific questions"""
        if not query or not self.providers:
            return None, 0.0
        
        best_provider = None
        best_score = 0.0
        
        for i, provider in enumerate(self.providers):
            # Calculate scores with priority on exact question matching
            exact_question_score = self._calculate_exact_question_match_score(query, provider)
            eldercare_score = self._calculate_eldercare_service_relevance_score(query, provider)
            direct_keyword_score = self._calculate_direct_keyword_score(query, provider)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 12  # Scale to 0-12
            
            # Combined score with heavy emphasis on exact question matching
            if exact_question_score > 0:
                # If we have an exact question match, prioritize it heavily
                combined_score = exact_question_score + (eldercare_score * 0.1) + (direct_keyword_score * 0.1)
            elif tfidf_score > 0.8:
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.2) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.7) + (direct_keyword_score * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_provider = provider
        
        return best_provider, best_score

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best service provider recommendation ID with strict threshold for the 35 specific questions"""
        provider, score = self.recommend_best_provider_with_score(query)
        
        # Balanced threshold to prevent unrelated recommendations while allowing legitimate eldercare queries
        # Based on analysis: exact matches get 70-100+, good eldercare matches get 15-25, weak matches get <10
        min_relevance_score = 10.0  # Adjusted from 12.0 to allow more legitimate eldercare queries
        
        # Additional validation: ensure the query contains eldercare-related terms
        query_lower = query.lower()
        eldercare_terms = [
            # Core eldercare terms
            'senior', 'elderly', 'aging', 'elder', 'geriatric', 'care', 'caregiver',
            # Health and medical terms
            'health', 'medical', 'medicine', 'therapy', 'nursing', 'chronic', 'medication',
            # Safety and mobility terms
            'safety', 'fall', 'mobility', 'grab', 'handrail', 'bathroom', 'shower', 'toilet',
            # Legal and financial planning terms
            'legal', 'attorney', 'will', 'trust', 'insurance', 'financial', 'planning',
            # Home and living terms
            'home', 'house', 'living', 'modification', 'accessible', 'bedroom', 'stairs',
            # Care and assistance terms
            'assistance', 'help', 'support', 'aid', 'service', 'provider',
            # Specific eldercare activities
            'bathing', 'dressing', 'toileting', 'eating', 'walking', 'transfer',
            # Health management terms
            'checkup', 'appointment', 'prescription', 'monitor', 'manage',
            # Emergency and safety terms
            'emergency', 'alert', 'response', 'monitoring', 'prevention'
        ]
        
        # Check if query contains eldercare-related terms
        has_eldercare_terms = any(term in query_lower for term in eldercare_terms)
        
        # Only recommend if we meet both criteria: high score AND eldercare-related query
        if provider and score >= min_relevance_score and has_eldercare_terms:
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