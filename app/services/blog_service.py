import json
import os
from typing import List, Dict, Optional
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.models.blog import BlogEntry, BlogList, BlogResponse
from app.core.config import settings


class BlogService:
    def __init__(self):
        """Initialize the blog service optimized for the 35 specific eldercare questions"""
        self.blogs: List[BlogEntry] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        
        # Manual mapping for the 35 specific questions to exact blog matches
        self.question_blog_mapping = {
            # SAFE Category Tasks (Questions 1-11)
            "handrails installed both sides stairs": ["aUe9fA1RMS2N4TMKda0b"],
            "install grab bars bathroom toilet shower": ["aUe9fA1RMS2N4TMKda0b", "KpYlyBvGpjp3JwUldZ8U"],
            "improve lighting areas hallways stairs": ["aUe9fA1RMS2N4TMKda0b"],
            "assess fall risks implement prevention strategies": ["aUe9fA1RMS2N4TMKda0b", "AaePzMFk2HdxVwRS4Ftz"],
            "provide appropriate mobility aids cane walker wheelchair": ["aUe9fA1RMS2N4TMKda0b"],
            "remove tripping hazards loose rugs clutter electrical cords": ["aUe9fA1RMS2N4TMKda0b"],
            "ensure bedroom accessible without using stairs": ["aUe9fA1RMS2N4TMKda0b"],
            "evaluate neighborhood safety accessibility walking": ["DYfLP9GMUbHU9PYNfjNs"],
            "post emergency numbers visibly near phones": ["aUe9fA1RMS2N4TMKda0b"],
            "install personal emergency response system": ["aUe9fA1RMS2N4TMKda0b"],
            "evaluate home care support needs consider professional help": ["6FjBf1DPoh3QilgTnhiA", "ZcTmu3Hnnt9tU6rzQwZc", "FoM0tYNWXSgSJD0dkUPK"],
            
            # Functional Ability Tasks (Questions 12-15)
            "ensure safe bathing practices provide assistance needed": ["VMpZbOMhSuzujtc7TrR4", "UkW6ehkqYMprcw1aPpBw"],
            "provide assistance dressing undressing necessary": ["VMpZbOMhSuzujtc7TrR4", "UkW6ehkqYMprcw1aPpBw"],
            "ensure safe movement bed chair without help": ["KpYlyBvGpjp3JwUldZ8U", "UkW6ehkqYMprcw1aPpBw"],
            "provide assistance toilet use needed": ["KpYlyBvGpjp3JwUldZ8U", "UkW6ehkqYMprcw1aPpBw"],
            
            # HEALTHY Category Tasks (Questions 1-10)
            "manage chronic medical conditions effectively": ["Tg4BdtWErPjs6wTk15RH", "Hf2wT2IZvYTFZAYq53ap", "7YGPzqY0itYRYsuRR7wT"],
            "organize manage daily medications": ["2yFFIe5O2yVwenG9LkW3", "VpO06tufLq2qZwlCKZNN"],
            "implement medication management system": ["2yFFIe5O2yVwenG9LkW3", "VpO06tufLq2qZwlCKZNN"],
            "schedule regular checkups primary care specialists": ["2jK9qSbIqZtj9dl325ZJ"],
            "ensure regular balanced meals proper nutrition": ["Tg4BdtWErPjs6wTk15RH"],
            "assist meal preparation grocery shopping needed": ["Tg4BdtWErPjs6wTk15RH"],
            "establish exercise routine regular physical activity": ["Tg4BdtWErPjs6wTk15RH"],
            "monitor cognitive health address memory issues": ["4lP0E98ZJHcU8STFN3ZK", "5IL2RRFKRXsd50NXhwEm"],
            "improve sleep quality address issues": ["Tg4BdtWErPjs6wTk15RH"],
            "schedule regular dental vision hearing checkups": ["2jK9qSbIqZtj9dl325ZJ"],
            
            # Emotional Health Tasks (Questions 11-14)
            "address feelings depression hopelessness": ["Ob5FILbCQ9z79K0FuuAq", "Tg4BdtWErPjs6wTk15RH"],
            "encourage participation enjoyable activities": ["Ob5FILbCQ9z79K0FuuAq"],
            "reduce feelings loneliness isolation": ["Ob5FILbCQ9z79K0FuuAq"],
            "ensure vaccinations up date": ["2jK9qSbIqZtj9dl325ZJ"],
            
            # PREPARED Category Tasks (Questions 1-10)
            "establish advance directives living will healthcare proxy": ["EPz4zRCSQS7MjWqB9P7I", "61a5jNtELAOW7J7xRaUP"],
            "set durable power attorney finances": ["EPz4zRCSQS7MjWqB9P7I", "5ReRS7m79R65VQPdZF3K"],
            "create will trust": ["EPz4zRCSQS7MjWqB9P7I", "5ReRS7m79R65VQPdZF3K"],
            "discuss end life care preferences family": ["61a5jNtELAOW7J7xRaUP", "NLNrsUUhCk2W3yKpjsKV"],
            "review update insurance coverage": ["OshTmi0FteModVJW6Gc7", "bxpvBiJSmLkiVHwWHQml"],
            "develop financial plan potential long term care needs": ["Smzoqnbmmm6UJKzqqw9G", "WL9Nbi6zXx6Jvl3GhQYA"],
            "consider living arrangement options future needs": ["aOflXZEzoKkfZS7meY3h", "Tp0tbrdoJh7F6EYvpmwV"],
            "implement system managing bills financial matters": ["Smzoqnbmmm6UJKzqqw9G"],
            "organize important documents easy access": ["Smzoqnbmmm6UJKzqqw9G"],
            "create communication plan family care decisions": ["61a5jNtELAOW7J7xRaUP", "NLNrsUUhCk2W3yKpjsKV"]
        }
        
        # Keyword mappings for the 35 specific questions
        self.eldercare_keywords = {
            # SAFE Category - Home Safety & Fall Prevention
            'handrails_stairs': [
                'handrails', 'handrail', 'stairs', 'stair', 'stairway', 'steps', 'step',
                'railing', 'banister', 'stair safety', 'stair rail', 'both sides'
            ],
            'grab_bars_bathroom': [
                'grab bars', 'grab bar', 'bathroom', 'shower', 'toilet', 'bath',
                'bathroom safety', 'shower safety', 'toilet safety', 'install grab bars',
                'bathroom modification', 'shower grab bar', 'bath rail'
            ],
            'lighting_improvement': [
                'lighting', 'lights', 'light', 'improve lighting', 'hallways', 'stairs',
                'adequate lighting', 'bright light', 'LED light', 'night light',
                'motion sensor', 'pathway light', 'stair light', 'hallway light'
            ],
            'fall_prevention': [
                'fall risks', 'fall prevention', 'falls', 'fall', 'prevention strategies',
                'accident prevention', 'injury prevention', 'safety assessment',
                'fall safety', 'assess fall risks', 'implement prevention'
            ],
            'mobility_aids': [
                'mobility aids', 'cane', 'walker', 'wheelchair', 'rollator',
                'mobility scooter', 'walking aid', 'mobility device', 'walking stick',
                'crutches', 'appropriate mobility aids', 'provide mobility aids'
            ],
            'tripping_hazards': [
                'tripping hazards', 'loose rugs', 'clutter', 'electrical cords',
                'remove hazards', 'trip hazards', 'hazard removal', 'declutter',
                'secure rugs', 'cord management'
            ],
            'bedroom_accessibility': [
                'bedroom accessible', 'without stairs', 'bedroom', 'accessible',
                'stairs', 'bedroom modification', 'accessible bedroom', 'ground floor'
            ],
            'neighborhood_safety': [
                'neighborhood safety', 'accessibility walking', 'walking safety',
                'outdoor safety', 'community safety', 'safe walking', 'evaluate neighborhood'
            ],
            'emergency_numbers': [
                'emergency numbers', 'post emergency', 'visible phones', 'emergency contacts',
                'phone numbers', 'emergency information', 'contact information'
            ],
            'emergency_response_system': [
                'personal emergency response', 'emergency response system', 'medical alert',
                'emergency system', 'alert system', 'help button', 'emergency button',
                'personal alarm', 'medical alarm'
            ],
            'home_care_support': [
                'home care support', 'professional help', 'in-home care', 'caregiver',
                'home health', 'care support', 'professional care', 'home assistance',
                'evaluate care needs', 'consider professional help'
            ],
            
            # Functional Ability Tasks
            'bathing_assistance': [
                'safe bathing', 'bathing practices', 'bathing assistance', 'shower safety',
                'bath safety', 'bathing help', 'shower assistance', 'bath assistance',
                'provide assistance bathing', 'ensure safe bathing'
            ],
            'dressing_assistance': [
                'dressing assistance', 'undressing', 'dressing help', 'clothing assistance',
                'provide assistance dressing', 'dressing aid', 'clothing aid'
            ],
            'transfer_mobility': [
                'safe movement', 'bed chair', 'transfer', 'mobility', 'movement assistance',
                'bed transfer', 'chair transfer', 'safe transfer', 'movement bed chair'
            ],
            'toilet_assistance': [
                'toilet assistance', 'toilet use', 'toileting', 'bathroom assistance',
                'toilet help', 'provide assistance toilet', 'toileting aid'
            ],
            
            # HEALTHY Category Tasks
            'chronic_condition_management': [
                'chronic medical conditions', 'manage chronic', 'chronic conditions',
                'chronic illness', 'condition management', 'medical conditions',
                'chronic disease', 'manage conditions effectively'
            ],
            'medication_management': [
                'daily medications', 'organize medications', 'manage medications',
                'medication management', 'pill organization', 'medication system',
                'prescription management', 'medication adherence', 'pill management'
            ],
            'healthcare_checkups': [
                'regular checkups', 'primary care', 'specialists', 'doctor visits',
                'medical appointments', 'health checkups', 'schedule checkups',
                'healthcare visits', 'medical care'
            ],
            'nutrition_meals': [
                'balanced meals', 'proper nutrition', 'regular meals', 'nutrition',
                'meal preparation', 'grocery shopping', 'healthy eating', 'diet',
                'nutritional needs', 'meal planning'
            ],
            'exercise_activity': [
                'exercise routine', 'physical activity', 'regular activity', 'exercise',
                'fitness', 'physical therapy', 'movement', 'activity routine'
            ],
            'cognitive_health': [
                'cognitive health', 'memory issues', 'memory problems', 'dementia',
                'cognitive problems', 'mental health', 'memory care', 'cognitive decline'
            ],
            'sleep_quality': [
                'sleep quality', 'sleep issues', 'sleep problems', 'sleeping',
                'insomnia', 'sleep disorders', 'improve sleep', 'sleep hygiene'
            ],
            'health_screenings': [
                'dental checkups', 'vision checkups', 'hearing checkups', 'regular checkups',
                'health screenings', 'preventive care', 'routine checkups'
            ],
            
            # Emotional Health Tasks
            'depression_support': [
                'depression', 'hopelessness', 'feelings depression', 'emotional health',
                'mental health', 'mood', 'address depression', 'emotional support'
            ],
            'social_activities': [
                'enjoyable activities', 'participation activities', 'social activities',
                'engagement', 'hobbies', 'recreation', 'encourage participation'
            ],
            'social_connection': [
                'loneliness', 'isolation', 'social connection', 'companionship',
                'social support', 'reduce loneliness', 'social interaction'
            ],
            'vaccinations': [
                'vaccinations', 'vaccines', 'immunizations', 'up to date',
                'vaccination schedule', 'preventive care'
            ],
            
            # PREPARED Category Tasks
            'advance_directives': [
                'advance directives', 'living will', 'healthcare proxy', 'advance planning',
                'end-of-life planning', 'healthcare decisions', 'medical directives'
            ],
            'power_of_attorney': [
                'power of attorney', 'durable power', 'financial power', 'POA',
                'attorney finances', 'legal documents', 'financial decisions'
            ],
            'estate_planning': [
                'will', 'trust', 'estate planning', 'estate documents', 'inheritance',
                'legal planning', 'create will', 'estate'
            ],
            'end_of_life_discussion': [
                'end of life', 'care preferences', 'family discussion', 'end-of-life care',
                'discuss preferences', 'care decisions', 'family planning'
            ],
            'insurance_review': [
                'insurance coverage', 'review insurance', 'update insurance',
                'health insurance', 'medicare', 'medicaid', 'insurance planning'
            ],
            'financial_planning': [
                'financial plan', 'long term care', 'care costs', 'financial planning',
                'care financing', 'long-term care planning', 'financial needs'
            ],
            'living_arrangements': [
                'living arrangements', 'living options', 'senior living', 'assisted living',
                'care options', 'housing options', 'future living needs'
            ],
            'bill_management': [
                'managing bills', 'financial matters', 'bill management', 'financial organization',
                'money management', 'financial system', 'bill paying'
            ],
            'document_organization': [
                'important documents', 'organize documents', 'document organization',
                'paperwork', 'records', 'document management', 'easy access documents'
            ],
            'family_communication': [
                'communication plan', 'family decisions', 'care decisions',
                'family planning', 'family communication', 'decision making'
            ]
        }
        
        self._load_blogs()
        self._build_search_index()
    
    def _parse_firestore_date(self, date_obj) -> Optional[datetime]:
        """Parse Firestore date format with __time__ field"""
        if isinstance(date_obj, dict) and "__time__" in date_obj:
            try:
                return datetime.fromisoformat(date_obj["__time__"].replace("Z", "+00:00"))
            except:
                return None
        elif isinstance(date_obj, str):
            try:
                return datetime.fromisoformat(date_obj.replace("Z", "+00:00"))
            except:
                return None
        return None
    
    def _load_blogs(self):
        """Load all blog entries from Firestore export JSON files"""
        blogs_dir = os.path.join(settings.BASE_DIR, "data", "blogs")
        if not os.path.exists(blogs_dir):
            return
        
        for filename in os.listdir(blogs_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(blogs_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        firestore_data = json.load(f)
                        
                        # Handle Firestore export format
                        if 'data' in firestore_data:
                            # This is a Firestore export
                            for blog_id, blog_data in firestore_data['data'].items():
                                try:
                                    # Clean and process the blog data
                                    processed_data = {
                                        'id': blog_data.get('id', blog_id),
                                        'title': blog_data.get('title', ''),
                                        'content': self._strip_html(blog_data.get('content', '')),
                                        'author': blog_data.get('author', ''),
                                        'category': blog_data.get('category', ''),
                                        'subcategory': blog_data.get('subcategory'),
                                        'summary': blog_data.get('summary'),
                                        'tags': blog_data.get('tags', []),
                                        'featured': blog_data.get('featured', False),
                                        'contentType': blog_data.get('contentType'),
                                        'createdAt': self._parse_firestore_date(blog_data.get('createdAt')) or datetime.now(),
                                        'updatedAt': self._parse_firestore_date(blog_data.get('updatedAt')),
                                        'publishedDate': self._parse_firestore_date(blog_data.get('publishedDate'))
                                    }
                                    
                                    # Use source_id as the blog ID for consistency
                                    processed_data['source_id'] = blog_id
                                    
                                    blog = BlogEntry(**processed_data)
                                    self.blogs.append(blog)
                                except Exception as e:
                                    print(f"Error processing blog {blog_id}: {e}")
                        else:
                            # Handle old format (array of blog objects)
                            if isinstance(firestore_data, list):
                                for item in firestore_data:
                                    blog = BlogEntry(**item)
                                    self.blogs.append(blog)
                            else:
                                # Single blog object
                                blog = BlogEntry(**firestore_data)
                                self.blogs.append(blog)
                                
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

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
        
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _calculate_exact_question_match_score(self, query: str, blog: BlogEntry) -> float:
        """Calculate exact match score for the 35 specific questions"""
        query_lower = self._preprocess_text(query)
        
        # Check for direct question mapping first
        for question_key, blog_ids in self.question_blog_mapping.items():
            if blog.source_id in blog_ids:
                # Calculate similarity between query and question key
                question_words = set(question_key.split())
                query_words = set(query_lower.split())
                
                # Calculate overlap
                overlap = len(question_words.intersection(query_words))
                if overlap > 0:
                    similarity = overlap / max(len(question_words), len(query_words))
                    if similarity > 0.3:  # Threshold for considering it a match
                        return 50.0 + (similarity * 50.0)  # Score between 50-100
        
        return 0.0

    def _calculate_eldercare_relevance_score(self, query: str, blog: BlogEntry) -> float:
        """Calculate eldercare relevance score based on domain-specific keywords"""
        query_lower = query.lower()
        blog_text = f"{blog.title} {blog.summary or ''} {' '.join(blog.tags)} {blog.category}".lower()
        
        total_score = 0.0
        
        # Check each eldercare category
        for category, keywords in self.eldercare_keywords.items():
            query_matches = sum(1 for keyword in keywords if keyword in query_lower)
            blog_matches = sum(1 for keyword in keywords if keyword in blog_text)
            
            if query_matches > 0 and blog_matches > 0:
                # Score based on relevance strength
                category_score = min(query_matches * blog_matches * 3.0, 15.0)
                total_score += category_score
        
        # Bonus for category matching
        if blog.category:
            category_keywords = {
                'safe': ['safety', 'fall', 'home', 'bathroom', 'lighting', 'mobility', 'emergency'],
                'healthy': ['health', 'medical', 'medication', 'chronic', 'nutrition', 'exercise', 'mental'],
                'prepared': ['planning', 'legal', 'financial', 'insurance', 'documents', 'care'],
                'caregiver': ['caregiver', 'support', 'stress', 'burnout', 'respite']
            }
            
            blog_category = blog.category.lower()
            if blog_category in category_keywords:
                category_words = category_keywords[blog_category]
                if any(word in query_lower for word in category_words):
                    total_score += 8.0
        
        return total_score

    def _calculate_direct_keyword_score(self, query: str, blog: BlogEntry) -> float:
        """Calculate direct keyword matching score"""
        query_lower = query.lower()
        score = 0.0
        
        # Split query into meaningful words (filter out very short words)
        query_words = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
        
        if not query_words:
            return 0.0
        
        # Check title for matches (highest weight)
        title_words = blog.title.lower().split()
        title_matches = 0
        for word in query_words:
            if any(word in title_word for title_word in title_words):
                title_matches += 1
        
        if title_matches > 0:
            score += (title_matches / len(query_words)) * 15.0
        
        # Check tags for matches
        tag_matches = 0
        for tag in blog.tags:
            tag_lower = tag.lower()
            for word in query_words:
                if word in tag_lower:
                    tag_matches += 1
        
        if tag_matches > 0:
            score += min(tag_matches / len(query_words), 1.0) * 12.0
        
        # Check summary for matches
        if blog.summary:
            summary_words = blog.summary.lower().split()
            summary_matches = 0
            for word in query_words:
                if any(word in summary_word for summary_word in summary_words):
                    summary_matches += 1
            
            if summary_matches > 0:
                score += (summary_matches / len(query_words)) * 8.0
        
        # Check category for matches
        if blog.category:
            category_lower = blog.category.lower()
            for word in query_words:
                if word in category_lower:
                    score += 6.0
        
        return score

    def _build_search_index(self):
        """Build TF-IDF index for all blog content"""
        if not self.blogs:
            return
        
        # Combine title, summary, tags, category for each blog with strategic emphasis
        self.processed_content = []
        for blog in self.blogs:
            # Strategic emphasis: category and title get highest weight
            category_text = blog.category.replace('_', ' ') if blog.category else ''
            subcategory_text = blog.subcategory.replace('_', ' ') if blog.subcategory else ''
            
            # Build content with strategic repetition for TF-IDF
            combined_text = f"{category_text} " * 8 + \
                           f"{subcategory_text} " * 6 + \
                           f"{blog.title} " * 6 + \
                           f"{blog.summary or ''} " * 4 + \
                           f"{' '.join(blog.tags)} " * 3 + \
                           f"{blog.content[:300]}"  # Limit content to avoid overwhelming
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Handle small datasets
        if len(self.processed_content) < 2:
            print(f"Skipping TF-IDF for small dataset ({len(self.processed_content)} blogs)")
            self.tfidf_matrix = None
            self.tfidf_vectorizer = None
            return
        
        # Initialize TF-IDF vectorizer with reasonable parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.85,
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True
        )
        
        # Build TF-IDF matrix
        if self.processed_content:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
                print(f"Built TF-IDF matrix for {len(self.processed_content)} blogs with {self.tfidf_matrix.shape[1]} features")
            except Exception as e:
                print(f"Error building TF-IDF matrix: {e}")
                # Fallback to simpler configuration
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                try:
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
                    print(f"Built fallback TF-IDF matrix for {len(self.processed_content)} blogs")
                except Exception as e2:
                    print(f"Failed to build TF-IDF matrix: {e2}")
                    self.tfidf_matrix = None
                    self.tfidf_vectorizer = None

    def get_all_blogs(self) -> BlogList:
        """Get all blogs"""
        return BlogList(blogs=self.blogs, total=len(self.blogs))

    def search_blogs(self, query: str, limit: int = 10) -> BlogList:
        """Search blogs using optimized scoring for the 35 specific questions"""
        if not query or not self.blogs:
            return BlogList(blogs=[], total=0)
        
        scored_blogs = []
        
        for i, blog in enumerate(self.blogs):
            # Calculate scores with priority on exact question matching
            exact_question_score = self._calculate_exact_question_match_score(query, blog)
            eldercare_score = self._calculate_eldercare_relevance_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    processed_query = self._preprocess_text(query)
                    if processed_query:
                        query_vector = self.tfidf_vectorizer.transform([processed_query])
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                        tfidf_score = similarities[i] * 15  # Scale for better differentiation
                except Exception as e:
                    print(f"Error calculating TF-IDF score: {e}")
                    tfidf_score = 0.0
            
            # Combined scoring with heavy emphasis on exact question matching
            if exact_question_score > 0:
                # If we have an exact question match, prioritize it heavily
                combined_score = exact_question_score + (eldercare_score * 0.3) + (direct_keyword_score * 0.2)
            elif tfidf_score > 0.5:  # If TF-IDF found meaningful similarity
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.3) + (tfidf_score * 0.3)
            else:  # Fall back to eldercare and keyword matching
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.4)
            
            # Lower threshold for search to return more results
            if combined_score > 1.0:
                scored_blogs.append((blog, combined_score))
        
        # Sort by score and return top results
        scored_blogs.sort(key=lambda x: x[1], reverse=True)
        filtered_results = [blog for blog, score in scored_blogs[:limit]]
        
        return BlogList(blogs=filtered_results, total=len(filtered_results))

    def get_blogs_by_category(self, category: str) -> BlogList:
        """Get blogs filtered by category"""
        filtered_blogs = [blog for blog in self.blogs if blog.category.lower() == category.lower()]
        return BlogList(blogs=filtered_blogs, total=len(filtered_blogs))

    def get_blogs_by_subcategory(self, subcategory: str) -> BlogList:
        """Get blogs filtered by subcategory"""
        filtered_blogs = [blog for blog in self.blogs 
                         if blog.subcategory and blog.subcategory.lower() == subcategory.lower()]
        return BlogList(blogs=filtered_blogs, total=len(filtered_blogs))

    def get_blogs_by_tag(self, tag: str) -> BlogList:
        """Get blogs filtered by tag"""
        filtered_blogs = [blog for blog in self.blogs 
                         if tag.lower() in [t.lower() for t in blog.tags]]
        return BlogList(blogs=filtered_blogs, total=len(filtered_blogs))

    def get_blog_by_source_id(self, source_id: str) -> Optional[BlogEntry]:
        """Get a specific blog by source ID"""
        for blog in self.blogs:
            if blog.source_id == source_id:
                return blog
        return None

    def get_recommendations(self, query: str, limit: int = 5) -> BlogList:
        """Get blog recommendations with optimized scoring for the 35 specific questions"""
        if not query or not self.blogs:
            return BlogList(blogs=[], total=0)
        
        scored_blogs = []
        
        for i, blog in enumerate(self.blogs):
            # Calculate scores with priority on exact question matching
            exact_question_score = self._calculate_exact_question_match_score(query, blog)
            eldercare_score = self._calculate_eldercare_relevance_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    processed_query = self._preprocess_text(query)
                    if processed_query:
                        query_vector = self.tfidf_vectorizer.transform([processed_query])
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                        tfidf_score = similarities[i] * 15  # Consistent scaling
                except Exception as e:
                    print(f"Error calculating TF-IDF score for recommendations: {e}")
                    tfidf_score = 0.0
            
            # Combined scoring with heavy emphasis on exact question matching
            if exact_question_score > 0:
                # If we have an exact question match, prioritize it heavily
                combined_score = exact_question_score + (eldercare_score * 0.2) + (direct_keyword_score * 0.1)
            elif tfidf_score > 0.8:  # Higher TF-IDF threshold for recommendations
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.3) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.7) + (direct_keyword_score * 0.3)
            
            # Lower threshold for recommendations to ensure results
            if combined_score > 2.0:
                scored_blogs.append((blog, combined_score))
        
        # Sort by score and return top results
        scored_blogs.sort(key=lambda x: x[1], reverse=True)
        recommended_blogs = [blog for blog, score in scored_blogs[:limit]]
        
        return BlogList(blogs=recommended_blogs, total=len(recommended_blogs))

    def recommend_best_blog_with_score(self, query: str) -> tuple[Optional[BlogEntry], float]:
        """Get the best blog recommendation with optimized scoring for the 35 specific questions"""
        if not query or not self.blogs:
            return None, 0.0
        
        best_blog = None
        best_score = 0.0
        
        for i, blog in enumerate(self.blogs):
            # Calculate scores with priority on exact question matching
            exact_question_score = self._calculate_exact_question_match_score(query, blog)
            eldercare_score = self._calculate_eldercare_relevance_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    processed_query = self._preprocess_text(query)
                    if processed_query:
                        query_vector = self.tfidf_vectorizer.transform([processed_query])
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                        tfidf_score = similarities[i] * 15  # Consistent scaling
                except Exception as e:
                    print(f"Error calculating TF-IDF score for best recommendation: {e}")
                    tfidf_score = 0.0
            
            # Combined scoring with heavy emphasis on exact question matching
            if exact_question_score > 0:
                # If we have an exact question match, prioritize it heavily
                combined_score = exact_question_score + (eldercare_score * 0.1) + (direct_keyword_score * 0.1)
            elif tfidf_score > 1.0:  # Meaningful TF-IDF similarity
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.2) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.8) + (direct_keyword_score * 0.2)
            
            if combined_score > best_score:
                best_score = combined_score
                best_blog = blog
        
        return best_blog, best_score

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best blog recommendation source_id with reasonable threshold"""
        blog, score = self.recommend_best_blog_with_score(query)
        
        # Much more reasonable threshold for the 35 specific questions
        min_relevance_score = 5.0  # Higher threshold due to exact question matching
        
        if blog and score >= min_relevance_score:
            return blog.source_id
        
        return None

    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        categories = set(blog.category for blog in self.blogs)
        return sorted(list(categories))
    
    def get_subcategories(self) -> List[str]:
        """Get all unique subcategories"""
        subcategories = set(blog.subcategory for blog in self.blogs if blog.subcategory)
        return sorted(list(subcategories))

    def get_statistics(self) -> Dict:
        """Get blog statistics"""
        categories = {}
        subcategories = {}
        
        for blog in self.blogs:
            categories[blog.category] = categories.get(blog.category, 0) + 1
            if blog.subcategory:
                subcategories[blog.subcategory] = subcategories.get(blog.subcategory, 0) + 1
        
        return {
            "total_blogs": len(self.blogs),
            "categories": categories,
            "subcategories": subcategories,
            "featured_count": len([blog for blog in self.blogs if blog.featured]),
            "authors": list(set(blog.author for blog in self.blogs))
        }

    def recommend_blogs_for_product(self, product_handle: str, limit: int = 3) -> BlogList:
        """
        Blog recommendations for products using optimized eldercare matching.
        
        Args:
            product_handle: The product handle.
            limit: Maximum number of blogs to return.
            
        Returns:
            BlogList with recommended blogs.
        """
        try:
            # Convert product handle to search query
            search_query = product_handle.replace('-', ' ')
            
            # Find relevant blogs using optimized recommendation system
            return self.get_recommendations(search_query, limit)
            
        except Exception as e:
            print(f"Error recommending blogs for product: {str(e)}")
            return BlogList(blogs=[], total=0) 